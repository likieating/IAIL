import copy
import os

import random
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from models.base import BaseLearner
from utils.draw import plot_confusion
from utils.toolkit import tensor2numpy, load_model, set_random, accuracy, early_stop
from convs.sinet import SiNet

EPSILON = 1e-8
epochs = 40
lrate = 0.01
step_size = 10
lrate_decay = 0.5
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2



class SPrompts(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        self._network = SiNet(args)

        self.args = args

        self.topk = 2  # origin is 5
        self.class_num = self._network.class_num
        self.best_model_path = []
        self.all_keys = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            resize_size=self.args["resize_size"],
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="val", mode="val", resize_size=self.args["resize_size"]
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test", resize_size=self.args["resize_size"]
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        out_dataset = data_manager.get_dataset(
            None, source="test_out", mode="test", resize_size=self.args["resize_size"]
        )
        self.out_loader = DataLoader(
            out_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        if self._cur_task == 0 and self.args['skip']:
            state = load_model(self._network, self.args, self.best_model_path)
            self._network.to(self._device)
            for name, param in self._network.named_parameters():
                param.requires_grad_(False)
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "prompt_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
        else:
            self._train(self.train_loader, self.val_loader)
        self.clustering(self.train_loader)
        self._eval(self.test_loader, self.out_loader, data_manager)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                param.requires_grad_(True)
            if "prompt_pool" + "." + str(self._network.numtask - 1) in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        # optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=lrate, weight_decay=weight_decay)
        optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)

        self.run_epoch = epochs
        self.train_function(train_loader, val_loader, optimizer, scheduler)

    def train_function(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        best_acc, patience, self.best_model_path = 0, 0, []
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets, orders) in enumerate(train_loader):
                inputs, targets, orders = inputs.to(self._device), targets.to(self._device), orders.to(self._device)

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = self._compute_accuracy_domain(self._network, val_loader)
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Val_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, val_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets, orders) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_order, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_order, y_pred, y_true, self._known_classes, self.args['increment'])
        # grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        for _, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            orders = orders.to(self._device)

            with torch.no_grad():
                feature = self._network.extract_vector(inputs)

                taskselection = []
                for task_centers in self.all_keys:
                    tmpcentersbatch = []
                    for center in task_centers:
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

                selection = torch.vstack(taskselection).min(0)[1]

                outputs = self._network.interface(inputs, selection)
            # predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            y_order.append(orders.cpu().numpy())

        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval(self, test_loader, out_loader, data_manager):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        self._network.eval()
        orders, _out_k, _pred_k, _labels_k, cnn_accy, nme_accy = self.eval_task(type='test')
        eval_type = 'NME' if nme_accy else 'CNN'
        if eval_type == 'NME':
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
            logging.info("closed-set NME: {}".format(nme_accy["grouped"]))
        else:
            logging.info("closed-set No NME accuracy.")
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
        # set_random(self.args['seed'])
        # if self._cur_task == data_manager.nb_tasks - 1:
        #     self.eval_open('CNN', final=True)
        # else:
        #     self.eval_open('CNN', final=False)

    def eval_task(self, type='test'):
        if type == 'out':
            loader = self.out_loader
        else:
            loader = self.test_loader

        y_order, y_out, y_true = self._eval_cnn(type, loader)
        y_pred = np.argmax(y_out, axis=1)
        cnn_accy = self._evaluate(y_order, y_pred, y_true)

        if 'out' in type:
            return y_order, y_out, y_pred, y_true

        if hasattr(self, "_class_means") and self.args['model_name'] != 'coil':
            # y_order, y_out, y_true = self._eval_nme(loader, self._class_means)
            # y_pred = np.argmin(y_out, axis=1)
            # nme_accy = self._evaluate(y_order, y_pred, y_true)
            nme_accy = None
        else:
            nme_accy = None

        _save_dir = os.path.join(self.args['logfilename'], "task" + str(self._cur_task))
        os.makedirs(_save_dir, exist_ok=True)
        _pred_path = os.path.join(_save_dir, "closed_set_pred.npy")
        _target_path = os.path.join(_save_dir, "closed_set_target.npy")
        np.save(_pred_path, y_pred)
        np.save(_target_path, y_true)
        _confusion_img_path = os.path.join(_save_dir, "closed_set_conf.png")
        plot_confusion(_confusion_img_path, confusion_matrix(y_true, y_pred))

        return y_order, y_out, y_pred, y_true, cnn_accy, nme_accy

    @property
    def feature_dim(self):
        return 768

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def build_rehearsal_memory(self, data_manager, per_class):
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        real_dummy_data, real_dummy_targets = copy.deepcopy(self._real_data_memory), copy.deepcopy(
            self._real_targets_memory
        )
        fake_dummy_data, fake_dummy_targets = copy.deepcopy(self._fake_data_memory), copy.deepcopy(
            self._fake_targets_memory
        )
        self._class_means = np.zeros((2, self._total_classes, self.feature_dim))
        self._real_data_memory, self._real_targets_memory = self._reduce_process(real_dummy_data, real_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 0)
        self._fake_data_memory, self._fake_targets_memory = self._reduce_process(fake_dummy_data, fake_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 1)

    def _extract_vectors(self, loader):
        real_vectors, fake_vectors, targets, orders = [], [], [], []
        for _, _inputs, _targets, _orders in loader:
            _targets = _targets.numpy()
            _orders = _orders.numpy()
            _vectors = tensor2numpy(
                self._network.extract_vector(_inputs.to(self._device)).view(_inputs.size()[0], -1)
            )
            real_vectors.append(_vectors[np.where(_targets == 0)])
            fake_vectors.append(_vectors[np.where(_targets == 1)])
            orders.append(_orders)
            targets.append(_targets)
        if len(real_vectors) == 0:
            return np.concatenate(orders), 0, np.concatenate(fake_vectors), np.concatenate(
                targets)
        elif len(fake_vectors) == 0:
            return np.concatenate(orders), np.concatenate(real_vectors), 0, np.concatenate(targets)
        else:
            return np.concatenate(orders), np.concatenate(real_vectors), np.concatenate(fake_vectors), np.concatenate(
                targets)

    def _reduce_process(self, dummy_data, dummy_targets, data_memory, targets_memory, data_manager, m, label):
        for class_idx in range(self._known_classes):
            set_random(self.args['seed'])
            logging.info("Reducing exemplars for label {} data class {}".format(label, class_idx))
            mask = np.where(dummy_targets == class_idx)[0]

            # trsf = transforms.Compose([transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
            # dataset = DummyDataset(None, np.array(dummy_data[mask]), np.array(dummy_targets[mask]), self.args["resize_size"], trsf)
            # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            # if label:
            #     _, _, vectors, _ = self._extract_vectors(loader)
            # else:
            #     _, vectors, _, _ = self._extract_vectors(loader)
            # index, _ = FPS(vectors, m)
            # dd = dummy_data[mask][index]
            # dt = dummy_targets[mask][:m]

            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

            data_memory = (
                np.concatenate((data_memory, dd))
                if len(data_memory) != 0
                else dd
            )
            targets_memory = (
                np.concatenate((targets_memory, dt))
                if len(targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt), resize_size=self.args["resize_size"]
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            # _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            vectors = real_vectors if not label else fake_vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[label, class_idx, :] = mean
        return data_memory, targets_memory

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            set_random(self.args['seed'])
            logging.info("Constructing exemplars for class {}".format(class_idx))
            real_data, fake_data = [], []
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                resize_size=self.args["resize_size"],
                ret_data=True
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            for i in range(len(data)):
                if 'real' in data[i]:
                    real_data.append(data[i])
                else:
                    fake_data.append(data[i])
            real_data = np.array(real_data)
            fake_data = np.array(fake_data)

            self._construct_process(class_idx, m // 2, real_vectors, data_manager, real_data, 0)
            self._construct_process(class_idx, m // 2, fake_vectors, data_manager, fake_data, 1)


    def _construct_process(self, class_idx, per_class, vectors, data_manager, data, label):
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # index, selected_vectors = FPS(vectors, per_class)
        # selected_exemplars = data[index]

        class_mean = np.mean(vectors, axis=0)

        # Select
        selected_exemplars = []
        exemplar_vectors = []  # [n, feature_dim]
        for k in range(1, per_class + 1):
            S = np.sum(
                exemplar_vectors, axis=0
            )  # [feature_dim] sum of selected exemplars vectors
            mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
            if i >= len(data):
                i = random.randint(0, len(data) - 1)
            selected_exemplars.append(
                np.array(data[i])
            )  # New object to avoid passing by inference
            exemplar_vectors.append(
                np.array(vectors[i])
            )  # New object to avoid passing by inference

            vectors = np.delete(
                vectors, i, axis=0
            )  # Remove it to avoid duplicative selection
            data = np.delete(
                data, i, axis=0
            )  # Remove it to avoid duplicative selection

            if len(vectors) == 0:
                break

        selected_exemplars = np.array(selected_exemplars)
        # exemplar_targets = np.full(m, class_idx)
        exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
        set_random(self.args['seed'])
        if not label:
            self._real_data_memory = (
                np.concatenate((self._real_data_memory, selected_exemplars))
                if len(self._real_data_memory) != 0
                else selected_exemplars
            )
            self._real_targets_memory = (
                np.concatenate((self._real_targets_memory, exemplar_targets))
                if len(self._real_targets_memory) != 0
                else exemplar_targets
            )
        else:
            self._fake_data_memory = (
                np.concatenate((self._fake_data_memory, selected_exemplars))
                if len(self._fake_data_memory) != 0
                else selected_exemplars
            )
            self._fake_targets_memory = (
                np.concatenate((self._fake_targets_memory, exemplar_targets))
                if len(self._fake_targets_memory) != 0
                else exemplar_targets
            )

        # Exemplar mean
        idx_dataset = data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            resize_size=self.args["resize_size"],
            appendent=(selected_exemplars, exemplar_targets),
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
        vectors = real_vectors if not label else fake_vectors
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)

        self._class_means[label, class_idx, :] = mean
