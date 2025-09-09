import copy
import logging
import random

import numpy as np
from sklearn.metrics import roc_auc_score
from timm.layers import SelectAdaptivePool2d
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from timm.models.swin_transformer import swin_base_patch4_window7_224

from convs.swin_vit_adapter import set_adapter
from models.base import BaseLearner
# from convs.vit_adapter_dynamic import vit_adapter_patch16_224
from utils.data_manager import DummyDataset
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random


EPSILON = 1e-8
epochs = 20
lrate = 5e-4
step_size = 10
lrate_decay = 0.5
batch_size = 64
weight_decay = 2e-4
num_workers = 8
T = 2


def get_old_net(model):
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    return old_model


class SwinViT(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_network_list = []
        self._network = swin_base_patch4_window7_224(True)
        num_ftrs = self._network.head.fc.in_features
        self._network.head.fc = nn.Linear(num_ftrs, 2)
        set_adapter(self._network)
        for name, p in self._network.named_parameters():
            if "adapter" in name or "head" in name:
            # if "head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.best_model_path = []

    def after_task(self):
        self._old_network = get_old_net(self._network)
        self._known_classes = self._total_classes

        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

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
            np.arange(0, self._total_classes), source="val", mode="val", resize_size=self.args["resize_size"]
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

        if self._cur_task == 0:
            if self.args['skip']:
                load_model(self._network, self.args, self.best_model_path)
                self._network.to(self._device)
            else:
                self._train(self.train_loader, self.val_loader)
        else:
            self._train(self.train_loader, self.val_loader)

        self._eval(self.test_loader, self.out_loader, data_manager)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        e_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, weight_decay=weight_decay)
        e_scheduler = optim.lr_scheduler.StepLR(optimizer=e_optimizer, step_size=step_size, gamma=lrate_decay)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            self._init_train(train_loader, val_loader, e_optimizer, e_scheduler)
        else:
            self._update_representation(train_loader, val_loader, e_optimizer, e_scheduler)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        torch.use_deterministic_algorithms(False)
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)
                cls_loss = F.cross_entropy(logits, targets)
                loss = cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = self._compute_accuracy(self._network, val_loader)
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            info = "Task {}, Epoch {}/{} => cls {:.4f}, train_acc {:.2f}, val_acc {:.2f} lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _get_kd_weight(self, old_logits, targets):
        weights = torch.zeros_like(targets).type(torch.float32)
        prob = torch.softmax(old_logits, dim=1)
        for i in range(len(targets)):
            weights[i] = prob[i][targets[i].data]
        return weights

    def _weight_KD_loss(self, pred, soft, T, weights):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        multi = torch.mul(soft, pred)
        return -1 * (weights * multi.sum(dim=1)).sum() / pred.shape[0]

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            correct, total = 0, 0
            kd_losses = 0.0
            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)
                old_logits = self._old_network(inputs)
                cls_loss = F.cross_entropy(logits, targets)
                # weights = self._get_kd_weight(old_logits, targets)
                # kd_loss = self._weight_KD_loss(logits, old_logits, T, weights)
                # kd_loss = _KD_loss(logits, old_logits, T)
                # loss = cls_loss + kd_loss
                loss = cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                # kd_losses += kd_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = self._compute_accuracy(self._network, val_loader)
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            info = "Task {}, Epoch {}/{} => cls {:.4f}, kd {:.4f}, train_acc {:.2f}, val_acc {:.2f} lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                kd_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, order) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            if type == 'out' or self.args['run_type'] == 'train' or self.args['run_type'] == 'train_bak':
                with torch.no_grad():
                    outputs = self._network(inputs)
                y_order.append(orders.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            else:
                for class_id in self.args['test_class']:
                    if class_id in orders.cpu().numpy():
                        with torch.no_grad():
                            outputs = self._network(inputs)
                        y_order.append(orders.cpu().numpy())
                        y_pred.append(outputs.cpu().numpy())
                        y_true.append(targets.cpu().numpy())
        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval(self, test_loader, out_loader, data_manager):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        self._network.eval()
        orders, _out_k, _pred_k, _labels_k, cnn_accy, nme_accy = self.eval_task(type='test')
        auc = self.each_auc(_out_k[:, 1], _labels_k, orders)
        logging.info("closed-set auc: {}".format(auc))
        eval_type = 'NME' if nme_accy else 'CNN'
        if eval_type == 'NME':
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
            logging.info("closed-set NME: {}".format(nme_accy["grouped"]))
        else:
            logging.info("closed-set No NME accuracy.")
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
        set_random(self.args['seed'])
        if self._cur_task == data_manager.nb_tasks - 1:
            self.eval_open('CNN', final=True)
        else:
            self.eval_open('CNN', final=False)

    def each_auc(self, prediction, labels, orders, increment=1):
        all_auc = {}
        all_auc["total"] = np.around(
            roc_auc_score(labels, prediction) * 100, decimals=2
        )
        # Grouped accuracy
        for class_id in range(0, np.max(orders) + 1, increment):
            label = "{}".format(
                str(class_id).rjust(2, "0")
            )
            idxes = np.where(orders == class_id)
            if len(idxes[0]) == 0:
                all_auc[label] = 'nan'
                continue
            all_auc[label] = np.around(
                roc_auc_score(labels[idxes], prediction[idxes]) * 100, decimals=2
            )
        return all_auc

    @property
    def feature_dim(self):
        return self._network.head.in_features

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
                self._network.forward_features(_inputs.to(self._device))[:, 0].view(_inputs.size()[0], -1)
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
        # uniques = np.unique(selected_exemplars, axis=0)
        # print('Unique elements: {}'.format(len(uniques)))
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


