import copy
import logging
import math
import random
from time import sleep
from utils.loss import SupConLoss
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base_new import BaseLearner
from convs.xception_m2tr import xception
from models.m2 import m2
from pytorch_metric_learning.losses import SupConLoss
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random
EPSILON = 1e-8
epochs = 20
lrate = 1e-4
step_size = 5
lrate_decay = 0.5
batch_size = 64
weight_decay = 2e-4
num_workers = 8
T = 20


class m2_train(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = m2()
        self.best_model_path = []
        self.data_memory_high, self.targets_memory_high = np.array([]), np.array([])

    def after_task(self):
        self._old_network = self._network.copy().freeze()
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
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._train(self.train_loader, self.val_loader)
                self._compute_accuracy(self._network, self.val_loader)
        else:
            self._train(self.train_loader, self.val_loader)
        self._eval(self.test_loader, self.out_loader, data_manager)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, val_loader, optimizer, scheduler)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses, consup_losses = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out["logits"]
                features = out["features"]

                cls_loss = F.cross_entropy(logits, targets.long())
                consup_loss = 0.1 * _ConSup_loss(features, targets)

                loss = cls_loss + consup_loss
                # loss = cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                consup_losses += consup_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = self._compute_accuracy(self._network, val_loader)
            info = "Task {}, Epoch {}/{} => cls_loss {:.4f}, consup_loss {:.4f} train_accy {:.2f}, val_accy {:.2f} lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                consup_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            if patience >= 5:
                logging.info("early stop! epoch {}/{}".format(epoch + 1, epochs))
                break
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

    def _weight_FD_loss(self, student_feature, teacher_feature, weights):
        student_feature = F.normalize(student_feature)
        teacher_feature = F.normalize(teacher_feature)
        student_feature, teacher_feature = torch.broadcast_tensors(student_feature, teacher_feature)
        bs, embs = student_feature.size()

        sub = torch.sub(student_feature, teacher_feature)
        square = torch.square(sub)
        harm = torch.mean(square, dim=1)
        weight_harm = torch.mul(harm, weights)
        result = torch.sum(weight_harm) / bs
        return result

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        set_random(self.args['seed'])
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses, consup_losses, kd_losses, fd_losses = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (paths, inputs, targets, orders) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)
                logits = out["logits"]
                features = out["features"]

                old_out = self._old_network(inputs)
                old_logits = old_out["logits"]
                old_features = old_out["features"]
                # weights = self._get_kd_weight(old_logits, targets)

                cls_loss = F.cross_entropy(logits, targets.long())
                consup_loss = _ConSup_loss(features, targets)
                kd_loss = _KD_loss(logits, old_logits, T)
                # kd_loss = self._weight_KD_loss(logits, old_logits, T, weights)
                fd_loss = _FD_loss(features, old_features)
                # fd_loss = self._weight_FD_loss(features, old_features, weights)

                loss = cls_loss + 0.1 * consup_loss + kd_loss + 0.01 * fd_loss
                # loss = cls_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                consup_losses += consup_loss.item()
                kd_losses += kd_loss.item()
                fd_losses += fd_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = self._compute_accuracy(self._network, val_loader)
            info = "Task {}, Epoch {}/{} => cls_loss {:.4f}, consup_loss {:.4f}, kd_loss {:.4f}, fd_loss {:.4f} train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                consup_losses / len(train_loader),
                kd_losses / len(train_loader),
                fd_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            # if patience >= 5:
            #     logging.info("early stop! epoch {}/{}".format(epoch + 1, epochs))
            #     break
            prog_bar.set_description(info)
        logging.info(info)

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
        #记得解开
        test_auc = roc_auc_score(_labels_k, _out_k[:, 1])
        print(test_auc)
        auc = self.each_auc(_out_k[:, 1], _labels_k, orders)
        logging.info("closed-set auc: {}".format(auc))


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
    def exemplar_size(self):
        data_length = len(np.concatenate((self._real_data_memory, self._fake_data_memory, self.data_memory_high)))
        targets_length = len(
            np.concatenate((self._real_targets_memory, self._fake_targets_memory, self.targets_memory_high)))
        assert data_length == targets_length, "Exemplar size error."
        return data_length

    def _get_memory(self):
        if len(self.data_memory_high) and len(self._real_data_memory) == 0 and len(self._fake_data_memory):
            return None
        else:
            return (np.concatenate((self._real_data_memory, self._fake_data_memory, self.data_memory_high)),
                    np.concatenate((self._real_targets_memory, self._fake_targets_memory, self.targets_memory_high)))

    def build_rehearsal_memory(self, data_manager, per_class):
        self._reduce_exemplar(data_manager, per_class // 2)
        self._reduce_exemplar_high(per_class // 2)
        self._construct_exemplar(data_manager, per_class // 2)
        self._construct_exemplar_high(data_manager, per_class // 2)
        # self._reduce_exemplar(data_manager, per_class)
        # self._reduce_exemplar_high(per_class // 2)
        # self._construct_exemplar(data_manager, per_class)
        # self._construct_exemplar_high(data_manager, per_class // 2)

    def _reduce_exemplar_high(self, m):
        logging.info("Reducing high entropy exemplars... ({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self.data_memory_high), copy.deepcopy(self.targets_memory_high)
        self.data_memory_high, self.targets_memory_high = np.array([]), np.array([])
        for class_idx in range(self._known_classes):
            set_random(self.args['seed'])
            logging.info("Reducing high entropy exemplars for class {}".format(class_idx))
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self.data_memory_high = (
                np.concatenate((self.data_memory_high, dd))
                if len(self.data_memory_high) != 0
                else dd
            )
            self.targets_memory_high = (
                np.concatenate((self.targets_memory_high, dt))
                if len(self.targets_memory_high) != 0
                else dt
            )

    def _construct_exemplar_high(self, data_manager, m):
        logging.info("Constructing high entropy exemplars... ({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            set_random(self.args['seed'])
            logging.info("Constructing high entropy exemplars for class {}".format(class_idx))
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                resize_size=self.args["resize_size"],
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True,
            )
            entropy_infos = self._get_entropy_info(idx_loader)
            entropy_high_order = np.argsort(-entropy_infos)
            sort_data = data[entropy_high_order]
            self.data_memory_high = np.concatenate((self.data_memory_high, sort_data[:m]))
            self.targets_memory_high = np.concatenate((self.targets_memory_high, np.full(m, class_idx)))

    def _extract_vectors(self, loader):
        real_vectors, fake_vectors, targets, orders = [], [], [], []
        for _, _inputs, _targets, _orders in loader:
            _targets = _targets.numpy()
            _orders = _orders.numpy()
            _vectors = tensor2numpy(
                self._network(_inputs.to(self._device))['features']
            )
            real_vectors.append(_vectors[np.where(_targets == 0)])
            fake_vectors.append(_vectors[np.where(_targets == 1)])
            orders.append(_orders)
            targets.append(_targets)

        if len(real_vectors) == 0:
            return np.concatenate(orders), 0, np.concatenate(fake_vectors), np.concatenate(targets)
        elif len(fake_vectors) == 0:
            return np.concatenate(orders), np.concatenate(real_vectors), 0, np.concatenate(targets)
        else:
            return np.concatenate(orders), np.concatenate(real_vectors), np.concatenate(fake_vectors), np.concatenate(
                targets)

    def _get_entropy_info(self, loader):
        self._network.eval()
        entropy_infos = []
        for _, inputs, targets, orders in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            prob = nn.functional.softmax(outputs.data, dim=1)
            for i in range(len(inputs)):
                prob[i][0] = 1e-40 if outputs[i][0].item() <= -40 else prob[i][0]
                prob[i][1] = 1e-40 if outputs[i][1].item() <= -40 else prob[i][1]
                entropy_info = -(prob[i][0].item() * math.log(prob[i][0].item()) + prob[i][1].item() * math.log(
                    prob[i][1].item()))
                entropy_infos.append(entropy_info)
        return np.array(entropy_infos)

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


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


# def _KD_loss(y, labels, teacher_scores, T, alpha):
#     return nn.KLDivLoss()(F.log_softmax(y / T, dim=1), F.log_softmax(teacher_scores / T, dim=1)) * (
#             T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def _FD_loss(student_feature, teacher_feature):
    return torch.nn.functional.mse_loss(F.normalize(student_feature, dim=1), F.normalize(teacher_feature, dim=1),
                                        reduction='mean')


def _ConSup_loss(features, labels):
    criterion_supcon = SupConLoss()
    loss = criterion_supcon(features, labels)
    return loss
