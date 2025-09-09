import copy
import logging
import math
import os
import random
from time import sleep

import pandas as pd

from utils.data_manager import DataManager
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from convs.xception_dmp import xception
from utils.loss import SupConLoss
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random
from trainer_bak import _train

EPSILON = 1e-8
epochs = 20
lrate = 2e-4
step_size = 5
lrate_decay = 0.95
batch_size = 64
weight_decay = 2e-4
num_prototypes = 20
replay_size = 500
stage_all = 4

# 线程数
num_workers = 8


class DMP_xception(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # 记得选一下pth路径
        self._network = xception(num_classes=2)

        # 初始化存储最佳模型的路径
        self.best_model_path = []
        self.incre_classes = 4
        self.known_classes = 0
        # 初始化输入数据和对应标签
        self.num_prototypes = num_prototypes
        self.memory_samples = []
        self.samples_class = 0
        self.batch_size = batch_size
        self.hards = pd.DataFrame(columns=['path', 'label', 'distance'])

    def after_task(self):
        self._old_network = copy.deepcopy(self._network)
        self._known_classes = self._total_classes
        for param in self._old_network.parameters():
            param.requires_grad = False
        self._old_network.eval()
        logging.info("Exemplar size: {}".format(len(self.hards)))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._network.set_stage(self._cur_task)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        train_dataset = data_manager.get_dataset(
            # 索引
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            resize_size=self.args["resize_size"],
            appendent=(self.hards['path'], self.hards['label'])
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )

        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="val", mode="val", resize_size=self.args["resize_size"]
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test", resize_size=self.args["resize_size"]
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if self._cur_task == 0:
            if self.args['skip']:
                load_model(self._network, self.args, self.best_model_path)
                self._network.to(self._device)
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._train(self.train_loader, self.val_loader)
        else:
            self._train(self.train_loader, self.val_loader)
        self._eval(self.test_loader, data_manager)
        self.hards = self.build_rehearsal_memory(data_manager, train_dataset, self.samples_class)

    def build_rehearsal_memory(self, data_manager, train_dataset, per_class):
        """
               使用 Prototype-Guided Replay (PGR) 策略构建重放集。
               参数:
               - data_manager: 数据管理器，负责提供数据集
               - per_class: 每个原型选择的样本数量
               """
        logging.info("Building rehearsal memory using Prototype-Guided Replay (PGR) strategy.")
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        train_dataset = data_manager.get_dataset(
            # 索引
            np.arange(self._cur_task, self._cur_task + 1),
            source="train",
            mode="test",
            resize_size=self.args["resize_size"],
            appendent=(self.hards['path'], self.hards['label'])
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        all_labels = []
        all_distance = []
        all_img_path = []
        with torch.no_grad():
            for i, (img_path, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                _, _, p_distance, _, _ = self._network(inputs, targets, return_feat=True)
                all_labels.append(order)
                all_distance.append(p_distance)
                all_img_path.extend(img_path)
        all_distance = torch.cat(all_distance, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        df_distance = pd.DataFrame(all_distance.cpu().numpy())
        sample_per_p = 500 // (4 * (self._cur_task + 2))
        cur_hards = pd.DataFrame()
        for i in range(df_distance.shape[1]):
            min_indices = df_distance.nsmallest(sample_per_p, i).index
            cur_group = pd.DataFrame({
                'path': [all_img_path[idx] for idx in min_indices],
                'label': all_labels[min_indices].cpu().numpy(),
                'distance': df_distance.loc[min_indices, i].values
            })
            cur_hards = pd.concat([cur_hards, cur_group], ignore_index=True)
        return cur_hards


        # if self.replay_set_data is not None:
        #     self.replay_set_data = np.concatenate((self.replay_set_data, data_total), axis=0)
        #     self.replay_set_targets = np.concatenate((self.replay_set_targets, targets_total), axis=0)
        # else:
        #     self.replay_set_data = data_total
        #     self.replay_set_targets = targets_total

    # 数据加载进去，开train
    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate,
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
            self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate,
                                          betas=(0.9, 0.95), eps=1e-08, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
            self._update_representation(train_loader, val_loader, optimizer, scheduler)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, order) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs, _ = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []
        for _, epoch in enumerate(prog_bar):
            # 设置成train()模式
            self._network.train()
            closses = 0.0
            plosses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                ploss, closs, _, logits = self._network(inputs, targets)
                loss = ploss * 0.1 + closs
                closses += closs
                plosses += ploss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算准确度和当前已经处理过的数数目
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = self._compute_accuracy(self._network, val_loader)
            info = "Task {}, Epoch {}/{} => closs {:.4f}, ploss {:.4f},  train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                closses / len(train_loader),
                plosses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        set_random(self.args['seed'])
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            plosses, kd_losses = 0.0, 0.0
            correct, total = 0, 0
            for i, (paths, inputs, targets, orders) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                ploss, _, p_distance, outputs, feature = self._network(inputs, targets, return_feat=True)
                _, _, _, outputs_t, feature_t = self._old_network(inputs, targets, return_feat=True)
                kd_loss = _KD_loss(outputs, outputs_t, T=20.0)

                # print(kl_loss," ",ce_loss)
                loss = ploss + 0.2 * kd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                plosses += ploss.item()
                kd_losses += kd_loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = self._compute_accuracy(self._network, val_loader)
            info = "Task {}, Epoch {}/{} => ploss {:.4f}, kd_loss {:.4f} train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                plosses / len(train_loader),
                kd_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits, _ = self._network(inputs)
            y_order.append(orders.cpu().numpy())
            y_pred.append(logits.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def eval_task(self, type='test'):
        loader = self.test_loader

        y_order, y_out, y_true = self._eval_cnn(type, loader)
        y_pred = np.argmax(y_out, axis=1)
        cnn_accy = self._evaluate(y_order, y_pred, y_true)

        _save_dir = os.path.join(self.args['logfilename'], "task" + str(self._cur_task))
        return y_order, y_out, y_pred, y_true, cnn_accy

    def _eval(self, test_loader, datamanager):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        self._network.eval()
        orders, _out_k, _pred_k, _labels_k, cnn_accy = self.eval_task(type='test')
        logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
        # test_auc = roc_auc_score(_labels_k, _out_k[:, 1])
        # auc = self.each_auc(_out_k[:, 1], _labels_k, orders)
        # logging.info("closed-set auc: {}".format(auc))


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

    def add_to_memory(self, samples):
        """
        将新的样本添加到内存中，用于重放。

        参数:
        - samples: Sample 对象的列表
        """
        self.memory_samples.extend(samples)


def generate_random_orthogonal_matrix(in_channels, num_prototypes):
    rand_mat = np.random.random(size=(in_channels, num_prototypes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_prototypes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_prototypes))))
    return orth_vec

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
