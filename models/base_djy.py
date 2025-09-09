import copy
import logging
from time import sleep

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, set_random
# from utils.draw import plot_confusion
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from models.retrive_shallow import RS_eval

EPSILON = 1e-8
batch_size = 100
num_workers = 8

#基础学习器
class BaseLearner(object):
    def __init__(self, args):
        #将传入的args作为自己的args
        self.args = args
        #表示任务参数编号，为-1就是没开始
        self._cur_task = -1
        #初始化已知类别数为0
        self._known_classes = 0
        #初始化总类别数为0
        self._total_classes = 0
        #初始化神经网络模型为none
        self._network = None
        self._old_network = None
        #初始化真实目标和伪造目标记忆为空
        self._real_data_memory, self._fake_data_memory = np.array([]), np.array([])
        self._real_targets_memory, self._fake_targets_memory = np.array([]), np.array([])
        #初始化存储记忆数据的总容量
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        #决定是否使用固定大小的内存
        self._fixed_memory = args.get("fixed_memory", False)
        #gpu设置
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]


    #计算存储在记忆中的样本总数
    @property
    def exemplar_size(self):
        data_length = len(np.concatenate((self._real_data_memory, self._fake_data_memory)))
        targets_length = len(np.concatenate((self._real_targets_memory, self._fake_targets_memory)))
        assert data_length == targets_length, "Exemplar size error."
        return data_length

    #每个类别的示例数量
    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes




    def _evaluate(self, y_order, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_order, y_pred, y_true, self._known_classes, self.args['increment'])
        ret["grouped"] = grouped
        return ret

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
        # plot_confusion(_confusion_img_path, confusion_matrix(y_true, y_pred))
        return y_order, y_out, y_pred, y_true, cnn_accy, nme_accy

    def eval_out_task(self, type='out'):
        orders, _out_u, _pred_u, _labels_u = self.eval_task(type)
        acc_score = np.around(np.sum(_pred_u == _labels_u) * 100 / len(_labels_u), decimals=2)
        fpr, tpr, _ = roc_curve(_labels_u, _out_u[:, 1])
        auc_score = np.around(auc(fpr, tpr) * 100, decimals=2)
        _save_dir = os.path.join(self.args['logfilename'], "task" + str(self._cur_task))
        os.makedirs(_save_dir, exist_ok=True)
        _save_path = os.path.join(_save_dir, "open-set-result.csv")
        with open(_save_path, "a+") as f:
            f.write(f"{type}, {'ACC:' + str(acc_score)},{'OSCR:' + str(auc_score)}\n")
        out_cnn_accy = self._evaluate(orders, _pred_u, _labels_u)
        logging.info("open-set CNN: {}".format(out_cnn_accy["grouped"]))
        return acc_score, auc_score


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, order) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        print(1111)
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            if type == 'out' or self.args['run_type'] == 'train' or self.args['run_type'] == 'train_bak' or type=='replay':
                with torch.no_grad():
                    outputs = self._network(inputs)["logits"]
                y_order.append(orders.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            else:
                for class_id in self.args['test_class'][self._cur_task]:
                    if class_id in orders.cpu().numpy():
                        with torch.no_grad():
                            outputs = self._network(inputs)["logits"]
                        y_order.append(orders.cpu().numpy())
                        y_pred.append(outputs.cpu().numpy())
                        y_true.append(targets.cpu().numpy())
        if(type=='replay'):
            print(len(y_pred))
            return y_pred
        else :
            return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


