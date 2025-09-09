import copy
import logging
import math
import random
from convs.xception import xception
import cv2
import numpy as np
from .multi_attention import MAT
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_metric_learning.losses import SupConLoss
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer_v2 import swinv2_base_window8_256
import timm
import os
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random
# from .retrive_shallow import RS_init,contrastive_loss,RS_eval
from models.AGDA import AGDA
from .dataset_based import img_selector

EPSILON = 1e-8
epochs = 10
lrate = 1e-4
step_size = 10
batch_size = 16
lrate_decay = 0.5
db_batch_size=10
weight_decay = 2e-4
num_workers = 8
T = 20


def _compute_accuracy(device, model, loader):
    model.eval()
    correct, total = 0, 0
    for i, (_, inputs, targets, order) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = model(inputs)
        predicts = torch.max(logits, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

def ACC(x,y):
    with torch.no_grad():
        a=torch.max(x,dim=1)[1]
        acc= torch.sum(a==y).float()/x.shape[0]
    #print(y,a,acc)
    return acc

def freeze_vit_except_adapters(model):
    # 首先，冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

def train_loss(loss_pack):
    ensemble_loss_weight = 1
    aux_loss_weight = 0.5
    AGDA_loss_weight = 1
    match_loss_weight = 0.1
    kd_loss_weight=1

    if 'loss' in loss_pack:
        return loss_pack['loss']
    loss=ensemble_loss_weight*loss_pack['ensemble_loss']+aux_loss_weight*loss_pack['aux_loss']
    loss+=AGDA_loss_weight*loss_pack['AGDA_ensemble_loss']+match_loss_weight*loss_pack['match_loss']
    if 'kd_loss' in loss_pack:
        loss+=kd_loss_weight*loss_pack['kd_loss']
    return loss

def get_old_net(model):
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    return old_model
class djy_try2(BaseLearner):
    def __init__(self, args,window_size=3):
        super().__init__(args)
        self.args = args
        self._old_network = None
        self._network =
        self.best_model_path = []
        self._old_adapter=[]
        self.data_selector=img_selector(self._network,self._device,self._cur_task+1)
        self.img_app_real=torch.zeros(0,3,299)
        self.img_app_fake=torch.zeros(0,3,299)
    def build_rehearsal_memory(self, data_manager, per_class):
        dataset = data_manager.get_dataset(
            np.arange(0, self._known_classes),
            source="train",
            mode="random_choose",
            resize_size=self.args["resize_size"]
        )
    def _get_appendent(self,images_total_real,images_total_fake,labels):
        if not images_total_fake:
            return None
        img_total_1=images_total_real+images_total_fake
        img_total=np.array(img_total_1)
        label=np.array(labels)
        return img_total,label


    def after_task(self):
        self._old_network = get_old_net(self._network)
        self._known_classes = self._total_classes
        # self._old_adapter.append(copy.deepcopy(get_old_adapter(self._network)))
        # self._old_filter.append(self.get_old_filter())
        # logging.info("old filter has been saved {}").format(len(self._old_filter))

    def incremental_train(self, data_manager,data_manager2,images_total_real,images_total_fake,labels):
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
            appendent=self._get_appendent(images_total_real,images_total_fake,labels),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        self.db_loader=DataLoader(
            train_dataset, batch_size=db_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
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
        if self._cur_task==1:
            print(1)
        img_real,img_fake,label=self.data_selector(data_manager2,self._cur_task)
        if self._cur_task==3:
            print(3)
        self._eval(self.test_loader, self.out_loader, data_manager)
        return img_real,img_fake,label

        # self.db_fake,self.db_real = self.RS(self.db_loader)
        # self._eval_rs(self.test_loader)
        # print(1111)




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
        torch.use_deterministic_algorithms(False)
        for _, epoch in enumerate(prog_bar):

            self._network.train()
            loss=0.0
            cls_losses, consup_losses = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                loss,logits = self._network(inputs,targets, train_batch=True, AG=self.AG)
                # loss = cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss_pack['ensemble_acc'] = ACC(loss_pack['ensemble_logit'], targets)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = _compute_accuracy(self._device,self._network, val_loader)
            info = "Task {}, Epoch {}/{} => loss {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                loss / len(train_loader),
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
        # logging.info(info)


    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(epochs))
        torch.use_deterministic_algorithms(False)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            kd_losses=0.0
            contra_loss=0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                loss, logits = self._network(inputs, targets, train_batch=True, AG=self.AG)
                # loss = cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss_pack['ensemble_acc'] = ACC(loss_pack['ensemble_logit'], targets)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = _compute_accuracy(self._device,self._network, val_loader)
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
        # 记得解开
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
    def feature_dim(self):
        return self._network.head.in_features

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

def _ConSup_loss(features, labels):
    criterion_supcon = SupConLoss()
    loss = criterion_supcon(features, labels)
    return loss

def _FD_loss(student_feature, teacher_feature):
    return torch.nn.functional.mse_loss(F.normalize(student_feature, dim=1), F.normalize(teacher_feature, dim=1),
                                        reduction='mean')