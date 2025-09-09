import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, early_stop, save_model, load_model, set_random

epochs = 60
lrate = 0.001
step_size = 30
lrate_decay = 0.1
batch_size = 128 
weight_decay = 2e-4
num_workers = 8


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args["convnet_type"], False)
        self.best_model_path = []

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(2)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            resize_size=self.args["resize_size"]
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
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        out_dataset = data_manager.get_dataset(
            None, source="test_out", mode="test", resize_size=self.args["resize_size"]
        )
        self.out_loader = DataLoader(
            out_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.val_loader)
        self._eval(self.test_loader, self.out_loader, data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
        if self._cur_task == 0:
            if self.args['skip']:
                if len(self._multiple_gpus) > 1:
                    self._network = self._network.module
                load_model(self._network, self.args, self.best_model_path)
                self._network.to(self._device)
                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
            else:
                self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            self._update_representation(train_loader, val_loader, optimizer, scheduler)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                cls_loss = F.cross_entropy(logits, targets.long())
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
            info = "Task {}, Epoch {}/{} => cls_loss {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path, self._cur_task, self._network)
            prog_bar.set_description(info)
        _ = save_model(self.args, best_acc, self._cur_task, self.best_model_path, self._network, skip_checkpoints=True)
        logging.info(info)

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        best_acc, patience, self.best_model_path = 0, 0, []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                cls_loss = F.cross_entropy(logits, targets.long())

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
            info = "Task {}, Epoch {}/{} => cls_loss {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                cls_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path, self._cur_task, self._network)
            # if patience >= 5:
            #     logging.info("early stop! epoch {}/{}".format(epoch+1, epochs))
            #     break
            prog_bar.set_description(info)
        logging.info(info)

    def _eval(self, test_loader, out_loader, data_manager):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        self._network.eval()
        _out_k, _pred_k, _labels_k, cnn_accy, nme_accy = self.eval_task(type='test')
        logging.info(_out_k[:10])
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