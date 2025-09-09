import copy
import logging
import math
import random
from convs.xception import xception
import cv2
import numpy as np
from models.identity.networks.ucf_detector import UCFDetector
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_metric_learning.losses import SupConLoss
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.identity.networks.base_ucf import BaseLearner
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random
# from .retrive_shallow import RS_init,contrastive_loss,RS_eval
from models.AGDA import AGDA
from models.dataset_based2 import img_selector

EPSILON = 1e-8
epochs =25
lrate = 1e-4
step_size = 10
batch_size = 16
lrate_decay = 0.5
db_batch_size=10
weight_decay = 5e-4
num_workers = 8
T = 2

def compute_acc(device,model,loader):
    model.eval()
    correct, total = 0, 0
    for i, (_, inputs, targets, order) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = torch.cat([inputs[:, 0], inputs[:, 1]], dim=0)
        targets = torch.cat([targets[:, 0], targets[:, 1]], dim=0)
        data_dict = {
            'image': inputs,
            'label': targets,
            'label_spe': targets,
        }
        with torch.no_grad():
            pred_dict = model(data_dict,inference=True)
            outputs=pred_dict['cls']
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def get_old_net(model):
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    return old_model
class identity(BaseLearner):
    def __init__(self, args,window_size=3):
        super().__init__(args)
        self.args = args
        self._old_network = None
        self._network = UCFDetector()
        # self.best_model_path = ["/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/identity/0126-20-04-55-782_OSMA_split1_pretrained_vit_b16_224_in21k_adapter_1993_B0_Inc1/task0/val_acc96.22_model.pth"]
        # self.best_model_path = [
        #    "/home/tangshuai/dmp-bk/useful/0126-20-04-55-782_OSMA_split1_pretrained_vit_b16_224_in21k_adapter_1993_B0_Inc1/task0/val_acc96.22_model.pth"]
        self.best_model_path = [
            "/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/identity2/0331-15-17-13-434_OSMA_split1_xception_1993_B0_Inc1/task0/val_acc96.1_model.pth"]
        self._old_adapter=[]
        self.data_selector=img_selector(self._network,self._device,self._cur_task+1)

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
            distinguish=True,
            resize_size=self.args["resize_size"],
            # appendent=self._get_appendent(images_total_real,images_total_fake,labels),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers, pin_memory=True
        )
        self.db_loader=DataLoader(
            train_dataset, batch_size=db_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # val_dataset = data_manager.get_dataset(
        #     np.arange(0, self._total_classes),distinguish=True, source="val", mode="val", resize_size=self.args["resize_size"]
        # )
        # self.val_loader = DataLoader(
        #     val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        # )
        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), distinguish=True,source="test", mode="test", resize_size=self.args["resize_size"]
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
            resize_size=self.args["resize_size"]
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
        # img_real,img_fake,label=self.data_selector(data_manager2,self._cur_task)
        self._eval(self.test_loader, self.out_loader, data_manager)
        # self.build_rehearsal_memory(data_manager, self.samples_per_class)
        # return img_real,img_fake,label

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


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)


        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.Adam(self._network.parameters(), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
            self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            for param in self._network.con_gan.parameters():
                param.requires_grad=False
            # for name, param in self._network.named_parameters():
            #     print(f'{name}: {param.requires_grad}')
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=lrate_decay)
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
            contra=0.05
            recon=0.3
            spe=0.1
            for i, (_, inputs, targets, order) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs=torch.cat([inputs[:,0],inputs[:,1]],dim=0)
                targets=torch.cat([targets[:,0],targets[:,1]],dim=0)
                data_dict={
                    'image':inputs,
                    'label':targets,
                    'label_spe':targets,
                }
                pred_dict= self._network(data_dict)
                loss_all=self._network.get_losses(data_dict,pred_dict)
                loss=loss_all['common']+loss_all['specific']*spe+loss_all['contrastive']*contra+loss_all['reconstruction']*recon

                # loss=loss_all['overall']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, preds = torch.max(pred_dict['cls'], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            print(train_acc)
            val_acc = compute_acc(self._device,self._network, val_loader)
            info = "Task {}, Epoch {}/{} => loss {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {},contra {},spe {},recon {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                loss / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
                contra,
                spe,
                recon,
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            # if patience >= 5:
            #     logging.info("early stop! epoch {}/{}".format(epoch + 1, epochs))
            #     break
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(epochs))
        torch.use_deterministic_algorithms(False)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            loss_all = 0.0
            loss_sha, loss_spe, loss_reconstruction, loss_con = 0.0, 0.0, 0.0, 0.0
            loss_kd = 0.0
            contra = 0.05
            spe = 0.2
            kd = 1
            fd = 0
            recon=0.3
            correct, total = 0, 0
            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs = torch.cat([inputs[:, 0], inputs[:, 1]], dim=0)
                targets = torch.cat([targets[:, 0], targets[:, 1]], dim=0)
                data_dict = {
                    'image': inputs,
                    'label': targets,
                    'label_spe': targets,
                }
                pred_dict = self._network(data_dict)
                pred_dict_old = self._old_network(data_dict)
                kd_loss = _KD_loss(pred_dict['cls'], pred_dict_old['cls'], T=20.0)
                fd_loss = _FD_loss(pred_dict['feat'], pred_dict_old['feat'])
                loss_all = self._network.get_losses(data_dict, pred_dict)
                loss = loss_all['common'] + loss_all['specific'] * spe + loss_all[
                    'contrastive'] * contra + kd * kd_loss + fd * fd_loss+loss_all['reconstruction']*recon
                # loss=loss_all['common']+0.2*kd_loss
                loss_sha += loss_all['common']
                loss_spe += loss_all['specific']
                loss_con += loss_all['contrastive']
                loss_kd += kd_loss
                # loss_reconstruction+=loss_all['reconstruction']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, preds = torch.max(pred_dict['cls'], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            print(train_acc)
            val_acc = compute_acc(self._device, self._network, val_loader)
            info = "Task {}, Epoch {}/{} => loss {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {},contra {},spe {},kd {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                loss / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
                contra,
                spe,
                kd,
            )

            info = "Task {}, Epoch {}/{} => loss_sha {:.4f},loss_spe {:.4f},loss_con {:.4f},loss_kd {:.4f},loss_reconstruction {:.4f}, train_accy {:.2f}, val_accy {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                loss_sha / len(train_loader),
                loss_spe / len(train_loader),
                loss_con / len(train_loader),
                loss_kd / len(train_loader),
                loss_reconstruction / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            prog_bar.set_description(info)
        # logging.info(info)


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

def _FD_loss(student_feature, teacher_feature):
    return torch.nn.functional.mse_loss(F.normalize(student_feature, dim=1), F.normalize(teacher_feature, dim=1),
                                        reduction='mean')