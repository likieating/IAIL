import copy
import logging
from time import sleep
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, set_random
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

EPSILON = 1e-8
batch_size = 100
num_workers = 8


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._real_data_memory, self._fake_data_memory = np.array([]), np.array([])
        self._real_targets_memory, self._fake_targets_memory = np.array([]), np.array([])

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        data_length = len(np.concatenate((self._real_data_memory, self._fake_data_memory)))
        targets_length = len(np.concatenate((self._real_targets_memory, self._fake_targets_memory)))
        assert data_length == targets_length, "Exemplar size error."
        return data_length

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)

    def after_task(self):
        pass

    def _evaluate(self, y_order, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_order, y_pred, y_true, self._known_classes, self.args['increment'])
        ret["grouped"] = grouped
        return ret

    def draw_spe(self,comfea,spefea,comlogits,spelogits):
        # 设置颜色映射，用于不同任务
        unique_tasks = len(comfea)  # 假设 comfea 和 spefea 中的任务数量相同
        cmap = plt.get_cmap('tab10')  # 使用 'tab10' 调色板

        # 初始化两个子图
        fig, (ax_com, ax_spe) = plt.subplots(1, 2, figsize=(16, 8))

        # t-SNE 降维设置
        tsne = TSNE(n_components=2, random_state=42)

        # 处理 comfea 和 spefea
        all_comfea = np.concatenate(comfea, axis=0)  # 合并所有任务的特征
        all_spefea = np.concatenate(spefea, axis=0)

        # 使用 t-SNE 降维
        comfea_2d = tsne.fit_transform(all_comfea)
        spefea_2d = tsne.fit_transform(all_spefea)

        start_idx = 0  # 用于记录每个任务的特征起始位置

        # 绘制 content features (comfea)
        for task_idx in range(unique_tasks):
            # 获取当前任务的 comfea 和 logits
            end_idx = start_idx + comfea[task_idx].shape[0]
            task_comfea_2d = comfea_2d[start_idx:end_idx]
            task_comlogits = comlogits[task_idx]

            # 判断哪些是真数据，哪些是假数据（logits 较大的为真数据）
            true_data_mask = task_comlogits[:, 0] > task_comlogits[:, 1]  # logits[:, 0] > logits[:, 1] 为真数据
            false_data_mask = ~true_data_mask

            # 绘制真数据
            ax_com.scatter(task_comfea_2d[true_data_mask, 0], task_comfea_2d[true_data_mask, 1],
                           color='blue', label=f'Task {task_idx} - True', alpha=0.6,s=10)

            # 绘制假数据
            ax_com.scatter(task_comfea_2d[false_data_mask, 0], task_comfea_2d[false_data_mask, 1],
                           color=cmap(task_idx / unique_tasks), label=f'Task {task_idx} - False', alpha=0.6,s=10)

            start_idx = end_idx  # 更新起始索引

        # 设置标题和标签
        ax_com.set_title('t-SNE Visualization of Content Features (comfea)')
        ax_com.set_xlabel('t-SNE Component 1')
        ax_com.set_ylabel('t-SNE Component 2')
        ax_com.legend()

        # 绘制 specific features (spefea)，与 comfea 相同的步骤
        start_idx = 0  # 重置索引
        for task_idx in range(unique_tasks):
            # 获取当前任务的 spefea 和 logits
            end_idx = start_idx + spefea[task_idx].shape[0]
            task_spefea_2d = spefea_2d[start_idx:end_idx]
            task_spelogits = spelogits[task_idx]

            # 判断哪些是真数据，哪些是假数据
            true_data_mask = task_spelogits[:, 0] > task_spelogits[:, task_idx+1]  # logits[:, 0] > logits[:, 1] 为真数据
            false_data_mask = ~true_data_mask

            # 绘制真数据
            ax_spe.scatter(task_spefea_2d[true_data_mask, 0], task_spefea_2d[true_data_mask, 1],
                           color='blue', label=f'Task {task_idx} - True', alpha=0.6,s=10)

            # 绘制假数据
            ax_spe.scatter(task_spefea_2d[false_data_mask, 0], task_spefea_2d[false_data_mask, 1],
                           color=cmap(task_idx / unique_tasks), label=f'Task {task_idx} - False', alpha=0.6,s=10)

            start_idx = end_idx  # 更新起始索引

        # 设置标题和标签
        ax_spe.set_title('t-SNE Visualization of Specific Features (spefea)')
        ax_spe.set_xlabel('t-SNE Component 1')
        ax_spe.set_ylabel('t-SNE Component 2')
        ax_spe.legend()
        return plt

    def load_fea(self,dir_name,cur_task):
        comfea,spefea=[],[]
        comlogits,spelogits=[],[]
        for i in range(cur_task+1):
            dir=os.path.join(dir_name,"task"+str(i))
            dir_com=os.path.join(dir,'_com.npy')
            dir_spe= os.path.join(dir, '_spe.npy')
            dir_com_logits=os.path.join(dir,'_com_logits.npy')
            dir_spe_logits = os.path.join(dir, '_spe_logits.npy')
            fea_com=np.load(dir_com)
            fea_spe=np.load(dir_spe)
            co_lo=np.load(dir_com_logits)
            sp_lo=np.load(dir_spe_logits)
            comlogits.append(co_lo)
            spelogits.append(sp_lo)
            comfea.append(fea_com)
            spefea.append(fea_spe)
        plt=self.draw_spe(comfea,spefea,comlogits,spelogits)
        dir = os.path.join(dir_name, "task" + str(cur_task))
        save_path=os.path.join(dir,'figure.png')
        plt.savefig(save_path,dpi=300)
        plt.tight_layout()
        plt.show()
        print(1)


    def save_fea(self,com_fea,spe_fea,logits_com,logits_spe,file_path):
        com_fea_path=os.path.join(file_path,'_com.npy')
        spe_fea_path=os.path.join(file_path,'_spe.npy')
        com_logits_path=os.path.join(file_path,'_com_logits.npy')
        spe_logits_path = os.path.join(file_path, '_spe_logits.npy')
            # 保存更新后的特征
        np.save(com_fea_path, com_fea)
        np.save(spe_fea_path,spe_fea)
        np.save(com_logits_path,logits_com)
        np.save(spe_logits_path,logits_spe)

    def eval_task(self, type='test'):
        if type == 'out':
            loader = self.out_loader
        else:
            loader = self.test_loader

        y_order, y_out, y_true,y_fea_com,y_fea_spe,logits_com,logits_spe = self._eval_cnn(type, loader)
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
        # 使用 t-SNE 进行降维
        self.save_fea(y_fea_com,y_fea_spe,logits_com,logits_spe,_save_dir)
        self.load_fea(self.args['logfilename'],self._cur_task)
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

    def eval_open(self, eval_type, final=False):
        self._network.eval()
        _log_dir = os.path.join("./results/", f"{self.args['prefix']}", "open-set")
        os.makedirs(_log_dir, exist_ok=True)
        out_acc, out_auc = self.eval_out_task(type='out')
        _log_path = os.path.join(_log_dir, f"{self.args['csv_name']}.csv")
        if self.args['prefix'] == 'benchmark' and final:
            with open(_log_path, "a+") as f:
                f.write(f"{self.args['time_str']}, {self.args['model_name']},")
                f.write(f"{out_acc}, {out_auc}\n")
        logging.info("open-set-out {} acc: {}, auc: {}".format(eval_type, out_acc, out_auc))
        return out_acc

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _eval(self):
        pass

    def _get_memory(self):
        if len(self._real_data_memory) == 0 and len(self._fake_data_memory):
            return None
        else:
            return (np.concatenate((self._real_data_memory, self._fake_data_memory)),
                    np.concatenate((self._real_targets_memory, self._fake_targets_memory)))

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
        y_fea_com,y_fea_spe=[],[]
        logits_spe,logits_com=[],[]
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            # inputs = torch.cat([inputs[:, 0], inputs[:, 1]], dim=0)
            # targets = torch.cat([targets[:, 0], targets[:, 1]], dim=0)
            data_dict = {
                'image': inputs,
                'label': targets,
                'label_spe': targets,
            }

            if type == 'out' or self.args['run_type'] == 'train' or 'train_bak' in self.args['run_type']:
                with torch.no_grad():
                    pred_dict = self._network(data_dict,inference=True)
                    logits=pred_dict['cls']
                    logitssp=pred_dict['cls_spe']
                    fea_spe=pred_dict['feat_spe']
                    fea=pred_dict['feat']
                    # outputs, loss = self._network(inputs)
                    # outputs = self._network(inputs)
                y_fea_spe.append(fea_spe.cpu().numpy())
                y_fea_com.append(fea.cpu().numpy())
                y_order.append(orders.cpu().numpy())
                y_pred.append(logits.cpu().numpy())
                logits_spe.append(logitssp.cpu().numpy())
                logits_com.append(logits.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            else:
                for class_id in self.args['test_class'][self._cur_task]:
                    if class_id in orders.cpu().numpy():
                        with torch.no_grad():
                            pred_dict = self._network(data_dict, inference=True)
                            logits = pred_dict['cls']
                            logitssp = pred_dict['cls_spe']
                            fea_spe = pred_dict['feat_spe']
                            fea = pred_dict['feat']
                            # outputs, loss = self._network(inputs)
                            # outputs = self._network(inputs)
                        y_fea_spe.append(fea_spe.cpu().numpy())
                        y_fea_com.append(fea.cpu().numpy())
                        y_order.append(orders.cpu().numpy())
                        logits_spe.append(logitssp.cpu().numpy())
                        logits_com.append(logits.cpu().numpy())
                        y_pred.append(logits.cpu().numpy())
                        y_true.append(targets.cpu().numpy())
        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true),np.concatenate(y_fea_com),np.concatenate(y_fea_spe),np.concatenate(logits_com),np.concatenate(logits_spe)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        y_order, real_vectors, fake_vectors, y_true = self._extract_vectors(loader)
        real_vectors = (real_vectors.T / (np.linalg.norm(real_vectors.T, axis=0) + EPSILON)).T
        fake_vectors = (fake_vectors.T / (np.linalg.norm(fake_vectors.T, axis=0) + EPSILON)).T
        real_dists = cdist(class_means[0], real_vectors, "sqeuclidean")
        fake_dists = cdist(class_means[1], fake_vectors, "sqeuclidean")
        scores = np.vstack((real_dists.T, fake_dists.T))

        return y_order, scores, y_true  # [N, topk]

    def get_curve_online(self, novel, stypes=['Bas']):
        tp, fp = dict(), dict()
        tnr_at_tpr95 = dict()
        for stype in stypes:
            if self.args['model_name'] == "icarl":
                novel = abs(np.sort(-novel))
                logging.info("model {} calculates auc using descending order!".format(self.args["model_name"]))
            else:
                novel.sort()
                logging.info("model {} calculates auc using ascending order!".format(self.args["model_name"]))
            num_n = novel.shape[0]
            print("num_n", num_n)
            tp[stype] = -np.ones([num_n + 1], dtype=int)
            fp[stype] = -np.ones([num_n + 1], dtype=int)
            tp[stype][0], fp[stype][0] = num_n, num_n
            print("tp[stype]:", tp[stype])
            print("fp[stype]:", fp[stype])
            for l in range(num_n):
                tp[stype][l + 1] = tp[stype][l]
                fp[stype][l + 1] = fp[stype][l] - 1
            tpr95_pos = np.abs(tp[stype] / num_n - .95).argmin()
            tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
        return tp, fp, tnr_at_tpr95

    def metric_ood(self, _out_u, stypes=['Bas'], verbose=True):
        if self.args['model_name'] == "icarl":
            x2 = np.min(_out_u, axis=1)
            logging.info("model {} get the minimum of the output in auc".format(self.args["model_name"]))
        else:
            x2 = np.max(_out_u, axis=1)
            logging.info("model {} get the maximum of the output in auc".format(self.args["model_name"]))
        tp, fp, tnr_at_tpr95 = self.get_curve_online(x2, stypes)
        for stype in stypes:
            # AUROC
            mtype = 'AUROC'
            tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
            fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
            # -np.traz(tpr, fpr)
            roc_auc = 100. * (-np.trapz(1. - fpr, tpr))
        return roc_auc

    # return order real_vec fake_vec targets
    def _extract_vectors(self, loader):
        real_vectors, fake_vectors, targets, orders = [], [], [], []
        for _, _inputs, _targets, _orders in loader:
            _targets = _targets.numpy()
            _orders = _orders.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )
            if _targets.all() == 0:
                real_vectors.append(_vectors)
            else:
                fake_vectors.append(_vectors)

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
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader, oneclass=True, inference=True)
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
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=data_manager.collate_fn
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
            appendent=(selected_exemplars, exemplar_targets),
            resize_size=self.args["resize_size"]
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader, oneclass=True, inference=True)
        vectors = real_vectors if not label else fake_vectors
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)

        self._class_means[label, class_idx, :] = mean
