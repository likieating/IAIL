import copy
import pdb
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.DER.resnet import resnet18
import numpy as np
from torch.nn import Module
from torch.nn.parameter import Parameter
class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                resnet18(nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = resnet18(nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:

            new_clf = resnet18(nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier


class BiC(nn.Module):
    def __init__(self, lr, scheduling, lr_decay_factor, weight_decay, batch_size, epochs):
        super(BiC, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))  #.cuda()
        self.gamma = torch.nn.Parameter(torch.zeros(1))  #.cuda()
        self.lr = lr
        self.scheduling = scheduling
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.class_specific = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.bic_flag = False

    def reset(self, lr=None, scheduling=None, lr_decay_factor=None, weight_decay=None, n_classes=-1):
        with torch.no_grad():
            if lr is None:
                lr = self.lr
            if scheduling is None:
                scheduling = self.scheduling
            if lr_decay_factor is None:
                lr_decay_factor = self.lr_decay_factor
            if weight_decay is None:
                weight_decay = self.weight_decay
            if self.class_specific:
                assert n_classes != -1
                self.beta = torch.nn.Parameter(torch.ones(n_classes).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(n_classes).cuda())
            else:
                self.beta = torch.nn.Parameter(torch.ones(1).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(1).cuda())
            self.optimizer = torch.optim.SGD([self.beta, self.gamma], lr=lr, momentum=0.9, weight_decay=weight_decay)
            # self.scheduler = CosineAnnealingLR(self.optimizer, 10)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, scheduling, gamma=lr_decay_factor)

    def extract_preds_and_targets(self, model, loader):
        preds, targets = [], []
        with torch.no_grad():
            for (x, y) in loader:
                preds.append(model(x.cuda())['logit'])
                targets.append(y.cuda())
        return torch.cat((preds)), torch.cat((targets))

    def update(self, logger, task_size, model, loader, loss_criterion=None):
        if task_size == 0:
            logger.info("no new task for BiC!")
            return
        if loss_criterion is None:
            loss_criterion = F.cross_entropy

        self.bic_flag = True
        logger.info("Begin BiC ...")
        model.eval()

        for epoch in range(self.epochs):
            preds_, targets_ = self.extract_preds_and_targets(model, loader)
            order = np.arange(preds_.shape[0])
            np.random.shuffle(order)

            preds, targets = preds_.clone(), targets_.clone()
            preds, targets = preds[order], targets[order]
            _loss = 0.0
            _correct = 0
            _count = 0
            for start in range(0, preds.shape[0], self.batch_size):
                if start + self.batch_size < preds.shape[0]:
                    out = preds[start:start + self.batch_size, :].clone()
                    lbls = targets[start:start + self.batch_size]
                else:
                    out = preds[start:, :].clone()
                    lbls = targets[start:]
                if self.class_specific is False:
                    out1 = out[:, :-task_size].clone()
                    out2 = out[:, -task_size:].clone()
                    outputs = torch.cat((out1, out2 * self.beta + self.gamma), 1)
                else:
                    outputs = out * self.beta + self.gamma
                loss = loss_criterion(outputs, lbls)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _, pred = outputs.max(1)
                _correct += (pred == lbls).sum()
                _count += lbls.size(0)
                _loss += loss.item() * outputs.shape[0]
            logger.info("epoch {} loss {:4f} acc {:4f}".format(epoch, _loss / preds.shape[0], _correct / _count))

            self.scheduler.step()
        logger.info("beta {:.4f} gamma {:.4f}".format(self.beta.cpu().item(), self.gamma.cpu().item()))

    @torch.no_grad()
    def post_process(self, preds, task_size):
        if self.class_specific is False:
            if task_size != 0:
                preds[:, -task_size:] = preds[:, -task_size:] * self.beta + self.gamma
        else:
            preds = preds * self.beta + self.gamma
        return preds


class WA(object):
    def __init__(self):
        self.gamma = None

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print(self.gamma.cpu().item())

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits


class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out
