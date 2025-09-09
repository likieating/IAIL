import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass


class CrossEntropyLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)

        return loss

class L1Loss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, inputs, targets):
        """
        Computes the l1 loss.
        """
        # Compute the l1 loss
        loss = self.loss_fn(inputs, targets)

        return loss


def swap_spe_features(type_list, value_list):
    type_list = type_list.cpu().numpy().tolist()
    # get index
    index_list = list(range(len(type_list)))

    # init a dict, where its key is the type and value is the index
    spe_dict = defaultdict(list)

    # do for-loop to get spe dict
    for i, one_type in enumerate(type_list):
        spe_dict[one_type].append(index_list[i])

    # shuffle the value list of each key
    for keys in spe_dict.keys():
        random.shuffle(spe_dict[keys])

    # generate a new index list for the value list
    new_index_list = []
    for one_type in type_list:
        value = spe_dict[one_type].pop()
        new_index_list.append(value)

    # swap the value_list by new_index_list
    value_list_new = value_list[new_index_list]

    return value_list_new


class ContrastiveLoss(AbstractLossClass):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def contrastive_loss(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        # Compute loss as the distance between anchor and negative minus the distance between anchor and positive
        loss = torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))
        return loss

    def forward(self, common, specific, spe_label):
        # prepare
        bs = common.shape[0]
        real_common, fake_common = common.chunk(2)
        ### common real
        idx_list = list(range(0, bs // 2))
        random.shuffle(idx_list)
        real_common_anchor = common[idx_list]
        ### common fake
        idx_list = list(range(bs // 2, bs))
        random.shuffle(idx_list)
        fake_common_anchor = common[idx_list]
        ### specific
        specific_anchor = swap_spe_features(spe_label, specific)
        real_specific_anchor, fake_specific_anchor = specific_anchor.chunk(2)
        real_specific, fake_specific = specific.chunk(2)

        # Compute the contrastive loss of common between real and fake
        loss_realcommon = self.contrastive_loss(real_common, real_common_anchor, fake_common_anchor)
        loss_fakecommon = self.contrastive_loss(fake_common, fake_common_anchor, real_common_anchor)

        # Comupte the constrastive loss of specific between real and fake
        loss_realspecific = self.contrastive_loss(real_specific, real_specific_anchor, fake_specific_anchor)
        loss_fakespecific = self.contrastive_loss(fake_specific, fake_specific_anchor, real_specific_anchor)

        # Compute the final loss as the sum of all contrastive losses
        loss = loss_realcommon + loss_fakecommon + loss_fakespecific + loss_realspecific
        # loss = loss_realcommon + loss_fakecommon
        return loss