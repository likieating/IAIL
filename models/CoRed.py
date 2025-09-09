import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
import pdb
from transformers.activations import ACT2FN
import math
import torch.nn.functional as F
VERY_SMALL_NUMBER = 1e-12
INF = 1e20

# Incorporating type-level event correlations
class WeightedCosine(nn.Module):
    def __init__(self, n_feature):
        super(WeightedCosine, self).__init__()
        self.W_pt = torch.Tensor(16, n_feature)  # multi-head for learnable cosine metric
        self.W_pt = nn.Parameter(nn.init.xavier_uniform_(self.W_pt))

    def forward(self, input_pt):
        '''
        input_pt: [l, e]
        '''
        W_pt = self.W_pt.pow(2).unsqueeze(1)  # [16, 1, e]
        input_pt = input_pt.unsqueeze(0)  # [1, l, e]
        h_pt = torch.mul(input_pt, W_pt)
        h_norm = F.normalize(h_pt, p=2, dim=-1)  # 2-norm on dim=-1
        scores_cosine = torch.matmul(h_norm, h_norm.transpose(-1, -2)).mean(0)  # sum on several heads=16
        return scores_cosine


class LabelGraph(nn.Module):
    def __init__(self, config, pt_feature, in_features, out_features, nlayer, dropout, att_dropout_rate):
        super(LabelGraph, self).__init__()
        self.dropout = dropout
        self.att_dropout = att_dropout_rate

        self.weighted_cosine = WeightedCosine(pt_feature)

        self.threshold = config.graph_learn_epsilon_label2label
        self.graph_encoders = nn.ModuleList()

        self.graph_encoders.append(GCNLayer(in_features, out_features, batch_norm=False))
        for _ in range(nlayer - 1):
            self.graph_encoders.append(GCNLayer(out_features, out_features, batch_norm=False))

    def forward(self, input, input_pt, adj=None):
        '''
        input: [l, e], predictive embeddings for types, randomly initialized
        input_pt: [l, e_pt], pretrained literal embedddings for types, initialized by GloVE on type name
        '''
        # metric-based graph structure learner
        scores_cosine = self.weighted_cosine(input_pt)
        mask_value = torch.ones_like(scores_cosine, device=scores_cosine.device) * -INF
        scores = torch.where(scores_cosine >= self.threshold, scores_cosine, mask_value)
        adj = torch.softmax(scores, dim=-1)

        # gnn for message passing among types
        x = input
        for _, encoder in enumerate(self.graph_encoders[:-1]):
            x = torch.relu(encoder(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.graph_encoders[-1](x, adj))
        return x


class LabelCorInc(nn.Module):
    '''
    used to incorporate type correlations;
    1. learn type graph, via metric-based graph structure learner
    2. messaging passing among types, via graph neural networks
    '''
    def __init__(self, config, dropout_rate, att_dropout_rate):
        super(LabelCorInc, self).__init__()
        self.gnn = LabelGraph(config,
                              pt_feature=300,
                              in_features=config.hidden_size,
                              out_features=config.hidden_size,
                              nlayer=config.nlayer,
                              dropout=dropout_rate,
                              att_dropout_rate=att_dropout_rate)

    def forward(self, label_embedding, pretrained_label_embedding):
        '''
        embedding: [b, t, e]
        '''
        label_output = self.gnn(label_embedding, pretrained_label_embedding, adj=None)
        return label_output


# Incorporating instance-level event correlations
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config

    def forward(self, hidden_span, type_embedding):
        '''
        hidden_span: [b, t, e]
        type_embedding_t2l: [l, e]
        '''
        scores = torch.matmul(hidden_span, type_embedding.transpose(-1, -2))  # [b,t,l]
        att = torch.softmax(scores, dim=-1)  # [b, t, l]
        type_mixed_embedding = torch.matmul(att, type_embedding)  # [b, t, e]
        return type_mixed_embedding, scores


class MSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_in = self.dim_k = config.hidden_size  # keep: dim_in = dim_k
        self.dim_v = config.label_emb
        self.tp = nn.Parameter(torch.randn(config.label_emb))  # here tp is initialized by [mask]

        self.device = config.device
        self.m_config = MaskedTransformerConfig(max_len=config.max_len,
                                                hidden_size=config.hidden_size,
                                                label_emb=config.label_emb,
                                                hidden_dropout_prob=config.decoder_dropout_rate,
                                                attention_probs_dropout_prob=0.1,
                                                num_heads=config.num_heads,
                                                partial_att=False)

        self.linear_w1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_w2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.msa = MaskedSALayer(self.m_config)
        self.attention = Attention(config)

    def forward(self, last_hidden_states, att_mask, type_emb=None, type_emb_re=None, label_truth=None, step=None, total=None):
        # generate pseudo-label representations
        type_embedding = type_emb
        first_label_emb, att_logits_first = self.attention(last_hidden_states, type_embedding)

        # conduct MLM-style masked self-attention
        hidden_states_truth = last_hidden_states + first_label_emb
        hidden_states_unknown = last_hidden_states + self.tp.expand_as(first_label_emb)
        outputs = self.msa(hidden_states_truth=hidden_states_truth,
                           hidden_states_unknown=hidden_states_unknown,
                           sim_first=None,
                           attention_mask=att_mask,
                           output_attentions=False)
        twice_hidden_states = outputs[0]
        output_hidden_states = twice_hidden_states + last_hidden_states + first_label_emb

        return output_hidden_states, att_logits_first  # [b, m, e]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)

        self.loss_func = nn.BCELoss(reduction='none')
        self.reset_parameters()
        self.device = self.config.device
        self.Attention = Attention(config=config)
        self.label2label_graph = LabelCorInc(config, config.decoder_dropout_rate, config.dropout_rate_att)
        self.linear_span = nn.Linear(config.hidden_size, config.hidden_size)
        self.MSA = MSA(config)

    def reset_parameters(self) -> None:
        self.type_embedding = nn.Parameter(torch.Tensor(self.config.label_num, self.config.hidden_size))
        nn.init.kaiming_uniform_(self.type_embedding, a=math.sqrt(5))

        type_embedding_none = nn.Parameter(torch.Tensor(1, 300), requires_grad=self.config.tune_emb)
        nn.init.zeros_(type_embedding_none)

        with open(self.config.f_type_embedding, 'rb') as f:
            type_embeddings = np.load(f)
        type_embedding = nn.Parameter(torch.from_numpy(type_embeddings).float(), requires_grad=self.config.tune_emb)
        self.pretrained_type_embedding = nn.Parameter(torch.cat([type_embedding_none, type_embedding], dim=0),
                                                      requires_grad=self.config.tune_emb)

    def forward(self, input_ids, att_mask, seg_ids, candidate_span, label_span, step=None, total=None):
        """
        :param input_ids:
        :param seg_ids:
        :param att_mask: [b, t], 0 is mask
        :param s_label:
        :param e_label:
        :param label_adjacent_matrix:
        :return:
        """

        bert_output = self.bert(input_ids=input_ids, attention_mask=att_mask, token_type_ids=seg_ids)
        last_hidden_states = bert_output[0]  # [b, t, e]

        type_embedding = self.type_embedding
        hidden_states = last_hidden_states
        candidate_span_padded_batched, mask_candidate_batched = self.convert_candidate_span_to_padded_tensor(
            candidate_span, len_sentence=last_hidden_states.size(1))
        candidate_span_padded_tensor = torch.tensor(candidate_span_padded_batched, device=self.device)  # [b, st, t]
        mask_candidate_tensor = torch.tensor(mask_candidate_batched, device=self.device)  # [b, st]

        agg = self.config.agg
        if agg == 'maxp':
            candidate_span_padded_tensor = candidate_span_padded_tensor.bool().logical_not().unsqueeze(-1)  # [b, st, t, 1]
            hidden_states = hidden_states.unsqueeze(1).expand(hidden_states.size(0), candidate_span_padded_tensor.size(1),
                                                              hidden_states.size(-2), hidden_states.size(-1))  # [b, st, t, e]
            candidate_rep_batched = torch.masked_fill(hidden_states, candidate_span_padded_tensor, hidden_states.min())
            candidate_rep = torch.max(candidate_rep_batched, dim=2)[0]  # [b, st, e]
        elif agg == 'meanp':
            candidate_span_padded_tensor = candidate_span_padded_tensor.unsqueeze(-1).float()  # [b, st, t, 1]
            hidden_states = hidden_states.unsqueeze(1).expand(hidden_states.size(0), candidate_span_padded_tensor.size(1),
                                                              hidden_states.size(-2), hidden_states.size(-1))  # [b, st, t, e]
            candidate_rep = torch.matmul(candidate_span_padded_tensor.transpose(-1, -2), hidden_states).squeeze(-2) / (
                torch.sum(candidate_span_padded_tensor.squeeze(-1), dim=-1, keepdim=True) + 1e-8)
        else:
            assert True

        candidate_rep = self.dropout(candidate_rep)
        candidate_rep = self.linear_span(candidate_rep)
        hidden_span = torch.relu(candidate_rep)

        loss = 0.
        type_embedding_l2l = self.label2label_graph(type_embedding, self.pretrained_type_embedding)
        type_embedding_l2l = type_embedding_l2l + type_embedding
        hidden_span = hidden_span
        type_embedding = type_embedding_l2l
        hidden_span, att_logits_first = self.MSA(hidden_span, att_mask=mask_candidate_tensor, type_emb=type_embedding)
        p_preds = self.predictor_dot(hidden_span, type_embedding)

        if label_span is not None:
            self.loss_func = nn.BCELoss(reduction='none')
            candidate_label_batched = self.convert_candidate_label_to_padded_tensor(label_span,
                                                                                    mask_candidate_tensor.size(-1))  # [b, st]
            candidate_label_tenser = torch.tensor(candidate_label_batched).long().to(self.device)  # [b, st]

            b, t, l = hidden_span.size(0), hidden_span.size(1), type_embedding.size(0)
            label_matrix_all = torch.zeros(b, t, l, device=self.device).scatter_(-1, candidate_label_tenser.unsqueeze(-1), 1)
            label_matrix = label_matrix_all[:, :, 1:]
            num_valid_candidate_without_pad = torch.sum(mask_candidate_tensor.float())

            self.loss_func_1 = nn.CrossEntropyLoss(reduction='none')
            loss_1 = self.loss_func_1(att_logits_first.view(-1, att_logits_first.size(-1)),
                                      candidate_label_tenser.view(-1))  # [b*st]
            loss_1 = torch.mul(loss_1, mask_candidate_tensor.view(-1))
            loss_1 = torch.sum(loss_1) / num_valid_candidate_without_pad

            # final loss
            loss = self.loss_func(p_preds.pow(self.config.pow_2), label_matrix)  # [b, st, l-1]
            loss = torch.sum(loss, dim=-1)  # [b, st]
            loss = torch.mul(loss.view(-1), mask_candidate_tensor.view(-1))
            loss = torch.sum(loss) / num_valid_candidate_without_pad

            loss = self.config.w1 * loss_1 + self.config.w2 * loss

        return loss, p_preds

    def predictot_vanilla_sigmoid(self, hidden_span, type_embedding):
        logit_all = torch.matmul(hidden_span, type_embedding.transpose(-1, -2))  # [b, st, e]*[b, e, l] -> [b, st, l]
        logit_type = logit_all[:, :, 1:]  # [b, st, l-1]
        p_type = torch.sigmoid(logit_type)
        p_preds = p_type
        return p_preds

    def predictor_dot(self, hidden_span, type_embedding):
        logit_all = torch.matmul(hidden_span, type_embedding.transpose(-1, -2))  # [b, st, e]*[b, e, l] -> [b, st, l]
        logit_none = logit_all[:, :, 0][:, :, None]  # [b, st, 1]
        logit_type = logit_all[:, :, 1:]  # [b, st, l-1]
        p_none = torch.sigmoid(logit_none)
        p_type = torch.sigmoid(logit_type)
        p_preds = torch.mul((1 - p_none), p_type)  # [b, st, l-1]
        return p_preds

    def predictor_nn(self, hidden_span, type_embedding):
        '''
        hidden_span: [b, t, e]
        type_embedding: [l, e]
        '''

        b, t, l, e = hidden_span.size(0), hidden_span.size(1), type_embedding.size(0), hidden_span.size(2)

        type_emb_o = type_embedding[0, :].view(1, e)
        type_emb_t = type_embedding[1:, :]

        z_mixed_t = torch.tanh(
            self.linear_infer_w1(hidden_span)[:, :, None, :] + self.linear_infer_w2(type_emb_t)[None, None, :, :])
        scores_t = self.linear_infer_v(z_mixed_t).squeeze(-1)
        p_type = torch.sigmoid(scores_t)

        z_mixed_o = torch.tanh(
            self.linear_infer_wo1(hidden_span)[:, :, None, :] + self.linear_infer_wo2(type_emb_o)[None, None, :, :])
        scores_o = self.linear_infer_vo(z_mixed_o).squeeze(-1)
        p_none = torch.sigmoid(scores_o)

        p_preds = torch.mul((1 - p_none), p_type)  # [b, st, l-1]
        return p_preds

    def convert_candidate_label_to_padded_tensor(self, candidate_label, max_num_candidate):
        candidate_label_batched = []
        for candidate_label_cur in candidate_label:
            num_cur = len(candidate_label_cur)
            candidate_label_padded_cur = list.copy(candidate_label_cur) + [0] * (max_num_candidate - num_cur)
            candidate_label_batched.append(candidate_label_padded_cur)
        return candidate_label_batched

    def convert_candidate_span_to_padded_tensor(self, candidate_span, len_sentence):
        num_batch = len(candidate_span)
        max_num_candidate_within_batch = 0
        for candidate_sent in candidate_span:
            if len(candidate_sent) > max_num_candidate_within_batch:
                max_num_candidate_within_batch = len(candidate_sent)

        mask_candidate_batched = []
        candidate_span_padded_batched = []
        for batch_id in range(num_batch):
            candidate_sent_cur = candidate_span[batch_id]
            num_candidate_sent_cur = len(candidate_sent_cur)
            mask_sent_cur = [1] * num_candidate_sent_cur + [0] * (max_num_candidate_within_batch - num_candidate_sent_cur
                                                                  )  # construct candidate padding mask
            candidate_sent_cur_padded = []
            for cand_id in range(len(candidate_sent_cur)):
                candidate_sent_pos_mask = [0] * len_sentence

                for span_id in range(candidate_sent_cur[cand_id][0],
                                     candidate_sent_cur[cand_id][1] + 1):  # note that there exist [cls] token
                    candidate_sent_pos_mask[span_id] = 1
                candidate_sent_cur_padded.append(candidate_sent_pos_mask)
            for _ in range(num_candidate_sent_cur, max_num_candidate_within_batch):
                candidate_sent_cur_padded.append([0] * len_sentence)
            assert len(candidate_sent_cur_padded) == len(mask_sent_cur)
            mask_candidate_batched.append(mask_sent_cur)
            candidate_span_padded_batched.append(candidate_sent_cur_padded)  # [st, t]
        return candidate_span_padded_batched, mask_candidate_batched

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)

        a_input1 = torch.matmul(h, self.a1)  # [b, t, 1]
        a_input2 = torch.matmul(h, self.a2)  # [b, t, 1]
        a_input1 = a_input1.expand_as(adj)
        a_input2 = a_input2.transpose(-1, -2).expand_as(adj)
        e = self.leakyrelu(a_input1 + a_input2)

        zero_vec = -9e15 * torch.ones_like(adj)
        attention = torch.where(adj != -INF, e, zero_vec)

        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return torch.tanh(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))
        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))

    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag_embed(r_inv_sqrt.view(mx.size(0), -1), dim1=-2, dim2=-1)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


BertLayerNorm = torch.nn.LayerNorm


class MaskedTransformerConfig(object):
    def __init__(self,
                 max_len,
                 hidden_size,
                 label_emb,
                 num_heads,
                 num_hidden_layers=1,
                 num_attention_heads=1,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 partial_att=False):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.label_emb = label_emb
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.num_heads = num_heads
        self.partial_att = partial_att


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LN
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor=None):
        # dense -> dp -> res -> LN
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if input_tensor is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MaskedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_att = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)
        self.linear_q = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)
        self.linear_k = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)
        self.linear_v = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        output_attentions=False,
    ):
        '''
        hidden_states_truth: [b, m, h+e]
        hidden_states_unknown: [b, m, h+e]
        '''
        n_batch, max_len, n_emb = hidden_states_truth.size()

        hidden_states_unknown_q = self.linear_q(hidden_states_unknown)  # [b, m, h+e]
        hidden_states_unknown_k = self.linear_k(hidden_states_unknown)  # [b, m, h+e]
        hidden_states_unknown_v = self.linear_v(hidden_states_unknown)  # [b, m, h+e]

        hidden_states_truth_k = self.linear_k(hidden_states_truth)  # [b, m, h+e]
        hidden_states_truth_v = self.linear_v(hidden_states_truth)  # [b, m, h+e]

        # attention matrix
        eye = torch.eye(max_len, max_len).to(hidden_states_truth.device)
        attention_scores_1 = torch.matmul(hidden_states_unknown_q,
                                          hidden_states_truth_k.transpose(-1, -2))  # [b, m, h+e] * [b, h+e, m] = [b, m, m]
        attention_scores_2 = torch.sum(hidden_states_unknown_q * hidden_states_unknown_k, dim=-1,
                                       keepdim=True).expand(n_batch, max_len, max_len)  # [b, m, m]
        attention_scores = attention_scores_1 * (1 - eye) + attention_scores_2 * eye  #  [b, m, m]
        attention_scores = attention_scores / math.sqrt(n_emb)  # final attention scores

        if attention_mask is not None:
            attention_mask_matrix = ((1 - attention_mask).float() * -1e9).unsqueeze(-1).expand(n_batch, max_len,
                                                                                               max_len)  # for padding
            attention_scores = attention_scores + attention_mask_matrix
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [b, m, m]
        attention_probs = self.dropout_att(attention_probs)  # [b, m, m]

        # for v matrix
        eye_exp = eye.unsqueeze(-1).expand(max_len, max_len, n_emb)  # [m, m] -> [m, m, h]
        hidden_states_truth_exp = hidden_states_truth_v.unsqueeze(1).expand(n_batch, max_len, max_len,
                                                                            n_emb)  # [b, m, h] -> [b, m, m, h]
        hidden_states_unknown_exp = hidden_states_unknown_v.unsqueeze(1).expand(n_batch, max_len, max_len,
                                                                                n_emb)  # [b, m, h] -> [b, m, m, h]
        hidden_states_exp = hidden_states_truth_exp * (1 - eye_exp) + hidden_states_unknown_exp * eye_exp  # [b, m, m, h]

        # fuse atteniton scores and v matrix
        attention_probs_exp = attention_probs.unsqueeze(-1)  # [b, m, m] -> [b, m, m, 1]
        context_layer = torch.matmul(hidden_states_exp.transpose(-1, -2), attention_probs_exp).squeeze(
            -1)  # [b, m, h+e, m] * [b, m, m, 1] -> [b, m, h+e, 1] -> [b, m, h+e]

        outputs = self.output(context_layer)  # [b, m, h]
        outputs = (outputs, attention_probs) if output_attentions else (outputs, )
        return outputs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dropout_att = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)
        self.num_heads = config.num_heads
        self.dim_in = self.dim_k = self.dim_v = config.hidden_size
        self.partial_att = config.partial_att

        self.linear_q = nn.Linear(self.dim_in, self.dim_k)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        sim_first=None,
        output_attentions=False,
    ):
        '''
        hidden_states_truth: [b, m, h+e]
        hidden_states_unknown: [b, m, h+e]
        '''
        n_batch, max_len, n_emb = hidden_states_truth.size()
        assert n_emb == self.dim_in

        num_heads = self.num_heads
        dk = self.dim_k // num_heads  # dim_k(dim_q) of each head
        dv = self.dim_v // num_heads  # dim_v of each head

        hidden_states_unknown_q = self.linear_q(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_unknown_k = self.linear_k(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_unknown_v = self.linear_v(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dv).transpose(1, 2)  # [b, nh, m, dv]

        hidden_states_truth_k = self.linear_k(hidden_states_truth).reshape(n_batch, max_len, num_heads,
                                                                           dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_truth_v = self.linear_v(hidden_states_truth).reshape(n_batch, max_len, num_heads,
                                                                           dv).transpose(1, 2)  # [b, nh, m, dv]

        # attention matrix
        eye = torch.eye(max_len, max_len).to(hidden_states_truth.device)
        attention_scores_1 = torch.matmul(hidden_states_unknown_q, hidden_states_truth_k.transpose(
            -1, -2))  # [b, nh, m, dk].dot[b, nh, dk, m] = [b, nh, m, m]
        attention_scores_2 = torch.sum(hidden_states_unknown_q * hidden_states_unknown_k, dim=-1,
                                       keepdim=True).expand_as(attention_scores_1)  # [b, nh, m, m]

        attention_scores = attention_scores_1 * (1 - eye) + attention_scores_2 * eye  #   [b, nh, m, m]
        attention_scores = attention_scores / math.sqrt(n_emb)  # final attention scores

        if attention_mask is not None:
            attention_mask_matrix = ((1 - attention_mask).float() * -1e9).view(n_batch, 1, 1,
                                                                               max_len).expand_as(attention_scores)  # padding
            attention_scores = attention_scores + attention_mask_matrix
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [b, nh, m, m]

        attention_probs = self.dropout_att(attention_probs)  # [b, nh, m, m]

        eye_exp = eye.unsqueeze(-1).expand(max_len, max_len, dv)  # [m, m] -> [m, m, dv]
        hidden_states_truth_exp = hidden_states_truth_v.unsqueeze(2).expand(n_batch, num_heads, max_len, max_len,
                                                                            dv)  # [b, nh, m, dv] -> [b, nh, m, m, dv]
        hidden_states_unknown_exp = hidden_states_unknown_v.unsqueeze(2).expand(n_batch, num_heads, max_len, max_len,
                                                                                dv)  # [b, nh, m, dv] -> [b, nh, m, m, dv]
        hidden_states_exp = hidden_states_truth_exp * (1 - eye_exp) + hidden_states_unknown_exp * eye_exp  # [b, nh, m, m, dv]

        # fuse attention and v matrix
        attention_probs_exp = attention_probs.unsqueeze(-1)  # [b, nh, m, m] -> [b, nh, m, m, 1]
        context_layer = torch.matmul(hidden_states_exp.transpose(-1, -2), attention_probs_exp).squeeze(
            -1)  # [b, nh, m, dv, m].dot[b, nh, m, m, 1] -> [b, nh, m, dv, 1] -> [b, nh, m, dv]
        context_layer = context_layer.transpose(1, 2).reshape(n_batch, max_len, self.dim_v)  # flaten multi-head

        outputs = self.output(context_layer)  # [b, m, h]
        outputs = (outputs, attention_probs) if output_attentions else (outputs, )
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # dense -> act
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # dense -> dp -> res -> LN
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MaskedTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MaskedAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states_truth,
            hidden_states_unknown,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs


class MaskedSALayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)  # multi-head
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states_truth, hidden_states_unknown, sim_first=None, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states_truth=hidden_states_truth,
                                                hidden_states_unknown=hidden_states_unknown,
                                                sim_first=sim_first,
                                                attention_mask=attention_mask,
                                                output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs
