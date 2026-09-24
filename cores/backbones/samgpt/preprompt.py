import torch
import torch.nn as nn
import torch.nn.functional as F
from cores.backbones.samgpt.models import GraphCL, Lp, GcnLayers, GatLayers
from cores.backbones.samgpt.layers import AvgReadout
import numpy as np
from sklearn.decomposition import PCA
from cores.backbones.samgpt.layers.prompt import *
import copy

class PrePrompt(nn.Module):

    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, dropout, type_, backbone='gcn', alpha=1.0):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)])
        self.structure_prompt_layers = nn.ModuleList([nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)]) for _ in range(num_pretrain_dataset_num)])
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        if backbone == 'gat':
            self.gcn = GatLayers(n_in, n_h, num_layers_num, dropout)
            str_prompt = [textprompt(n_h * self.gcn.heads, type_) for _ in range(num_layers_num)]
            self.structure_prompt_layers = nn.ModuleList([nn.ModuleList(copy.deepcopy(str_prompt)) for _ in range(num_pretrain_dataset_num)])
        self.combine = alpha
        self.loss = nn.BCEWithLogitsLoss()

    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, sparse=False):
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            fea_prelogits = self.lp(self.gcn, fea_pretext(seq), adj, sparse)
            str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
            yield (fea_prelogits + self.combine * str_prelogits)

    def compute_prelogits_GRAPHCL(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        for fea_pretext, str_layers, seq, adj in zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))]
            fea_prelogits = self.graphcledge(self.gcn, preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], adj[0], adj[1], adj[2], sparse, msk, samp_bias1, samp_bias2, aug_type='edge')
            str_prelogits = self.graphcledge(self.gcn, seq[0], seq[1], seq[2], seq[3], adj[0], adj[1], adj[2], sparse, msk, samp_bias1, samp_bias2, 'edge', str_layers)
            yield (fea_prelogits + self.combine * str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)
        return (h_1.detach(), c.detach())

    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [[layer.weight.detach() for layer in structure_prompt_layer] for structure_prompt_layer in self.structure_prompt_layers]
        combines = [self.combine]
        return (fea_pretext_weights, str_pretext_weights, combines)

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2, lbl, samples=None):
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        if samples == None:
            logits = list(self.compute_prelogits_GRAPHCL(self.feature_prompt_layers, self.structure_prompt_layers, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2))
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i])
                total_loss += loss
        else:
            logits = list(self.compute_prelogits_LP(self.feature_prompt_layers, self.structure_prompt_layers, seq_list, adj_list, sparse))
            if type(samples) == list:
                samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device) for sample in samples]
                for i in range(len(logits)):
                    loss = compareloss(logits[i], samples[i], temperature=1)
                    total_loss += loss
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)
        return total_loss

def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:, :k].shape)
    print(VT[:k, :].shape)
    res = U[:, :k].dot(np.diag(Sigma[:k]))
    return res

def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))

def compareloss(feature, tuples, temperature):
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def prompt_pretrain_sample(adj, n):
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n))
    whole = np.array(range(nodenum))
    for i in range(nodenum):
        nonzero_index_i_row = indices[indptr[i]:indptr[i + 1]]
        zero_index_i_row = np.setdiff1d(whole, nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row) == 0:
            res[i][0] = i
        else:
            res[i][0] = nonzero_index_i_row[0]
        res[i][1:1 + n] = zero_index_i_row[0:n]
    return torch.tensor(res.astype(int))
