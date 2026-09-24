import torch
import torch.nn as nn
import torch.nn.functional as F
from cores.backbones.bridge.models import Lp, GcnLayers
from cores.backbones.bridge.layers import AvgReadout
import numpy as np
from sklearn.decomposition import PCA

def spectral_regularization_smooth(x, x0, eivec, eival, thres):
    relu = torch.nn.ReLU()
    x_out = torch.einsum('nm,md->nd', eivec, x)
    x0_out = torch.einsum('nm,md->nd', eivec, x0)
    delta = (x_out[:-1].t() * eival[:-1] - x_out[1:].t() * eival[1:]).t().abs()
    delta0 = (x0_out[:-1].t() * eival[:-1] - x0_out[1:].t() * eival[1:]).t().abs()
    loss = relu(delta - thres * delta0)[eival[:-1] - eival[1:] > 0.01].mean()
    return loss

class RoutingNetwork(nn.Module):

    def __init__(self, input_dim=50, dropout_rate=0.1):
        super(RoutingNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        x = F.softmax(x, dim=0)
        return x

class downprompt(nn.Module):

    def __init__(self, weights_list, ft_in, nb_classes, type, feature_dim, num_tokens=4, dropout_rate=0.1):
        super(downprompt, self).__init__()
        self.prefeature = prefeatureprompt(weights_list, dim=feature_dim, type=type, num_tokens=num_tokens, dropout_rate=dropout_rate)
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in).cuda()
        self.ave = torch.FloatTensor(nb_classes, ft_in).cuda()

    def forward(self, eivec, eival, thres, features, adj, sparse, gcn, idx, seq, labels=None, train=0):
        features1 = self.prefeature(features)
        reg_loss = spectral_regularization_smooth(features1, features, eivec, eival, thres)
        embeds1 = gcn(features1, adj, sparse, None).squeeze()
        pretrain_embs1 = embeds1[idx]
        rawret = pretrain_embs1
        rawret = rawret.cuda()
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes)
        numerator = rawret @ self.ave.t()
        denominator = torch.outer(rawret.norm(dim=1), self.ave.norm(dim=1)).clamp_min(1e-08)
        ret = numerator / denominator
        ret = F.softmax(ret, dim=1)
        en_result = self.calculate_uncertainty(ret)
        ret = ret.float().cuda()
        en_result = en_result.float().cuda()
        return (ret, en_result, reg_loss)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def calculate_uncertainty(self, softmax_output):
        epsilon = 1e-08
        entropy = -torch.sum(softmax_output * torch.log(softmax_output + epsilon), dim=1)
        return entropy

class downstreamprompt(nn.Module):

    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        graph_embedding = self.weight * graph_embedding
        return graph_embedding

class composedtoken(nn.Module):

    def __init__(self, weights_list, type: str, num_tokens=4, dropout_rate=0.1):
        super(composedtoken, self).__init__()
        self.texttoken = weights_list
        self.prompt = weighted_prompt(num_tokens, dropout_rate).cuda()
        self.type = type

    def forward(self, seq):
        texttoken = self.prompt(self.texttoken)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0], 1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets

class prefeatureprompt(nn.Module):

    def __init__(self, weights_list, dim, type: str, num_tokens=4, dropout_rate=0.1):
        super(prefeatureprompt, self).__init__()
        self.precomposedfeature = composedtoken(weights_list, type, num_tokens, dropout_rate=dropout_rate)
        self.preopenfeature = downstreamprompt(dim)
        self.combineprompt = combineprompt()

    def forward(self, seq):
        seq1 = self.precomposedfeature(seq)
        seq2 = self.preopenfeature(seq)
        ret = self.combineprompt(seq1, seq2)
        return ret

class combineprompt(nn.Module):

    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding1, graph_embedding2):
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)

class weighted_prompt(nn.Module):

    def __init__(self, weightednum, dropout_rate=0.1):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
        self.routingnet = RoutingNetwork(dropout_rate=dropout_rate)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        graph_embedding = self.weight.T * graph_embedding
        guide_weights = self.routingnet(graph_embedding)
        graph_embedding = torch.mm(guide_weights.T, graph_embedding)
        return graph_embedding

def averageemb(labels, rawret, nb_class):
    import torch_scatter
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel

class PrePrompt(nn.Module):

    def __init__(self, n_in, n_h, activation, sample, num_layers_num, p, type, variance_weight, num_tokens=4, n_samples=3):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, p)
        self.read = AvgReadout()
        self.prompttype = type
        self.register_buffer('negative_sample', torch.tensor(sample, dtype=int), persistent=False)
        self.loss = nn.BCEWithLogitsLoss()
        self.masks_logits = nn.Parameter(torch.randn(num_tokens, n_in))
        self.n_samples = n_samples
        self.variance_weight = variance_weight

    def forward(self, seq_list, adj_list, sparse, msk, samp_bias1, samp_bias2):
        mask_prob = torch.sigmoid(self.masks_logits)
        masked_features = [seq * mask_prob[i].unsqueeze(0) for i, seq in enumerate(seq_list)]
        prelogits = [self.lp(self.gcn, preseq, adj, sparse) for preseq, adj in zip(masked_features, adj_list)]
        logits = torch.cat(prelogits, dim=0)
        lploss = compareloss(logits, self.negative_sample, temperature=1)
        loss_variances = []
        for i in range(self.n_samples):
            random_noise = torch.randn(1, 50, device=seq_list[0].device)
            random_noise = random_noise * (1 - mask_prob[i].unsqueeze(0))
            noisy_features = [masked + random_noise.expand_as(masked) * (1 - mask_prob[j].unsqueeze(0)) for j, masked in enumerate(masked_features)]
            noisy_prelogits = [self.lp(self.gcn, noisy_seq, adj, sparse) for noisy_seq, adj in zip(noisy_features, adj_list)]
            noisy_logits = torch.cat(noisy_prelogits, dim=0)
            noisy_loss = compareloss(noisy_logits, self.negative_sample, temperature=1)
            loss_variances.append(noisy_loss)
        loss_variances = torch.stack(loss_variances)
        variance_loss = torch.var(loss_variances)
        total_loss = lploss + self.variance_weight * variance_loss
        print(f'lploss: {lploss.item()}, variance_loss: {variance_loss.item()}')
        return total_loss

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)
        return (h_1.detach(), c.detach())

def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))

def compareloss(feature, tuples, temperature):
    h_tuples = mygather(feature, tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp = temp.cuda()
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    print(pca.explained_variance_ratio_.sum())
    return seq

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
    return res.astype(int)
