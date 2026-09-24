import torch
import torch.nn as nn
from cores.backbones.samgpt.layers import AvgReadout, Discriminator

class GraphCL(nn.Module):

    def __init__(self, n_in, n_h, activation):
        super(GraphCL, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type, prompt_layers=None):
        h_0 = gcn(seq1, adj, sparse)
        if aug_type == 'edge':
            h_1 = gcn(seq1, aug_adj1, sparse, False, prompt_layers)
            h_3 = gcn(seq1, aug_adj2, sparse, False, prompt_layers)
        elif aug_type == 'mask':
            h_1 = gcn(seq3, adj, sparse, False, prompt_layers)
            h_3 = gcn(seq4, adj, sparse, False, prompt_layers)
        elif aug_type == 'node' or aug_type == 'subgraph':
            h_1 = gcn(seq3, aug_adj1, sparse, False, prompt_layers)
            h_3 = gcn(seq4, aug_adj2, sparse, False, prompt_layers)
        else:
            assert False
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)
        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)
        h_2 = gcn(seq2, adj, sparse, False, prompt_layers)
        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
