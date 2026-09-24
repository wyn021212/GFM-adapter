import torch
from torch.nn import functional as F
from cores.backbones.mdgfm.preprompt import PrePrompt, textprompt
from cores.backbones.mdgfm.utils import Calbound

class MultiDomainPrePrompt(PrePrompt):

    def __init__(self, n_in=50, n_h=256, num_layers=3, dropout=0.1, prompt_type='mul', num_sources=7):
        if num_sources < 1:
            raise ValueError('num_sources must be positive')
        super().__init__(n_in, n_h, 'prelu', 1, num_layers, dropout, prompt_type)
        self.num_sources = num_sources
        for prefix, dim in [('pretext', n_in), ('texttoken', n_h), ('balancetoken', 2 * n_in)]:
            for i in range(6, num_sources + 1):
                setattr(self, f'{prefix}{i}', textprompt(dim, prompt_type))
            for i in range(num_sources + 1, 6):
                delattr(self, f'{prefix}{i}')

    def forward(self, features, adjs, ks):
        if not len(features) == len(adjs) == len(ks) == self.num_sources:
            raise ValueError('Every source requires features, adjacency and k')
        pre, refined = ([], [])
        for i, (x, adj, k) in enumerate(zip(features, adjs, ks), 1):
            x = x.squeeze(0)
            h = self.sumtext(F.relu(getattr(self, f'pretext{i}')(x)))
            r = getattr(self, f'balancetoken{i}')(torch.cat((h, torch.sparse.mm(adj, h)), dim=1))
            pre.append(h)
            refined.append(self.learner.graph_process(k, r))
        z = [self.lp(self.gcn, x, a, True) for x, a in zip(pre, refined)]
        v = [self.lp(self.gcn, x, a, True) for x, a in zip(pre, adjs)]
        identity_losses = [Calbound.calc_lower_bound(a, b, torch.eye(a.shape[0], device=a.device)) for a, b in zip(z, v)]
        topology_losses = [Calbound.calc_lower_bound(a, b, c.detach()) for a, b, c in zip(z, v, refined)]
        return sum(identity_losses) + sum(topology_losses)

    def embedding(self, features, adjs):
        if len(features) != self.num_sources or len(adjs) != self.num_sources:
            raise ValueError('Incorrect source count')
        return tuple((self.lp(self.gcn, getattr(self, f'pretext{i}')(x.squeeze(0)), adj, True).detach() for i, (x, adj) in enumerate(zip(features, adjs), 1)))
