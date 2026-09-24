import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from cores.backbones.gcope.model import FAGCN
from data.data_loader import load_tensor

class InputGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, encoder):
        ctx.encoder = encoder
        ctx.save_for_backward(features)
        with torch.no_grad():
            return encoder.all_embeddings(features)

    @staticmethod
    def backward(ctx, gradient):
        features, = ctx.saved_tensors
        encoder = ctx.encoder
        result = torch.zeros_like(features)
        with torch.enable_grad():
            for batch in encoder.batches:
                ids = batch.n_id.to(features.device)
                local = features[ids].detach().requires_grad_(True)
                embedding = encoder.batch_embedding(local, batch)
                output_ids = batch.center_ids.to(features.device)
                partial = torch.autograd.grad(embedding, local, gradient[output_ids])[0]
                result.index_add_(0, ids, partial)
        return (result, None)

class GcopeEncoder:

    def __init__(self, data_root, target, weights, device):
        self.model = FAGCN(100, 128, 2, 0.2, 0.1).to(device)
        if not all((key.startswith('backbone.') for key in weights)):
            raise ValueError('Unexpected GCOPE checkpoint keys')
        self.model.load_state_dict({k.removeprefix('backbone.'): v for k, v in weights.items()}, strict=True)
        self.model.eval().requires_grad_(False)
        cached = load_tensor(Path(data_root) / target / 'prepared/gcope_input.pt')
        self.x = cached['x'].to(device)
        self.batches = cached['batches']
        ids = torch.cat([b.center_ids for b in self.batches])
        if not torch.equal(ids, torch.arange(len(self.x))):
            raise ValueError('GCOPE input cache has inconsistent node order')

    def batch_embedding(self, local, batch):
        model = self.model
        batch = batch.to(local.device)
        hidden = F.dropout(local, p=model.dropout, training=model.training)
        hidden = F.relu(model.t1(hidden))
        hidden = F.dropout(hidden, p=model.dropout, training=model.training)
        raw = hidden
        for layer in model.layers:
            hidden = layer(hidden, raw, batch.edge_index)
        return model.global_pool(model.t2(hidden), batch.batch)

    def all_embeddings(self, features):
        return torch.cat([self.batch_embedding(features[b.n_id.to(features.device)], b) for b in self.batches])

    def encode(self, features):
        if features.requires_grad:
            return InputGradient.apply(features, self)
        return self.all_embeddings(features)

    def with_graph(self, raw):
        incoming = raw.T.tocsr()
        rng = np.random.RandomState(39)
        all_ids = np.arange(raw.shape[0])
        graphs = []
        for node in all_ids:
            subset = np.array([node])
            frontier = subset
            for hop in range(1, 6):
                adjacent = [incoming.indices[incoming.indptr[i]:incoming.indptr[i + 1]] for i in frontier]
                next_ids = np.unique(np.concatenate(adjacent)) if adjacent else np.array([], dtype=np.int64)
                frontier = np.setdiff1d(next_ids, subset)
                subset = np.union1d(subset, next_ids)
                if hop >= 2 and (len(subset) >= 10 or hop == 5):
                    break
            if len(subset) < 10:
                fallback_ids = np.setdiff1d(all_ids, subset)
                subset = np.union1d(subset, rng.choice(fallback_ids, min(10 - len(subset), len(fallback_ids)), replace=False))
            if len(subset) > 30:
                subset = np.sort(np.r_[node, rng.choice(subset[subset != node], 29, replace=False)])
            subgraph = raw[subset][:, subset].tocoo()
            graphs.append(Data(n_id=torch.tensor(subset, dtype=torch.long), edge_index=torch.tensor(np.stack([subgraph.row, subgraph.col]), dtype=torch.long), num_nodes=len(subset), center_ids=torch.tensor([node])))
        encoder = copy.copy(self)
        encoder.batches = [Batch.from_data_list(graphs[i:i + 512]) for i in range(0, len(graphs), 512)]
        return encoder
