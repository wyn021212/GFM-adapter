import numpy as np
import scipy.sparse as sp
import torch
from configs.base_config import DATASETS
from data.data_loader import load_dataset, load_tensor
from data.graph_utils import normalized
from downstream.adapter import set_seed
from utils.checkpoints import checkpoint_metadata

def load_model(name, device):
    if name == 'BRIDGE':
        from cores.backbones.bridge.model import PrePrompt
        model = PrePrompt(50, 256, 'prelu', np.zeros((8, 51), dtype=np.int64), 3, 0.1, 'mul', 1521434.9368374627, 6, 2)
    elif name == 'MDGFM':
        from cores.backbones.mdgfm.multidomain import MultiDomainPrePrompt
        model = MultiDomainPrePrompt(num_sources=6)
    elif name == 'SAMGPT':
        from cores.backbones.samgpt.preprompt import PrePrompt
        model = PrePrompt(50, 256, 'prelu', 6, 3, 0.1, 'mul')
    elif name == 'GRAVER':
        from cores.backbones.graver.model import PrePrompt
        model = PrePrompt(50, 256, 2, 0, 1, 1.0, 0.2, 'prelu', np.zeros((1, 51), dtype=np.int64), 1, 'mul', 6)
    else:
        raise ValueError(f'Unsupported backbone: {name}')
    return model.to(device)

class FrozenBackbone:

    def __init__(self, name, target, data_root, checkpoint_root, device):
        set_seed(39)
        self.name, self.target, self.device = (name, target, torch.device(device))
        self.checkpoint, self.metadata = checkpoint_metadata(checkpoint_root, name, target)
        if set(self.metadata['sources']) != set(DATASETS) - {target}:
            raise ValueError('Checkpoint source domains do not match the target fold')
        self.x, self.y, self.raw = load_dataset(data_root, target, self.device)
        weights = load_tensor(self.checkpoint)
        if name == 'GCOPE':
            from cores.backbones.gcope.encoder import GcopeEncoder
            self.graph = GcopeEncoder(data_root, target, weights, self.device)
            self.model, self.x = (self.graph.model, self.graph.x)
        else:
            self.model = load_model(name, self.device)
            self.model.load_state_dict(weights, strict=True)
            self.graph = self.numeric_graph(self.raw)
        self.model.eval().requires_grad_(False)
        self.initial_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def numeric_graph(self, raw):
        if self.name != 'GRAVER':
            return normalized(raw).to(self.device)
        matrix = raw.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
        values = torch.from_numpy(matrix.data)
        return torch.sparse_coo_tensor(indices, values, matrix.shape, device=self.device).coalesce()

    def encode(self, features, graph):
        if self.name == 'GCOPE':
            return graph.encode(features)
        if self.name == 'GRAVER':
            return self.model.gcn(features, graph)
        return self.model.gcn(features, graph, True, False).squeeze(0)

    def embed(self, features, edge_index, edge_weight):
        return self.encode(features, self.graph)

    def adapted(self, path):
        payload = load_tensor(path)
        if payload['checkpoint_sha256'] != self.metadata['sha256'] or payload['model'] != self.name or payload['target'] != self.target:
            raise ValueError('Adapted graph does not match checkpoint and dataset')
        if payload.get('smoke', False):
            raise ValueError('Incomplete graph assets cannot be used for evaluation')
        edges = payload['edge_index'].numpy()
        weights = payload['edge_weight'].numpy()
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Invalid edge weights')
        raw = sp.csr_matrix((weights, (edges[0], edges[1])), shape=self.raw.shape)
        raw.eliminate_zeros()
        if self.name == 'GCOPE':
            topology = raw.copy()
            topology.data[:] = 1
            graph = self.graph.with_graph(topology)
        else:
            graph = self.numeric_graph(raw)
        return (graph, raw)

    def verify_frozen(self):
        for key, value in self.model.state_dict().items():
            if not torch.equal(value.cpu(), self.initial_state[key]):
                raise RuntimeError(f'Frozen backbone state changed: {key}')
