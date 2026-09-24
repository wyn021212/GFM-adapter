import json
from pathlib import Path
import numpy as np
import torch

def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def load_dataset(root, name, device):
    folder = Path(root) / name / 'prepared'
    features = load_tensor(folder / 'feature.pt').float().to(device)
    labels = load_tensor(folder / 'labels.pt').long().reshape(-1).to(device)
    adjacency = load_tensor(folder / 'adj.pt').tocsr().astype(np.float32)
    if features.shape[0] != len(labels) or adjacency.shape != (len(labels), len(labels)):
        raise ValueError(f'Inconsistent dataset shapes: {name}')
    return (features, labels, adjacency)

def load_split(root, name, shot, episode, labels):
    path = Path(root) / name / 'splits' / f'{shot}_shot' / f'{episode}.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    support = torch.tensor(plan['support'], dtype=torch.long, device=labels.device)
    query = torch.tensor(plan['query'], dtype=torch.long, device=labels.device)
    if len(support.unique()) != len(support) or len(query.unique()) != len(query):
        raise ValueError('Repeated indices in split')
    if set(query.tolist()) != set(range(len(labels))) - set(support.tolist()):
        raise ValueError('Query indices must be the support complement')
    expected = labels.bincount().clamp(max=shot)
    if not torch.equal(labels[support].bincount(minlength=len(expected)), expected):
        raise ValueError('Support class counts do not match the split protocol')
    return (support, query, path)
