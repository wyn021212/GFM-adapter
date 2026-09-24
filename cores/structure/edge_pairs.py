import numpy as np
import scipy.sparse as sp
import torch

def two_hop_edge_weights(edge_index, num_nodes, dtype, device):
    edges = edge_index.detach().cpu().numpy()
    adjacency = sp.csr_matrix((np.ones(edges.shape[1], dtype=np.float32), edges), shape=(num_nodes, num_nodes))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1
    edge_pairs = sp.triu(adjacency + adjacency @ adjacency, k=1).tocoo()
    keys = edge_pairs.row.astype(np.int64) * num_nodes + edge_pairs.col
    order = np.argsort(keys)
    row, col = (edge_pairs.row[order], edge_pairs.col[order])
    pairs = torch.as_tensor(np.stack([row, col]), dtype=torch.long, device=device)
    weights = torch.as_tensor(np.asarray(adjacency[row, col]).reshape(-1), dtype=dtype, device=device)
    assert pairs.shape[1] == len(np.unique(keys))
    return (pairs, weights)
