import numpy as np
import scipy.sparse as sp

def normalized(adj):
    import torch
    a = adj.astype(np.float32) + sp.eye(adj.shape[0], dtype=np.float32, format='csr')
    d = np.asarray(a.sum(1)).reshape(-1)
    inv = np.power(d, -0.5)
    inv[np.isinf(inv)] = 0
    a = (a @ sp.diags(inv)).T @ sp.diags(inv)
    a = a.tocoo()
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack([a.row, a.col]).astype(np.int64)), torch.from_numpy(a.data.astype(np.float32)), a.shape).coalesce()
