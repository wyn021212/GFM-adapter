import torch

def pool_matrix(raw, device):
    import numpy as np
    import scipy.sparse as sp
    import torch
    raw = raw.tocsr(copy=True)
    raw.eliminate_zeros()
    raw.sort_indices()
    rows, cols, weights = ([], [], [])
    for center in range(raw.shape[0]):
        one = [int(v) for v in raw.indices[raw.indptr[center]:raw.indptr[center + 1]] if v != center][:10]
        ids = [center] + one
        for neighbor in one:
            ids.extend([int(v) for v in raw.indices[raw.indptr[neighbor]:raw.indptr[neighbor + 1]] if v != neighbor][:4])
        rows.extend([center] * len(ids))
        cols.extend(ids)
        weights.extend([1.0 / len(ids)] * len(ids))
    mat = sp.coo_matrix((np.asarray(weights, dtype=np.float32), (rows, cols)), shape=raw.shape)
    return torch.sparse_coo_tensor(torch.tensor(np.stack([mat.row, mat.col])), torch.tensor(mat.data), mat.shape, device=device).coalesce()
