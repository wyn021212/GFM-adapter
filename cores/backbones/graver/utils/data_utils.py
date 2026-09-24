import os
from torch_geometric.data import Data
import torch
import scipy.sparse as sp
import numpy as np
from typing import List, Tuple

def inject_graphs_return_sparse(generated_graphs: List[Data], target_x: torch.Tensor, target_adj: torch.Tensor, idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(generated_graphs) == len(idx)
    device = target_x.device
    new_x = target_x.clone()
    N = target_x.size(0)
    row, col = target_adj.coalesce().indices() if target_adj.is_sparse else torch.nonzero(target_adj, as_tuple=True)
    row = row.tolist()
    col = col.tolist()
    for graph, tgt_node in zip(generated_graphs, idx):
        if graph.edge_index.numel() == 0:
            continue
        graph = graph.to(device)
        num_nodes = graph.num_nodes
        deg = torch.bincount(graph.edge_index[0], minlength=num_nodes)
        max_deg_node = torch.argmax(deg).item()
        new_node_indices = [j for j in range(num_nodes) if j != max_deg_node]
        n_add = len(new_node_indices)
        if n_add == 0:
            continue
        old2new = {old: N + i for i, old in enumerate(new_node_indices)}
        new_x = torch.cat([new_x, graph.x[new_node_indices]], dim=0)
        for s, t in graph.edge_index.t().tolist():
            s_new = tgt_node if s == max_deg_node else old2new.get(s)
            t_new = tgt_node if t == max_deg_node else old2new.get(t)
            if s_new is not None and t_new is not None:
                row.extend([s_new, t_new])
                col.extend([t_new, s_new])
        N += n_add
    edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
    edge_values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    new_adj = torch.sparse_coo_tensor(edge_index, edge_values, size=(new_x.size(0), new_x.size(0)))
    return (new_x, new_adj)
