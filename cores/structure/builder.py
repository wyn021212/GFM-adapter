from pathlib import Path
from types import SimpleNamespace
import torch
from cores.models import FrozenBackbone
from cores.structure import spectral
from cores.structure.edge_pairs import two_hop_edge_weights
from utils.checkpoints import save_json

def build_graph(config):
    backbone = FrozenBackbone(config.backbone, config.data_name, config.data_root, config.checkpoint_root, config.device)
    raw = backbone.raw.tocoo()
    edges = torch.tensor([raw.row.tolist(), raw.col.tolist()], dtype=torch.long, device=backbone.device)
    weights = torch.ones(edges.shape[1], dtype=backbone.x.dtype, device=backbone.device)
    centers, coefficients = spectral.gaussian_chebyshev_coefficients(16, 20, 0.1067, 256, backbone.device, backbone.x.dtype)
    response = spectral.blackbox_spectral_probe(backbone, backbone.x, edges, weights, coefficients, 16, 0.02, 1.0, 39 + 1709)
    pairs, initial = two_hop_edge_weights(edges, len(backbone.y), backbone.x.dtype, backbone.device)
    parameters = SimpleNamespace(adapt_epochs=2, adapt_lr=0.03, gradient_clip=5.0, maximum_edge_weight=2.0, preserve_edge_mass=True, sinkhorn_regularization=0.05, sinkhorn_iterations=30, transport_gap_closure=1.0)
    signal = spectral.whiten_features(backbone.x, max_rank=16)
    edge_index, edge_weight = spectral.adapt_full_graph(pairs, initial, signal, coefficients, centers, response, parameters)
    if not torch.isfinite(edge_weight).all() or (edge_weight < 0).any():
        raise RuntimeError('Invalid generated edge weights')
    backbone.verify_frozen()
    payload = dict(model=config.backbone, target=config.data_name, checkpoint_sha256=backbone.metadata['sha256'], sources=backbone.metadata['sources'], edge_index=edge_index.cpu(), edge_weight=edge_weight.cpu(), config=vars(parameters), probe_count=16, smoke=False)
    path = Path(config.output_dir) / 'graphs' / config.backbone / f'{config.data_name}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    save_json(path.with_suffix('.json'), {key: value for key, value in payload.items() if key not in ['edge_index', 'edge_weight']})
    print(path)
