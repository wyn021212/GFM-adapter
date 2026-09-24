from dataclasses import asdict
from pathlib import Path
import statistics
import torch
from cores.models import FrozenBackbone
from data.data_loader import load_split
from data.graph_utils import normalized
from downstream import adapter
from downstream.tasks import pool_matrix
from utils.checkpoints import save_json, sha256

class AdaptTrainer:

    def __init__(self, config):
        self.config = config
        self.backbone = FrozenBackbone(config.backbone, config.data_name, config.data_root, config.checkpoint_root, config.device)

    def run(self):
        args, backbone = (self.config, self.backbone)
        graph_hash = sha256(args.graph)
        if args.expected_graph_hash and graph_hash != args.expected_graph_hash:
            raise ValueError('Adapted graph checksum mismatch')
        graph, raw = backbone.adapted(args.graph)
        propagation = normalized(raw).to(backbone.device)
        pool = pool_matrix(backbone.raw, backbone.device) if args.task_type == 'graph_cls' else None

        def readout(embedding):
            return torch.sparse.mm(pool, embedding) if pool is not None else embedding
        with torch.no_grad():
            original = readout(backbone.encode(backbone.x, backbone.graph)).detach()
        sentinel = object()

        def encode(features, input_graph):
            return original if input_graph is sentinel else readout(backbone.encode(features, input_graph))
        results = []
        destination = Path(args.output_dir) / args.task_type / args.backbone / args.data_name / f'{args.k_shot}_shot'
        for episode in range(args.split_start, args.split_start + args.num_splits):
            support, query, split_path = load_split(args.data_root, args.data_name, args.k_shot, episode, backbone.y)
            cfg = adapter.make_full_mlp_mcr_config(**dict(args.adapter_config, seed=39 + episode))
            original_loss = adapter.feature_mcr_loss
            if pool is not None:
                if cfg.mcr_assignment_source != 'original_embedding' and cfg.feature_regularizer == 'mcr':
                    raise ValueError('Graph classification requires original-embedding MCR assignments')

                def graph_loss(features, *values, **keywords):
                    population = readout(features) if cfg.mcr_target in ['input', 'low_rank'] else features
                    return original_loss(population, *values, **keywords)
                adapter.feature_mcr_loss = graph_loss
            try:
                output = adapter.fit_direct_adapter(encode_fn=encode, features=backbone.x, labels=None, support_index=support, support_labels=backbone.y[support], num_classes=int(backbone.y.max()) + 1, original_graph=sentinel, adapted_graph=graph, diffusion_propagation=propagation, config=cfg)
            finally:
                adapter.feature_mcr_loss = original_loss
            if not torch.isfinite(output.logits).all():
                raise RuntimeError('Non-finite prediction scores')
            predictions = output.logits[query].argmax(-1)
            score = float((predictions == backbone.y[query]).float().mean()) * 100
            backbone.verify_frozen()
            result = dict(task=args.task_type, backbone=args.backbone, dataset=args.data_name, shot=args.k_shot, split=episode, accuracy=score, support=support.tolist(), query=query.tolist(), predictions=predictions.tolist(), support_per_class=backbone.y[support].bincount().tolist(), config=asdict(cfg), checkpoint_sha256=backbone.metadata['sha256'], graph_sha256=graph_hash, split_sha256=sha256(split_path), sources=backbone.metadata['sources'])
            save_json(destination / f'{episode}.json', result)
            results.append(score)
            print(f'{args.task_type} {args.backbone} {args.data_name} {args.k_shot}-shot split={episode}: {score:.4f}', flush=True)
        summary = dict(task=args.task_type, backbone=args.backbone, dataset=args.data_name, shot=args.k_shot, splits=list(range(args.split_start, args.split_start + args.num_splits)), mean=statistics.mean(results), std=statistics.stdev(results) if len(results) > 1 else 0.0)
        save_json(destination / f'summary_{args.split_start}_{args.split_start + args.num_splits}.json', summary)
        print(f'Accuracy: {summary['mean']:.4f} +/- {summary['std']:.4f}', flush=True)
        return summary
