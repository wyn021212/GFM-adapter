import argparse
import json
from pathlib import Path
from configs.base_config import ROOT, BACKBONES, DATASETS

def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='TAGFM: feature and structure adaptation for frozen graph backbones')
    parser.add_argument('--run_type', choices=['adapt', 'build_graph'], default='adapt')
    parser.add_argument('--task_type', choices=['node_cls', 'graph_cls'], default='node_cls')
    parser.add_argument('--backbone', choices=BACKBONES, default='BRIDGE')
    parser.add_argument('--data_name', choices=DATASETS, default='Cora')
    parser.add_argument('--k_shot', type=int, choices=[1, 5], default=1)
    parser.add_argument('--split_start', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data_root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--checkpoint_root', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.split_start < 0 or args.num_splits < 1 or args.split_start + args.num_splits > 50:
        parser.error('The split interval must be contained in [0, 50).')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive.')
    config_path = args.config or ROOT / 'configs/adapt' / args.backbone / args.data_name / f'{args.k_shot}_shot.json'
    settings = json.loads(config_path.read_text(encoding='utf-8'))
    graph = Path(settings['graph'])
    args.graph = (args.graph or (graph if graph.is_absolute() else ROOT / graph)).resolve()
    args.settings = settings
    args.expected_graph_hash = settings.get('graph_sha256') if args.graph == (ROOT / graph).resolve() else None
    args.adapter_config = settings['adapter']
    if args.epochs is not None:
        args.adapter_config = dict(args.adapter_config, epochs=args.epochs)
    return args
