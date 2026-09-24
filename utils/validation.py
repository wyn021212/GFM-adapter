import json
from configs.base_config import DATASETS
from utils.checkpoints import checkpoint_metadata, sha256

def check_assets(args):
    checkpoint, metadata = checkpoint_metadata(args.checkpoint_root, args.backbone, args.data_name)
    if set(metadata['sources']) != set(DATASETS) - {args.data_name}:
        raise ValueError('Invalid checkpoint source domains')
    folder = args.data_root / args.data_name
    for name in ['feature.pt', 'adj.pt', 'labels.pt']:
        if not (folder / 'prepared' / name).is_file():
            raise FileNotFoundError(folder / 'prepared' / name)
    if args.backbone == 'GCOPE' and (not (folder / 'prepared/gcope_input.pt').is_file()):
        raise FileNotFoundError(folder / 'prepared/gcope_input.pt')
    for episode in range(args.split_start, args.split_start + args.num_splits):
        path = folder / 'splits' / f'{args.k_shot}_shot' / f'{episode}.json'
        split = json.loads(path.read_text(encoding='utf-8'))
        if set(split['support']) & set(split['query']):
            raise ValueError(f'Overlapping split indices: {path}')
    graph_hash = sha256(args.graph)
    if args.expected_graph_hash and args.expected_graph_hash != graph_hash:
        raise ValueError('Graph checksum mismatch')
    return dict(status='ok', task=args.task_type, backbone=args.backbone, dataset=args.data_name, shot=args.k_shot, num_splits=args.num_splits, checkpoint_sha256=metadata['sha256'], graph_sha256=graph_hash)
