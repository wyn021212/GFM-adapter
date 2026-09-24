import json
from configs.adapt_config import parse_config

def main(argv=None):
    args = parse_config(argv)
    if args.check:
        from utils.validation import check_assets
        print(json.dumps(check_assets(args), indent=2))
        return
    import torch
    torch.set_num_threads(2)
    if str(args.device).startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable. Install the matching PyTorch build or use --device cpu.')
    if args.run_type == 'build_graph':
        from cores.structure.builder import build_graph
        build_graph(args)
    else:
        from downstream.adapt_trainer import AdaptTrainer
        AdaptTrainer(args).run()
if __name__ == '__main__':
    main()
