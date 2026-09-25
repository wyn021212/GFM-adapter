# TAGFM

TAGFM adapts node features and graph structure for frozen graph backbones. It supports node classification and ego-graph classification with BRIDGE, MDGFM, SAMGPT, GRAVER, and GCOPE.

Feature adaptation uses a residual MLP and input-space MCR regularization with fixed soft assignments from the original backbone embeddings. Layer-normalized original and adapted embeddings are added, then classified with class prototypes, top-k pseudo-label refinement, and score diffusion. Per-dataset configuration files contain the runtime parameters for this method.

## Get Started

Install Git LFS, clone this repository using the URL shown by the **Code** button, then download the data and model assets from the repository directory:

```bash
git lfs install
git lfs pull
```

Use Python 3.12. Install PyTorch and the remaining dependencies:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The datasets, frozen checkpoints, adapted graphs, and fixed splits are included locally. Run all commands from this directory.

## Node Classification

```bash
python main.py --run_type adapt --task_type node_cls \
  --backbone BRIDGE --data_name Cora --k_shot 1
```

## Graph Classification

```bash
python main.py --run_type adapt --task_type graph_cls \
  --backbone BRIDGE --data_name Cora --k_shot 5
```

The graph task uses one capped two-hop ego-graph per center node. Samples use the center label and the released 10-by-4 neighbor rule, including repeated-neighbor weighting. Neighborhoods may overlap. The graph task shares the original dataset and fixed center splits with the node task.

Both commands run 50 splits. Use `--split_start 0 --num_splits 1` for one split. Add `--device cpu` to run on CPU. Results contain split indices, predictions, accuracy, configuration, and file checksums, followed by mean and sample standard deviation.

```bash
python main.py --backbone BRIDGE --data_name Cora --k_shot 1 --check
```

The check command verifies the requested local assets without starting training.

## Data and Configuration

The included datasets are Cora, Citeseer, Pubmed, Photo, Computers, Cornell, and Chameleon. Each dataset has 50 fixed 1-shot and 50 fixed 5-shot splits. The query set contains all centers outside the support set. Cornell uses `min(k, class_size)`; its 5-shot support counts are `[5, 1, 5, 5, 5]`.

Each checkpoint uses the other six domains as source domains. Parameters are stored in `configs/adapt/<backbone>/<dataset>/<shot>_shot.json`. Use `--config`, `--data_root`, `--checkpoint_root`, `--graph`, and `--output_dir` to supply alternate paths.

## Structure Generation

The packaged graphs are ready for evaluation. To generate a graph using the base structural recipe:

```bash
python main.py --run_type build_graph --backbone BRIDGE --data_name Cora
```

This writes a new graph under `outputs/graphs/`. Use `--graph` to evaluate that file. This command does not overwrite the packaged graph assets.

## Repository Structure

```text
configs/          Runtime options and per-dataset parameters
cores/            Frozen backbones and structural adaptation
data/             Dataset loading, splits, and graph operations
downstream/       Feature adapter and task training
utils/            Checkpoint, output, and asset utilities
datasets/         Raw and prepared datasets, fixed splits
checkpoints/      Frozen backbone weights
adapted_graphs/   Packaged graph structures
licenses/         Third-party license texts
main.py           Shared command-line entry
```

The repository layout follows a functional separation of configuration, models, data, and downstream tasks. Third-party code is identified in `THIRD_PARTY_NOTICES.md`.
