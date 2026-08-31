# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

All responses must be in **Traditional Chinese (繁體中文)**.

---

## Commands

All commands are run from the `CI-RCT/` subdirectory.

### Install dependencies

```bash
pip install -r CI-RCT/requirements.txt
# PyG extras (torch-scatter, torch-sparse) must be installed separately:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### Run tests

```bash
cd CI-RCT
pytest tests/                          # all tests
pytest tests/test_typed_causal_graph.py  # single file
pytest tests/ -v --cov=model --cov=utils --cov-report=term-missing
```

### One-command pipeline (recommended)

`run_pipeline.py` chains train → evaluate → export → static viewer, using the
frozen configuration in `pipeline/config.py`. Stages whose output already
exists are skipped, so re-running is cheap.

```bash
cd CI-RCT

python run_pipeline.py --dry-run          # show the plan + exact commands
python run_pipeline.py --device cuda      # full run, all three variants
python run_pipeline.py --from evaluate    # reuse existing checkpoints
python run_pipeline.py --force evaluate   # redo a stage (and everything downstream)
python run_pipeline.py --only frontend    # rebuild just the viewer
```

Two flags in that frozen config deliberately differ from the CLI defaults, and
from each other:

| Flag | Train | Evaluate | Why |
|------|-------|----------|-----|
| `--ncm_baseline` | `type_mean` | `marginal` | `marginal` hangs during training (per-step MLP over 822k wallets); it is an eval-time choice only |
| `--tracer_score` | — | `ce_signed` | matches the reported results; the argparse default is `ce` |

Outputs land where the viewer already looks (`frontend_temp/vite.config.ts`
serves `../viz` and `../results` directly), so no paths need matching by hand.

### Training (individual stages)

```bash
cd CI-RCT

# Phase 1 — backbone + NCM only (sanity check / ablation)
python train.py --dataset elliptic++ --epochs 100 --use_gan false

# Full training on Elliptic++ (best config, Exp-05: F1=0.8110)
python train.py \
  --dataset elliptic++ \
  --data_root data \
  --epochs 200 \
  --use_gan true \
  --include_addr_addr true \
  --hidden_dim 128 \
  --lambda_adversarial 0.1 \
  --lambda_stability 0.5

# Per-variant training (transaction / wallet / joint detection head)
python train.py --dataset elliptic++ --variant joint --epochs 200 --use_gan true
```

### Evaluation

```bash
cd CI-RCT

# Standard evaluation (all four metric dimensions)
python evaluate.py \
  --dataset elliptic++ \
  --checkpoint checkpoints/ci_rct_elliptic++_best.pt \
  --lfpn_mode both

# With debug diagnostics (CE distribution, chain depth histogram)
python evaluate.py --dataset elliptic++ --checkpoint <path> --debug
```

### Memory-constrained Elliptic++ options

| Flag | Purpose |
|------|---------|
| `--subsample_tx 20000` | Keep all fraud + random licit tx; fits 16 GB GPU with GAN |
| `--labeled_only true` | Only labeled tx + 1-hop neighbors (~1/10 graph size) |
| `--fraud_subgraph true` | Full tx, but wallet set restricted to BFS neighbors of labeled tx |
| `--include_addr_addr true` | Include 2.87M wallet→wallet edges (best F1, needs more RAM) |

---

## Architecture

The repository has two top-level packages:

- **`CXGNN/`** — upstream reference implementation (ECCV 2024); used only as a Related Work baseline. `CI-RCT/_cxgnn_path.py` appends it to `sys.path`.
- **`CI-RCT/`** — the main codebase described below.

### Four-module pipeline (`CI-RCT/model/`)

```
HeteroData (PyG)
    |
    v
[Module 1] HeteroGNNBackbone   — hetero_backbone.py
           HGT with per-relation attention heads
           Output: h_dict (node embeddings), logits (node classification)
    |
    +-------> [Module 4, training only] CausalAdversarialGAN — causal_adversarial_gan.py
    |          Generator: creates camouflage fraud nodes (DAG-topology constrained)
    |          Discriminator: the backbone itself (WGAN-GP)
    |
    v
[Module 2] Causal Intervention Engine
           typed_causal_graph.py  — TypedCausalGraph: directed, typed SCM (DAG with timestamps)
           hetero_ncm.py          — HeteroNCM: per-edge-type NNModel; do-calculus via parent-edge cutoff
           causal_shapley.py      — compute_asymmetric_causal_shapley(): O(n) prefix-coalition formula
           Output: CE(u→v) scores, φ^asym per parent node
    |
    v
[Module 3] RootCauseTracer      — root_cause_tracer.py
           Backward BFS on CE scores; four stop conditions
           Output: root cause node + full causal chain
```

### Entry points

| File | Purpose |
|------|---------|
| `run_pipeline.py` | One-command pipeline: train → evaluate → export → static viewer |
| `train.py` | CLI training; supports Phase 1 (no GAN) and Phase 2 (WGAN-GP) |
| `evaluate.py` | Four-dimension evaluation: classification, root cause, explanation quality, φ-stability |
| `infer.py` | Single-graph inference |

### Configuration

The pipeline's own frozen flag sets live in `pipeline/config.py` (transcribed from `ablation_plan.md` §0). All model hyperparameters live in `configs/config.py` as the frozen `CI_RCT_Config` dataclass. Nothing is hardcoded in model files. CLI flags in `train.py` / `evaluate.py` map 1-to-1 to config fields.

### Data loaders (`CI-RCT/utils/`)

| Loader | Dataset | Node types |
|--------|---------|-----------|
| `elliptic_plus_loader.py` | Elliptic++ (Bitcoin fraud) | `transaction`, `wallet` |
| `elliptic_plus_wallet_loader.py` | Elliptic++, wallet-as-target variant | `wallet`, `transaction` |
| `elliptic_plus_joint_loader.py` | Elliptic++, joint tx+wallet detection heads | `transaction`, `wallet` |
| `data_utils.py` | `build_typed_causal_graph_from_hetero()`, `compute_type_offsets()` |
| `lfpn_utils.py` | LFPN ground-truth for Elliptic++ Metric C |

### Dataset placement

Datasets are **not** included in the repository and must be downloaded manually:

```
CI-RCT/data/
  Elliptic++/
    txs_features.csv
    txs_classes.csv
    wallets_features.csv
    wallets_classes.csv
    AddrTx_edgelist.csv
    TxAddr_edgelist.csv
    TxTx_edgelist.csv
    AddrAddr_edgelist.csv
```

### Joint loss function

```
L_total = L_detection + λ1 · L_adversarial + λ2 · L_stability + λ3 · L_ncm

Defaults: λ1=0.1  λ2=0.5  λ3=0.1
```

- `L_detection`: CrossEntropy on target node type (class-weighted for imbalanced fraud data)
- `L_adversarial`: WGAN-GP critic/generator loss (Module 4)
- `L_stability`: ‖φ_t − φ_{t-1}‖² — prevents adversarial training from destabilising Causal Shapley values
- `L_ncm`: BCE supervising the NCM to predict fraud probability from parent embeddings

### Global node ID convention

`TypedCausalGraph` and `RootCauseTracer` use **global node IDs** that concatenate all node types in sorted order. `compute_type_offsets(data)` returns `{node_type: offset}` to convert local indices. Always pass global IDs when calling `causal_graph.parents()`, `causal_effects[(src, dst)]`, or `tracer.trace_root_cause()`.

### Key experimental results

Best configuration (Exp-05): `--include_addr_addr true` with bidirectional BFS for `bfs_adj` and unidirectional for `causal_adj`, giving **F1=0.8110 / AUC=0.9703** on Elliptic++.