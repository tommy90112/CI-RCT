# CI-RCT

**Causal Intervention-based Root Cause Tracing on heterogeneous graphs.**

繁體中文版：[README.zh-TW.md](README.zh-TW.md)

Given a transaction flagged as suspicious, CI-RCT walks *backwards* along a
time-respecting money-flow DAG — crossing between transaction and wallet node
types — to the entity that actually controlled the funds, and returns an
auditable causal chain rather than a single score.

---

## Why

Graph neural networks detect fraud well. They are much weaker at answering the
question an investigator actually asks next: **where did this start, how did it
propagate, and who is responsible?** Three gaps motivate this work.

| Gap | What is missing |
|-----|-----------------|
| **Heterogeneity** | Real networks mix node and edge types. Treating them as equivalent erases the type semantics that make a money trail readable. |
| **Correlation, not causation** | Explainers such as GNNExplainer and PGExplainer surface the subgraph most *statistically associated* with a prediction. That is not the same as the subgraph that *caused* it. |
| **No backward tracing** | Most pipelines stop at detection. Few can walk from an observed anomaly back up to its origin across layers of a system. |

CI-RCT addresses all three: a type-aware directed graph, causal effects
estimated by intervention (Pearl's do-calculus) rather than association, and a
tracer that reconstructs the chain from effect back to cause.

## What makes it different

- **Interventional, not associational.** Edge-level causal effects come from
  cutting parent edges and re-running the model, not from gradient saliency or
  learned edge masks.
- **Cross-type root causes.** The traced chain alternates between wallets and
  transactions, so a flagged transaction can be attributed to a controlling
  wallet. A detector that only scores nodes cannot produce this in principle.
- **Time-respecting by construction.** Edges are oriented by timestamp and the
  causal graph rejects any edge that would let a cause follow its effect.
- **Attribution that survives depth.** Per-hop local causal responsibility is
  computed with a rolling readout, so nodes further from the target than the
  backbone's receptive field still receive a measurable attribution.

## Architecture

Four modules. The GAN participates in training only and is inert at inference.

![CI-RCT architecture: a temporal hetero-graph feeds Module 1 (HeteroGNN backbone, HGTConv), whose embeddings drive Module 2 (Causal Intervention Engine — TypedCausalGraph, HeteroNCM producing CE, and asymmetric causal Shapley via coalition do-intervention); Module 3 (RootCauseTracer) walks backward over signed CE to emit the crime chain; Module 4 (CausalAdversarialGAN) adds WGAN-GP camouflaged samples during training only.](CI-RCT/viz/ch3_architecture_v3.svg)

Module 2 produces two parallel signals with distinct jobs: **CE** (edge-level
causal effect) ranks upstream candidates and drives the trace, while **φ**
(asymmetric causal Shapley) quantifies how much local causal responsibility
each upstream node bears for its immediate downstream transaction. CE traces;
φ explains.

## How it is evaluated

Four dimensions, all implemented in `evaluate.py`:

| Dimension | Question | Ground truth |
|-----------|----------|--------------|
| **A — Classification** | Is detection competitive with comparable methods? | Dataset labels |
| **B — Root cause tracing** | Does the trace terminate on a real fraud entity? | Labelled fraud entity set |
| **C — Explanation quality** | Does the chain contain the true originating wallet? | LFPN (Labeled Fraud Propagation Neighborhood), strict and k-hop extended |
| **D — Attribution robustness** | Does the attribution survive input perturbation? | Gaussian noise sweep over node embeddings |

Dimension A is a precondition check, not the contribution: tracing and
explanation only mean something if detection is not being traded away to get
them.

Three supervision variants share one graph and one explanation mechanism and
differ only in where the supervision signal is attached — `transaction`,
`wallet`, and `joint` (transaction primary, wallet auxiliary). Comparing them
side by side doubles as a test of how the supervision target affects each
dimension.

> Quantitative results are reported in the thesis and are not reproduced here
> yet. Everything needed to regenerate them is in this repository.

## Getting started

### Requirements

- Python 3.10+
- PyTorch 2.0+ and PyTorch Geometric 2.4+
- Node.js 18+ (only for the explainability viewer)

### Install

```bash
pip install -r CI-RCT/requirements.txt
```

> **The PyG extras are the one real installation trap.** `torch-scatter` and
> `torch-sparse` are compiled against an exact torch build. If they are
> mismatched, `import torch_geometric` **segfaults** instead of raising, which
> is confusing to debug. Install them following the
> [official PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> for your exact torch and CUDA version. `run_pipeline.py` checks this for you
> before starting any long job.

### Dataset

Elliptic++ is not redistributed here. Download it and place the tables as:

```
CI-RCT/data/Elliptic++/
  txs_features.csv       txs_classes.csv       txs_edgelist.csv
  wallets_features.csv   wallets_classes.csv
  AddrTx_edgelist.csv    TxAddr_edgelist.csv   AddrAddr_edgelist.csv
```

## Running it

The whole pipeline is one command. It chains training, evaluation, chain
export and the viewer build, wiring each stage's output into the next.

```bash
cd CI-RCT

python run_pipeline.py --dry-run        # show the plan and exact commands; changes nothing
python run_pipeline.py --device cuda    # full run: all three variants, then build the viewer
python run_pipeline.py --from evaluate  # reuse existing checkpoints
python run_pipeline.py --force evaluate # redo a stage (and everything downstream of it)
python run_pipeline.py --only frontend  # rebuild just the viewer
```

A stage is skipped when its outputs already exist, so re-running is cheap.
Forcing a stage also re-runs everything downstream — an export built from a
model you just retrained would otherwise be silently stale.

Outputs land where the viewer already looks for them:

```
checkpoints/ci_rct_elliptic++[_variant]_best.pt
viz/crime_chains[_variant].json     traced chains + per-node φ + feature attribution
results/crime_chains.csv            flat one-row-per-chain table
results/chain_neighbors.json        real 1-hop neighbourhood overlay
frontend_temp/dist/                 self-contained static viewer
```

### Running stages individually

```bash
cd CI-RCT

python train.py --dataset elliptic++ --variant joint --epochs 400 --use_gan true
python evaluate.py --dataset elliptic++ --checkpoint <path> --lfpn_mode both
python infer.py --dataset elliptic++ --checkpoint <path> --target_node <idx>
```

> **Two flags differ from their CLI defaults on purpose, and from each other.**
> `run_pipeline.py` sets them for you; if you invoke the scripts directly you
> have to set them yourself.
>
> | Flag | Training | Evaluation | Why |
> |------|----------|------------|-----|
> | `--ncm_baseline` | `type_mean` | `marginal` | `marginal` hangs during training (it fits a per-step MLP over ~800k wallets). It is an eval-time choice only. |
> | `--tracer_score` | — | `ce_signed` | Matches the reported configuration; the argparse default is `ce`. |
> | `--shapley_topk` | — | bounded (e.g. `8`) | Each coalition is a full backbone forward. Unbounded, the φ export does not terminate in practice. |

### Memory-constrained options

| Flag | Effect |
|------|--------|
| `--subsample_tx 20000` | Keep all fraud plus random licit transactions |
| `--labeled_only true` | Labelled transactions and their 1-hop neighbours only |
| `--fraud_subgraph true` | All transactions, wallets restricted to a BFS neighbourhood of labelled ones |
| `--include_addr_addr true` | Include wallet→wallet edges (needed for the reported configuration; costs RAM) |

## Explainability viewer

`frontend_temp/` is a React + Vite viewer for the exported chains, presenting
three layers:

- **L1** — the money chain as a graph, with causal responsibility shown per node
- **L2** — per-node contribution bars across the chain
- **L3** — per-feature causal attribution at the responsibility pivot

It reads the exports directly from `viz/` and `results/` (see
`frontend_temp/vite.config.ts`), so regenerating the data is picked up without
copying files around.

```bash
cd CI-RCT/frontend_temp
npm install
npm run dev       # live, reads the exports as they are regenerated
npm run build     # static bundle in dist/, data included
```

## Repository layout

```
CI-RCT/
  run_pipeline.py      one-command pipeline
  pipeline/            stage graph, frozen config, preflight checks
  train.py             training (Phase 1 no-GAN / Phase 2 WGAN-GP)
  evaluate.py          four-dimension evaluation and chain export
  infer.py             single-graph inference
  model/               the four modules; tracer_strategies/ holds tracer variants
  utils/               Elliptic++ loaders, causal graph construction, metrics, LFPN
  configs/config.py    frozen CI_RCT_Config dataclass — nothing is hardcoded in model files
  scripts/             ablation drivers, figure and export utilities
  tests/               pytest suite
  frontend_temp/       explainability viewer
CXGNN/                 upstream reference implementation, used as a baseline
```

### A note on global node IDs

`TypedCausalGraph` and `RootCauseTracer` address nodes by **global IDs** that
concatenate all node types in sorted order. `compute_type_offsets(data)`
converts local indices to global ones. Passing a local index where a global
one is expected is the most common source of confusing results.

## Tests

```bash
cd CI-RCT
pytest tests/
pytest tests/ -v --cov=model --cov=utils --cov=pipeline
```

## Related work and attribution

`CXGNN/` vendors the reference implementation of **"Graph Neural Network Causal
Explanation via Neural Causal Models"** (ECCV 2024,
[arXiv:2407.09378](https://arxiv.org/pdf/2407.09378)), used here as a related-work
baseline. It is MIT licensed and carries its own `LICENSE` file; it is not
authored by this project.

## Citation

<!-- TODO: fill in author, thesis title, institution and year before publishing. -->

```bibtex
@mastersthesis{circt,
  title  = {{TODO: thesis title}},
  author = {{TODO: author}},
  school = {{TODO: institution}},
  year   = {{TODO}}
}
```

## License

<!-- TODO: choose a license and add a LICENSE file at the repository root.
     Without one, the default is "all rights reserved" and nobody may reuse
     the code. Note that CXGNN/ is separately MIT licensed. -->

TODO.
