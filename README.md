# CI-RCT：Explainable Root Cause Tracing on Heterogeneous Graphs Based on Causal Intervention

繁體中文版：[README.zh-TW.md](README.zh-TW.md)

Graph neural networks are good at flagging anomalies. They are much weaker at the
question that follows: *where did this start, and who caused it?*

CI-RCT (Causal Intervention-based Root Cause Tracing) is a **methodological
framework** built to add that second half. Given a node flagged as anomalous, it
walks **backward** along a time-respecting directed graph — crossing freely between node types — until it reaches the entity the
evidence points to, and attaches a causal justification to every hop. The result is
a chain a human can audit, not a saliency heatmap.

![A causal chain traced by CI-RCT. From the flagged target on the left, the trace runs backward through alternating addresses and transactions to the traced source on the right, each selected edge labelled with its causal effect. Grey edges show the neighbouring nodes that were available at each hop but not selected.](CI-RCT/viz/fraud_chain.png)

*A traced chain. The labelled edges are what the tracer chose; the grey ones are the
alternatives it rejected at each hop. This instance comes from the Bitcoin dataset
used for validation — the method itself is defined over any typed, time-ordered
heterogeneous graph.*

## How it works

The core idea is to replace **correlation** with **intervention**. Where explainers
such as GNNExplainer or PGExplainer ask *which subgraph is most associated with this
prediction*, CI-RCT asks *what happens to the prediction if I cut this edge* —
Pearl's do-calculus, applied per edge.

Four modules. The GAN runs during training only and is inert at inference.

![CI-RCT architecture: a temporal hetero-graph feeds Module 1 (HeteroGNN backbone, HGTConv), whose embeddings drive Module 2 (Causal Intervention Engine — TypedCausalGraph, HeteroNCM producing CE, and asymmetric causal Shapley via coalition do-intervention); Module 3 (RootCauseTracer) walks backward over signed CE to emit the chain; Module 4 (CausalAdversarialGAN) adds WGAN-GP camouflaged samples during training only.](CI-RCT/viz/ch3_architecture_v3.svg)

| Module | Role |
|--------|------|
| **1 · HeteroGNN Backbone** | HGT with per-relation attention; produces node embeddings and detection logits |
| **2 · Causal Intervention Engine** | Builds a typed, timestamped causal DAG; estimates per-edge causal effects by parent-edge cutoff; computes asymmetric causal Shapley attribution by coalition intervention |
| **3 · RootCauseTracer** | Walks backward over causal effects to the root, with four stop conditions |
| **4 · CausalAdversarialGAN** | Training only: generates camouflaged anomalies under DAG constraints to harden detection (WGAN-GP) |

Module 2 emits two signals with different jobs. **CE** (edge-level causal effect)
ranks upstream candidates and drives the trace. **φ** (asymmetric causal Shapley,
computed by coalition intervention on the backbone) quantifies how much local causal
responsibility each upstream node bears. In short: **CE traces, φ explains.**

Two design choices make the chains hold up. Edges are oriented by timestamp and the
causal graph **rejects any edge that would let a cause follow its effect**, so a
chain cannot run backward through time. And attribution uses a rolling readout, so
nodes further from the target than the backbone's receptive field still receive a
measurable value instead of collapsing to zero.

## What the method needs from a graph

CI-RCT is not tied to a domain. Any setting that needs to answer *where an anomaly
began* — the origin of a fraudulent money flow, the entry point of lateral movement
in a security incident, the source of a fault propagating through an industrial
process — is within the framework's scope, provided the data can be expressed as a
PyG `HeteroData` object satisfying five conditions:

1. **At least two node types** — cross-type tracing is the point; a homogeneous graph
   reduces the method to ordinary backward search.
2. **Directed edges with timestamps**, forming a time-respecting DAG.
3. **Anomaly labels on the target node type.**
4. **Node feature vectors** per type.
5. **A root-cause criterion that can be operationalised** — some rule that decides
   whether a traced endpoint counts as correct.

This repository ships the **Elliptic++ instantiation**. Porting to another domain
means writing a loader that returns `HeteroData` plus a target type, and a
ground-truth builder for condition 5. The four model modules, the tracer and the
evaluation harness are unchanged.

## Validation on Elliptic++

Bitcoin fraud is a demanding test for this method: the graph has two node types that
genuinely alternate (transactions and wallet addresses), money flow gives edges an
unambiguous direction and time order, and the entity that ultimately controls funds
is usually several hops away from the transaction that gets flagged.

The framework is evaluated across four dimensions — detection performance,
root-cause tracing, explanation quality, and the stability of the attribution under
input perturbation — over three supervision variants (`transaction`, `wallet`,
`joint`) that share one graph and one explanation mechanism.

> Metric definitions, the ground-truth construction and the quantitative results are
> reported in the thesis. Everything needed to reproduce them is in this repository.

### Do the traces mean anything?

A backward walk will always produce *a* path. The question is whether it lands on a
structure an investigator would recognise.

Two things suggest it does. The tracer is **selecting, not drifting** — at every hop
it ranks all available upstream neighbours by causal effect and takes the strongest;
the grey branches in the figure at the top are the candidates it passed over. And the
chains it produces line up with laundering patterns already documented in the
literature.

**Example — a peeling chain.** A peeling chain is a well-known Bitcoin laundering
pattern: a large holding moves through a long series of transactions, each one
"peeling" a small amount off to a service or exchange while the bulk moves on to a
fresh change address. Repeated enough times, the trail becomes tedious to follow by
hand.

The trace below was produced with no knowledge of this pattern — the model only
followed causal effect upstream. The result carries the peeling signature throughout:
**strict alternation** between transactions and addresses over nine hops,
**timestamps non-decreasing** along the direction of flow, and **amounts decreasing
at every hop**.

![A depth-9 traced chain shown as a money flow: from the detected fraudulent transaction at top right, the chain runs backward through alternating addresses and transactions, each edge labelled with its causal effect and the BTC amount transferred, ending at the traced source address at bottom left. The transferred amount decreases at every hop.](CI-RCT/figures/fig_case5_peeling_210646674.png)

The chain reads as a scenario rather than a list of node IDs: funds leave the traced
source, are split down across a series of hops, and arrive at the transaction that
was ultimately flagged.

> Elliptic++ carries no typology labels, so this is a **structural signature match** —
> the chain is *consistent with* a peeling pattern, not confirmed to be one. The
> matching criteria are implemented in `scripts/typology_scan.py`.

## Roadmap

- **Second-domain validation.** The natural next step is a domain with the same
  shape but different semantics.
- **Loader contributions.** Any dataset meeting the five conditions above can be
  wired in without touching the model code.
- **Viewer.** Broader coverage of the explanation layers and a hosted demo.

## Requirements

### Hardware

Trained and evaluated on an **NVIDIA RTX PRO 6000 Blackwell (96 GB)**.

The reported configuration keeps all 2.87M wallet→wallet edges resident, which is
what drives the memory footprint. Training on CPU is not practical.

### Software

- Python 3.10+
- PyTorch 2.0+ and PyTorch Geometric 2.4+
- Node.js 18+ (only for the viewer)

## Installation

```bash
git clone <repo-url>
cd CI-RCT
pip install -r CI-RCT/requirements.txt
```

> **Install the PyG extras first.** `torch-scatter` and `torch-sparse` are compiled
> against an exact torch build. If they are mismatched, `import torch_geometric`
> **segfaults** instead of raising an error. Follow the
> [official PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> for your torch and CUDA version. `run_pipeline.py` verifies this before starting
> any long job.

### Dataset

Elliptic++ is not redistributed here. [Download it](https://github.com/git-disl/EllipticPlusPlus)
and place the tables as:

```
CI-RCT/data/Elliptic++/
  txs_features.csv       txs_classes.csv       txs_edgelist.csv
  wallets_features.csv   wallets_classes.csv
  AddrTx_edgelist.csv    TxAddr_edgelist.csv   AddrAddr_edgelist.csv
```

## Usage

One command runs everything — training, evaluation, chain export and the viewer
build — with the reported configuration already applied.

```bash
cd CI-RCT

python run_pipeline.py --dry-run        # show the plan; changes nothing
python run_pipeline.py --device cuda    # full run, all three variants
```

### What you get

```
checkpoints/ci_rct_elliptic++[_variant]_best.pt
viz/crime_chains[_variant].json     traced chains, per-node φ, feature attribution
results/crime_chains.csv            one row per chain
results/chain_neighbors.json        1-hop neighbourhood overlay
frontend_temp/dist/                 self-contained static viewer
```

### Running stages separately

```bash
python train.py    --dataset elliptic++ --variant joint --epochs 400 --use_gan true
python evaluate.py --dataset elliptic++ --checkpoint <path>
python infer.py    --dataset elliptic++ --checkpoint <path> --target_node <idx>
```

`run_pipeline.py --dry-run` prints the exact commands it would run, including every
flag — the easiest way to see the full configuration.

## Explainability viewer

`frontend_temp/` is a React + Vite viewer for the exported chains, in three layers:
the chain as a graph (L1), per-node contribution bars (L2), and per-feature causal
attribution at the responsibility pivot (L3).

```bash
cd CI-RCT/frontend_temp
npm install
npm run dev       # live — picks up regenerated exports automatically
npm run build     # static bundle in dist/, data included
```

## Project structure

```
CI-RCT/
  run_pipeline.py      one-command pipeline
  pipeline/            stage graph, frozen config, preflight checks
  train.py             training (Phase 1 no-GAN / Phase 2 WGAN-GP)
  evaluate.py          four-dimension evaluation and chain export
  infer.py             single-graph inference
  model/               the four modules
  utils/               loaders, causal graph construction, metrics
  configs/config.py    frozen hyperparameters
  scripts/             ablation drivers, figure and export utilities
  tests/               pytest suite
  frontend_temp/       explainability viewer
```

## Tests

```bash
cd CI-RCT
pytest tests/
```

## Citation

If you use this work in your research, please cite the thesis.

**APA 7th**

> Shih, Y. (2026). *CI-RCT: Explainable root cause tracing on heterogeneous
> graphs based on causal intervention* [Master's thesis, Tamkang University].
> Tamkang University Institutional Repository. <!-- TODO: thesis URL -->

**BibTeX**

```bibtex
@mastersthesis{shih2026circt,
  title   = {{CI-RCT: Explainable Root Cause Tracing on Heterogeneous Graphs Based on Causal Intervention}},
  author  = {Shih, Yuhung},
  school  = {Tamkang University},
  type    = {Master's thesis},
  address = {New Taipei City, Taiwan},
  year    = {2026},
  url     = {TODO}
}
```

## License

Released under the [MIT License](LICENSE). Third-party components are covered
by their own terms; see [NOTICE](NOTICE).
