"""
Frozen configuration for the end-to-end CI-RCT pipeline.

Every flag set here is transcribed from the configuration that produced the
thesis numbers (`ablation_plan.md` §0, mirrored in `scripts/run_loss_ablation.sh`).
Nothing is invented: if a value differs from the CLI default of `train.py` /
`evaluate.py`, it is because the default is *not* the frozen choice and the
flag must be passed explicitly.

Two traps are encoded here on purpose:

1. ``--ncm_baseline`` differs between train and eval.  Training MUST use
   ``type_mean``; ``marginal`` hangs during training because it fits a per-step
   MLP over 822k wallets.  ``marginal`` is an eval-time choice only.
2. ``--shapley_topk`` must be bounded for the φ_asym dump.  Each coalition is a
   full backbone forward, so an unbounded high-in-degree node makes the export
   effectively non-terminating on Elliptic++.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# ── Paths ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelinePaths:
    """Absolute locations the pipeline reads from and writes to.

    `root` is the CI-RCT/ package directory (the cwd every underlying script
    already assumes). The frontend resolves `../viz` and `../results` itself
    via vite.config.ts, so the dump destinations below are what wire the
    viewer up — they are not arbitrary.
    """

    root: Path
    data_root: Path
    checkpoint_dir: Path
    viz_dir: Path
    results_dir: Path
    frontend_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "PipelinePaths":
        root = root.resolve()
        return cls(
            root=root,
            data_root=root / "data",
            checkpoint_dir=root / "checkpoints",
            viz_dir=root / "viz",
            results_dir=root / "results",
            frontend_dir=root / "frontend_temp",
        )

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "Elliptic++"


# Elliptic++ tables that must be present before anything can run.
REQUIRED_DATASET_FILES: Tuple[str, ...] = (
    "txs_features.csv",
    "txs_classes.csv",
    "txs_edgelist.csv",
    "wallets_features.csv",
    "wallets_classes.csv",
    "AddrTx_edgelist.csv",
    "TxAddr_edgelist.csv",
    "AddrAddr_edgelist.csv",
)


# ── Detection variants ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    """One detection head configuration, trained and evaluated independently."""

    name: str
    #: `train.py` names the checkpoint ci_rct_<dataset><suffix>_best.pt, where
    #: the suffix is empty for the transaction variant.
    checkpoint_suffix: str
    #: Filename the viewer expects under viz/. The frontend loads
    #: `crime_chains.json` on mount and switches to `crime_chains_<name>.json`.
    chains_filename: str
    #: The joint dump is the viewer's landing view and the source of the flat
    #: CSV that the neighbour export reads.
    is_primary: bool

    def checkpoint_path(self, paths: PipelinePaths, dataset: str) -> Path:
        return paths.checkpoint_dir / f"ci_rct_{dataset}{self.checkpoint_suffix}_best.pt"

    def chains_path(self, paths: PipelinePaths) -> Path:
        return paths.viz_dir / self.chains_filename


VARIANTS: Tuple[Variant, ...] = (
    Variant("joint", "_joint", "crime_chains.json", is_primary=True),
    Variant("transaction", "", "crime_chains_transaction.json", is_primary=False),
    Variant("wallet", "_wallet", "crime_chains_wallet.json", is_primary=False),
)

VARIANT_NAMES: Tuple[str, ...] = tuple(v.name for v in VARIANTS)

DATASET = "elliptic++"
DEFAULT_EPOCHS = 400


def variant_by_name(name: str) -> Variant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(
        f"unknown variant {name!r}; expected one of {', '.join(VARIANT_NAMES)}"
    )


# ── Frozen flag sets ───────────────────────────────────────────────────────

#: Shared by train and eval. Determines which graph is built, so it MUST match
#: on both sides or the checkpoint will not line up with the evaluated graph.
DATA_FLAGS: Tuple[str, ...] = (
    "--dataset", DATASET,
    "--data_root", "data",
    "--include_addr_addr", "true",
)

#: ablation_plan.md §0 "Full" configuration, minus the per-variant and
#: per-run flags supplied by stages.py.
TRAIN_FLAGS: Tuple[str, ...] = (
    "--hidden_dim", "128",
    "--num_hgt_layers", "3",
    "--use_gan", "true",
    "--lambda_adversarial", "0.1",
    "--lambda_ncm", "0.3",
    "--lambda_stability", "0.5",
    "--lambda_aux_detection", "1.0",
    "--use_reconstruction", "false",
    "--symmetric_joint", "true",
    # type_mean, NOT marginal — see the module docstring.
    "--ncm_baseline", "type_mean",
    "--ncm_edge_balance", "sqrt",
)

#: Frozen evaluation configuration. `marginal` + `ce_signed` is the combination
#: the reported results use; both differ from the argparse defaults.
EVAL_FLAGS: Tuple[str, ...] = (
    "--num_hgt_layers", "3",
    "--max_explain", "2000",
    "--max_hops", "20",
    "--node_limit", "1000000",
    "--ce_threshold", "0.0001",
    "--threshold_tuning", "val",
    "--threshold_objective", "fraud_f1",
    "--prefer_root_types", "wallet",
    "--prefer_reachable_depth", "3",
    "--lfpn_mode", "both",
    "--ncm_baseline", "marginal",
    "--tracer_score", "ce_signed",
)

#: Extra flags that turn a numbers-only evaluation into a viewer export.
#: shapley_topk bounds the coalition count — see the module docstring.
DUMP_FLAGS: Tuple[str, ...] = (
    "--dump_phi",
    "--dump_feature_attribution",
    "--shapley_topk", "8",
)

#: Neighbour-overlay export. Reads the flat CSV produced by the primary
#: variant and streams the (large) raw edge lists once.
NEIGHBOR_CAP = 30
