"""
SMT2020 HeteroData loader for CI-RCT.

Thin PyG layer over utils.smt2020_graph: reads the simulator trace and the
injection tables from disk, builds GraphTables (torch-free) and packs them
into a HeteroData with the attribute conventions the rest of CI-RCT expects
(x / y / train_mask / val_mask / test_mask / time per node type, edge_index
per (src, rel, dst)).  Also exposes the Metric B / C ground truth in the
global-ID convention of utils.data_utils.compute_type_offsets.

Directory layout (produced by scripts/smt2020/run_simulation.py and
scripts/smt2020/inject_excursion.py):

    <data_root>/lot_trace.csv  tool_events.csv  lots.csv  meta.json
    <data_root>/<excursion_subdir>/excursions.csv run_labels.csv lot_labels.csv gt_runs.csv

Target node type is "run"; y = 1 anomaly observed at metrology, 0 clean
metrology.  Unlabelled process runs (UNKNOWN_LABEL in GraphTables) are stored
as y = 0 with every mask False — the Elliptic++ class-3 convention — because
the NCM supervision (model.hetero_ncm.supervised_ncm_loss) applies BCE to the
labels of ALL destination nodes in the causal subgraph, not only masked ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils.smt2020_excursion import EXCURSION_COLUMNS, GT_RUN_COLUMNS, LOT_LABEL_COLUMNS, RUN_LABEL_COLUMNS
from utils.smt2020_graph import EDGE_TYPES, SPLITS, GraphConfig, GraphTables, build_graph_tables, compute_local_offsets
from utils.smt2020_gt import compute_excursion_ground_truth, ground_truth_to_global

TARGET_NODE_TYPE = "run"
DEFAULT_EXCURSION_SUBDIR = "excursion_seed0"
TRACE_FILES = ("lot_trace.csv", "tool_events.csv", "lots.csv")
INJECTION_FILES = ("excursions.csv", "run_labels.csv", "lot_labels.csv", "gt_runs.csv")
GT_MODES = ("strict", "extended")


@dataclass(frozen=True)
class Smt2020Paths:
    trace_dir: Path
    excursion_dir: Path


@dataclass(frozen=True)
class InjectionTables:
    excursions: pd.DataFrame
    run_labels: pd.DataFrame
    lot_labels: pd.DataFrame
    gt_runs: pd.DataFrame


# ── Disk I/O ───────────────────────────────────────────────────────────────────

def resolve_paths(data_root: str | Path, excursion_subdir: str = DEFAULT_EXCURSION_SUBDIR) -> Smt2020Paths:
    trace_dir = Path(data_root)
    excursion_dir = trace_dir / excursion_subdir
    for folder, files in ((trace_dir, TRACE_FILES), (excursion_dir, INJECTION_FILES)):
        missing = [f for f in files if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"SMT2020 data incomplete in {folder}: missing {missing}. Run "
                "scripts/smt2020/run_simulation.py then scripts/smt2020/inject_excursion.py."
            )
    return Smt2020Paths(trace_dir=trace_dir, excursion_dir=excursion_dir)


def read_trace(paths: Smt2020Paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(paths.trace_dir / "lot_trace.csv")
    events = pd.read_csv(paths.trace_dir / "tool_events.csv")
    lots = pd.read_csv(paths.trace_dir / "lots.csv")
    return runs, events, lots


def read_injection(paths: Smt2020Paths) -> InjectionTables:
    frames = {name: pd.read_csv(paths.excursion_dir / f"{name}.csv")
              for name in ("excursions", "run_labels", "lot_labels", "gt_runs")}
    for name, cols in (("excursions", EXCURSION_COLUMNS), ("run_labels", RUN_LABEL_COLUMNS),
                       ("lot_labels", LOT_LABEL_COLUMNS), ("gt_runs", GT_RUN_COLUMNS)):
        missing = [c for c in cols if c not in frames[name].columns]
        if missing:
            raise ValueError(f"{name}.csv is missing columns {missing}")
    return InjectionTables(**frames)


def load_smt2020_tables(data_root: str | Path, excursion_subdir: str,
                        cfg: GraphConfig) -> Tuple[GraphTables, InjectionTables]:
    paths = resolve_paths(data_root, excursion_subdir)
    runs, events, lots = read_trace(paths)
    inj = read_injection(paths)
    tables = build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions, cfg)
    return tables, inj


# ── HeteroData ─────────────────────────────────────────────────────────────────

def graph_tables_to_heterodata(tables: GraphTables) -> HeteroData:
    data = HeteroData()
    for ntype, node in tables.nodes.items():
        store = data[ntype]
        store.num_nodes = len(node.ids)
        store.x = torch.from_numpy(node.x)
        store.time = torch.from_numpy(node.time)
        if node.y is not None:
            store.y = torch.from_numpy(np.where(node.y < 0, 0, node.y).astype(np.int64))
        for split in SPLITS:
            if ntype in tables.masks:
                store[f"{split}_mask"] = torch.from_numpy(tables.masks[ntype][split])
    for name, ei in tables.edges.items():
        src, dst = EDGE_TYPES[name]
        data[(src, name, dst)].edge_index = torch.from_numpy(ei)
    return data


def load_smt2020_dataset(data_root: str | Path, excursion_subdir: str = DEFAULT_EXCURSION_SUBDIR,
                         window_hours: float = 8.0, metrology_signal: float = 2.0,
                         tool_signal: float = 0.0, drop_before_days: float = 2.0,
                         feature_seed: int = 0, split_seed: int = 0) -> Tuple[HeteroData, str]:
    """Build and return (HeteroData, target_node_type) — the train.py / evaluate.py contract."""
    cfg = GraphConfig(window_hours=window_hours, metrology_signal=metrology_signal,
                      tool_signal=tool_signal, drop_before_days=drop_before_days,
                      feature_seed=feature_seed, split_seed=split_seed)
    tables, _ = load_smt2020_tables(data_root, excursion_subdir, cfg)
    _print_summary(tables)
    return graph_tables_to_heterodata(tables), TARGET_NODE_TYPE


# ── Ground truth in global-ID space ────────────────────────────────────────────

def load_smt2020_ground_truth(data_root: str | Path, excursion_subdir: str, cfg: GraphConfig,
                              type_offsets: Dict[str, int], mode: str) -> Dict[int, Set[int]]:
    """Metric C ground truth keyed by global run ID (mode = strict | extended)."""
    if mode not in GT_MODES:
        raise ValueError(f"mode must be one of {GT_MODES}, got {mode!r}")
    tables, inj = load_smt2020_tables(data_root, excursion_subdir, cfg)
    _check_offsets(tables, type_offsets)
    gt = compute_excursion_ground_truth(tables, inj.run_labels, inj.gt_runs, inj.excursions)
    strict, extended = ground_truth_to_global(gt, type_offsets)
    return strict if mode == "strict" else extended


def load_smt2020_anomaly_entities(data_root: str | Path, excursion_subdir: str, cfg: GraphConfig,
                                  type_offsets: Dict[str, int]) -> Set[int]:
    """Global IDs of excursion tool_states ∪ root-defect runs ∪ observed runs (Metric B set)."""
    tables, inj = load_smt2020_tables(data_root, excursion_subdir, cfg)
    _check_offsets(tables, type_offsets)
    gt = compute_excursion_ground_truth(tables, inj.run_labels, inj.gt_runs, inj.excursions)
    return {type_offsets[t] + i for t, i in gt.anomaly_entities}


def _check_offsets(tables: GraphTables, type_offsets: Dict[str, int]) -> None:
    expected = compute_local_offsets(tables)
    if {k: type_offsets.get(k) for k in expected} != expected:
        raise ValueError(
            f"type_offsets {type_offsets} do not match the rebuilt tables {expected}; "
            "pass the same loader flags (window_hours, drop_before_days, excursion_subdir) "
            "as at load time"
        )


def _print_summary(tables: GraphTables) -> None:
    run = tables.nodes["run"]
    print(f"  SMT2020 graph: " + ", ".join(f"{k}={len(v.ids):,}" for k, v in tables.nodes.items()))
    print(f"  run labels: y=1 {int((run.y == 1).sum()):,}  y=0 {int((run.y == 0).sum()):,}  "
          f"unlabelled {int((run.y < 0).sum()):,}; edges " +
          ", ".join(f"{k}={v.shape[1]:,}" for k, v in tables.edges.items()))
