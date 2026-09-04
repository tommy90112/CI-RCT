"""
Graph tables for the SMT2020 domain-transfer dataset (torch-free).

Builds node tables, feature matrices, integer timestamps, labels, split masks
and typed edge lists from the simulator trace + injection outputs.  Everything
here is pandas / numpy so the structural invariants (time-respecting edges,
acyclicity, GT alignment) can be tested on machines where the PyG extensions
do not load; utils.smt2020_loader turns a GraphTables into a HeteroData.

Node types (alphabetical — the order compute_type_offsets uses)
    lot         one per lot with ≥1 run.  time = completion or last run end.
    run         one per (lot, dispatch) event.  time = t_start.  Target type:
                y ∈ {1 anomaly observed, 0 clean metrology, -1 process/unknown}.
    tool_state  one per (machine, window) — contiguous windows per machine
                from the first to the last referenced window.  time = window
                start.

Edge types (src → dst, all time-respecting by construction)
    flows_to    run → run            consecutive runs of the same lot
    executes    tool_state → run     window containing the run's start
    wears       run → tool_state     window AFTER the one containing t_end
    progresses  tool_state → tool_state   consecutive windows, same machine
    belongs_to  run → lot

Synthetic measurements (the only features that are not derived from the trace)
    run.measurement        = metrology_signal · 1[y == 1] + N(0, 1) on metrology
                             runs, 0 elsewhere — the inline reading whose
                             signal-to-noise ratio E2 sweeps.
    tool_state.sensor      = tool_signal · 1[window overlaps an excursion on
                             that machine] + N(0, 1) — equipment FDC evidence;
                             0 by default so the root cause must be inferred
                             from topology and timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.smt2020_excursion import DEFAULT_METROLOGY_GROUPS, UNKNOWN_LABEL

NODE_TYPES = ("lot", "run", "tool_state")
EDGE_TYPES: Dict[str, Tuple[str, str]] = {
    "flows_to": ("run", "run"),
    "executes": ("tool_state", "run"),
    "wears": ("run", "tool_state"),
    "progresses": ("tool_state", "tool_state"),
    "belongs_to": ("run", "lot"),
}
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphConfig:
    window_hours: float = 8.0
    metrology_signal: float = 2.0
    tool_signal: float = 0.0
    feature_seed: int = 0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    split_seed: int = 0
    drop_before_days: float = 0.0
    metrology_groups: Tuple[str, ...] = DEFAULT_METROLOGY_GROUPS

    def __post_init__(self) -> None:
        if not self.window_hours > 0:
            raise ValueError(f"window_hours must be > 0, got {self.window_hours}")
        if self.metrology_signal < 0 or self.tool_signal < 0:
            raise ValueError("signal strengths must be >= 0")
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1
                and self.train_ratio + self.val_ratio < 1):
            raise ValueError("train_ratio and val_ratio must be in (0, 1) and sum to < 1")
        if self.drop_before_days < 0:
            raise ValueError("drop_before_days must be >= 0")

    @property
    def window_seconds(self) -> float:
        return self.window_hours * SECONDS_PER_HOUR


@dataclass(frozen=True)
class NodeTable:
    ids: pd.DataFrame               # identity columns; row position = local id
    x: np.ndarray                   # (n, d) float32
    feature_names: Tuple[str, ...]
    time: np.ndarray                # (n,) int64 seconds
    y: Optional[np.ndarray] = None  # (n,) int64 or None


@dataclass(frozen=True)
class GraphTables:
    config: GraphConfig
    nodes: Dict[str, NodeTable]
    edges: Dict[str, np.ndarray]                 # name -> (2, E) int64 local ids
    masks: Dict[str, Dict[str, np.ndarray]]      # node type -> split -> bool array


# ── Public API ─────────────────────────────────────────────────────────────────

def build_graph_tables(runs: pd.DataFrame, events: pd.DataFrame, lots: pd.DataFrame,
                       run_labels: pd.DataFrame, lot_labels: pd.DataFrame,
                       excursions: pd.DataFrame, cfg: GraphConfig) -> GraphTables:
    rng = np.random.default_rng(cfg.feature_seed)
    run_ids = _run_ids(runs, run_labels, cfg)
    if run_ids.empty:
        raise ValueError(
            f"no runs left after drop_before_days={cfg.drop_before_days} "
            f"(trace spans {runs['t_start'].min() / SECONDS_PER_DAY:.2f}–"
            f"{runs['t_end'].max() / SECONDS_PER_DAY:.2f} days)"
        )
    lot_ids = _lot_ids(lots, lot_labels, run_ids)
    tool_ids = _tool_state_ids(run_ids, cfg)

    nodes = {
        "lot": _lot_table(lot_ids, run_ids),
        "run": _run_table(run_ids, cfg, rng),
        "tool_state": _tool_state_table(tool_ids, run_ids, events, excursions, cfg, rng),
    }
    edges = _build_edges(run_ids, lot_ids, tool_ids, cfg)
    masks = {
        "run": _stratified_masks(nodes["run"].y, cfg),
        "lot": _stratified_masks(nodes["lot"].y, cfg),
    }
    return GraphTables(config=cfg, nodes=nodes, edges=edges, masks=masks)


def compute_local_offsets(tables: GraphTables) -> Dict[str, int]:
    """Global-ID offsets in alphabetical type order (mirrors compute_type_offsets)."""
    offsets, total = {}, 0
    for ntype in sorted(tables.nodes):
        offsets[ntype] = total
        total += len(tables.nodes[ntype].ids)
    return offsets


# ── Identity tables ────────────────────────────────────────────────────────────

def _run_ids(runs, run_labels, cfg) -> pd.DataFrame:
    keep = runs[runs["t_start"] >= cfg.drop_before_days * SECONDS_PER_DAY]
    keep = keep.sort_values("run_id").reset_index(drop=True)
    y = run_labels.set_index("run_id")["y"].reindex(keep["run_id"]).fillna(UNKNOWN_LABEL)
    out = keep.copy()
    out["y"] = y.to_numpy().astype(np.int64)
    out["is_metrology"] = out["machine_group"].isin(cfg.metrology_groups)
    out["w_start"] = np.floor(out["t_start"] / cfg.window_seconds).astype(np.int64)
    out["w_wear"] = np.floor(out["t_end"] / cfg.window_seconds).astype(np.int64) + 1
    return out


def _lot_ids(lots, lot_labels, run_ids) -> pd.DataFrame:
    present = lots[lots["lot_idx"].isin(run_ids["lot_idx"])].sort_values("lot_idx").reset_index(drop=True)
    fail = lot_labels.set_index("lot_idx")["fail"].reindex(present["lot_idx"]).fillna(UNKNOWN_LABEL)
    out = present.copy()
    out["y"] = fail.to_numpy().astype(np.int64)
    last_end = run_ids.groupby("lot_idx")["t_end"].max().reindex(out["lot_idx"]).to_numpy()
    done = out["done_at"].to_numpy(dtype=float)
    out["time"] = np.where(np.isnan(done), last_end, np.maximum(done, last_end)).astype(np.int64)
    out["n_runs"] = run_ids.groupby("lot_idx").size().reindex(out["lot_idx"]).to_numpy()
    return out


def _tool_state_ids(run_ids, cfg) -> pd.DataFrame:
    per_machine = run_ids.groupby("machine_idx").agg(
        w_min=("w_start", "min"), w_max=("w_wear", "max"),
        machine_group=("machine_group", "first"), machine_family=("machine_family", "first"))
    rows = []
    for machine_idx, r in per_machine.sort_index().iterrows():
        for w in range(int(r.w_min), int(r.w_max) + 1):
            rows.append((int(machine_idx), w, r.machine_group, r.machine_family))
    out = pd.DataFrame(rows, columns=["machine_idx", "window_idx", "machine_group", "machine_family"])
    out["time"] = (out["window_idx"] * cfg.window_seconds).astype(np.int64)
    return out


# ── Feature tables ─────────────────────────────────────────────────────────────

def _one_hot(values: pd.Series, prefix: str) -> Tuple[np.ndarray, list]:
    cats = sorted(values.unique())
    mat = np.stack([(values == c).to_numpy() for c in cats], axis=1).astype(np.float32)
    return mat, [f"{prefix}={c}" for c in cats]


def _run_table(run_ids, cfg, rng) -> NodeTable:
    group_x, group_names = _one_hot(run_ids["machine_group"], "group")
    route_len = run_ids.groupby("part")["step_order"].transform("max").to_numpy(dtype=float)
    y = run_ids["y"].to_numpy()
    is_met = run_ids["is_metrology"].to_numpy()
    measurement = np.where(is_met, cfg.metrology_signal * (y == 1) + rng.standard_normal(len(y)), 0.0)
    scalars = np.column_stack([
        is_met, (run_ids["t_end"] - run_ids["t_start"]) / SECONDS_PER_HOUR,
        run_ids["batch_size"], run_ids["priority"], run_ids["n_processed"] / route_len,
        run_ids["pm_triggered"].astype(float), measurement,
    ]).astype(np.float32)
    names = group_names + ["is_metrology", "duration_hours", "batch_size", "priority",
                           "step_progress", "pm_triggered", "measurement"]
    ids = run_ids[["run_id", "lot_idx", "machine_idx", "t_start", "t_end", "w_start", "w_wear"]].copy()
    return NodeTable(ids=ids, x=np.hstack([group_x, scalars]), feature_names=tuple(names),
                     time=run_ids["t_start"].to_numpy().astype(np.int64), y=y.astype(np.int64))


def _lot_table(lot_ids, run_ids) -> NodeTable:
    part_x, part_names = _one_hot(lot_ids["part"], "part")
    scalars = np.column_stack([
        lot_ids["priority"], lot_ids["n_runs"], lot_ids["done_at"].notna().astype(float),
        lot_ids["release_at"] / SECONDS_PER_DAY,
    ]).astype(np.float32)
    names = part_names + ["priority", "n_runs", "is_done", "release_day"]
    ids = lot_ids[["lot_idx", "part"]].copy()
    return NodeTable(ids=ids, x=np.hstack([part_x, scalars]), feature_names=tuple(names),
                     time=lot_ids["time"].to_numpy().astype(np.int64),
                     y=lot_ids["y"].to_numpy().astype(np.int64))


def _tool_state_table(tool_ids, run_ids, events, excursions, cfg, rng) -> NodeTable:
    group_x, group_names = _one_hot(tool_ids["machine_group"], "group")
    family_x, family_names = _one_hot(tool_ids["machine_family"], "family")
    key = pd.MultiIndex.from_frame(tool_ids[["machine_idx", "window_idx"]])
    n_runs = run_ids.groupby(["machine_idx", "w_start"]).size().reindex(key, fill_value=0).to_numpy()
    busy = _busy_hours(tool_ids, run_ids, cfg)
    ev = _event_features(tool_ids, events, cfg)
    active = _excursion_active(tool_ids, excursions, cfg)
    sensor = cfg.tool_signal * active + rng.standard_normal(len(tool_ids))
    scalars = np.column_stack([
        n_runs, busy / cfg.window_hours, ev["breakdown_in_window"], ev["pm_in_window"],
        ev["hours_since_repair"], sensor,
    ]).astype(np.float32)
    names = group_names + family_names + ["n_runs", "utilization", "breakdown_in_window",
                                           "pm_in_window", "hours_since_repair", "sensor"]
    ids = tool_ids[["machine_idx", "window_idx", "machine_group", "machine_family"]].copy()
    return NodeTable(ids=ids, x=np.hstack([group_x, family_x, scalars]), feature_names=tuple(names),
                     time=tool_ids["time"].to_numpy().astype(np.int64))


def _busy_hours(tool_ids, run_ids, cfg) -> np.ndarray:
    """Machine-busy hours inside each window (overlap of dispatch intervals)."""
    w = cfg.window_seconds
    rows = run_ids[["machine_idx", "t_start", "t_machine_end"]]
    parts = []
    for win in ("w_start", "w_end"):
        idx = np.floor((rows["t_start"] if win == "w_start" else rows["t_machine_end"]) / w).astype(np.int64)
        parts.append(rows.assign(window_idx=idx))
    span = pd.concat(parts).drop_duplicates(["machine_idx", "t_start", "window_idx"])
    lo = np.maximum(span["t_start"], span["window_idx"] * w)
    hi = np.minimum(span["t_machine_end"], (span["window_idx"] + 1) * w)
    span = span.assign(busy=np.clip(hi - lo, 0, None))
    key = pd.MultiIndex.from_frame(tool_ids[["machine_idx", "window_idx"]])
    return span.groupby(["machine_idx", "window_idx"])["busy"].sum().reindex(key, fill_value=0.0).to_numpy() / SECONDS_PER_HOUR


def _event_features(tool_ids, events, cfg) -> Dict[str, np.ndarray]:
    w = cfg.window_seconds
    ev = events.assign(window_idx=np.floor(events["t_start"] / w).astype(np.int64),
                       t_end=events["t_start"] + events["duration"])
    key = pd.MultiIndex.from_frame(tool_ids[["machine_idx", "window_idx"]])
    flags = {}
    for etype in ("breakdown", "pm"):
        hit = ev[ev["event_type"] == etype].groupby(["machine_idx", "window_idx"]).size()
        flags[f"{etype}_in_window"] = (hit.reindex(key, fill_value=0) > 0).to_numpy().astype(float)
    since = np.empty(len(tool_ids))
    ev_by_machine = {m: g["t_end"].sort_values().to_numpy() for m, g in ev.groupby("machine_idx")}
    for i, (m, win) in enumerate(zip(tool_ids["machine_idx"], tool_ids["window_idx"])):
        start = win * w
        ends = ev_by_machine.get(m)
        prior = ends[ends <= start] if ends is not None else np.empty(0)
        since[i] = (start - prior[-1]) / SECONDS_PER_HOUR if len(prior) else start / SECONDS_PER_HOUR
    flags["hours_since_repair"] = since
    return flags


def _excursion_active(tool_ids, excursions, cfg) -> np.ndarray:
    w = cfg.window_seconds
    active = np.zeros(len(tool_ids), dtype=bool)
    for ex in excursions.itertuples():
        hit = ((tool_ids["machine_idx"] == ex.machine_idx)
               & (tool_ids["window_idx"] * w < ex.t1) & ((tool_ids["window_idx"] + 1) * w > ex.t0))
        active |= hit.to_numpy()
    return active.astype(float)


# ── Edges ──────────────────────────────────────────────────────────────────────

def _build_edges(run_ids, lot_ids, tool_ids, cfg) -> Dict[str, np.ndarray]:
    tool_local = {(m, w): i for i, (m, w) in enumerate(zip(tool_ids["machine_idx"], tool_ids["window_idx"]))}
    lot_local = {l: i for i, l in enumerate(lot_ids["lot_idx"])}
    run_local = np.arange(len(run_ids))

    order = run_ids.sort_values(["lot_idx", "t_start", "run_id"])
    same_lot = order["lot_idx"].to_numpy()[1:] == order["lot_idx"].to_numpy()[:-1]
    pos = order.index.to_numpy()
    flows = np.stack([pos[:-1][same_lot], pos[1:][same_lot]])

    exec_src = np.array([tool_local[(m, w)] for m, w in zip(run_ids["machine_idx"], run_ids["w_start"])])
    wear_dst = np.array([tool_local[(m, w)] for m, w in zip(run_ids["machine_idx"], run_ids["w_wear"])])
    lot_dst = np.array([lot_local[l] for l in run_ids["lot_idx"]])

    same_machine = tool_ids["machine_idx"].to_numpy()[1:] == tool_ids["machine_idx"].to_numpy()[:-1]
    tpos = np.arange(len(tool_ids))
    progress = np.stack([tpos[:-1][same_machine], tpos[1:][same_machine]])

    return {name: arr.astype(np.int64) for name, arr in {
        "flows_to": flows,
        "executes": np.stack([exec_src, run_local]),
        "wears": np.stack([run_local, wear_dst]),
        "progresses": progress,
        "belongs_to": np.stack([run_local, lot_dst]),
    }.items()}


# ── Splits ─────────────────────────────────────────────────────────────────────

def _stratified_masks(y: np.ndarray, cfg: GraphConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.split_seed)
    masks = {s: np.zeros(len(y), dtype=bool) for s in SPLITS}
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_train = int(len(idx) * cfg.train_ratio)
        n_val = int(len(idx) * cfg.val_ratio)
        masks["train"][idx[:n_train]] = True
        masks["val"][idx[n_train:n_train + n_val]] = True
        masks["test"][idx[n_train + n_val:]] = True
    return masks
