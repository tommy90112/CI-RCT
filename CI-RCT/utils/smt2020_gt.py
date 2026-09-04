"""
Root-cause ground truth for the SMT2020 dataset (Metric B / C inputs).

Mirrors utils.lfpn_utils for Elliptic++, but here the truth is known by
construction from the injection tables instead of being operationalised:

    strict(r)    for an observed anomaly run r of lot L: the tool_state that
                 EXECUTED the lot's first root-defect run (the direct culprit).
    extended(r)  every tool_state window of that excursion's machine inside
                 [t0, t1)  ∪  the lot's root-defect runs that precede r.

Only runs whose lot carries an *excursion* defect get an entry; runs made
anomalous by background defects have no root cause to recover and are left
out (evaluate.py then skips them in Metric C, like unlabeled nodes).

anomaly_entities is the SMT2020 analogue of Elliptic++'s labelled-fraud set
used by root-cause precision: excursion tool_states ∪ root-defect runs ∪
observed anomaly runs.  Nodes are (node_type, local_id); use
ground_truth_to_global for the evaluate.py global-ID convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from utils.smt2020_graph import GraphTables

Node = Tuple[str, int]


@dataclass(frozen=True)
class ExcursionGroundTruth:
    strict: Dict[int, Set[Node]]
    extended: Dict[int, Set[Node]]
    excursion_tool_states: Dict[int, Set[int]]
    anomaly_entities: Set[Node]


def compute_excursion_ground_truth(tables: GraphTables, run_labels: pd.DataFrame,
                                   gt_runs: pd.DataFrame, excursions: pd.DataFrame) -> ExcursionGroundTruth:
    run_ids = tables.nodes["run"].ids
    ts_ids = tables.nodes["tool_state"].ids
    run_local = {rid: i for i, rid in enumerate(run_ids["run_id"])}
    ts_local = {(m, w): i for i, (m, w) in enumerate(zip(ts_ids["machine_idx"], ts_ids["window_idx"]))}
    w = tables.config.window_seconds

    roots = gt_runs[gt_runs["run_id"].isin(run_local)].sort_values(["lot_idx", "t_start", "run_id"])
    ex_windows = _excursion_windows(ts_ids, ts_local, excursions, set(roots["excursion_id"]), w)

    strict, extended = {}, {}
    observed = np.flatnonzero(tables.nodes["run"].y == 1)
    for local in observed:
        lot, t_obs = run_ids["lot_idx"].iat[local], run_ids["t_start"].iat[local]
        prior = roots[(roots["lot_idx"] == lot) & (roots["t_start"] < t_obs)]
        if prior.empty:
            continue
        first = prior.iloc[0]
        ex_id = int(first["excursion_id"])
        strict[int(local)] = {("tool_state", ts_local[(int(first["machine_idx"]), int(first["t_start"] // w))])}
        same_ex = prior[prior["excursion_id"] == ex_id]
        extended[int(local)] = ({("tool_state", i) for i in ex_windows[ex_id]}
                                | {("run", run_local[r]) for r in same_ex["run_id"]})

    entities = ({("tool_state", i) for ids in ex_windows.values() for i in ids}
                | {("run", run_local[r]) for r in roots["run_id"]}
                | {("run", int(i)) for i in observed})
    return ExcursionGroundTruth(strict=strict, extended=extended,
                                excursion_tool_states=ex_windows, anomaly_entities=entities)


def ground_truth_to_global(gt: ExcursionGroundTruth,
                           offsets: Dict[str, int]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Re-key strict / extended by global run ID with global-ID value sets."""
    def convert(mapping: Dict[int, Set[Node]]) -> Dict[int, Set[int]]:
        return {offsets["run"] + local: {offsets[t] + i for t, i in nodes}
                for local, nodes in mapping.items()}
    return convert(gt.strict), convert(gt.extended)


def _excursion_windows(ts_ids, ts_local, excursions, effective_ids, w) -> Dict[int, Set[int]]:
    """tool_state locals overlapping [t0, t1) for each excursion that produced ≥1 root defect."""
    out: Dict[int, Set[int]] = {}
    for ex in excursions.itertuples():
        if int(ex.excursion_id) not in effective_ids:
            continue
        hit = ((ts_ids["machine_idx"] == ex.machine_idx)
               & (ts_ids["window_idx"] * w < ex.t1) & ((ts_ids["window_idx"] + 1) * w > ex.t0))
        out[int(ex.excursion_id)] = {ts_local[(int(m), int(win))]
                                     for m, win in zip(ts_ids.loc[hit, "machine_idx"], ts_ids.loc[hit, "window_idx"])}
    return out
