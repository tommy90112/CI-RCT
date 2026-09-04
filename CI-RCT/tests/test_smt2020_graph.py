"""
Tests for utils.smt2020_graph (pure pandas/numpy graph tables) and
utils.smt2020_gt (root-cause ground truth), using tests/smt2020_toy.py.

No torch / PyG import here on purpose: these invariants must be checkable on
a machine where the PyG extensions cannot load.

Toy geometry with a 4 h window: machine 0 has process runs starting at hours
{0,2,4,...,18} (lots 0 and 2), each 1 h long, so its tool_state windows run
contiguously from window 0 to window 5 (last run ends at 19 h → next window).
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from tests.smt2020_toy import H, N_LOTS, RUNS_PER_LOT, toy_config, toy_events, toy_lots, toy_runs
from utils.smt2020_excursion import UNKNOWN_LABEL, inject_excursions
from utils.smt2020_graph import (
    EDGE_TYPES,
    NODE_TYPES,
    GraphConfig,
    build_graph_tables,
    compute_local_offsets,
)
from utils.smt2020_gt import compute_excursion_ground_truth, ground_truth_to_global

WINDOW_H = 4.0


@pytest.fixture(scope="module")
def injected():
    runs, events, lots = toy_runs(), toy_events(), toy_lots()
    return runs, events, lots, inject_excursions(runs, events, lots, toy_config())


@pytest.fixture(scope="module")
def tables(injected):
    runs, events, lots, inj = injected
    cfg = GraphConfig(window_hours=WINDOW_H, metrology_signal=5.0, tool_signal=5.0, feature_seed=0)
    return build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions, cfg)


def _run_time(tables, local):
    return tables.nodes["run"].time[local]


# ── nodes ──────────────────────────────────────────────────────────────────────

def test_node_types_are_sorted_and_counts_match(tables):
    assert NODE_TYPES == ("lot", "run", "tool_state") == tuple(sorted(NODE_TYPES))
    assert set(tables.nodes) == set(NODE_TYPES)
    assert len(tables.nodes["run"].ids) == N_LOTS * RUNS_PER_LOT
    assert len(tables.nodes["lot"].ids) == N_LOTS
    ts = tables.nodes["tool_state"].ids
    assert (ts.machine_idx == 0).sum() == 6           # windows 0..5, hand-computed
    for _, g in ts.groupby("machine_idx"):
        w = sorted(g.window_idx)
        assert w == list(range(w[0], w[-1] + 1))    # contiguous per machine


def test_node_tables_have_aligned_features_and_times(tables):
    for ntype, node in tables.nodes.items():
        n = len(node.ids)
        assert node.x.shape == (n, len(node.feature_names)) and node.x.dtype == np.float32
        assert node.time.shape == (n,) and node.time.dtype == np.int64
        assert np.isfinite(node.x).all()
    ts = tables.nodes["tool_state"]
    assert (ts.time == (ts.ids.window_idx * WINDOW_H * H).astype(np.int64)).all()
    run = tables.nodes["run"]
    assert (run.time == run.ids.t_start.astype(np.int64)).all()


def test_lot_time_is_after_every_run_of_the_lot(tables):
    run, lot = tables.nodes["run"], tables.nodes["lot"]
    last_end = run.ids.groupby("lot_idx").t_end.max()
    for local, row in lot.ids.iterrows():
        assert lot.time[local] >= last_end[row.lot_idx]


# ── edges ──────────────────────────────────────────────────────────────────────

def test_edge_types_and_local_index_ranges(tables):
    assert set(tables.edges) == set(EDGE_TYPES)
    for name, ei in tables.edges.items():
        src_t, dst_t = EDGE_TYPES[name]
        assert ei.shape[0] == 2 and ei.dtype == np.int64
        assert ei[0].max() < len(tables.nodes[src_t].ids) and ei[0].min() >= 0
        assert ei[1].max() < len(tables.nodes[dst_t].ids) and ei[1].min() >= 0


def test_every_edge_respects_time(tables):
    for name, ei in tables.edges.items():
        src_t, dst_t = EDGE_TYPES[name]
        t_src = tables.nodes[src_t].time[ei[0]]
        t_dst = tables.nodes[dst_t].time[ei[1]]
        assert (t_src <= t_dst).all(), name
    wears = tables.edges["wears"]
    assert (tables.nodes["run"].time[wears[0]] < tables.nodes["tool_state"].time[wears[1]]).all()


def test_flows_to_links_consecutive_runs_of_each_lot(tables):
    ei = tables.edges["flows_to"]
    assert ei.shape[1] == N_LOTS * (RUNS_PER_LOT - 1)
    ids = tables.nodes["run"].ids
    for s, d in ei.T:
        assert ids.lot_idx[s] == ids.lot_idx[d]
        assert ids.t_start[s] < ids.t_start[d]
    # no lot skips a run: out-degree ≤ 1 and in-degree ≤ 1 per run
    assert pd.Series(ei[0]).value_counts().max() == 1
    assert pd.Series(ei[1]).value_counts().max() == 1


def test_executes_points_to_the_window_containing_the_run_start(tables):
    ei = tables.edges["executes"]
    assert ei.shape[1] == N_LOTS * RUNS_PER_LOT           # exactly one per run
    ts, run = tables.nodes["tool_state"].ids, tables.nodes["run"].ids
    for s, d in ei.T:
        assert ts.machine_idx[s] == run.machine_idx[d]
        assert ts.window_idx[s] == int(run.t_start[d] // (WINDOW_H * H))


def test_wears_points_to_the_window_after_the_run_end(tables):
    ei = tables.edges["wears"]
    ts, run = tables.nodes["tool_state"].ids, tables.nodes["run"].ids
    for s, d in ei.T:
        assert ts.machine_idx[d] == run.machine_idx[s]
        assert ts.window_idx[d] == int(run.t_end[s] // (WINDOW_H * H)) + 1


def test_progresses_chains_consecutive_windows(tables):
    ei = tables.edges["progresses"]
    ts = tables.nodes["tool_state"].ids
    n_expected = sum(len(g) - 1 for _, g in ts.groupby("machine_idx"))
    assert ei.shape[1] == n_expected
    for s, d in ei.T:
        assert ts.machine_idx[s] == ts.machine_idx[d]
        assert ts.window_idx[d] == ts.window_idx[s] + 1


def test_belongs_to_covers_every_run(tables):
    ei = tables.edges["belongs_to"]
    assert ei.shape[1] == N_LOTS * RUNS_PER_LOT
    run, lot = tables.nodes["run"].ids, tables.nodes["lot"].ids
    for s, d in ei.T:
        assert run.lot_idx[s] == lot.lot_idx[d]


def test_graph_is_acyclic(tables):
    offsets = compute_local_offsets(tables)
    adj, indeg = defaultdict(list), defaultdict(int)
    nodes = set()
    for name, ei in tables.edges.items():
        src_t, dst_t = EDGE_TYPES[name]
        for s, d in ei.T:
            u, v = offsets[src_t] + int(s), offsets[dst_t] + int(d)
            adj[u].append(v); indeg[v] += 1; nodes.update((u, v))
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        u = queue.pop(); seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    assert seen == len(nodes)


# ── labels, masks, features ────────────────────────────────────────────────────

def test_run_labels_and_masks(injected, tables):
    _, _, _, inj = injected
    run = tables.nodes["run"]
    expected = inj.run_labels.set_index("run_id").y.reindex(run.ids.run_id).to_numpy()
    assert (run.y == expected).all()
    labelled = run.y != UNKNOWN_LABEL
    m = tables.masks["run"]
    assert set(m) == {"train", "val", "test"}
    union = m["train"] | m["val"] | m["test"]
    assert (union == labelled).all()
    assert not (m["train"] & m["val"]).any() and not (m["train"] & m["test"]).any() \
        and not (m["val"] & m["test"]).any()
    assert m["train"].sum() >= m["val"].sum() and m["train"].sum() >= m["test"].sum()


def test_lot_labels_follow_final_fail(injected, tables):
    _, _, _, inj = injected
    lot = tables.nodes["lot"]
    expected = inj.lot_labels.set_index("lot_idx").fail.reindex(lot.ids.lot_idx).to_numpy()
    assert (lot.y == expected).all()
    assert (tables.masks["lot"]["train"] | tables.masks["lot"]["val"] | tables.masks["lot"]["test"]
            == (lot.y != UNKNOWN_LABEL)).all()


def test_metrology_signal_separates_observed_runs(tables):
    run = tables.nodes["run"]
    names = list(run.feature_names)
    assert "measurement" in names and "is_metrology" in names
    meas = run.x[:, names.index("measurement")]
    is_met = run.x[:, names.index("is_metrology")] == 1
    assert (meas[~is_met] == 0).all()
    assert meas[is_met & (run.y == 1)].mean() > meas[is_met & (run.y == 0)].mean() + 3


def test_tool_signal_marks_excursion_windows(injected, tables):
    _, _, _, inj = injected
    ts = tables.nodes["tool_state"]
    names = list(ts.feature_names)
    sensor = ts.x[:, names.index("sensor")]
    ex = inj.excursions.iloc[0]
    w = WINDOW_H * H
    active = ((ts.ids.machine_idx == ex.machine_idx)
              & (ts.ids.window_idx * w < ex.t1) & ((ts.ids.window_idx + 1) * w > ex.t0)).to_numpy()
    assert active.sum() == 2                              # windows [4,8) and [8,12) overlap [5,11)
    assert sensor[active].mean() > sensor[~active].mean() + 3
    for col in ("n_runs", "breakdown_in_window", "hours_since_repair"):
        assert col in names


def test_feature_seed_controls_noise_only(injected):
    runs, events, lots, inj = injected
    a = build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions,
                           GraphConfig(window_hours=WINDOW_H, feature_seed=0))
    b = build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions,
                           GraphConfig(window_hours=WINDOW_H, feature_seed=1))
    for name in EDGE_TYPES:
        assert np.array_equal(a.edges[name], b.edges[name])
    assert not np.array_equal(a.nodes["run"].x, b.nodes["run"].x)


def test_drop_before_days_trims_early_runs(injected):
    runs, events, lots, inj = injected
    t = build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions,
                           GraphConfig(window_hours=WINDOW_H, drop_before_days=0.25))   # 6 h
    run = t.nodes["run"]
    assert (run.ids.t_start >= 6 * H).all()
    assert len(run.ids) == (runs.t_start >= 6 * H).sum()
    assert set(t.nodes["lot"].ids.lot_idx) == set(run.ids.lot_idx)
    ei = t.edges["flows_to"]
    assert ei.shape[1] == sum(len(g) - 1 for _, g in run.ids.groupby("lot_idx"))


def test_trimming_everything_is_a_clear_error(injected):
    runs, events, lots, inj = injected
    with pytest.raises(ValueError, match="drop_before_days"):
        build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions,
                           GraphConfig(window_hours=WINDOW_H, drop_before_days=5.0))


def test_config_validation():
    with pytest.raises(ValueError):
        GraphConfig(window_hours=0)
    with pytest.raises(ValueError):
        GraphConfig(train_ratio=0.9, val_ratio=0.2)
    with pytest.raises(ValueError):
        GraphConfig(metrology_signal=-1)


# ── ground truth ───────────────────────────────────────────────────────────────

def test_local_offsets_follow_alphabetical_order(tables):
    off = compute_local_offsets(tables)
    assert off["lot"] == 0
    assert off["run"] == len(tables.nodes["lot"].ids)
    assert off["tool_state"] == off["run"] + len(tables.nodes["run"].ids)


def test_ground_truth_for_observed_runs(injected, tables):
    runs, _, _, inj = injected
    gt = compute_excursion_ground_truth(tables, inj.run_labels, inj.gt_runs, inj.excursions)
    run, ts = tables.nodes["run"].ids, tables.nodes["tool_state"].ids
    observed = run.index[tables.nodes["run"].y == 1]
    assert set(gt.strict) == set(observed) == set(gt.extended)
    ts_local = {(m, w): i for i, (m, w) in enumerate(zip(ts.machine_idx, ts.window_idx))}
    run_local = {rid: i for i, rid in enumerate(run.run_id)}
    w = WINDOW_H * H
    for local in observed:
        lot = run.lot_idx[local]
        first_root_t = 8 * H if lot == 0 else 6 * H
        assert gt.strict[local] == {("tool_state", ts_local[(0, int(first_root_t // w))])}
        prior = inj.gt_runs[(inj.gt_runs.lot_idx == lot) & (inj.gt_runs.t_start < run.t_start[local])]
        root_runs = {("run", run_local[r]) for r in prior.run_id}   # only roots BEFORE the observation
        assert gt.extended[local] == {("tool_state", ts_local[(0, 1)]), ("tool_state", ts_local[(0, 2)])} | root_runs
        assert gt.strict[local] <= gt.extended[local]
    assert gt.excursion_tool_states == {0: {ts_local[(0, 1)], ts_local[(0, 2)]}}
    expected_entities = ({("tool_state", ts_local[(0, 1)]), ("tool_state", ts_local[(0, 2)])}
                         | {("run", run_local[r]) for r in inj.gt_runs.run_id}
                         | {("run", int(l)) for l in observed})
    assert gt.anomaly_entities == expected_entities


def test_background_only_lots_have_no_ground_truth(injected):
    runs, events, lots, _ = injected
    inj = inject_excursions(runs, events, lots, toy_config(p_root=0.0, p_bg=1.0))
    t = build_graph_tables(runs, events, lots, inj.run_labels, inj.lot_labels, inj.excursions,
                           GraphConfig(window_hours=WINDOW_H))
    gt = compute_excursion_ground_truth(t, inj.run_labels, inj.gt_runs, inj.excursions)
    assert gt.strict == {} and gt.extended == {}
    assert (t.nodes["run"].y == 1).sum() > 0
    assert all(k == "run" for k, _ in gt.anomaly_entities)


def test_ground_truth_to_global(tables, injected):
    _, _, _, inj = injected
    gt = compute_excursion_ground_truth(tables, inj.run_labels, inj.gt_runs, inj.excursions)
    off = compute_local_offsets(tables)
    g_strict, g_ext = ground_truth_to_global(gt, off)
    for local, nodes in gt.strict.items():
        assert g_strict[off["run"] + local] == {off[t] + i for t, i in nodes}
    for local, nodes in gt.extended.items():
        assert g_ext[off["run"] + local] == {off[t] + i for t, i in nodes}
