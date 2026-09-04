"""
Tests for utils.smt2020_excursion — the excursion-injection model that turns a
clean SMT2020 lot trace into a labelled dataset with ground-truth root causes.

The injector never sees the model; it only reads the trace tables. Most tests
use the hand-computable toy trace from tests/smt2020_toy.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.smt2020_excursion import (
    EXCURSION_COLUMNS,
    GT_RUN_COLUMNS,
    LOT_LABEL_COLUMNS,
    RUN_LABEL_COLUMNS,
    UNKNOWN_LABEL,
    ExcursionConfig,
    inject_excursions,
    write_injection,
)
from tests.smt2020_toy import H, toy_config as _cfg, toy_events as _toy_events, toy_lots as _toy_lots, toy_runs as _toy_runs


@pytest.fixture
def toy():
    return _toy_runs(), _toy_events(), _toy_lots()


@pytest.fixture
def deterministic(toy):
    runs, events, lots = toy
    return inject_excursions(runs, events, lots, _cfg())


def test_aligned_excursion_lands_on_event_end(deterministic):
    ex = deterministic.excursions
    assert list(ex.columns) == list(EXCURSION_COLUMNS)
    assert len(ex) == 1
    row = ex.iloc[0]
    assert row.machine_idx == 0 and row.machine_group == "Dry_Etch"
    assert row.t0 == 5 * H and row.t1 == 11 * H
    assert row.aligned_event_id == 0
    assert row.n_runs_in_window == 3


def test_root_defects_are_exactly_the_runs_in_window(deterministic):
    gt = deterministic.gt_runs
    assert list(gt.columns) == list(GT_RUN_COLUMNS)
    got = {(r.lot_idx, r.t_start) for r in gt.itertuples()}
    assert got == {(0, 8 * H), (2, 6 * H), (2, 10 * H)}
    labels = deterministic.run_labels
    assert labels.defect.sum() == 3
    assert set(labels.loc[labels.defect, "run_id"]) == set(gt.run_id)
    assert (labels.loc[labels.defect, "defect_source"] == 0).all()


def test_carrying_is_monotone_and_starts_after_defect_run(deterministic):
    labels = deterministic.run_labels.merge(_toy_runs()[["run_id", "lot_idx", "t_start"]], on="run_id")
    for lot, g in labels.sort_values("t_start").groupby("lot_idx"):
        carrying = g.carrying.tolist()
        assert carrying == sorted(carrying)  # False... then True...
        if g.defect.any():
            first_defect_t = g.loc[g.defect, "t_start"].min()
            assert not g.loc[g.t_start <= first_defect_t, "carrying"].any()
            assert g.loc[g.t_start > first_defect_t, "carrying"].all()
        else:
            assert not g.carrying.any()


def test_observation_only_on_metrology_runs_of_carrying_lots(deterministic):
    labels = deterministic.run_labels.merge(_toy_runs()[["run_id", "lot_idx", "t_start", "machine_group"]], on="run_id")
    met = labels.machine_group == "Def_Met"
    assert (labels.loc[~met, "y"] == UNKNOWN_LABEL).all()
    assert set(labels.loc[met, "y"]) <= {0, 1}
    # q = 1 → every metrology run after the defect is observed, nothing else
    assert (labels.loc[met, "y"] == labels.loc[met, "carrying"].astype(int)).all()
    observed = labels[labels.y == 1]
    assert set(observed.lot_idx) == {0, 2}
    assert set(observed.loc[observed.lot_idx == 0, "t_start"]) == {10 * H, 14 * H, 18 * H}
    assert set(observed.loc[observed.lot_idx == 2, "t_start"]) == {8 * H, 12 * H, 16 * H, 20 * H}


def test_q_zero_observes_nothing(toy):
    runs, events, lots = toy
    res = inject_excursions(runs, events, lots, _cfg(q_observe=0.0))
    assert res.run_labels.defect.sum() == 3
    assert (res.run_labels.y != 1).all()


def test_lot_labels(deterministic):
    lots = deterministic.lot_labels.set_index("lot_idx")
    assert list(deterministic.lot_labels.columns) == list(LOT_LABEL_COLUMNS)
    assert lots.loc[0, "defective"] and lots.loc[2, "defective"]
    assert not lots.loc[1, "defective"] and not lots.loc[3, "defective"]
    assert lots.loc[0, "fail"] == 1 and lots.loc[1, "fail"] == 0      # finished lots, r = 1
    assert lots.loc[2, "fail"] == UNKNOWN_LABEL and lots.loc[3, "fail"] == UNKNOWN_LABEL
    assert lots.loc[0, "first_defect_run"] == deterministic.gt_runs.loc[
        deterministic.gt_runs.lot_idx == 0, "run_id"].min()


def test_background_defects_never_hit_metrology_and_respect_p_bg(toy):
    runs, events, lots = toy
    res = inject_excursions(runs, events, lots, _cfg(p_bg=1.0, p_root=0.0))
    labels = res.run_labels.merge(runs[["run_id", "machine_group"]], on="run_id")
    process = labels.machine_group != "Def_Met"
    assert labels.loc[process, "defect"].all()
    assert not labels.loc[~process, "defect"].any()
    assert (labels.loc[labels.defect, "defect_source"] == -1).all()
    assert res.gt_runs.empty


def test_unaligned_window_stays_inside_horizon_and_bounds(toy):
    runs, events, lots = toy
    cfg = _cfg(align_to_events=0.0, window_hours_min=2.0, window_hours_max=4.0, warmup_days=0.1)
    res = inject_excursions(runs, events, lots, cfg)
    row = res.excursions.iloc[0]
    assert pd.isna(row.aligned_event_id)
    assert row.t0 >= 0.1 * 86400
    assert 2 * H <= row.t1 - row.t0 <= 4 * H
    assert row.t1 <= runs.t_end.max()
    assert row.machine_group != "Def_Met"


def test_seed_determinism(toy):
    runs, events, lots = toy
    cfg = _cfg(align_to_events=0.0, window_hours_min=2.0, window_hours_max=8.0, p_root=0.5, q_observe=0.5)
    a = inject_excursions(runs, events, lots, cfg)
    b = inject_excursions(runs, events, lots, cfg)
    pd.testing.assert_frame_equal(a.excursions, b.excursions)
    pd.testing.assert_frame_equal(a.run_labels, b.run_labels)
    c = inject_excursions(runs, events, lots, _cfg(align_to_events=0.0, window_hours_min=2.0,
                                                   window_hours_max=8.0, p_root=0.5, q_observe=0.5, seed=7))
    assert not (a.excursions.equals(c.excursions) and a.run_labels.equals(c.run_labels))


def test_multiple_excursions_use_distinct_tools(toy):
    runs, events, lots = toy
    res = inject_excursions(runs, events, lots, _cfg(n_excursions=2, align_to_events=0.0))
    assert res.excursions.machine_idx.nunique() == 2
    assert set(res.excursions.machine_idx) == {0, 1}


def test_too_many_excursions_is_a_clear_error(toy):
    runs, events, lots = toy
    with pytest.raises(ValueError, match="candidate"):
        inject_excursions(runs, events, lots, _cfg(n_excursions=3, align_to_events=0.0))


@pytest.mark.parametrize("field,value", [("p_root", 1.5), ("p_bg", -0.1), ("q_observe", 2), ("align_to_events", -1),
                                          ("n_excursions", 0), ("window_hours_min", 0), ("min_runs_in_window", 0)])
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        _cfg(**{field: value})


def test_window_max_below_min_is_rejected():
    with pytest.raises(ValueError):
        _cfg(window_hours_min=8.0, window_hours_max=4.0)


def test_write_injection_round_trips(deterministic, tmp_path: Path):
    paths = write_injection(deterministic, tmp_path)
    ex = pd.read_csv(paths["excursions"])
    rl = pd.read_csv(paths["run_labels"])
    ll = pd.read_csv(paths["lot_labels"])
    gt = pd.read_csv(paths["gt_runs"])
    assert list(ex.columns) == list(EXCURSION_COLUMNS)
    assert list(rl.columns) == list(RUN_LABEL_COLUMNS)
    assert list(ll.columns) == list(LOT_LABEL_COLUMNS)
    assert list(gt.columns) == list(GT_RUN_COLUMNS)
    assert len(rl) == 40 and len(ll) == 4 and len(gt) == 3
    assert "p_root" in paths["meta"].read_text()


def test_end_to_end_on_real_simulator_trace():
    sim = pytest.importorskip("utils.smt2020_sim")
    if not sim.simulator_available():
        pytest.skip("PySCFabSim not present")
    result = sim.run_simulation(sim.SimulationConfig(dataset="SMT2020_HVLM", days=1.0, seed=0))
    runs, events, lots = sim.result_to_frames(result)
    cfg = ExcursionConfig(n_excursions=3, window_hours_min=4, window_hours_max=8, p_root=0.8,
                          p_bg=0.001, q_observe=0.5, r_final_fail=0.9, align_to_events=0.5,
                          warmup_days=0.2, min_runs_in_window=5, seed=1)
    res = inject_excursions(runs, events, lots, cfg)
    assert len(res.excursions) == 3
    assert (res.excursions.n_runs_in_window >= 5).all()
    assert res.gt_runs.excursion_id.isin(res.excursions.excursion_id).all()
    assert (res.run_labels.y == 1).sum() > 0
    assert len(res.run_labels) == len(runs)
