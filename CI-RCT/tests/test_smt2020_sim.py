"""
Tests for utils.smt2020_sim — the PySCFabSim driver that records per-lot
processing history (run events) and machine events for CI-RCT's SMT2020
domain-transfer experiments.

The simulator itself lives in third_party/PySCFabSim (MIT, not vendored into
git). Every test in this module is skipped when it is absent so the rest of the
suite stays green on machines that never cloned it.

The fixture runs the HVLM fab for a fraction of a simulated day (well under a
second of wall time) and the tests check the invariants the downstream loader
relies on: batch consistency, temporal ordering, lot coverage, determinism and
CSV round-trip.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from utils.smt2020_sim import (
    LOT_COLUMNS,
    RUN_COLUMNS,
    TOOL_EVENT_COLUMNS,
    SimulationConfig,
    ensure_simulator_on_path,
    run_simulation,
    simulator_available,
    write_result,
)

pytestmark = pytest.mark.skipif(
    not simulator_available(), reason="PySCFabSim not present in third_party/"
)

SHORT_DAYS = 0.05  # ~72 simulated minutes: thousands of dispatches, <1s wall


@pytest.fixture(scope="module")
def result():
    return run_simulation(SimulationConfig(dataset="SMT2020_HVLM", days=SHORT_DAYS, seed=0))


def test_runs_have_valid_schema(result):
    assert len(result.runs) > 100
    for r in result.runs:
        assert 0 <= r.t_start < r.t_end
        assert 0 <= r.machine_idx < result.n_machines
        assert r.step_order >= 1
        assert r.batch_size >= 1
        assert r.step_family and r.machine_group


def test_batches_are_internally_consistent(result):
    by_batch = defaultdict(list)
    for r in result.runs:
        by_batch[r.batch_id].append(r)
    for rows in by_batch.values():
        assert len(rows) == rows[0].batch_size
        assert len({(r.machine_idx, r.t_start, r.step_name) for r in rows}) == 1


def test_run_ids_are_unique_and_sequential(result):
    ids = [r.run_id for r in result.runs]
    assert ids == list(range(len(ids)))


def test_per_lot_time_is_monotone(result):
    by_lot = defaultdict(list)
    for r in result.runs:
        by_lot[r.lot_idx].append(r)
    for rows in by_lot.values():
        starts = [r.t_start for r in sorted(rows, key=lambda r: r.run_id)]
        assert starts == sorted(starts)


def test_every_run_lot_is_in_lot_table(result):
    lot_ids = {l.lot_idx for l in result.lots}
    assert {r.lot_idx for r in result.runs} <= lot_ids
    for l in result.lots:
        assert l.deadline_at > l.release_at
        assert l.done_at is None or l.done_at >= l.release_at


def test_tool_events_are_well_formed(result):
    assert result.tool_events, "HVLM has exponential breakdowns; a few must occur"
    for e in result.tool_events:
        assert e.event_type in {"breakdown", "pm"}
        assert e.t_start >= 0
        assert e.duration > 0
        assert 0 <= e.machine_idx < result.n_machines


def test_same_seed_is_deterministic_and_seed_matters(result):
    again = run_simulation(SimulationConfig(dataset="SMT2020_HVLM", days=SHORT_DAYS, seed=0))
    assert again.runs == result.runs
    assert again.tool_events == result.tool_events
    other = run_simulation(SimulationConfig(dataset="SMT2020_HVLM", days=SHORT_DAYS, seed=1))
    assert other.runs != result.runs


def test_write_result_round_trips(result, tmp_path: Path):
    paths = write_result(result, tmp_path)
    runs = pd.read_csv(paths["lot_trace"])
    events = pd.read_csv(paths["tool_events"])
    lots = pd.read_csv(paths["lots"])
    assert list(runs.columns) == list(RUN_COLUMNS)
    assert list(events.columns) == list(TOOL_EVENT_COLUMNS)
    assert list(lots.columns) == list(LOT_COLUMNS)
    assert len(runs) == len(result.runs)
    assert len(events) == len(result.tool_events)
    assert len(lots) == len(result.lots)
    meta = paths["meta"]
    assert meta.exists() and "seed" in meta.read_text()


def test_missing_simulator_root_is_reported_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="PySCFabSim"):
        ensure_simulator_on_path(tmp_path / "nowhere")
