"""
SMT2020 simulation driver for CI-RCT (domain-transfer experiments).

Wraps PySCFabSim (third_party/PySCFabSim, MIT licence, NOT vendored into git)
with a trace-recording plugin so that every lot processing event and every
machine breakdown / preventive-maintenance event is exported as a flat table.
The simulator's own source is never modified: it exposes a plugin protocol
(``on_dispatch``, ``on_breakdown``, ...) that is sufficient.

Outputs (see write_result)
──────────────────────────
    lot_trace.csv    one row per (lot, dispatch)   — the future ``run`` nodes
    tool_events.csv  breakdown / PM events         — future ``tool_state`` cues
    lots.csv         one row per released lot      — the future ``lot`` nodes
    meta.json        config + counts + simulator commit

Time unit: simulated seconds since fab start (float).

Locating the simulator: ``$PYSCFABSIM_ROOT`` if set, otherwise
``<package root>/third_party/PySCFabSim``.  Clone it with
    git clone https://github.com/prosysscience/PySCFabSim-release third_party/PySCFabSim
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

PYSCFABSIM_ROOT_ENV = "PYSCFABSIM_ROOT"
DEFAULT_SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "PySCFabSim"
SUPPORTED_DATASETS = ("SMT2020_HVLM", "SMT2020_LVHM")
SUPPORTED_DISPATCHERS = ("fifo", "cr")
SECONDS_PER_DAY = 86_400.0

EVENT_BREAKDOWN = "breakdown"
EVENT_PM = "pm"


# ── Records ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunRecord:
    """One lot processed once on one machine (a batch yields several rows)."""
    run_id: int
    batch_id: int
    batch_size: int
    lot_idx: int
    lot_name: str
    part: str
    priority: int
    step_idx: int          # 0-based position in the route definition
    step_order: int        # 1-based STEP column of the route file
    step_name: str
    step_family: str       # STNFAM the step requires
    machine_idx: int
    machine_group: str     # STNGRP (coarse tool type, e.g. Dry_Etch)
    machine_family: str    # STNFAM (fine tool group, e.g. DE_BE_11)
    t_start: float
    t_end: float           # lot leaves the machine (incl. load/unload/transport)
    t_machine_end: float   # machine becomes free again
    n_processed: int       # steps completed by the lot before this run
    setup: str
    pm_triggered: bool     # piece-count PM ran inside this dispatch


@dataclass(frozen=True)
class ToolEvent:
    event_id: int
    machine_idx: int
    machine_group: str
    machine_family: str
    event_type: str        # EVENT_BREAKDOWN | EVENT_PM
    t_start: float
    duration: float


@dataclass(frozen=True)
class LotRecord:
    lot_idx: int
    lot_name: str
    part: str
    priority: int
    release_at: float
    deadline_at: float
    done_at: Optional[float]


RUN_COLUMNS = tuple(f.name for f in fields(RunRecord))
TOOL_EVENT_COLUMNS = tuple(f.name for f in fields(ToolEvent))
LOT_COLUMNS = tuple(f.name for f in fields(LotRecord))


@dataclass(frozen=True)
class SimulationConfig:
    dataset: str = "SMT2020_HVLM"
    days: float = 7.0
    seed: int = 0
    dispatcher: str = "fifo"
    simulator_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {self.dataset!r}")
        if self.dispatcher not in SUPPORTED_DISPATCHERS:
            raise ValueError(f"dispatcher must be one of {SUPPORTED_DISPATCHERS}, got {self.dispatcher!r}")
        if not self.days > 0:
            raise ValueError(f"days must be > 0, got {self.days}")
        if int(self.seed) != self.seed or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    runs: Tuple[RunRecord, ...]
    tool_events: Tuple[ToolEvent, ...]
    lots: Tuple[LotRecord, ...]
    sim_days: float
    n_machines: int
    n_families: int


# ── Simulator location ─────────────────────────────────────────────────────────

def resolve_simulator_root(root: Optional[Path] = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(PYSCFABSIM_ROOT_ENV)
    return Path(env) if env else DEFAULT_SIMULATOR_ROOT


def simulator_available(root: Optional[Path] = None) -> bool:
    return (resolve_simulator_root(root) / "simulation" / "instance.py").is_file()


def ensure_simulator_on_path(root: Optional[Path] = None) -> Path:
    """Validate the PySCFabSim checkout and make ``import simulation`` work."""
    resolved = resolve_simulator_root(root)
    if not simulator_available(resolved):
        raise FileNotFoundError(
            f"PySCFabSim not found at {resolved}. Clone "
            "https://github.com/prosysscience/PySCFabSim-release into third_party/PySCFabSim "
            f"or point {PYSCFABSIM_ROOT_ENV} at an existing checkout."
        )
    as_str = str(resolved)
    if as_str not in sys.path:
        sys.path.insert(0, as_str)
    return resolved


def simulator_commit(root: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(resolve_simulator_root(root)), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ── Trace plugin ───────────────────────────────────────────────────────────────

class _TracePlugin:
    """
    Duck-typed PySCFabSim plugin (same method names as
    simulation.plugins.interface.IPlugin) that accumulates immutable records.

    Piece-count preventive maintenance is applied silently inside
    Instance.dispatch (it never reaches on_preventive_maintenance), so PM is
    detected from the growth of ``machine.pmed_time`` between dispatches.
    Breakdown lengths are likewise recovered from ``machine.bred_time`` deltas
    because the event object only carries the length *distribution*.
    """

    def __init__(self) -> None:
        self._runs: List[RunRecord] = []
        self._events: List[ToolEvent] = []
        self._next_batch = 0
        self._pmed_seen: Dict[int, float] = {}
        self._bred_seen: Dict[int, float] = {}

    # -- hooks the simulator calls --------------------------------------------
    def on_sim_init(self, instance) -> None: ...
    def on_sim_done(self, instance) -> None: ...
    def on_lots_release(self, instance, lots) -> None: ...
    def on_lot_done(self, instance, lot) -> None: ...
    def on_step_done(self, instance, lot, step) -> None: ...
    def on_machine_free(self, instance, machine) -> None: ...
    def on_lot_free(self, instance, lot) -> None: ...
    def on_cqt_violated(self, instance, machine, lot) -> None: ...

    def get_output_name(self):
        return None

    def on_dispatch(self, instance, machine, lots, machine_end_time, lot_end_time) -> None:
        pm_delta = self._consume_delta(self._pmed_seen, machine.idx, machine.pmed_time)
        if pm_delta > 0:
            self._add_event(machine, EVENT_PM, float(instance.current_time), pm_delta)
        batch_id = self._next_batch
        self._next_batch += 1
        for lot in lots:
            self._runs.append(self._run_record(
                batch_id, len(lots), lot, machine, instance.current_time,
                lot_end_time, machine_end_time, pm_delta > 0,
            ))

    def on_breakdown(self, instance, event) -> None:
        machine = event.machine
        delta = self._consume_delta(self._bred_seen, machine.idx, machine.bred_time)
        self._add_event(machine, EVENT_BREAKDOWN, float(event.timestamp), delta)

    def on_preventive_maintenance(self, instance, event) -> None:
        machine = event.machine
        delta = self._consume_delta(self._pmed_seen, machine.idx, machine.pmed_time)
        self._add_event(machine, EVENT_PM, float(event.timestamp), delta)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _consume_delta(seen: Dict[int, float], idx: int, total: float) -> float:
        delta = float(total) - seen.get(idx, 0.0)
        seen[idx] = float(total)
        return delta

    def _add_event(self, machine, event_type: str, t_start: float, duration: float) -> None:
        self._events.append(ToolEvent(
            event_id=len(self._events), machine_idx=machine.idx,
            machine_group=machine.group, machine_family=machine.family,
            event_type=event_type, t_start=t_start, duration=float(duration),
        ))

    def _run_record(self, batch_id, batch_size, lot, machine, t_start,
                    lot_end, machine_end, pm_triggered) -> RunRecord:
        step = lot.actual_step
        return RunRecord(
            run_id=len(self._runs), batch_id=batch_id, batch_size=batch_size,
            lot_idx=lot.idx, lot_name=lot.name, part=lot.part_name, priority=lot.priority,
            step_idx=step.idx, step_order=step.order, step_name=step.step_name,
            step_family=step.family, machine_idx=machine.idx,
            machine_group=machine.group, machine_family=machine.family,
            t_start=float(t_start), t_end=float(lot_end), t_machine_end=float(machine_end),
            n_processed=len(lot.processed_steps), setup=machine.current_setup,
            pm_triggered=bool(pm_triggered),
        )

    def runs(self) -> Tuple[RunRecord, ...]:
        return tuple(self._runs)

    def events(self) -> Tuple[ToolEvent, ...]:
        return tuple(self._events)


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run the greedy lot-for-machine loop of PySCFabSim and collect the trace."""
    root = ensure_simulator_on_path(cfg.simulator_root)
    dataset_dir = root / "datasets" / cfg.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"SMT2020 dataset directory missing: {dataset_dir}")

    from simulation.dispatching.dispatcher import dispatcher_map
    from simulation.file_instance import FileInstance
    from simulation.greedy import get_lots_to_dispatch_by_machine
    from simulation.randomizer import Randomizer
    from simulation.read import read_all

    files = read_all(str(dataset_dir), preprocessors=[])  # ignore NOWIP/... env switches
    run_to = SECONDS_PER_DAY * cfg.days
    Randomizer().random.seed(cfg.seed)
    plugin = _TracePlugin()
    instance = FileInstance(files, run_to, True, [plugin])
    dispatcher = dispatcher_map[cfg.dispatcher]

    while not instance.done:
        finished = instance.next_decision_point()
        if finished or instance.current_time > run_to:
            break
        machine, lots = get_lots_to_dispatch_by_machine(instance, dispatcher)
        if lots is None:
            instance.usable_machines.remove(machine)
        else:
            instance.dispatch(machine, lots)
    instance.finalize()

    return SimulationResult(
        config=cfg,
        runs=plugin.runs(),
        tool_events=plugin.events(),
        lots=_lot_records(instance),
        sim_days=float(instance.current_time_days),
        n_machines=len(instance.machines),
        n_families=len(instance.family_machines),
    )


def _lot_records(instance) -> Tuple[LotRecord, ...]:
    released = list(instance.done_lots) + list(instance.active_lots)
    return tuple(sorted(
        (LotRecord(
            lot_idx=lot.idx, lot_name=lot.name, part=lot.part_name, priority=lot.priority,
            release_at=float(lot.release_at), deadline_at=float(lot.deadline_at),
            done_at=None if lot.done_at is None else float(lot.done_at),
        ) for lot in released),
        key=lambda l: l.lot_idx,
    ))


# ── Output ─────────────────────────────────────────────────────────────────────

def result_to_frames(result: SimulationResult) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.DataFrame([asdict(r) for r in result.runs], columns=list(RUN_COLUMNS))
    events = pd.DataFrame([asdict(e) for e in result.tool_events], columns=list(TOOL_EVENT_COLUMNS))
    lots = pd.DataFrame([asdict(l) for l in result.lots], columns=list(LOT_COLUMNS))
    return runs, events, lots


def write_result(result: SimulationResult, out_dir: Path) -> Dict[str, Path]:
    """Write lot_trace.csv, tool_events.csv, lots.csv and meta.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, events, lots = result_to_frames(result)
    paths = {
        "lot_trace": out / "lot_trace.csv",
        "tool_events": out / "tool_events.csv",
        "lots": out / "lots.csv",
        "meta": out / "meta.json",
    }
    runs.to_csv(paths["lot_trace"], index=False)
    events.to_csv(paths["tool_events"], index=False)
    lots.to_csv(paths["lots"], index=False)
    meta = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(result.config).items()},
        "sim_days": result.sim_days,
        "n_machines": result.n_machines,
        "n_families": result.n_families,
        "n_runs": len(result.runs),
        "n_tool_events": len(result.tool_events),
        "n_lots": len(result.lots),
        "simulator_commit": simulator_commit(result.config.simulator_root),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return paths
