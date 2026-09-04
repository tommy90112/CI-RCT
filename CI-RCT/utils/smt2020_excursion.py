"""
Excursion injection for the SMT2020 domain-transfer experiments.

Turns a *clean* PySCFabSim lot trace (utils.smt2020_sim) into a labelled
dataset whose root causes are known by construction:

    1. Pick n process tools k and time windows W = [t0, t1).  A window may be
       aligned to the end of a breakdown / PM on that tool ("bad repair").
    2. Every process run on k inside W creates a defect with prob p_root.
       Every other process run creates one with prob p_bg (background noise).
    3. A lot carries its defect from the first defective run onward.  Each
       later *metrology* run of a carrying lot reveals it with prob q_observe
       → run label y = 1.  Metrology runs of clean lots get y = 0; process
       runs are unlabelled (UNKNOWN_LABEL), mirroring Elliptic++ class 3.
    4. A finished defective lot fails final test with prob r_final_fail.

Background rate: p_bg is *per process run*.  A lot performs ~117 runs in a
14-day window (~580 over a full HVLM route), so the share of lots with a
background defect is 1 - (1 - p_bg)^runs: p_bg = 0.0002 gives ~2 % per 14
days, p_bg = 0.005 would give ~44 % and drown the excursion signal.

The injector reads only the trace tables and a seed — never the model — so the
ground truth is independent of the method under evaluation.  Root-cause
ground truth per excursion: (machine_idx, t0, t1) and the exact defect runs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

UNKNOWN_LABEL = -1
BACKGROUND_SOURCE = -1          # defect_source value for background defects
DEFAULT_METROLOGY_GROUPS = ("Def_Met", "Litho_Met", "TF_Met")
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0

EXCURSION_COLUMNS = ("excursion_id", "machine_idx", "machine_group", "machine_family",
                     "t0", "t1", "aligned_event_id", "n_runs_in_window",
                     "n_root_defects", "n_affected_lots")
RUN_LABEL_COLUMNS = ("run_id", "defect", "defect_source", "carrying", "observed", "y")
LOT_LABEL_COLUMNS = ("lot_idx", "defective", "first_defect_run", "defect_source", "fail")
GT_RUN_COLUMNS = ("excursion_id", "run_id", "lot_idx", "machine_idx", "t_start")

_REQUIRED_RUN_COLS = ("run_id", "lot_idx", "machine_idx", "machine_group", "machine_family",
                      "t_start", "t_end")
_REQUIRED_EVENT_COLS = ("event_id", "machine_idx", "t_start", "duration")
_REQUIRED_LOT_COLS = ("lot_idx", "done_at")


# ── Config / result ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExcursionConfig:
    n_excursions: int = 3
    window_hours_min: float = 4.0
    window_hours_max: float = 48.0
    p_root: float = 0.8
    p_bg: float = 0.0002          # per process run; ~2% of lots over 14 days (see docstring)
    q_observe: float = 0.5
    r_final_fail: float = 0.9
    align_to_events: float = 0.5     # fraction of excursions starting at a repair end
    warmup_days: float = 2.0         # no excursion starts before this
    min_runs_in_window: int = 20     # a window must contain at least this many runs
    metrology_groups: Tuple[str, ...] = DEFAULT_METROLOGY_GROUPS
    seed: int = 0
    max_draw_attempts: int = 200

    def __post_init__(self) -> None:
        for name in ("p_root", "p_bg", "q_observe", "r_final_fail", "align_to_events"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
        if self.n_excursions < 1:
            raise ValueError(f"n_excursions must be >= 1, got {self.n_excursions}")
        if not self.window_hours_min > 0:
            raise ValueError(f"window_hours_min must be > 0, got {self.window_hours_min}")
        if self.window_hours_max < self.window_hours_min:
            raise ValueError("window_hours_max must be >= window_hours_min")
        if self.warmup_days < 0:
            raise ValueError(f"warmup_days must be >= 0, got {self.warmup_days}")
        if self.min_runs_in_window < 1:
            raise ValueError(f"min_runs_in_window must be >= 1, got {self.min_runs_in_window}")


@dataclass(frozen=True)
class InjectionResult:
    config: ExcursionConfig
    excursions: pd.DataFrame
    run_labels: pd.DataFrame
    lot_labels: pd.DataFrame
    gt_runs: pd.DataFrame


@dataclass(frozen=True)
class _Excursion:
    excursion_id: int
    machine_idx: int
    machine_group: str
    machine_family: str
    t0: float
    t1: float
    aligned_event_id: Optional[int]
    n_runs_in_window: int


# ── Public API ─────────────────────────────────────────────────────────────────

def inject_excursions(runs: pd.DataFrame, events: pd.DataFrame, lots: pd.DataFrame,
                      cfg: ExcursionConfig) -> InjectionResult:
    """Label a clean trace with excursions, defects, observations and lot outcomes."""
    _validate_inputs(runs, events, lots)
    rng = np.random.default_rng(cfg.seed)
    runs = runs.sort_values("run_id").reset_index(drop=True)
    is_met = runs["machine_group"].isin(cfg.metrology_groups).to_numpy()
    horizon = float(runs["t_end"].max())
    warmup = cfg.warmup_days * SECONDS_PER_DAY

    excursions = _draw_excursions(runs[~is_met], events, cfg, rng, warmup, horizon)
    defect, source = _draw_defects(runs, is_met, excursions, cfg, rng)
    carrying = _carrying_mask(runs, defect)
    observed = is_met & carrying & (rng.random(len(runs)) < cfg.q_observe)
    y = np.where(is_met, observed.astype(int), UNKNOWN_LABEL)

    run_labels = pd.DataFrame({
        "run_id": runs["run_id"].to_numpy(), "defect": defect, "defect_source": source,
        "carrying": carrying, "observed": observed, "y": y,
    }, columns=list(RUN_LABEL_COLUMNS))
    gt_runs = _gt_runs(runs, defect, source)
    return InjectionResult(
        config=cfg,
        excursions=_excursion_frame(excursions, gt_runs),
        run_labels=run_labels,
        lot_labels=_lot_labels(runs, lots, defect, source, cfg, rng),
        gt_runs=gt_runs,
    )


def write_injection(result: InjectionResult, out_dir: Path) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.csv" for name in ("excursions", "run_labels", "lot_labels", "gt_runs")}
    for name, path in paths.items():
        getattr(result, name).to_csv(path, index=False)
    paths["meta"] = out / "injection_meta.json"
    meta = {
        **asdict(result.config),
        "n_runs": int(len(result.run_labels)),
        "n_defect_runs": int(result.run_labels["defect"].sum()),
        "n_observed_runs": int((result.run_labels["y"] == 1).sum()),
        "n_metrology_runs": int((result.run_labels["y"] != UNKNOWN_LABEL).sum()),
        "n_defective_lots": int(result.lot_labels["defective"].sum()),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return paths


# ── Excursion drawing ──────────────────────────────────────────────────────────

def _draw_excursions(process_runs, events, cfg, rng, warmup, horizon) -> List[_Excursion]:
    tools = (process_runs.drop_duplicates("machine_idx")
             .set_index("machine_idx")[["machine_group", "machine_family"]].sort_index())
    if len(tools) < cfg.n_excursions:
        raise ValueError(
            f"n_excursions={cfg.n_excursions} but only {len(tools)} candidate process tools "
            "have runs in the trace"
        )
    remaining = list(tools.index)
    chosen: List[_Excursion] = []
    for excursion_id in range(cfg.n_excursions):
        ex = _draw_one(excursion_id, remaining, tools, process_runs, events, cfg, rng, warmup, horizon)
        chosen.append(ex)
        remaining.remove(ex.machine_idx)
    return chosen


def _draw_one(excursion_id, remaining, tools, process_runs, events, cfg, rng, warmup, horizon) -> _Excursion:
    wmin, wmax = cfg.window_hours_min * SECONDS_PER_HOUR, cfg.window_hours_max * SECONDS_PER_HOUR
    if horizon - wmin < warmup:
        raise ValueError("trace is shorter than warmup + window_hours_min; nothing can be injected")
    want_aligned = rng.random() < cfg.align_to_events
    for _ in range(cfg.max_draw_attempts):
        length = float(rng.uniform(wmin, wmax))
        window = _draw_window(remaining, events, rng, warmup, horizon, wmin, length, want_aligned)
        if window is None:
            continue
        machine_idx, t0, t1, event_id = window
        n_in = int(((process_runs["machine_idx"] == machine_idx)
                    & (process_runs["t_start"] >= t0) & (process_runs["t_start"] < t1)).sum())
        if n_in >= cfg.min_runs_in_window:
            group, family = tools.loc[machine_idx, ["machine_group", "machine_family"]]
            return _Excursion(excursion_id, int(machine_idx), str(group), str(family),
                              t0, t1, event_id, n_in)
    raise ValueError(
        f"could not place excursion {excursion_id}: no window with >= {cfg.min_runs_in_window} "
        f"runs found in {cfg.max_draw_attempts} attempts (try a longer trace or a smaller minimum)"
    )


def _draw_window(remaining, events, rng, warmup, horizon, wmin, length, want_aligned):
    """Return (machine_idx, t0, t1, aligned_event_id) or None if the draw is impossible."""
    if want_aligned:
        eligible = events[events["machine_idx"].isin(remaining)].copy()
        eligible["t_end"] = eligible["t_start"] + eligible["duration"]
        eligible = eligible[(eligible["t_end"] >= warmup) & (eligible["t_end"] + wmin <= horizon)]
        if len(eligible):
            ev = eligible.iloc[int(rng.integers(len(eligible)))]
            t0 = float(ev["t_end"])
            return int(ev["machine_idx"]), t0, min(t0 + length, horizon), int(ev["event_id"])
    machine_idx = int(remaining[int(rng.integers(len(remaining)))])
    latest_start = horizon - length
    if latest_start < warmup:
        return None
    t0 = float(rng.uniform(warmup, latest_start))
    return machine_idx, t0, t0 + length, None


# ── Defects, propagation, labels ───────────────────────────────────────────────

def _draw_defects(runs, is_met, excursions, cfg, rng) -> Tuple[np.ndarray, np.ndarray]:
    defect = np.zeros(len(runs), dtype=bool)
    source = np.full(len(runs), np.nan)
    machine = runs["machine_idx"].to_numpy()
    t_start = runs["t_start"].to_numpy()
    for ex in excursions:
        in_window = (machine == ex.machine_idx) & (t_start >= ex.t0) & (t_start < ex.t1) & ~is_met
        hit = in_window & (rng.random(len(runs)) < cfg.p_root)
        defect |= hit
        source[hit] = ex.excursion_id
    background = ~is_met & ~defect & (rng.random(len(runs)) < cfg.p_bg)
    defect |= background
    source[background] = BACKGROUND_SOURCE
    return defect, source


def _carrying_mask(runs, defect) -> np.ndarray:
    """True for runs that start strictly after the lot's first defective run."""
    first = (runs.loc[defect, ["lot_idx", "t_start"]].groupby("lot_idx")["t_start"].min())
    first_t = runs["lot_idx"].map(first).to_numpy()
    return np.nan_to_num(runs["t_start"].to_numpy() > first_t, nan=False)


def _gt_runs(runs, defect, source) -> pd.DataFrame:
    root = defect & (source >= 0)
    frame = runs.loc[root, ["run_id", "lot_idx", "machine_idx", "t_start"]].copy()
    frame.insert(0, "excursion_id", source[root].astype(int))
    return frame.reset_index(drop=True)[list(GT_RUN_COLUMNS)]


def _excursion_frame(excursions: List[_Excursion], gt_runs: pd.DataFrame) -> pd.DataFrame:
    counts = gt_runs.groupby("excursion_id").agg(n_root_defects=("run_id", "size"),
                                                  n_affected_lots=("lot_idx", "nunique"))
    rows = []
    for ex in excursions:
        c = counts.loc[ex.excursion_id] if ex.excursion_id in counts.index else None
        rows.append({**asdict(ex),
                     "n_root_defects": int(c["n_root_defects"]) if c is not None else 0,
                     "n_affected_lots": int(c["n_affected_lots"]) if c is not None else 0})
    frame = pd.DataFrame(rows, columns=list(EXCURSION_COLUMNS))
    frame["aligned_event_id"] = frame["aligned_event_id"].astype("float")
    return frame


def _lot_labels(runs, lots, defect, source, cfg, rng) -> pd.DataFrame:
    first_rows = (runs.loc[defect, ["run_id", "lot_idx", "t_start"]]
                  .sort_values(["t_start", "run_id"]).drop_duplicates("lot_idx").set_index("lot_idx"))
    src_by_run = pd.Series(source, index=runs["run_id"].to_numpy())
    out = lots[["lot_idx", "done_at"]].copy()
    out["defective"] = out["lot_idx"].isin(first_rows.index)
    out["first_defect_run"] = out["lot_idx"].map(first_rows["run_id"])
    out["defect_source"] = out["first_defect_run"].map(src_by_run)
    finished = out["done_at"].notna().to_numpy()
    fails = out["defective"].to_numpy() & (rng.random(len(out)) < cfg.r_final_fail)
    out["fail"] = np.where(finished, fails.astype(int), UNKNOWN_LABEL)
    return out[list(LOT_LABEL_COLUMNS)].reset_index(drop=True)


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_inputs(runs, events, lots) -> None:
    for name, frame, cols in (("runs", runs, _REQUIRED_RUN_COLS),
                              ("events", events, _REQUIRED_EVENT_COLS),
                              ("lots", lots, _REQUIRED_LOT_COLS)):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} table is missing columns {missing}")
    if runs.empty:
        raise ValueError("runs table is empty")
    if runs["run_id"].duplicated().any():
        raise ValueError("runs table has duplicate run_id values")
