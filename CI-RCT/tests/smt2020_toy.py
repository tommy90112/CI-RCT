"""
Hand-computable toy SMT2020 trace shared by the smt2020_* test modules.

    machines : 0 (Dry_Etch / DE_1), 1 (Dry_Etch / DE_2), 2 (Def_Met / DM_1)
    lots     : 0..3, lot i released at hour i, one run every 2 h for 10 runs
               even runs = process step on machine (0 if lot even else 1)
               odd runs  = metrology on machine 2; every run lasts 1 h
    events   : breakdown on machine 0 at hour 4 lasting 1 h (ends hour 5)
    lots 0,1 finish at hour base+20; lots 2,3 are still active

With a 6 h excursion aligned to that breakdown the window is [5 h, 11 h) on
machine 0, containing exactly three process runs: lot 0 @ 8 h, lot 2 @ 6 h
and lot 2 @ 10 h.
"""
import pandas as pd

from utils.smt2020_excursion import ExcursionConfig
from utils.smt2020_sim import LOT_COLUMNS, RUN_COLUMNS, TOOL_EVENT_COLUMNS

H = 3600.0
N_LOTS = 4
RUNS_PER_LOT = 10


def _machine(idx):
    return {0: ("Dry_Etch", "DE_1"), 1: ("Dry_Etch", "DE_2"), 2: ("Def_Met", "DM_1")}[idx]


def toy_runs() -> pd.DataFrame:
    rows = []
    run_id = 0
    for lot in range(N_LOTS):
        for i in range(RUNS_PER_LOT):
            is_met = i % 2 == 1
            m = 2 if is_met else (0 if lot % 2 == 0 else 1)
            group, family = _machine(m)
            t = (lot + 2 * i) * H
            rows.append(dict(
                run_id=run_id, batch_id=run_id, batch_size=1, lot_idx=lot, lot_name=f"Lot_{lot}",
                part="part_3", priority=10, step_idx=i, step_order=i + 1,
                step_name=f"{i:03d}_{'Met' if is_met else 'Etch'}", step_family=family,
                machine_idx=m, machine_group=group, machine_family=family,
                t_start=t, t_end=t + H, t_machine_end=t + H, n_processed=i, setup="",
                pm_triggered=False,
            ))
            run_id += 1
    return pd.DataFrame(rows, columns=list(RUN_COLUMNS))


def toy_events() -> pd.DataFrame:
    return pd.DataFrame([dict(event_id=0, machine_idx=0, machine_group="Dry_Etch",
                              machine_family="DE_1", event_type="breakdown",
                              t_start=4 * H, duration=1 * H)], columns=list(TOOL_EVENT_COLUMNS))


def toy_lots() -> pd.DataFrame:
    rows = []
    for lot in range(N_LOTS):
        rows.append(dict(lot_idx=lot, lot_name=f"Lot_{lot}", part="part_3", priority=10,
                         release_at=lot * H, deadline_at=(lot + 40) * H,
                         done_at=(lot + 20) * H if lot < 2 else None))
    return pd.DataFrame(rows, columns=list(LOT_COLUMNS))


def toy_config(**kw) -> ExcursionConfig:
    base = dict(n_excursions=1, window_hours_min=6.0, window_hours_max=6.0, p_root=1.0,
                p_bg=0.0, q_observe=1.0, r_final_fail=1.0, align_to_events=1.0,
                warmup_days=0.0, min_runs_in_window=1, seed=0)
    base.update(kw)
    return ExcursionConfig(**base)
