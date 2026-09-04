"""
Inject excursions into an exported SMT2020 lot trace.

Usage (from CI-RCT/):
    python scripts/smt2020/inject_excursion.py --trace_dir data/smt2020_sim/SMT2020_HVLM/seed0_14d \
        --n_excursions 3 --p_root 0.8 --q_observe 0.5 --seed 0

Reads lot_trace.csv / tool_events.csv / lots.csv from --trace_dir and writes
excursions.csv, run_labels.csv, lot_labels.csv, gt_runs.csv, injection_meta.json
into --out (default: <trace_dir>/excursion_seed<seed>/).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.smt2020_excursion import ExcursionConfig, inject_excursions, write_injection  # noqa: E402

_CLI_FIELDS = ("n_excursions", "window_hours_min", "window_hours_max", "p_root", "p_bg",
               "q_observe", "r_final_fail", "align_to_events", "warmup_days",
               "min_runs_in_window", "seed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    defaults = {f.name: f.default for f in fields(ExcursionConfig)}
    for name in _CLI_FIELDS:
        p.add_argument(f"--{name}", type=type(defaults[name]), default=defaults[name])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExcursionConfig(**{name: getattr(args, name) for name in _CLI_FIELDS})
    runs = pd.read_csv(args.trace_dir / "lot_trace.csv")
    events = pd.read_csv(args.trace_dir / "tool_events.csv")
    lots = pd.read_csv(args.trace_dir / "lots.csv")
    out_dir = args.out or (args.trace_dir / f"excursion_seed{cfg.seed}")
    result = inject_excursions(runs, events, lots, cfg)
    paths = write_injection(result, out_dir)
    print(result.excursions.to_string(index=False))
    labels = result.run_labels
    print(f"\n  runs            : {len(labels):,}")
    print(f"  defect runs     : {int(labels.defect.sum()):,}  (root {len(result.gt_runs):,})")
    print(f"  metrology runs  : {int((labels.y != -1).sum()):,}   observed y=1: {int((labels.y == 1).sum()):,}")
    print(f"  defective lots  : {int(result.lot_labels.defective.sum()):,} / {len(result.lot_labels):,}")
    print(f"  written to      : {paths['excursions'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
