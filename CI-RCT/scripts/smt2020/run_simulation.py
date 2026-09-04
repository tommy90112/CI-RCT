"""
Run a PySCFabSim / SMT2020 fab simulation and export the lot trace.

Usage (from CI-RCT/):
    python scripts/smt2020/run_simulation.py --dataset SMT2020_HVLM --days 7 --seed 0
    python scripts/smt2020/run_simulation.py --days 14 --seed 3 --out data/smt2020_sim/HVLM/s3_14d

Default output directory: data/smt2020_sim/<dataset>/seed<seed>_<days>d/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.smt2020_sim import (  # noqa: E402
    SUPPORTED_DATASETS,
    SUPPORTED_DISPATCHERS,
    SimulationConfig,
    run_simulation,
    write_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="SMT2020_HVLM", choices=SUPPORTED_DATASETS)
    p.add_argument("--days", type=float, default=7.0, help="simulated days (float allowed)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dispatcher", default="fifo", choices=SUPPORTED_DISPATCHERS)
    p.add_argument("--simulator_root", type=Path, default=None,
                   help="PySCFabSim checkout (default: $PYSCFABSIM_ROOT or third_party/PySCFabSim)")
    p.add_argument("--out", type=Path, default=None, help="output directory")
    return p.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    days = int(args.days) if float(args.days).is_integer() else args.days
    return PACKAGE_ROOT / "data" / "smt2020_sim" / args.dataset / f"seed{args.seed}_{days}d"


def main() -> int:
    args = parse_args()
    cfg = SimulationConfig(dataset=args.dataset, days=args.days, seed=args.seed,
                           dispatcher=args.dispatcher, simulator_root=args.simulator_root)
    out_dir = args.out or default_out_dir(args)
    print(f"Simulating {cfg.dataset} for {cfg.days} days (seed {cfg.seed}, {cfg.dispatcher}) …")
    result = run_simulation(cfg)
    paths = write_result(result, out_dir)
    print(f"  simulated days : {result.sim_days:.2f}")
    print(f"  machines       : {result.n_machines} in {result.n_families} families")
    print(f"  runs           : {len(result.runs):,}")
    print(f"  tool events    : {len(result.tool_events):,}")
    print(f"  lots released  : {len(result.lots):,}")
    print(f"  written to     : {paths['lot_trace'].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
