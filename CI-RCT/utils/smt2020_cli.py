"""
Shared --smt2020_* CLI plumbing for train.py and evaluate.py.

Keeps the flag definitions, the loader kwargs and the GraphConfig used for
ground truth in ONE place so evaluate.py rebuilds exactly the graph that
train.py trained on (the type offsets must match for Metric B / C).
"""
from __future__ import annotations

import argparse
import os

from utils.smt2020_graph import GraphConfig

DEFAULT_SMT2020_DIR = os.path.join("data", "smt2020_sim", "SMT2020_HVLM", "seed0_14d")
DEFAULT_EXCURSION_SUBDIR = "excursion_seed0"


def add_smt2020_args(parser: argparse.ArgumentParser) -> None:
    """--smt2020_* flags (see smt2020_plan.md §3.3 for the semantics)."""
    parser.add_argument("--smt2020_dir", type=str, default=DEFAULT_SMT2020_DIR,
                        help="Directory holding lot_trace.csv / tool_events.csv / lots.csv "
                             "(output of scripts/smt2020/run_simulation.py).")
    parser.add_argument("--smt2020_excursion", type=str, default=DEFAULT_EXCURSION_SUBDIR,
                        help="Sub-directory with the injection tables "
                             "(output of scripts/smt2020/inject_excursion.py).")
    parser.add_argument("--smt2020_window_hours", type=float, default=8.0,
                        help="tool_state window length in hours.")
    parser.add_argument("--smt2020_metrology_signal", type=float, default=2.0,
                        help="SNR of the synthetic inline measurement on metrology runs.")
    parser.add_argument("--smt2020_tool_signal", type=float, default=0.0,
                        help="SNR of the synthetic equipment sensor on excursion windows "
                             "(0 = root cause must be inferred from topology/timing).")
    parser.add_argument("--smt2020_drop_before_days", type=float, default=2.0,
                        help="Drop runs starting before this many simulated days (warm-up).")


def smt2020_graph_config(args: argparse.Namespace) -> GraphConfig:
    return GraphConfig(
        window_hours=args.smt2020_window_hours,
        metrology_signal=args.smt2020_metrology_signal,
        tool_signal=args.smt2020_tool_signal,
        drop_before_days=args.smt2020_drop_before_days,
        feature_seed=args.seed,
        split_seed=args.seed,
    )


def smt2020_loader_kwargs(args: argparse.Namespace) -> dict:
    """kwargs for utils.smt2020_loader.load_smt2020_dataset."""
    cfg = smt2020_graph_config(args)
    return dict(
        data_root=args.smt2020_dir,
        excursion_subdir=args.smt2020_excursion,
        window_hours=cfg.window_hours,
        metrology_signal=cfg.metrology_signal,
        tool_signal=cfg.tool_signal,
        drop_before_days=cfg.drop_before_days,
        feature_seed=cfg.feature_seed,
        split_seed=cfg.split_seed,
    )
