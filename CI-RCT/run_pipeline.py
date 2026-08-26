#!/usr/bin/env python3
"""
CI-RCT one-command pipeline: train → evaluate → export → static viewer.

Runs every stage in order with the frozen configuration that produced the
reported results (see pipeline/config.py), wiring each stage's output into the
next so no paths have to be matched up by hand.

Quick start
-----------
    cd CI-RCT

    # See what would happen — no work is done, but every decision is real.
    python run_pipeline.py --dry-run

    # Full run: three variants, then build the static viewer.
    python run_pipeline.py --device cuda

    # Already have checkpoints? Everything trained is skipped automatically.
    python run_pipeline.py --from evaluate

    # Redo one stage.
    python run_pipeline.py --force evaluate

Stages already satisfied by an existing artifact are skipped, so re-running is
cheap. Forcing a stage also re-runs everything downstream of it, because an
export built from a model you just retrained would otherwise be stale.

Output lands where the viewer already looks for it (frontend_temp/vite.config.ts
serves ../viz and ../results directly):

    checkpoints/ci_rct_elliptic++[_variant]_best.pt
    viz/crime_chains[_variant].json      per-node φ_asym + L3 feature attribution
    results/crime_chains.csv             flat one-row-per-chain table
    results/chain_neighbors.json         real 1-hop neighbourhood overlay
    frontend_temp/dist/                  self-contained static viewer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import (
    DEFAULT_EPOCHS,
    PipelinePaths,
    VARIANT_NAMES,
    variant_by_name,
)
from pipeline.preflight import describe, ensure_directories, run_checks
from pipeline.process import PipelineError
from pipeline.runner import RUN, execute, plan, render_commands, render_plan, render_summary
from pipeline.stages import STAGE_ORDER, BuildOptions, build_steps, filter_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Run the whole CI-RCT pipeline with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stage names for --from / --force / --only: "
               + ", ".join(STAGE_ORDER),
    )
    parser.add_argument(
        "--variants", type=str, default=",".join(VARIANT_NAMES),
        help="Comma-separated detection variants to run (default: all three). "
             "The 'joint' variant is the viewer's landing view and the source "
             "of the CSV that the neighbour overlay needs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs per variant (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device passed to train.py / evaluate.py (default: cuda).",
    )
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Interpreter used for the Python stages (default: the one "
             "running this script).",
    )
    parser.add_argument(
        "--force", type=str, default="", metavar="STAGE[,STAGE...]",
        help="Re-run these stages even when their outputs exist. Use 'all' "
             "for everything. Downstream stages re-run too.",
    )
    parser.add_argument(
        "--from", dest="from_stage", type=str, default=None,
        choices=STAGE_ORDER,
        help="Start at this stage, ignoring earlier ones entirely.",
    )
    parser.add_argument(
        "--only", type=str, default=None, choices=STAGE_ORDER,
        help="Run just this stage.",
    )
    parser.add_argument(
        "--skip-frontend", action="store_true",
        help="Stop after the data exports; do not touch npm.",
    )
    parser.add_argument(
        "--dump-topn", type=int, default=0,
        help="Keep only the top-N chains in each dump (0 = all, the default).",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Drop --debug from evaluate.py (less diagnostic output).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and the exact commands, then exit.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Do not pause for confirmation before a run that trains.",
    )
    return parser.parse_args()


def resolve_variants(raw: str):
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if not names:
        raise ValueError("--variants is empty")
    unknown = [n for n in names if n not in VARIANT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown variant(s): {', '.join(unknown)}; "
            f"expected any of {', '.join(VARIANT_NAMES)}"
        )
    # Deduplicate while preserving the order the user asked for.
    seen, ordered = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(variant_by_name(name))
    return tuple(ordered)


def resolve_forced(raw: str) -> set:
    tokens = {t.strip() for t in raw.split(",") if t.strip()}
    if "all" in tokens:
        return set(STAGE_ORDER)
    unknown = tokens - set(STAGE_ORDER)
    if unknown:
        raise ValueError(
            f"unknown stage(s) for --force: {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(STAGE_ORDER)} or 'all'"
        )
    return tokens


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def main() -> int:
    args = parse_args()

    try:
        variants = resolve_variants(args.variants)
        forced = resolve_forced(args.force)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.only and args.from_stage:
        print("error: --only and --from are mutually exclusive", file=sys.stderr)
        return 2

    paths = PipelinePaths.from_root(Path(__file__).parent)
    options = BuildOptions(
        python=args.python,
        epochs=args.epochs,
        device=args.device,
        debug=not args.no_debug,
        dump_topn=args.dump_topn,
    )

    steps = build_steps(
        variants, paths, options, include_frontend=not args.skip_frontend
    )
    steps = filter_steps(steps, args.from_stage, args.only)
    planned = plan(steps, forced)

    running = [p for p in planned if p.action == RUN]
    will_train = any(p.step.stage == "train" for p in running)
    needs_frontend = any(p.step.stage == "frontend" for p in running)
    needs_torch = any(
        p.step.stage in ("train", "evaluate", "neighbors") for p in running
    )

    print(f"\nCI-RCT pipeline — variants: {', '.join(v.name for v in variants)}")
    print(f"root: {paths.root}\n")
    print("Plan")
    print(render_plan(planned))

    if args.dry_run:
        print("\nCommands that would run")
        print(render_commands(planned))
        return 0

    if not running:
        print("\nEverything is already up to date. "
              "Use --force <stage> to rebuild.")
        return 0

    problems = run_checks(
        paths=paths,
        python=args.python,
        device=args.device,
        will_train=will_train,
        needs_frontend=needs_frontend,
        needs_torch=needs_torch,
    )
    if problems:
        fatal = [p for p in problems if p.fatal]
        print(f"\n{'Preflight problems' if fatal else 'Preflight warnings'}")
        print(describe(problems))
        if fatal:
            return 1

    if will_train and not args.yes:
        n_train = sum(1 for p in running if p.step.stage == "train")
        print(
            f"\nThis run retrains {n_train} model(s) at {args.epochs} epochs "
            f"on {args.device}."
        )
        if not confirm("Continue?"):
            print("Aborted. Use --from evaluate to reuse existing checkpoints.")
            return 130

    ensure_directories(paths)

    try:
        results = execute(planned)
    except PipelineError as exc:
        print(f"\n✗ pipeline failed\n{exc}", file=sys.stderr)
        return 1

    print(f"\n{'━' * 72}\nSummary")
    print(render_summary(results))

    if not args.skip_frontend and any(
        r.key == "frontend:build" and r.action == RUN for r in results
    ):
        dist = paths.frontend_dir / "dist"
        print(
            f"\nStatic viewer ready: {dist}\n"
            f"  Preview locally:  cd {paths.frontend_dir} && npm run preview\n"
            f"  Or serve dist/ with any static file server."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
