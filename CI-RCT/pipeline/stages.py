"""
Stage graph for the end-to-end pipeline.

    train:<variant>  ──▶  evaluate:<variant>  ──┬──▶  neighbors  ──┐
                                                │                  ├──▶  frontend:build
                                                └──────────────────┘

Each step declares the artifacts that prove it has already run. The runner
skips a step whose outputs all exist, unless it is forced or an upstream step
actually re-ran (which would leave the downstream artifact stale).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import (
    DATA_FLAGS,
    DATASET,
    DUMP_FLAGS,
    EVAL_FLAGS,
    NEIGHBOR_CAP,
    PipelinePaths,
    TRAIN_FLAGS,
    Variant,
)
from .process import Command

#: Stage kinds in execution order — the vocabulary for --from / --only.
STAGE_ORDER: Tuple[str, ...] = ("train", "evaluate", "neighbors", "frontend")


@dataclass(frozen=True)
class Step:
    """A single unit of work with declared inputs, outputs and dependencies."""

    key: str
    stage: str
    label: str
    commands: Tuple[Command, ...]
    outputs: Tuple[Path, ...]
    depends_on: Tuple[str, ...] = ()
    #: Shown when the step is skipped, to explain what would otherwise happen.
    cost_hint: str = ""

    def is_satisfied(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)

    def missing_outputs(self) -> Tuple[Path, ...]:
        return tuple(p for p in self.outputs if not p.exists())


@dataclass(frozen=True)
class BuildOptions:
    """Knobs the CLI exposes that change how commands are constructed."""

    python: str
    epochs: int
    device: str
    debug: bool = True
    dump_topn: int = 0


def _train_step(
    variant: Variant, paths: PipelinePaths, options: BuildOptions
) -> Step:
    checkpoint = variant.checkpoint_path(paths, DATASET)
    argv = (
        options.python, "-u", "train.py",
        "--variant", variant.name,
        *DATA_FLAGS,
        *TRAIN_FLAGS,
        "--epochs", str(options.epochs),
        "--device", options.device,
        "--checkpoint_dir", str(paths.checkpoint_dir),
    )
    return Step(
        key=f"train:{variant.name}",
        stage="train",
        label=f"Train {variant.name} head ({options.epochs} epochs)",
        commands=(Command(argv=argv, cwd=paths.root),),
        outputs=(checkpoint,),
        cost_hint="a full retrain; hours on GPU, impractical on CPU",
    )


def _evaluate_step(
    variant: Variant, paths: PipelinePaths, options: BuildOptions
) -> Step:
    checkpoint = variant.checkpoint_path(paths, DATASET)
    chains = variant.chains_path(paths)
    csv_path = paths.results_dir / "crime_chains.csv"

    argv: List[str] = [
        options.python, "-u", "evaluate.py",
        "--variant", variant.name,
        *DATA_FLAGS,
        *EVAL_FLAGS,
        *DUMP_FLAGS,
        "--checkpoint", str(checkpoint),
        "--dump_chains", str(chains),
        "--device", options.device,
    ]
    if options.dump_topn > 0:
        argv += ["--dump_chains_topn", str(options.dump_topn)]
    if options.debug:
        argv.append("--debug")

    outputs = [chains]
    # Only the primary variant writes the flat CSV: the viewer serves a single
    # CSV fallback and the neighbour export reads exactly this file.
    if variant.is_primary:
        argv += ["--dump_csv", str(csv_path)]
        outputs.append(csv_path)

    return Step(
        key=f"evaluate:{variant.name}",
        stage="evaluate",
        label=f"Evaluate {variant.name} + export chains with φ/L3",
        commands=(Command(argv=tuple(argv), cwd=paths.root),),
        outputs=tuple(outputs),
        depends_on=(f"train:{variant.name}",),
        cost_hint="re-traces 2000 chains and recomputes φ_asym",
    )


def _neighbors_step(
    primary: Variant, paths: PipelinePaths, options: BuildOptions
) -> Step:
    out = paths.results_dir / "chain_neighbors.json"
    argv = (
        options.python, "-u", "scripts/export_chain_neighbors.py",
        "--chains", str(paths.results_dir / "crime_chains.csv"),
        "--data_root", str(paths.dataset_dir),
        "--out", str(out),
        "--cap", str(NEIGHBOR_CAP),
    )
    return Step(
        key="neighbors",
        stage="neighbors",
        label="Export real 1-hop neighbourhood overlay",
        commands=(Command(argv=argv, cwd=paths.root),),
        outputs=(out,),
        depends_on=(f"evaluate:{primary.name}",),
        cost_hint="streams AddrAddr_edgelist.csv (~200 MB) once",
    )


def _frontend_steps(
    paths: PipelinePaths, options: BuildOptions, upstream: Sequence[str]
) -> Tuple[Step, ...]:
    frontend = paths.frontend_dir
    deps = Step(
        key="frontend:deps",
        stage="frontend",
        label="Install frontend dependencies",
        commands=(Command(argv=("npm", "install"), cwd=frontend),),
        outputs=(frontend / "node_modules",),
        cost_hint="npm install",
    )
    build = Step(
        key="frontend:build",
        stage="frontend",
        label="Build static viewer into frontend_temp/dist",
        commands=(Command(argv=("npm", "run", "build"), cwd=frontend),),
        outputs=(frontend / "dist" / "index.html",),
        depends_on=("frontend:deps", *upstream),
        cost_hint="vite build; also copies the dumps into dist/",
    )
    return (deps, build)


def build_steps(
    variants: Sequence[Variant],
    paths: PipelinePaths,
    options: BuildOptions,
    include_frontend: bool = True,
) -> Tuple[Step, ...]:
    """Assemble the full ordered step list for the requested variants."""
    if not variants:
        raise ValueError("at least one variant is required")

    primary = next((v for v in variants if v.is_primary), None)
    steps: List[Step] = []
    for variant in variants:
        steps.append(_train_step(variant, paths, options))
        steps.append(_evaluate_step(variant, paths, options))

    upstream = [f"evaluate:{v.name}" for v in variants]
    # The neighbour overlay is keyed off the primary variant's CSV. Without the
    # primary variant in this run there is nothing to build it from, so the
    # step is omitted rather than silently reusing a stale CSV.
    if primary is not None:
        steps.append(_neighbors_step(primary, paths, options))
        upstream.append("neighbors")

    if include_frontend:
        steps.extend(_frontend_steps(paths, options, upstream))

    return tuple(steps)


def filter_steps(
    steps: Sequence[Step], from_stage: str | None, only_stage: str | None
) -> Tuple[Step, ...]:
    """Narrow the plan to a stage window, preserving order."""
    if only_stage:
        return tuple(s for s in steps if s.stage == only_stage)
    if from_stage:
        start = STAGE_ORDER.index(from_stage)
        allowed = set(STAGE_ORDER[start:])
        return tuple(s for s in steps if s.stage in allowed)
    return tuple(steps)
