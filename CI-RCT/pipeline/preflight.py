"""
Preconditions checked once, before any long-running stage starts.

The point is to fail in seconds rather than three hours into a training run,
so every check here is cheap and every message says what to do about it.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .config import PipelinePaths, REQUIRED_DATASET_FILES


@dataclass(frozen=True)
class Problem:
    """One failed precondition. `fatal` stops the run; otherwise it warns."""

    message: str
    remedy: str
    fatal: bool = True


def check_dataset(paths: PipelinePaths) -> Tuple[Problem, ...]:
    if not paths.dataset_dir.is_dir():
        return (
            Problem(
                f"Elliptic++ dataset directory not found: {paths.dataset_dir}",
                "Download Elliptic++ and place the CSVs there (see CLAUDE.md "
                "§ Dataset placement).",
            ),
        )
    missing = [
        name
        for name in REQUIRED_DATASET_FILES
        if not (paths.dataset_dir / name).is_file()
    ]
    if missing:
        return (
            Problem(
                f"Elliptic++ is missing {len(missing)} table(s): "
                + ", ".join(missing),
                f"Place the missing CSVs in {paths.dataset_dir}.",
            ),
        )
    return ()


def check_torch(python: str) -> Tuple[Problem, ...]:
    """Import torch_geometric in a subprocess.

    A mismatched torch / torch-scatter pair segfaults the interpreter on
    import rather than raising, so this must not run in-process.
    """
    probe = "import torch_geometric, torch_scatter"
    try:
        completed = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            Problem(
                f"could not probe the PyTorch environment with {python!r}: {exc}",
                "Check that the interpreter exists and torch is installed.",
            ),
        )

    if completed.returncode == 0:
        return ()

    crashed = completed.returncode < 0 or "Segmentation fault" in completed.stderr
    detail = "segfaulted" if crashed else "failed"
    tail = completed.stderr.strip().splitlines()[-1:] or [""]
    return (
        Problem(
            f"`import torch_geometric` {detail} under {python}: {tail[0]}",
            "torch and torch-scatter/torch-sparse are built against different "
            "torch versions. Reinstall the PyG extras for your exact torch "
            "build, or point the pipeline at a working interpreter with "
            "--python /path/to/python.",
        ),
    )


def check_node(paths: PipelinePaths) -> Tuple[Problem, ...]:
    problems: List[Problem] = []
    if shutil.which("npm") is None:
        problems.append(
            Problem(
                "npm was not found on PATH",
                "Install Node.js (>= 18), or pass --skip-frontend to stop "
                "after the data export.",
            )
        )
    if not (paths.frontend_dir / "package.json").is_file():
        problems.append(
            Problem(
                f"frontend project not found at {paths.frontend_dir}",
                "Pass --skip-frontend, or restore frontend_temp/.",
            )
        )
    return tuple(problems)


def check_device(device: str, will_train: bool) -> Tuple[Problem, ...]:
    if will_train and device == "cpu":
        return (
            Problem(
                "training on CPU was requested",
                "A 400-epoch Elliptic++ arm with the GAN is impractical on "
                "CPU. Use --device cuda (or mps), or supply a pre-trained "
                "checkpoint so the train stage is skipped.",
                fatal=False,
            ),
        )
    return ()


def run_checks(
    paths: PipelinePaths,
    python: str,
    device: str,
    will_train: bool,
    needs_frontend: bool,
    needs_torch: bool,
) -> Tuple[Problem, ...]:
    """Collect every applicable problem so the user sees them all at once."""
    problems: List[Problem] = []
    if needs_torch:
        problems.extend(check_dataset(paths))
        problems.extend(check_torch(python))
    problems.extend(check_device(device, will_train))
    if needs_frontend:
        problems.extend(check_node(paths))
    return tuple(problems)


def ensure_directories(paths: PipelinePaths) -> None:
    """Create the output directories the stages write into."""
    for directory in (paths.checkpoint_dir, paths.viz_dir, paths.results_dir):
        directory.mkdir(parents=True, exist_ok=True)


def describe(problems: Sequence[Problem]) -> str:
    lines: List[str] = []
    for problem in problems:
        marker = "✗" if problem.fatal else "!"
        lines.append(f"  {marker} {problem.message}")
        lines.append(f"      → {problem.remedy}")
    return "\n".join(lines)
