"""
Plan → execute → report.

Planning is separated from execution so `--dry-run` shows exactly what a real
run would do, using the same decision logic rather than a parallel description
that can drift out of sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from .process import Command, PipelineError, format_duration, run
from .stages import Step

SKIP = "skip"
RUN = "run"


@dataclass(frozen=True)
class PlannedStep:
    """A step plus the decision made about it and why."""

    step: Step
    action: str
    reason: str


@dataclass(frozen=True)
class StepResult:
    key: str
    action: str
    seconds: float


def plan(steps: Sequence[Step], forced: Set[str]) -> Tuple[PlannedStep, ...]:
    """Decide run/skip for every step.

    A step runs when it is forced, when an output is missing, or when a
    dependency is itself going to run — the last case matters because reusing
    an artifact built from a model that is about to be retrained would quietly
    mix two different runs into one viewer.
    """
    decisions: Dict[str, str] = {}
    planned: List[PlannedStep] = []

    for step in steps:
        stale_deps = [
            dep for dep in step.depends_on if decisions.get(dep) == RUN
        ]
        if step.stage in forced or step.key in forced:
            action, reason = RUN, "forced"
        elif stale_deps:
            action = RUN
            reason = f"upstream re-ran ({', '.join(stale_deps)})"
        elif step.is_satisfied():
            action = SKIP
            reason = "outputs already present"
        else:
            missing = step.missing_outputs()
            action = RUN
            reason = f"missing {_names(missing)}"

        decisions[step.key] = action
        planned.append(PlannedStep(step=step, action=action, reason=reason))

    return tuple(planned)


def _names(paths: Sequence[Path]) -> str:
    return ", ".join(p.name for p in paths)


def render_plan(planned: Sequence[PlannedStep]) -> str:
    """Format the plan as an ordered checklist."""
    if not planned:
        return "  (nothing to do — the stage filter matched no steps)"
    width = max(len(p.step.key) for p in planned)
    lines: List[str] = []
    for index, item in enumerate(planned, start=1):
        marker = "▶" if item.action == RUN else "·"
        lines.append(
            f"  {index}. {marker} {item.step.key.ljust(width)}  "
            f"{item.step.label}"
        )
        detail = f"        {item.action.upper()}: {item.reason}"
        if item.action == SKIP and item.step.cost_hint:
            detail += f"  (would cost: {item.step.cost_hint})"
        lines.append(detail)
    return "\n".join(lines)


def render_commands(planned: Sequence[PlannedStep]) -> str:
    """Show the exact commands a real run would execute."""
    lines: List[str] = []
    for item in planned:
        if item.action != RUN:
            continue
        lines.append(f"# {item.step.key} — {item.step.label}")
        for command in item.step.commands:
            lines.append(f"cd {command.cwd}")
            lines.append(f"  {command.display()}")
        lines.append("")
    return "\n".join(lines) if lines else "# (every step is up to date)"


def execute(planned: Sequence[PlannedStep]) -> Tuple[StepResult, ...]:
    """Run the plan in order. Raises PipelineError on the first failure."""
    results: List[StepResult] = []
    to_run = [p for p in planned if p.action == RUN]
    total = len(to_run)
    position = 0

    for item in planned:
        if item.action == SKIP:
            print(f"·  skip  {item.step.key}  ({item.reason})", flush=True)
            results.append(StepResult(item.step.key, SKIP, 0.0))
            continue

        position += 1
        print(
            f"\n{'━' * 72}\n"
            f"▶  [{position}/{total}] {item.step.key} — {item.step.label}\n"
            f"   reason: {item.reason}\n"
            f"{'━' * 72}",
            flush=True,
        )
        elapsed = 0.0
        for command in item.step.commands:
            elapsed += run(command)
        print(
            f"✓  {item.step.key} finished in {format_duration(elapsed)}",
            flush=True,
        )
        results.append(StepResult(item.step.key, RUN, elapsed))

    return tuple(results)


def render_summary(results: Sequence[StepResult]) -> str:
    if not results:
        return "  (no steps)"
    width = max(len(r.key) for r in results)
    lines: List[str] = []
    total = 0.0
    for result in results:
        if result.action == SKIP:
            lines.append(f"  ·  {result.key.ljust(width)}  skipped")
        else:
            total += result.seconds
            lines.append(
                f"  ✓  {result.key.ljust(width)}  "
                f"{format_duration(result.seconds)}"
            )
    lines.append(f"\n  total run time: {format_duration(total)}")
    return "\n".join(lines)
