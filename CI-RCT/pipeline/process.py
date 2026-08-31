"""
Subprocess execution with live output.

Stages are long-running (a training arm is hours), so output is streamed to the
terminal as it arrives rather than captured and replayed at the end.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple


class PipelineError(RuntimeError):
    """A stage failed. Carries enough context to act on without a stack trace."""


@dataclass(frozen=True)
class Command:
    """One executable invocation, pinned to a working directory."""

    argv: Tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)

    def display(self) -> str:
        """Render as a copy-pasteable shell line, wrapped for readability."""
        parts = [_quote(a) for a in self.argv]
        lines, current = [], []
        for part in parts:
            # 76 leaves room for the two-space indent and the trailing " \".
            if current and sum(len(p) + 1 for p in current) + len(part) > 76:
                lines.append(" ".join(current))
                current = [part]
            else:
                current.append(part)
        if current:
            lines.append(" ".join(current))
        return " \\\n    ".join(lines)


def _quote(arg: str) -> str:
    """Quote only when a shell would need it, so commands stay readable."""
    if arg and all(c.isalnum() or c in "-_=./+:@," for c in arg):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"


def run(command: Command) -> float:
    """Run `command`, streaming its output. Returns elapsed seconds.

    Raises PipelineError on a non-zero exit or a missing executable, so the
    runner can report which stage failed without unwinding a traceback.
    """
    import os

    env = {**os.environ, **command.env} if command.env else None
    started = time.monotonic()
    try:
        completed = subprocess.run(command.argv, cwd=str(command.cwd), env=env)
    except FileNotFoundError as exc:
        raise PipelineError(
            f"executable not found: {command.argv[0]!r}\n"
            f"  while running: {command.display()}"
        ) from exc
    except KeyboardInterrupt:
        raise PipelineError("interrupted by user") from None

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise PipelineError(
            f"command exited with status {completed.returncode}:\n"
            f"  {command.display()}\n"
            f"  (cwd: {command.cwd})"
        )
    return elapsed


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time: 45s, 3m12s, 2h05m."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"
