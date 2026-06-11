"""
Pluggable RootCauseTracer search strategies.

Each strategy is a pure ``trace(...)`` function (see ``base`` for the contract).
``greedy`` and ``beam`` are NOT here — they are served by the legacy code paths
inside ``RootCauseTracer`` (``trace_root_cause`` / ``trace_top_k_paths``) so the
default ``greedy`` route stays byte-for-byte identical to the pre-ablation tracer.
"""
from model.tracer_strategies import bfs, dag_dp, dfs, dijkstra

# name -> trace callable.  Algorithms handled by the legacy tracer itself
# (``greedy``, ``beam``) are intentionally absent.
STRATEGY_REGISTRY = {
    "dag_dp": dag_dp.trace,
    "dijkstra": dijkstra.trace,
    "bfs": bfs.trace,
    "dfs": dfs.trace,
}

# Algorithms served in-place by RootCauseTracer rather than the registry.
LEGACY_ALGORITHMS = ("greedy", "beam")

# Every selectable --tracer_algorithm value.
ALL_ALGORITHMS = LEGACY_ALGORITHMS + tuple(sorted(STRATEGY_REGISTRY))


def resolve(name: str):
    """Return the strategy callable for ``name`` (registry algorithms only)."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown tracer strategy '{name}'; "
            f"registry choices: {', '.join(sorted(STRATEGY_REGISTRY))} "
            f"(greedy/beam are served by RootCauseTracer directly)"
        )
    return STRATEGY_REGISTRY[name]
