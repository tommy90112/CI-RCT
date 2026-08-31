"""
Tests for the injectable upstream score (φ-weighted tracing).

Verifies that ``trace_root_cause(..., upstream_score_fn=...)`` ranks parents by
the injected score instead of |CE|, while leaving the default (None) path
byte-identical to the legacy |CE| ranking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.root_cause_tracer import RootCauseTracer  # noqa: E402
from model.typed_causal_graph import TypedCausalGraph  # noqa: E402


def _diamond():
    """target=0 with two parents a=1 (high |CE|) and b=2 (low |CE|)."""
    g = TypedCausalGraph(V=[0, 1, 2], node_types={0: "tx", 1: "tx", 2: "tx"})
    g.add_edge(1, 0, "tx__to__tx")
    g.add_edge(2, 0, "tx__to__tx")
    return g


def test_default_path_uses_ce_ranking():
    g = _diamond()
    ce = {(1, 0): 0.9, (2, 0): 0.1}
    tracer = RootCauseTracer(g, max_hops=3, threshold=0.05)
    root, chain = tracer.trace_root_cause(0, ce)
    assert chain[1] == 1  # picks the high-|CE| parent a


def test_injected_score_overrides_ranking():
    g = _diamond()
    ce = {(1, 0): 0.9, (2, 0): 0.1}  # |CE| favours a=1

    # φ favours b=2 instead — the tracer must follow φ, not |CE|.
    def phi_fn(current, upstream):
        return {1: 0.1, 2: 0.9}

    tracer = RootCauseTracer(g, max_hops=3, threshold=0.05)
    root, chain = tracer.trace_root_cause(0, ce, upstream_score_fn=phi_fn)
    assert chain[1] == 2  # picks the high-φ parent b
    assert root == 2


def test_injected_score_threshold_uses_injected_value():
    g = _diamond()
    ce = {(1, 0): 0.9, (2, 0): 0.9}

    # Both parents score below threshold under φ → trace must stop immediately.
    def phi_fn(current, upstream):
        return {u: 0.01 for u in upstream}

    tracer = RootCauseTracer(g, max_hops=3, threshold=0.5)
    root, chain = tracer.trace_root_cause(0, ce, upstream_score_fn=phi_fn)
    assert chain == [0]  # no hop taken
    assert root == 0


def test_signed_score_picks_max_positive_not_abs_max_negative():
    """ce_signed mode: max() over RAW signed scores must prefer the strongest
    positive promoter, NOT the abs-larger but negative (suppressor) parent."""
    g = _diamond()
    ce = {(1, 0): 0.1, (2, 0): 0.1}  # |CE| ties; injected score decides.

    # a=1 has the largest magnitude but is NEGATIVE (a suppressor);
    # b=2 is the strongest positive promoter. Signed max must pick b=2.
    def signed_fn(current, upstream):
        return {1: -0.9, 2: 0.3}

    tracer = RootCauseTracer(g, max_hops=3, threshold=0.05)
    root, chain = tracer.trace_root_cause(0, ce, upstream_score_fn=signed_fn)
    assert chain[1] == 2  # positive promoter, not the abs-larger negative a=1
    assert root == 2


def test_signed_score_negative_below_threshold_stops_trace():
    """ce_signed mode: when every signed score is below threshold (e.g. all
    parents are suppressors), no hop is taken and the trace stops at the seed."""
    g = _diamond()
    ce = {(1, 0): 0.9, (2, 0): 0.9}  # |CE| would happily hop; signed must not.

    def signed_fn(current, upstream):
        return {u: -0.9 for u in upstream}  # all negative → below any +threshold

    tracer = RootCauseTracer(g, max_hops=3, threshold=0.05)
    root, chain = tracer.trace_root_cause(0, ce, upstream_score_fn=signed_fn)
    assert chain == [0]  # no hop taken — negatives never pass the threshold
    assert root == 0


def test_score_fn_receives_current_and_upstream():
    g = _diamond()
    ce = {(1, 0): 0.9, (2, 0): 0.1}
    calls = []

    def phi_fn(current, upstream):
        calls.append((current, sorted(upstream)))
        return {u: 1.0 for u in upstream}

    tracer = RootCauseTracer(g, max_hops=1, threshold=0.0)
    tracer.trace_root_cause(0, ce, upstream_score_fn=phi_fn)
    assert calls[0][0] == 0
    assert calls[0][1] == [1, 2]
