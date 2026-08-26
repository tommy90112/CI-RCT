"""
Unit tests for RootCauseTracer.

Uses a hand-crafted 5-node linear causal chain:
    target(0) ← 1 ← 2 ← 3 ← root(4)
with CE values decreasing along the chain.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.typed_causal_graph import TypedCausalGraph
from model.root_cause_tracer import RootCauseTracer


@pytest.fixture
def linear_graph():
    """Linear chain: 4 → 3 → 2 → 1 → 0 (target)."""
    V = [0, 1, 2, 3, 4]
    node_types = {i: "transaction" for i in V}
    tcg = TypedCausalGraph(V, node_types)
    for src, dst in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        tcg.add_edge(src, dst, "tx__to__tx")
    return tcg


@pytest.fixture
def strong_ces():
    """Strong CE values along the chain (above 0.1 threshold)."""
    return {
        (1, 0): 0.9, (0, 1): 0.1,
        (2, 1): 0.8, (1, 2): 0.1,
        (3, 2): 0.7, (2, 3): 0.1,
        (4, 3): 0.6, (3, 4): 0.1,
    }


@pytest.fixture
def tracer(linear_graph):
    return RootCauseTracer(linear_graph, max_hops=5, threshold=0.1)


class TestInit:
    def test_invalid_max_hops(self, linear_graph):
        with pytest.raises(ValueError, match="max_hops"):
            RootCauseTracer(linear_graph, max_hops=0)

    def test_invalid_threshold(self, linear_graph):
        with pytest.raises(ValueError, match="threshold"):
            RootCauseTracer(linear_graph, threshold=1.5)


class TestTraceRootCause:
    def test_traces_to_true_root(self, tracer, strong_ces):
        root, chain = tracer.trace_root_cause(0, strong_ces)
        assert root == 4
        assert chain[0] == 0   # starts at target
        assert chain[-1] == 4  # ends at root

    def test_chain_is_monotonic(self, tracer, strong_ces):
        """Chain should go from target toward root without revisiting."""
        _, chain = tracer.trace_root_cause(0, strong_ces)
        assert len(chain) == len(set(chain)), "chain must have no repeated nodes"

    def test_stops_at_threshold(self, tracer):
        """CE below threshold: tracer should stop early."""
        weak_ces = {(1, 0): 0.05}  # below threshold of 0.1
        root, chain = tracer.trace_root_cause(0, weak_ces)
        assert root == 0  # no progress — target is root
        assert chain == [0]

    def test_stops_at_no_upstream(self):
        """Isolated target node (no upstream) immediately returns itself as root."""
        V = [0]
        tcg = TypedCausalGraph(V, {0: "tx"})
        tracer = RootCauseTracer(tcg, max_hops=5, threshold=0.1)
        root, chain = tracer.trace_root_cause(0, {})
        assert root == 0
        assert chain == [0]

    def test_cycle_guard(self):
        """Tracer must not loop on a cyclic graph."""
        V = [0, 1, 2]
        tcg = TypedCausalGraph(V, {i: "tx" for i in V})
        tcg.add_edge(1, 0, "e")
        tcg.add_edge(2, 1, "e")
        tcg.add_edge(0, 2, "e")  # creates cycle 0→2→1→0
        ce = {(1, 0): 0.9, (2, 1): 0.9, (0, 2): 0.9}
        tracer = RootCauseTracer(tcg, max_hops=10, threshold=0.1)
        root, chain = tracer.trace_root_cause(0, ce)
        # Should stop after cycle detection, not loop forever
        assert len(chain) <= 4  # max_hops bound + cycle guard


class TestTraceTopKPaths:
    def test_returns_k_paths(self, tracer, strong_ces):
        results = tracer.trace_top_k_paths(0, strong_ces, k=2)
        assert len(results) <= 2

    def test_sorted_by_score_descending(self, tracer, strong_ces):
        results = tracer.trace_top_k_paths(0, strong_ces, k=3)
        scores = [r[2] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_structure(self, tracer, strong_ces):
        results = tracer.trace_top_k_paths(0, strong_ces, k=1)
        assert len(results) == 1
        root, chain, score = results[0]
        assert isinstance(chain, list)
        assert isinstance(score, float)
        assert chain[0] == 0


class TestScoreChain:
    def test_empty_chain(self):
        assert RootCauseTracer.score_chain([], {}) == 0.0

    def test_single_node_chain(self):
        assert RootCauseTracer.score_chain([0], {}) == 0.0

    def test_two_node_chain(self):
        ce = {(1, 0): 0.7}
        assert RootCauseTracer.score_chain([0, 1], ce) == pytest.approx(0.7)

    def test_multi_hop_chain(self):
        ce = {(1, 0): 0.9, (2, 1): 0.8}
        score = RootCauseTracer.score_chain([0, 1, 2], ce)
        assert score == pytest.approx(1.7)
