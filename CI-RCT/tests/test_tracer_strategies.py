"""
Unit tests for the pluggable RootCauseTracer search strategies (ablation).

Scope note: the dead-end fixture below is a SYNTHETIC correctness unit test
(an MG24-style hub/bridge topology) used only to prove that the global-optimal
strategies resolve the greedy dead-end WITHOUT the prefer_reachable_depth
lookahead patch. It is NOT the thesis experimental validation — the ablation
numbers (RCP / CCV / RHR / MTD) are produced on Elliptic++ via evaluate.py
--tracer_algorithm. See tracer_ablation_plan.md.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.root_cause_tracer import RootCauseTracer
from model.tracer_strategies import dag_dp, dijkstra
from model.typed_causal_graph import TypedCausalGraph


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def linear_graph():
    """Linear chain 4 → 3 → 2 → 1 → 0 (target=0, true root=4)."""
    V = [0, 1, 2, 3, 4]
    tcg = TypedCausalGraph(V, {i: "transaction" for i in V})
    for src, dst in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        tcg.add_edge(src, dst, "tx__to__tx")
    return tcg


@pytest.fixture
def linear_ces():
    return {(1, 0): 0.9, (2, 1): 0.8, (3, 2): 0.7, (4, 3): 0.6}


@pytest.fixture
def branch_graph():
    """
    Branching DAG (target=0); wallet sources {5, 6} are the root-capable type.
        5→3→1→0 , 6→4→1→0 , 6→4→2→0
    """
    V = [0, 1, 2, 3, 4, 5, 6]
    types = {0: "transaction", 1: "transaction", 2: "transaction",
             3: "transaction", 4: "transaction", 5: "wallet", 6: "wallet"}
    tcg = TypedCausalGraph(V, types)
    for src, dst in [(1, 0), (2, 0), (3, 1), (4, 1), (4, 2), (5, 3), (6, 4)]:
        tcg.add_edge(src, dst, "edge")
    return tcg


@pytest.fixture
def branch_ces():
    # best max-product path to a wallet: 0←2←4←6 = 0.5*0.9*0.95 = 0.4275
    return {(1, 0): 0.6, (2, 0): 0.5, (3, 1): 0.7,
            (4, 1): 0.4, (4, 2): 0.9, (5, 3): 0.8, (6, 4): 0.95}


@pytest.fixture
def deadend_graph():
    """
    MG24-style dd18 dead-end (SYNTHETIC unit-test topology):
        H→T ; Bdead→H ; Blive→H ; R→Blive
    The flow target T's only parent is hub host H. H has two host parents:
    Bdead (a 0-parent dead-end host, highest |CE|) and Blive (slightly smaller
    |CE|) which leads to the true root process R. Greedy |CE|-max dead-ends at
    Bdead; a whole-path optimiser with a process-type terminal reaches R.
    """
    V = ["T", "H", "Bdead", "Blive", "R"]
    types = {"T": "flow", "H": "host", "Bdead": "host", "Blive": "host", "R": "process"}
    tcg = TypedCausalGraph(V, types)
    for src, dst in [("H", "T"), ("Bdead", "H"), ("Blive", "H"), ("R", "Blive")]:
        tcg.add_edge(src, dst, "edge")
    return tcg


@pytest.fixture
def deadend_ces():
    return {("H", "T"): 0.9, ("Bdead", "H"): 0.8, ("Blive", "H"): 0.7, ("R", "Blive"): 0.95}


# ── Greedy stays byte-identical ───────────────────────────────────────────────

class TestGreedyByteIdentical:
    def test_default_equals_explicit_greedy_linear(self, linear_graph, linear_ces):
        legacy = RootCauseTracer(linear_graph, max_hops=5, threshold=0.1)
        explicit = RootCauseTracer(linear_graph, max_hops=5, threshold=0.1,
                                   tracer_algorithm="greedy")
        assert legacy.trace_root_cause(0, linear_ces) == explicit.trace_root_cause(0, linear_ces)

    def test_default_equals_explicit_greedy_deadend(self, deadend_graph, deadend_ces):
        legacy = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1,
                                 prefer_root_types={"process"})
        explicit = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1,
                                   prefer_root_types={"process"},
                                   tracer_algorithm="greedy")
        assert legacy.trace_root_cause("T", deadend_ces) == explicit.trace_root_cause("T", deadend_ces)


class TestInvalidAlgorithm:
    def test_unknown_algorithm_raises(self, linear_graph):
        with pytest.raises(ValueError, match="tracer_algorithm"):
            RootCauseTracer(linear_graph, tracer_algorithm="nope")

    def test_unknown_objective_raises(self, linear_graph):
        with pytest.raises(ValueError, match="tracer_objective"):
            RootCauseTracer(linear_graph, tracer_objective="median")


# ── DAG-DP / Dijkstra correctness ─────────────────────────────────────────────

class TestDagDpLinear:
    @pytest.mark.parametrize("algo", ["dag_dp", "dijkstra", "bfs", "dfs"])
    def test_all_reach_true_root_on_linear(self, linear_graph, linear_ces, algo):
        tracer = RootCauseTracer(linear_graph, max_hops=5, threshold=0.1,
                                 tracer_algorithm=algo)
        root, chain = tracer.trace_root_cause(0, linear_ces)
        assert root == 4
        assert chain[0] == 0 and chain[-1] == 4

    @pytest.mark.parametrize("objective", ["product", "sum"])
    def test_dag_dp_full_chain(self, linear_graph, linear_ces, objective):
        root, chain = dag_dp.trace(0, linear_ces, graph=linear_graph, max_hops=5,
                                   threshold=0.1, objective=objective)
        assert chain == [0, 1, 2, 3, 4]


class TestDagDpEqualsDijkstra:
    def test_same_optimum_on_branching_dag(self, branch_graph, branch_ces):
        kw = dict(graph=branch_graph, max_hops=5, threshold=0.1,
                  prefer_root_types={"wallet"}, objective="product")
        r1, c1 = dag_dp.trace(0, branch_ces, **kw)
        r2, c2 = dijkstra.trace(0, branch_ces, **kw)
        assert (r1, c1) == (r2, c2)
        assert r1 == 6 and c1 == [0, 2, 4, 6]   # the max-product wallet path


# ── dd18 dead-end: global optimisers fix it WITHOUT lookahead ─────────────────

class TestDeadEndFixture:
    def test_greedy_deadends_at_bridge(self, deadend_graph, deadend_ces):
        # legacy greedy, no lookahead → walks into the highest-|CE| dead-end host
        tracer = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1,
                                 prefer_root_types={"process"})
        root, _ = tracer.trace_root_cause("T", deadend_ces)
        assert root == "Bdead"

    @pytest.mark.parametrize("algo", ["dag_dp", "dijkstra", "bfs"])
    def test_global_optimisers_reach_true_root(self, deadend_graph, deadend_ces, algo):
        tracer = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1,
                                 prefer_root_types={"process"},
                                 tracer_algorithm=algo)
        root, chain = tracer.trace_root_cause("T", deadend_ces)
        assert root == "R"
        assert chain == ["T", "H", "Blive", "R"]


class TestDfsEqualsGreedy:
    def test_dfs_matches_legacy_greedy_no_tiebreak(self, deadend_graph, deadend_ces):
        # dfs (single-deep |CE|-max) must equal greedy with NO prefer/lookahead
        greedy = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1)
        dfs_tracer = RootCauseTracer(deadend_graph, max_hops=5, threshold=0.1,
                                     tracer_algorithm="dfs")
        assert greedy.trace_root_cause("T", deadend_ces) == \
            dfs_tracer.trace_root_cause("T", deadend_ces)


# ── Immutability: strategies must not mutate inputs ───────────────────────────

class TestImmutability:
    @pytest.mark.parametrize("algo", ["dag_dp", "dijkstra", "bfs", "dfs", "beam"])
    def test_inputs_unchanged(self, branch_graph, branch_ces, algo):
        ces_snapshot = dict(branch_ces)
        pa_snapshot = {v: set(branch_graph.pa[v]) for v in branch_graph.v}
        tracer = RootCauseTracer(branch_graph, max_hops=5, threshold=0.1,
                                 prefer_root_types={"wallet"}, tracer_algorithm=algo)
        tracer.trace_root_cause(0, branch_ces)
        assert branch_ces == ces_snapshot
        assert {v: set(branch_graph.pa[v]) for v in branch_graph.v} == pa_snapshot
