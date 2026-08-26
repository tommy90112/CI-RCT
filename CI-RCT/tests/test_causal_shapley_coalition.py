"""
Tests for non-additive coalition-value Shapley (Ablation route A).

These tests pin down the *raison d'être* of the asymmetric-Shapley ablation:

  * Under an ADDITIVE coalition value v(S) = Σ_{u∈S} c_u, the symmetric and
    asymmetric Shapley values are proportional to c_u for every parent → the
    two are rank-identical and the ablation is a no-op.  (This is exactly the
    degenerate regime the per-edge HeteroNCM forces, see hetero_ncm.py.)

  * Under a NON-additive v(S) (parent interactions), the prefix-restricted
    asymmetric value and the all-permutation symmetric value DIVERGE — which
    is what makes "asymmetric vs symmetric" a meaningful comparison.

The coalition value here is a plain callable v: frozenset -> float, so these
tests need neither a backbone nor a NCM — they verify the Shapley algebra in
isolation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.causal_shapley import (  # noqa: E402
    compute_asymmetric_causal_shapley,
    compute_symmetric_causal_shapley,
)
from model.typed_causal_graph import TypedCausalGraph  # noqa: E402


def _two_parent_graph():
    """target=2 with parents {0, 1}; topo order 0 < 1 < 2 (0 = most upstream)."""
    node_types = {0: "wallet", 1: "wallet", 2: "transaction"}
    g = TypedCausalGraph(V=[0, 1, 2], node_types=node_types)
    g.add_edge(0, 2, "wallet__to__transaction")
    g.add_edge(1, 2, "wallet__to__transaction")
    return g


def test_additive_coalition_makes_sym_equal_asym_ranking():
    """Additive v(S) ⇒ symmetric and asymmetric agree (degenerate regime)."""
    g = _two_parent_graph()
    c = {0: 0.3, 1: 0.7}  # per-parent additive contributions

    def v_additive(S):
        return sum(c[u] for u in S)

    asym = compute_asymmetric_causal_shapley(
        {}, g, target_node=2, coalition_value_fn=v_additive
    )
    sym = compute_symmetric_causal_shapley(
        g, target_node=2, coalition_value_fn=v_additive, n_permutations=64
    )

    # Both must rank parent 1 above parent 0 (0.7 > 0.3), identically.
    assert asym[1] > asym[0]
    assert sym[1] > sym[0]
    # Symmetric Shapley of an additive game returns each parent's own c_u.
    assert abs(sym[0] - 0.3) < 1e-6
    assert abs(sym[1] - 0.7) < 1e-6


def test_nonadditive_coalition_makes_sym_diverge_from_asym():
    """Strong parent interaction ⇒ asymmetric (prefix) ≠ symmetric.

    v({a,b}) ≫ v({a}) + v({b}): the pair only "fires" together. The prefix
    asymmetric value (topo order 0→1) credits almost the entire interaction to
    the *later* parent 1, whereas the symmetric value splits it evenly.
    """
    g = _two_parent_graph()
    table = {
        frozenset(): 0.0,
        frozenset({0}): 0.1,
        frozenset({1}): 0.1,
        frozenset({0, 1}): 1.0,
    }

    def v_inter(S):
        return table[frozenset(S)]

    asym = compute_asymmetric_causal_shapley(
        {}, g, target_node=2, coalition_value_fn=v_inter
    )
    sym = compute_symmetric_causal_shapley(
        g, target_node=2, coalition_value_fn=v_inter, n_permutations=256
    )

    # Symmetric: by symmetry of the game both parents get exactly 0.5.
    assert abs(sym[0] - 0.5) < 1e-6
    assert abs(sym[1] - 0.5) < 1e-6

    # Asymmetric prefix (0 enters first): parent 1's marginal carries the
    # interaction, so φ_1^asym ≫ φ_0^asym — a clear divergence from symmetric.
    assert asym[1] > asym[0]
    assert asym[1] > asym[0] + 0.3  # strongly asymmetric, not a rounding wobble


def test_additive_fallback_preserves_legacy_behaviour():
    """coalition_value_fn=None ⇒ legacy CE/n additive approximation, unchanged."""
    g = _two_parent_graph()
    ce = {(0, 2): 0.3, (1, 2): 0.7}
    phi = compute_asymmetric_causal_shapley(ce, g, target_node=2)
    # Legacy formula: φ = CE / n_parents, n=2.
    assert abs(phi[0] - 0.15) < 1e-9
    assert abs(phi[1] - 0.35) < 1e-9


def test_empty_parents_returns_empty():
    g = TypedCausalGraph(V=[0], node_types={0: "transaction"})
    assert compute_asymmetric_causal_shapley({}, g, 0) == {}
    assert compute_symmetric_causal_shapley(
        g, 0, coalition_value_fn=lambda S: 0.0
    ) == {}
