"""
Unit tests for HeteroNCM.

Uses mock tensors; no real graph data required.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.hetero_ncm import HeteroNCM
from model.typed_causal_graph import TypedCausalGraph


EMB_DIM = 8
TYPE_EMB_DIM = 4
ALL_NODE_TYPES = ["actor", "transaction"]
ALL_EDGE_TYPES = ["actor__to__transaction", "transaction__to__actor", "transaction__to__transaction"]


@pytest.fixture
def ncm():
    return HeteroNCM(
        node_emb_dim=EMB_DIM,
        all_node_types=ALL_NODE_TYPES,
        all_edge_types=ALL_EDGE_TYPES,
        node_type_emb_dim=TYPE_EMB_DIM,
        ncm_h_size=16,
        ncm_h_layers=1,
    )


@pytest.fixture
def simple_graph():
    V = [0, 1, 2, 3]
    node_types = {0: "transaction", 1: "transaction", 2: "actor", 3: "actor"}
    tcg = TypedCausalGraph(V, node_types)
    tcg.add_edge(0, 1, "transaction__to__transaction")
    tcg.add_edge(0, 2, "transaction__to__actor")
    tcg.add_edge(2, 3, "actor__to__transaction")
    return tcg


class TestInit:
    def test_edge_type_models_created(self, ncm):
        for etype in ALL_EDGE_TYPES:
            assert etype in ncm.edge_type_models

    def test_type_embeddings_shape(self, ncm):
        assert ncm.type_embeddings.num_embeddings == len(ALL_NODE_TYPES)
        assert ncm.type_embeddings.embedding_dim == TYPE_EMB_DIM

    def test_empty_node_types_raises(self):
        with pytest.raises(ValueError, match="all_node_types"):
            HeteroNCM(EMB_DIM, [], ALL_EDGE_TYPES)

    def test_empty_edge_types_raises(self):
        with pytest.raises(ValueError, match="all_edge_types"):
            HeteroNCM(EMB_DIM, ALL_NODE_TYPES, [])


class TestComputeCausalEffect:
    def test_output_is_scalar(self, ncm):
        h_source = torch.randn(EMB_DIM)
        ce = ncm.compute_causal_effect(h_source, "transaction__to__actor", "transaction")
        assert ce.shape == torch.Size([])  # scalar

    def test_null_intervention_effect(self, ncm):
        """Zero source should produce lower CE than actual source (on average)."""
        h_zero = torch.zeros(EMB_DIM)
        h_actual = torch.randn(EMB_DIM)
        ce_zero = ncm.compute_causal_effect(h_zero, "transaction__to__actor", "transaction")
        ce_actual = ncm.compute_causal_effect(h_actual, "transaction__to__actor", "transaction")
        # CE = p(actual) - p(null) for h_actual;  p(null) - p(null) = 0 for h_zero
        assert ce_zero.item() == pytest.approx(0.0, abs=1e-5)
        _ = ce_actual  # just verify it runs without error

    def test_unknown_edge_type_raises(self, ncm):
        h = torch.randn(EMB_DIM)
        with pytest.raises(KeyError):
            ncm.compute_causal_effect(h, "nonexistent__edge", "transaction")


class TestForward:
    def test_forward_returns_dict(self, ncm, simple_graph):
        flat_h = {i: torch.randn(EMB_DIM) for i in range(4)}
        result = ncm.forward(flat_h, simple_graph)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_forward_all_keys_are_tuples(self, ncm, simple_graph):
        flat_h = {i: torch.randn(EMB_DIM) for i in range(4)}
        result = ncm.forward(flat_h, simple_graph)
        for key in result:
            assert isinstance(key, tuple) and len(key) == 2

    def test_detached_returns_floats(self, ncm, simple_graph):
        flat_h = {i: torch.randn(EMB_DIM) for i in range(4)}
        result = ncm.detached_causal_effects(flat_h, simple_graph)
        for v in result.values():
            assert isinstance(v, float)

    def test_missing_node_skipped(self, ncm, simple_graph):
        """If a node's embedding is absent, that edge is silently skipped."""
        flat_h = {0: torch.randn(EMB_DIM)}  # only node 0
        result = ncm.forward(flat_h, simple_graph)
        # Most edges involve nodes 1-3 which are absent, so result should be small
        assert isinstance(result, dict)


class TestTxToWalletSupervision:
    """Path 2b: tx→wallet edges must be supervised by the dst wallet label,
    otherwise that edge-type MLP never trains and its CE collapses to ~0."""

    def _wallet_ncm(self):
        return HeteroNCM(
            node_emb_dim=EMB_DIM,
            all_node_types=["transaction", "wallet"],
            all_edge_types=["wallet__to__transaction", "transaction__to__wallet"],
            node_type_emb_dim=TYPE_EMB_DIM,
            ncm_h_size=16,
            ncm_h_layers=1,
        )

    def _tx_to_wallet_graph(self):
        # node 0 = transaction, node 1 = wallet;  edge 0 → 1 (tx → wallet)
        tcg = TypedCausalGraph([0, 1], {0: "transaction", 1: "wallet"})
        tcg.add_edge(0, 1, "transaction__to__wallet")
        return tcg

    def test_tx_to_wallet_is_supervised(self):
        """A graph with ONLY a tx→wallet edge yields a non-zero NCM loss once
        wallet_labels are supplied (was silently skipped → 0 before Path 2b)."""
        ncm = self._wallet_ncm()
        tcg = self._tx_to_wallet_graph()
        flat_h = {0: torch.randn(EMB_DIM), 1: torch.randn(EMB_DIM)}
        loss = ncm.supervised_ncm_loss(
            flat_h, tcg,
            node_labels=torch.tensor([0.0]),   # transaction labels (unused here)
            target_type_offset=0,
            wallet_labels=torch.tensor([1.0]),  # the dst wallet (local 0) is fraud
            wallet_type_offset=1,               # wallet global id 1 → local 0
        )
        assert loss.item() > 0.0
        assert loss.requires_grad

    def test_tx_to_wallet_skipped_without_wallet_labels(self):
        """No wallet_labels → tx→wallet falls through and contributes nothing."""
        ncm = self._wallet_ncm()
        tcg = self._tx_to_wallet_graph()
        flat_h = {0: torch.randn(EMB_DIM), 1: torch.randn(EMB_DIM)}
        loss = ncm.supervised_ncm_loss(
            flat_h, tcg,
            node_labels=torch.tensor([0.0]),
            target_type_offset=0,
            wallet_labels=None,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


def _make_ncm(baseline="zero"):
    return HeteroNCM(
        node_emb_dim=EMB_DIM,
        all_node_types=ALL_NODE_TYPES,
        all_edge_types=ALL_EDGE_TYPES,
        node_type_emb_dim=TYPE_EMB_DIM,
        ncm_h_size=16,
        ncm_h_layers=1,
        baseline=baseline,
    )


class TestBaselineMode:
    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            _make_ncm(baseline="banana")

    def test_default_is_zero(self, ncm):
        assert ncm.baseline == "zero"

    def test_zero_baseline_byte_identical(self):
        """baseline=None in compute_causal_effect == legacy zero vector path."""
        torch.manual_seed(0)
        ncm = _make_ncm("zero")
        h = torch.randn(EMB_DIM)
        explicit_zero = ncm.compute_causal_effect(
            h, "transaction__to__actor", "transaction",
            baseline_h=torch.zeros(EMB_DIM),
        )
        implicit = ncm.compute_causal_effect(
            h, "transaction__to__actor", "transaction",
        )
        assert implicit.item() == pytest.approx(explicit_zero.item(), abs=1e-7)

    def test_ce_zero_when_source_equals_baseline(self):
        """CE = p(h) - p(h_base) collapses to 0 when h_source == h_base."""
        ncm = _make_ncm("type_mean")
        h = torch.randn(EMB_DIM)
        ce = ncm.compute_causal_effect(
            h, "transaction__to__actor", "transaction", baseline_h=h,
        )
        assert ce.item() == pytest.approx(0.0, abs=1e-6)

    def test_type_mean_recenters_to_zero(self, simple_graph):
        """If every node of a type shares one embedding, the type mean equals it,
        so CE on edges leaving that type collapses to 0 — proving the baseline is
        the per-type mean, not the zero vector."""
        ncm = _make_ncm("type_mean")
        shared = torch.randn(EMB_DIM)
        flat_h = {i: shared.clone() for i in range(4)}  # all nodes identical
        result = ncm.detached_causal_effects(flat_h, simple_graph)
        assert len(result) > 0
        for ce in result.values():
            assert ce == pytest.approx(0.0, abs=1e-6)

    def test_type_mean_differs_from_zero(self, simple_graph):
        """Same non-zero embeddings give different CE under the two baselines."""
        torch.manual_seed(1)
        flat_h = {i: torch.randn(EMB_DIM) + 3.0 for i in range(4)}
        ce_zero = _make_ncm("zero").detached_causal_effects(flat_h, simple_graph)
        ce_mean = _make_ncm("type_mean").detached_causal_effects(flat_h, simple_graph)
        # at least one shared edge must differ between the two baselines
        shared_keys = set(ce_zero) & set(ce_mean)
        assert shared_keys
        assert any(
            abs(ce_zero[k] - ce_mean[k]) > 1e-4 for k in shared_keys
        )


class TestMarginalBaseline:
    """baseline="marginal": p_null = E[MLP(h)] — the mean of PREDICTIONS over
    all same-type source nodes — instead of MLP(E[h]) (prediction AT the mean
    embedding, which carries a Jensen gap through the nonlinear MLP). This is
    the true marginal null intervention do(h_u ~ P(h_type)); its defining
    property is that CE averaged over all same-type sources is exactly 0."""

    def _fanin_graph(self):
        # Two transactions both feeding one actor — every node of the source
        # type is a source of the same edge type, so E[CE] over the two edges
        # must vanish under the marginal baseline.
        tcg = TypedCausalGraph(
            [0, 1, 2], {0: "transaction", 1: "transaction", 2: "actor"}
        )
        tcg.add_edge(0, 2, "transaction__to__actor")
        tcg.add_edge(1, 2, "transaction__to__actor")
        return tcg

    def test_marginal_accepted(self):
        assert _make_ncm("marginal").baseline == "marginal"

    def test_marginal_ce_mean_is_zero(self):
        torch.manual_seed(2)
        ncm = _make_ncm("marginal")
        tcg = self._fanin_graph()
        flat_h = {i: torch.randn(EMB_DIM) * 3.0 for i in range(3)}
        ces = ncm.detached_causal_effects(flat_h, tcg)
        assert set(ces) == {(0, 2), (1, 2)}
        assert (ces[(0, 2)] + ces[(1, 2)]) / 2 == pytest.approx(0.0, abs=1e-6)

    def test_marginal_ce_zero_when_sources_identical(self):
        """All same-type sources share one embedding ⇒ every prediction equals
        the mean prediction ⇒ CE = 0 on every edge."""
        ncm = _make_ncm("marginal")
        tcg = self._fanin_graph()
        shared = torch.randn(EMB_DIM)
        flat_h = {i: shared.clone() for i in range(3)}
        for ce in ncm.detached_causal_effects(flat_h, tcg).values():
            assert ce == pytest.approx(0.0, abs=1e-6)

    def test_marginal_differs_from_type_mean(self):
        """Identical weights, spread-out embeddings: the Jensen gap makes
        E[MLP(h)] ≠ MLP(E[h]), so the two baselines give different CE."""
        torch.manual_seed(7)
        ncm_mean = _make_ncm("type_mean")
        torch.manual_seed(7)
        ncm_marg = _make_ncm("marginal")
        tcg = self._fanin_graph()
        torch.manual_seed(8)
        flat_h = {i: torch.randn(EMB_DIM) * 4.0 for i in range(3)}
        ce_mean = ncm_mean.detached_causal_effects(flat_h, tcg)
        ce_marg = ncm_marg.detached_causal_effects(flat_h, tcg)
        assert any(
            abs(ce_mean[k] - ce_marg[k]) > 1e-6 for k in ce_mean
        )


class TestSupervisionCounts:
    """supervised_ncm_loss must record how many labelled edges each edge-type
    MLP actually trained on — zero-count types are exactly the undertrained
    MLPs whose CE collapses to ~0 (the tx→wallet RCP bottleneck)."""

    def test_counts_recorded_including_zeros(self):
        ncm = HeteroNCM(
            node_emb_dim=EMB_DIM,
            all_node_types=["transaction", "wallet"],
            all_edge_types=["transaction__to__wallet", "wallet__to__transaction"],
            node_type_emb_dim=TYPE_EMB_DIM,
            ncm_h_size=16,
            ncm_h_layers=1,
        )
        tcg = TypedCausalGraph([0, 1], {0: "transaction", 1: "wallet"})
        tcg.add_edge(0, 1, "transaction__to__wallet")
        flat_h = {0: torch.randn(EMB_DIM), 1: torch.randn(EMB_DIM)}
        ncm.supervised_ncm_loss(
            flat_h, tcg,
            node_labels=torch.tensor([0.0]),
            target_type_offset=0,
            wallet_labels=torch.tensor([1.0]),
            wallet_type_offset=1,
        )
        assert ncm.last_supervision_counts == {
            "transaction__to__wallet": 1,
            "wallet__to__transaction": 0,
        }
