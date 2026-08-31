"""L3 causal feature attribution (model.feature_attribution).

Uses a tiny stub backbone + 2-type graph so the do-intervention path runs on CPU
without the real model/dataset. The stub's target-transaction fraud logit depends
on its own feature 0 AND on connected wallets' feature 0, so intervening on those
features moves the readout (CFE ≠ 0).
"""
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from model.feature_attribution import compute_causal_feature_attribution

W2T = ("wallet", "to", "transaction")


class StubModel(torch.nn.Module):
    """Target-tx fraud logit = tx[:,0] + 2·(sum of connected wallet[:,0])."""

    def forward(self, graph):
        tx = graph["transaction"].x
        wal = graph["wallet"].x
        ei = graph[W2T].edge_index
        agg = torch.zeros(tx.size(0), wal.size(1))
        if ei.numel() > 0:
            agg = agg.index_add(0, ei[1], wal[ei[0]])
        score = tx[:, 0] + 2.0 * agg[:, 0]
        logits = torch.stack([torch.zeros_like(score), score], dim=1)
        return logits, None


def _build():
    # 2 transactions (global 0,1), 3 wallets (global 2,3,4).
    data = HeteroData()
    data["transaction"].x = torch.tensor([[1.0, 0, 0, 0], [0.5, 0, 0, 0]])
    data["wallet"].x = torch.tensor([[2.0, 1, 0], [1.0, 0, 0], [3.0, 0, 0]])
    # wallet0,1 → tx0 ;  wallet2 → tx1
    data[W2T].edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    type_offsets = {"transaction": 0, "wallet": 2}
    node_type = {0: "transaction", 1: "transaction", 2: "wallet", 3: "wallet", 4: "wallet"}
    cg = SimpleNamespace(node_type=node_type)
    names = {
        "transaction": ["Local_feature_1", "Local_feature_2", "tx_stat_a", "tx_stat_b"],
        "wallet": ["btc_total", "num_txs", "fees"],
    }
    return data, type_offsets, cg, names


def _attr(pivot_global, target_global=0):
    data, type_offsets, cg, names = _build()
    return compute_causal_feature_attribution(
        model=StubModel(), data=data, causal_graph=cg,
        target_node=target_global, pivot_node=pivot_global,
        type_offsets=type_offsets, target_node_type="transaction",
        feature_names=names, use_subgraph=True, num_layers=1, top_k=12,
    )


def test_wallet_pivot_gets_causal_attribution():
    rec = _attr(pivot_global=2)  # wallet0 → tx0
    assert rec["node_type"] == "wallet"
    assert rec["method"] == "causal_do"
    assert rec["anonymous"] is False
    assert rec["features"], "expected non-empty attribution"
    top = rec["features"][0]
    # wallet feature 0 (btc_total) drives the readout, so it must rank first.
    assert top["name"] == "btc_total"
    assert abs(top["value"]) > 1e-4
    assert "saliency" in top


def test_out_of_receptive_field_returns_empty():
    # wallet2 (global 4) connects to tx1, NOT the target tx0 → unreachable.
    rec = _attr(pivot_global=4, target_global=0)
    assert rec["features"] == []


def test_wrong_type_target_returns_empty_not_crash():
    """Joint dual-seed regression: a WALLET-seeded chain hands its wallet
    global id as the target while target_node_type stays 'transaction'.
    Before the type guard this became a bogus transaction-local index and
    crashed the receptive-field build (IndexError 250865 vs 203769)."""
    rec = _attr(pivot_global=2, target_global=3)  # global 3 is a wallet
    assert rec["features"] == []


def test_tx_pivot_anonymity_flag():
    # tx pivot's names are a mix (2 Local_* + 2 named) → not ALL anonymous.
    rec = _attr(pivot_global=1, target_global=1)  # tx1 explains itself
    assert rec["node_type"] == "transaction"
    assert rec["anonymous"] is False


def test_feature_values_are_signed_and_named():
    rec = _attr(pivot_global=2)
    for f in rec["features"]:
        assert isinstance(f["name"], str)
        assert isinstance(f["value"], float)
