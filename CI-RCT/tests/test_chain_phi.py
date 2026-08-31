"""Unit tests for utils.chain_phi.attach_phi_to_records."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.chain_phi import attach_phi_to_records  # noqa: E402


class _StubGraph:
    """Minimal causal graph for the additive φ path.

    compute_asymmetric_causal_shapley (additive branch) only needs parents() and
    topological_index(); node_type is used by the seed-type gate in the caller.
    """

    def __init__(self, parents_map, topo, node_type):
        self._parents = parents_map
        self._topo = topo
        self.node_type = node_type

    def parents(self, n):
        return self._parents.get(n, [])

    def topological_index(self):
        return self._topo


def _records():
    # chain [target=10, mid=11, root=12]; each node carries its global id.
    return [{
        "target_txid": "tx10",
        "depth": 2,
        "root_type": "wallet",
        "root_real_id": "addr12",
        "root_is_fraud": True,
        "is_true_positive": True,
        "nodes": [
            {"pos": 0, "global": 10, "type": "transaction", "real_id": "tx10",
             "fraud": True, "is_target": True},
            {"pos": 1, "global": 11, "type": "transaction", "real_id": "tx11",
             "fraud": False, "ce": 0.8},
            {"pos": 2, "global": 12, "type": "wallet", "real_id": "addr12",
             "fraud": True, "ce": 0.4},
        ],
    }]


def _graph():
    # 11 is the only parent of 10; 12 is the only parent of 11.
    return _StubGraph(
        parents_map={10: [11], 11: [12], 12: []},
        topo={12: 0, 11: 1, 10: 2},
        node_type={10: "transaction", 11: "transaction", 12: "wallet"},
    )


_CE = {(11, 10): 0.8, (12, 11): 0.4}


def test_additive_phi_is_ce_over_parent_count():
    out = attach_phi_to_records(
        _records(), causal_graph=_graph(), causal_effects=_CE,
    )
    nodes = out[0]["nodes"]
    assert "phi_add" not in nodes[0]          # target has no φ
    assert nodes[1]["phi_add"] == 0.8         # 11 → child 10, single parent
    assert nodes[2]["phi_add"] == 0.4         # 12 → child 11, single parent


def test_asym_unavailable_leaves_none():
    # No asym_phi_fn / readout_type supplied → asym pass is off.
    out = attach_phi_to_records(
        _records(), causal_graph=_graph(), causal_effects=_CE,
    )
    nodes = out[0]["nodes"]
    assert nodes[1]["phi_asym"] is None
    assert nodes[2]["phi_asym"] is None


def test_asym_phi_rolling_readout_and_keeps_sign():
    # In _records() every child (10, 11) is a transaction, so the rolling
    # readout resolves to the child itself — readout == intervene each hop.
    table = {10: {11: -0.5}, 11: {12: 0.9}}   # intervene -> {parent: phi_asym}
    seen = []

    def asym_phi_fn(readout, intervene):
        seen.append((readout, intervene))
        return table.get(intervene, {})

    out = attach_phi_to_records(
        _records(), causal_graph=_graph(), causal_effects=_CE,
        asym_phi_fn=asym_phi_fn, readout_type="transaction",
    )
    nodes = out[0]["nodes"]
    assert nodes[1]["phi_asym"] == -0.5       # signed value preserved
    assert nodes[2]["phi_asym"] == 0.9
    assert nodes[1]["phi_add"] == 0.8         # additive still filled
    # tx children → readout == intervene (distance 0).
    assert (10, 10) in seen and (11, 11) in seen


def test_asym_rolls_readout_to_downstream_tx_for_wallet_child():
    # chain [tx20(target), wallet21, tx22]: the hop scoring parent 22 has child
    # 21 (a wallet, no fraud head), so the readout must roll one step downstream
    # to tx20 (nodes[pos-2]) while the coalition still controls node 21's parents.
    recs = [{
        "target_txid": "tx20", "depth": 2, "root_type": "transaction",
        "root_real_id": "tx22", "root_is_fraud": True, "is_true_positive": True,
        "nodes": [
            {"pos": 0, "global": 20, "type": "transaction", "is_target": True},
            {"pos": 1, "global": 21, "type": "wallet"},
            {"pos": 2, "global": 22, "type": "transaction"},
        ],
    }]
    graph = _StubGraph(
        parents_map={20: [21], 21: [22], 22: []},
        topo={22: 0, 21: 1, 20: 2},
        node_type={20: "transaction", 21: "wallet", 22: "transaction"},
    )
    ce = {(21, 20): 0.5, (22, 21): 0.3}
    seen = []

    def asym_phi_fn(readout, intervene):
        seen.append((readout, intervene))
        return {20: {21: 0.7}, 21: {22: 0.2}}.get(intervene, {})

    out = attach_phi_to_records(
        recs, causal_graph=graph, causal_effects=ce,
        asym_phi_fn=asym_phi_fn, readout_type="transaction",
    )
    nodes = out[0]["nodes"]
    # pos=1: child=20 (tx) → readout==intervene==20.
    assert (20, 20) in seen
    # pos=2: child=21 (wallet) → readout rolls to downstream tx 20, intervene=21.
    assert (20, 21) in seen
    assert nodes[1]["phi_asym"] == 0.7
    assert nodes[2]["phi_asym"] == 0.2


def test_asym_none_when_no_head_readout_resolves():
    # A wallet-only front of the chain with no downstream head node → phi_asym
    # stays None for that hop (readout unresolved), phi_add still filled.
    recs = [{
        "target_txid": "w30", "depth": 1, "root_type": "wallet",
        "root_real_id": "w31", "root_is_fraud": True, "is_true_positive": True,
        "nodes": [
            {"pos": 0, "global": 30, "type": "wallet", "is_target": True},
            {"pos": 1, "global": 31, "type": "wallet"},
        ],
    }]
    graph = _StubGraph(
        parents_map={30: [31], 31: []},
        topo={31: 0, 30: 1},
        node_type={30: "wallet", 31: "wallet"},
    )
    out = attach_phi_to_records(
        recs, causal_graph=graph, causal_effects={(31, 30): 0.6},
        asym_phi_fn=lambda readout, intervene: {30: {31: 0.9}},
        readout_type="transaction",
    )
    nodes = out[0]["nodes"]
    assert nodes[1]["phi_asym"] is None       # no tx readout reachable
    assert nodes[1]["phi_add"] == 0.6         # additive still filled


def test_input_records_not_mutated():
    recs = _records()
    attach_phi_to_records(
        recs, causal_graph=_graph(), causal_effects=_CE,
    )
    for n in recs[0]["nodes"]:
        assert "phi_add" not in n
        assert "phi_asym" not in n
