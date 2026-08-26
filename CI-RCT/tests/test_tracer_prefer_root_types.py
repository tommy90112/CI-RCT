"""Tests for the type-aware tie-break in RootCauseTracer (prefer_root_types).

Reproduces the dd18 regression in miniature: at a host node the host→host
bridge edge has a marginally larger |CE| than process→host, so legacy greedy
ranking dead-ends at a host (wrong root), while prefer_root_types climbs to
the malicious process (correct root).

Graph (ids): 0=flow, 1=host_a (relay), 2=host_b (bridge dead-end), 3=process.
Edges (cause→effect): host_a→flow, host_b→host_a (bridge), process→host_a.
Backward from flow: flow←host_a←{host_b (bridge), process}.
"""
from model.root_cause_tracer import RootCauseTracer
from model.typed_causal_graph import TypedCausalGraph

_NODE_TYPES = {0: "flow_node", 1: "host_node", 2: "host_node", 3: "process_node"}

# CE keyed (parent, child). Bridge |CE| (0.0005) marginally beats
# process→host (0.0003), exactly like the dd18 CE table.
_CE = {
    (1, 0): 0.089,    # host_a → flow  (huge, like dd18)
    (2, 1): 0.0005,   # host_b → host_a  (bridge)
    (3, 1): 0.0003,   # process → host_a
}


def _bridge_graph():
    g = TypedCausalGraph(V=[0, 1, 2, 3], node_types=_NODE_TYPES)
    g.add_edge(1, 0, "host_to_flow")
    g.add_edge(2, 1, "host_to_host")     # DD-11 bridge
    g.add_edge(3, 1, "process_to_host")
    return g


def test_legacy_ranking_deadends_at_host():
    g = _bridge_graph()
    tracer = RootCauseTracer(g, max_hops=5, threshold=0.0001)
    root, _ = tracer.trace_root_cause(0, _CE)
    # Greedy |CE| picks the bridge (0.0005 > 0.0003) → ends at host_b (id 2).
    assert root == 2
    assert g.node_type[root] == "host_node"


def test_prefer_root_types_climbs_to_process():
    g = _bridge_graph()
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001, prefer_root_types={"process_node"}
    )
    root, chain = tracer.trace_root_cause(0, _CE)
    assert root == 3                       # the malicious process
    assert g.node_type[root] == "process_node"
    assert chain == [0, 1, 3]


def test_prefer_falls_back_when_no_preferred_parent():
    # host_a has only the bridge parent (no process). Must still trace, not crash.
    g = TypedCausalGraph(
        V=[0, 1, 2],
        node_types={0: "flow_node", 1: "host_node", 2: "host_node"},
    )
    g.add_edge(1, 0, "host_to_flow")
    g.add_edge(2, 1, "host_to_host")
    ce = {(1, 0): 0.089, (2, 1): 0.0005}
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001, prefer_root_types={"process_node"}
    )
    root, _ = tracer.trace_root_cause(0, ce)
    assert root == 2                       # falls back to |CE|-max bridge


def test_preferred_parent_below_threshold_is_not_chosen():
    # If the only process parent is below threshold, it must NOT be picked.
    g = _bridge_graph()
    ce = {(1, 0): 0.089, (2, 1): 0.0005, (3, 1): 0.00001}  # process below thr
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001, prefer_root_types={"process_node"}
    )
    root, _ = tracer.trace_root_cause(0, ce)
    assert root == 2                       # bridge, not the weak process
