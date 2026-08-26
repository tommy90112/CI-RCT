"""Tests for the LOOKAHEAD tie-break in RootCauseTracer (prefer_reachable_depth).

Reproduces the dd18 RCP collapse that prefer_root_types alone could NOT fix.

Unlike test_tracer_prefer_root_types.py — where the malicious process is a
*direct* upstream of the relay host, so the immediate-type tie-break sees it —
here the process sits ONE HOP PAST a bridge host. At the hub host the entire
upstream set is host_node (10 dead-end bridges + 2 live bridges), so:

  - legacy / prefer_root_types-only: no process in the immediate upstream, so
    the greedy |CE|-max picks the dead-end bridge with the largest single-hop
    |CE| and stops at a host (RCP miss) — exactly the dd18 diagnostic finding.
  - prefer_reachable_depth >= 2: among the threshold-passing host candidates,
    prefers the ones from which a process is still REACHABLE within d hops,
    so it climbs through a live bridge to the process (RCP hit).

Graph (ids):
  0   = flow
  1   = hub host (flow's only parent; all its parents are host bridges)
  2..11 = 10 dead-end bridge hosts (0 parents) — largest single-hop |CE|
  12, 13 = 2 live bridge hosts → each has a process parent
  14, 15 = malicious processes (true roots)

Backward from flow: flow←hub←{dead_2..11 (big |CE|), live_12/13 (smaller |CE|)}
                    live_12←process_14 ; live_13←process_15
"""
from model.root_cause_tracer import RootCauseTracer
from model.typed_causal_graph import TypedCausalGraph

_DEAD = list(range(2, 12))      # 10 dead-end bridge hosts
_LIVE = [12, 13]                # 2 live bridge hosts
_PROC = {12: 14, 13: 15}        # live host → its process parent

_NODE_TYPES = {0: "flow_node", 1: "host_node"}
for _h in _DEAD + _LIVE:
    _NODE_TYPES[_h] = "host_node"
for _p in _PROC.values():
    _NODE_TYPES[_p] = "process_node"


def _dd18_graph():
    g = TypedCausalGraph(V=list(_NODE_TYPES.keys()), node_types=_NODE_TYPES)
    g.add_edge(1, 0, "host_node__to__flow_node")            # hub → flow
    for h in _DEAD:
        g.add_edge(h, 1, "host_node__to__host_node")        # dead bridge → hub
    for h in _LIVE:
        g.add_edge(h, 1, "host_node__to__host_node")        # live bridge → hub
    for h, p in _PROC.items():
        g.add_edge(p, h, "process_node__to__host_node")     # process → live bridge
    return g


def _ce():
    """Dead-end bridges have the LARGEST single-hop |CE| (like dd18's 0.00067),
    live bridges slightly smaller — so greedy |CE|-max is lured into a dead end."""
    ce = {(1, 0): 0.089}
    for i, h in enumerate(_DEAD):
        ce[(h, 1)] = 0.00060 + i * 1e-6     # dead bridges: ~0.0006, largest
    for h in _LIVE:
        ce[(h, 1)] = 0.00040                # live bridges: smaller
    for h, p in _PROC.items():
        ce[(p, h)] = 0.00030                # process → live host
    return ce


def test_legacy_deadends_at_bridge_host():
    g, ce = _dd18_graph(), _ce()
    tracer = RootCauseTracer(g, max_hops=5, threshold=0.0001)
    root, _ = tracer.trace_root_cause(0, ce)
    assert g.node_type[root] == "host_node"     # stuck at a dead-end bridge
    assert root in _DEAD


def test_prefer_root_types_alone_still_deadends():
    # prefer_root_types can't help: the hub's whole upstream is host_node.
    g, ce = _dd18_graph(), _ce()
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001, prefer_root_types={"process_node"}
    )
    root, _ = tracer.trace_root_cause(0, ce)
    assert g.node_type[root] == "host_node"     # still a host — the dd18 bug
    assert root in _DEAD


def test_lookahead_climbs_through_live_bridge_to_process():
    g, ce = _dd18_graph(), _ce()
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001,
        prefer_root_types={"process_node"},
        prefer_reachable_depth=2,
    )
    root, chain = tracer.trace_root_cause(0, ce)
    assert g.node_type[root] == "process_node"  # RCP hit
    assert root in _PROC.values()
    # chain: flow → hub → live bridge → process
    assert chain[0] == 0 and chain[1] == 1
    assert g.node_type[chain[2]] == "host_node" and chain[2] in _LIVE
    assert chain[3] == _PROC[chain[2]]


def test_lookahead_depth1_enough_when_process_is_direct_parent():
    # In the dd18 structure the live bridge's process is its DIRECT parent,
    # so reachability is satisfied at depth 1 (process_14 is 1 hop above
    # live host_12). This mirrors the real MG24 graph (L1.7: live-bridge
    # ancestor contains process within 1 hop).
    g, ce = _dd18_graph(), _ce()
    tracer = RootCauseTracer(
        g, max_hops=5, threshold=0.0001,
        prefer_root_types={"process_node"},
        prefer_reachable_depth=1,
    )
    root, _ = tracer.trace_root_cause(0, ce)
    assert g.node_type[root] == "process_node"  # depth 1 already rescues RCP


def test_lookahead_depth_too_shallow_does_not_help():
    # Push the process TWO hops above the live bridge (insert an extra relay
    # host). Then depth=1 cannot see it → falls back to the |CE|-max dead end;
    # depth=2 rescues. Validates the depth budget is actually enforced.
    g = TypedCausalGraph(
        V=[0, 1, 2, 3, 20, 21],
        node_types={
            0: "flow_node", 1: "host_node",
            2: "host_node",            # dead-end bridge (0 parents)
            3: "host_node",            # live bridge
            20: "host_node",           # extra relay between live bridge & process
            21: "process_node",        # true root, 2 hops above the live bridge
        },
    )
    g.add_edge(1, 0, "host_node__to__flow_node")
    g.add_edge(2, 1, "host_node__to__host_node")   # dead bridge (bigger |CE|)
    g.add_edge(3, 1, "host_node__to__host_node")   # live bridge (smaller |CE|)
    g.add_edge(20, 3, "host_node__to__host_node")  # relay above live bridge
    g.add_edge(21, 20, "process_node__to__host_node")  # process 2 hops up
    ce = {(1, 0): 0.089, (2, 1): 0.0006, (3, 1): 0.0004,
          (20, 3): 0.0003, (21, 20): 0.0002}

    shallow = RootCauseTracer(
        g, max_hops=5, threshold=0.0001,
        prefer_root_types={"process_node"}, prefer_reachable_depth=1,
    )
    assert g.node_type[shallow.trace_root_cause(0, ce)[0]] == "host_node"  # too shallow

    deep = RootCauseTracer(
        g, max_hops=5, threshold=0.0001,
        prefer_root_types={"process_node"}, prefer_reachable_depth=2,
    )
    assert g.node_type[deep.trace_root_cause(0, ce)[0]] == "process_node"  # depth 2 reaches


def test_lookahead_disabled_by_default_is_legacy():
    # prefer_reachable_depth=0 (default) must be byte-for-byte legacy.
    g, ce = _dd18_graph(), _ce()
    legacy = RootCauseTracer(g, max_hops=5, threshold=0.0001)
    look0 = RootCauseTracer(g, max_hops=5, threshold=0.0001,
                            prefer_reachable_depth=0)
    assert legacy.trace_root_cause(0, ce)[0] == look0.trace_root_cause(0, ce)[0]


def test_negative_depth_rejected():
    g = _dd18_graph()
    try:
        RootCauseTracer(g, max_hops=5, threshold=0.0001, prefer_reachable_depth=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative prefer_reachable_depth")
