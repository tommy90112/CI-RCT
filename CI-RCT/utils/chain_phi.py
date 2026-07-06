"""Attach per-node Causal Shapley φ to dumped chain records.

For a chain record (``utils.elliptic_identity.chain_to_record``) we attach, to
each non-target node, its per-hop LOCAL causal responsibility φ toward its
downstream child on the chain — the value ch3 calls "逐段局部因果責任":

  - ``phi_add``  : additive approximation φ = CE(node→child) / |Pa(child)|
                   (``compute_asymmetric_causal_shapley`` with no coalition
                   value). Used by L_stability / Metric D; unaffected here.
  - ``phi_asym`` : asymmetric Causal Shapley via the backbone coalition value,
                   read out at the hop's *rolling readout* — the child itself, or
                   its nearest downstream head node (see ``_resolve_readout``).
                   This replaces the retired fixed-seed readout, whose deep hops
                   collapsed to φ≈0 outside the backbone's receptive field.
                   ``None`` when no readout resolves for a hop.

φ is **signed** (a suppressor parent has negative φ) to preserve the causal
responsibility semantics — magnitudes are the |CE| tracer's concern, not this
export.

This module is pure w.r.t. its input: it returns NEW records and never mutates
the originals (or their node dicts).
"""
from typing import Callable, Dict, List, Optional, Tuple

from model.causal_shapley import compute_asymmetric_causal_shapley

# {parent_global_id: phi} for the parents of one intervene node.
PhiByParent = Dict[int, float]
# A per-hop φ fn: (readout_global, intervene_global) -> {parent_global: phi_asym}
# (or None when φ cannot be read out for that hop).
AsymPhiFn = Callable[[int, int], Optional[PhiByParent]]


def _resolve_readout(nodes: List[dict], pos: int, causal_graph, readout_type: str):
    """Rolling φ readout for the hop whose intervene node is ``nodes[pos-1]``.

    φ_asym reads out a fraud probability, which only lives on ``readout_type``
    nodes (the backbone carries a single classifier head on that type). The
    intervene node is the hop's downstream child ``nodes[pos-1]``:
      - if it already carries the head → read out there (distance 0);
      - else roll one step further downstream on the chain (``nodes[pos-2]``),
        which the tx/wallet alternation guarantees is a head node (≤1 extra hop,
        well inside the backbone's receptive-field radius).

    Returns the readout global id, or ``None`` when neither is a head node (e.g.
    a seed whose type is not the readout type) → ``phi_asym`` is left ``None``.
    """
    child = nodes[pos - 1]["global"]
    if causal_graph.node_type.get(child) == readout_type:
        return child
    if pos - 2 >= 0:
        downstream = nodes[pos - 2]["global"]
        if causal_graph.node_type.get(downstream) == readout_type:
            return downstream
    return None


def attach_phi_to_records(
    records: List[dict],
    *,
    causal_graph,
    causal_effects: Dict[Tuple[int, int], float],
    asym_phi_fn: Optional[AsymPhiFn] = None,
    readout_type: Optional[str] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """Return new records with per-node ``phi_add`` / ``phi_asym`` attached.

    For the node at position ``i >= 1`` (parent) and its downstream child
    ``nodes[i - 1]`` we record:
      - ``phi_add``  — always (additive φ = CE(parent→child) / |Pa(child)|);
      - ``phi_asym`` — the per-hop LOCAL causal responsibility: the parent's
        asymmetric Causal Shapley contribution to the fraud probability of the
        hop's *rolling readout* (the child, or its nearest downstream head node;
        see :func:`_resolve_readout`). Filled only when both ``asym_phi_fn`` and
        ``readout_type`` are supplied and a readout resolves; ``None`` otherwise.

    ``asym_phi_fn(readout_global, intervene_global)`` returns
    ``{parent_global: phi_asym}`` for the intervene node's parents, read out at
    ``readout_global`` — or ``None`` when unavailable for that hop.

    Unlike the retired fixed-seed readout (which anchored every hop to the
    detected node and forced φ≈0 beyond the backbone's receptive-field radius),
    the rolling readout keeps every hop measurable and gives each hop its own
    Shapley-efficient decomposition (Σ_parents φ = readout's with/without-parents
    fraud-probability gap).

    ``on_progress(done, total)``, when given, is called after each record so the
    caller can surface progress (the asym pass is slow and otherwise silent).
    """
    # Additive φ depends only on (parent, child) + CE, so it is shared across
    # every chain. Asymmetric φ depends on the hop's rolling readout, so its
    # cache is keyed by the intervene node and lives per chain.
    add_cache: Dict[int, PhiByParent] = {}
    asym_on = asym_phi_fn is not None and readout_type is not None

    new_records: List[dict] = []
    for rec in records:
        nodes = rec.get("nodes", [])
        if not nodes:
            new_records.append(dict(rec))
            continue

        asym_cache: Dict[int, Optional[PhiByParent]] = {}

        new_nodes: List[dict] = []
        for pos, node in enumerate(nodes):
            nn = dict(node)
            if pos >= 1:
                child = nodes[pos - 1]["global"]
                parent = node["global"]
                if child not in add_cache:
                    add_cache[child] = compute_asymmetric_causal_shapley(
                        causal_effects, causal_graph, child
                    )
                nn["phi_add"] = float(add_cache[child].get(parent, 0.0))
                if asym_on:
                    if child not in asym_cache:
                        readout = _resolve_readout(
                            nodes, pos, causal_graph, readout_type
                        )
                        asym_cache[child] = (
                            (asym_phi_fn(readout, child) or {})
                            if readout is not None else None
                        )
                    table = asym_cache[child]
                    nn["phi_asym"] = (
                        None if table is None else float(table.get(parent, 0.0))
                    )
                else:
                    nn["phi_asym"] = None
            new_nodes.append(nn)

        nr = dict(rec)
        nr["nodes"] = new_nodes
        new_records.append(nr)
        if on_progress is not None:
            on_progress(len(new_records), len(records))

    return new_records
