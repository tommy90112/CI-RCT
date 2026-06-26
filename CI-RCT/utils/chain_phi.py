"""Attach per-node Causal Shapley φ to dumped chain records.

For a chain record (``utils.elliptic_identity.chain_to_record``) we attach, to
each non-target node, its causal-responsibility φ toward its downstream child on
the chain — the value ch3 calls "賦予因果責任 / 凸顯上游源頭":

  - ``phi_add``  : legacy additive approximation φ = CE(node→child) / |Pa(child)|
                   (``compute_asymmetric_causal_shapley`` with no coalition value).
  - ``phi_asym`` : true asymmetric Causal Shapley via the backbone coalition
                   value (the ch3 attribution).  Computed by a caller-supplied
                   per-seed score function; ``None`` when unavailable (e.g. a
                   joint seed whose type is not the φ readout type).

φ is **signed** (a suppressor parent has negative φ) to preserve the causal
responsibility semantics — magnitudes are the |CE| tracer's concern, not this
export.

This module is pure w.r.t. its input: it returns NEW records and never mutates
the originals (or their node dicts).
"""
from typing import Callable, Dict, List, Optional, Tuple

from model.causal_shapley import compute_asymmetric_causal_shapley

# {parent_global_id: phi} for the parents of one child node.
PhiByParent = Dict[int, float]
# A per-seed function: child_global_id -> {parent_global_id: phi_asym}.
AsymScoreFn = Callable[[int], PhiByParent]


def attach_phi_to_records(
    records: List[dict],
    *,
    causal_graph,
    causal_effects: Dict[Tuple[int, int], float],
    asym_score_fn_for_seed: Callable[[int], Optional[AsymScoreFn]],
) -> List[dict]:
    """Return new records with per-node ``phi_add`` / ``phi_asym`` attached.

    Each chain's seed is ``nodes[0]["global"]``.  For node at position ``i >= 1``
    (parent) and its downstream child ``nodes[i - 1]["global"]`` we record:
      - ``phi_add``  — always (additive, depends only on the child + CE);
      - ``phi_asym`` — when ``asym_score_fn_for_seed(seed)`` is not None.

    ``asym_score_fn_for_seed(seed_global)`` returns either a callable
    ``child_global -> {parent_global: phi_asym}`` or ``None`` (true φ
    unavailable for that seed; ``phi_asym`` is then left as ``None``).
    """
    # Additive φ depends only on (parent, child) + CE, so it is shared across
    # every chain.  Asymmetric φ depends on the seed (the coalition reads out the
    # seed's fraud probability), so its cache must live per chain/seed.
    add_cache: Dict[int, PhiByParent] = {}

    new_records: List[dict] = []
    for rec in records:
        nodes = rec.get("nodes", [])
        if not nodes:
            new_records.append(dict(rec))
            continue

        seed = nodes[0]["global"]
        asym_fn = asym_score_fn_for_seed(seed)
        asym_cache: Dict[int, PhiByParent] = {}

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
                if asym_fn is not None:
                    if child not in asym_cache:
                        asym_cache[child] = asym_fn(child) or {}
                    nn["phi_asym"] = float(asym_cache[child].get(parent, 0.0))
                else:
                    nn["phi_asym"] = None
            new_nodes.append(nn)

        nr = dict(rec)
        nr["nodes"] = new_nodes
        new_records.append(nr)

    return new_records
