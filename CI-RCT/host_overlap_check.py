"""
P0.5 diagnostic: cross-split host overlap analysis.

Hypothesis H2 (suspicious): CI-RCT's 60 pp gain over LogReg/RF on hybrid
test set might be driven by host_node embeddings memorising attacker IPs
seen in train, not by genuine graph-based generalisation.

This script measures, for the SAME hybrid split that produced the CI-RCT
training run, how much each split's set of "attack-touching" host_nodes
overlaps with the train split's. If overlap is high (e.g. >50%), most
attacker IPs in test were already seen in train — host_node embeddings
would carry train labels into test inference. If overlap is low (<20%),
CI-RCT's gain is legitimate cross-session generalisation.

Definitions
───────────
  "attack host"   any host_node connected by an edge (in either direction)
                  to a malicious flow_node in that split.
  "benign host"   any host_node connected to a benign flow_node in
                  that split.

Reported metrics
────────────────
  |split ∩ train| / |split|       fraction of split's attack hosts seen
                                  in train. (the "memorisation rate")
  Jaccard                         |A ∩ B| / |A ∪ B|  (symmetric overlap)

Verdict thresholds
──────────────────
  attack-host  memorisation rate
    > 70%   ⚠  high risk of leakage; CI-RCT's gain may be spurious
    30-70%  ?  ambiguous; recommend host-masking ablation
    < 30%   ✓  genuine cross-session generalisation likely

Reference: unsw_mg24_plan.md DD-8 + P0 baseline_compare.py findings
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Set, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_root", default="data")
    p.add_argument("--split_mode", default="hybrid",
                   choices=("row", "by_file", "hybrid"))
    p.add_argument("--mg24_subsample_ddos", type=float, default=0.1)
    p.add_argument("--mg24_min_host_flows", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _hosts_touching_flows(
    flow_idxs: np.ndarray,
    flow_to_host: np.ndarray,
    host_to_flow: np.ndarray,
) -> Set[int]:
    """
    Return the set of host_node indices that connect to any flow in
    `flow_idxs`, via either direction of the bipartite edges.

    Args:
        flow_idxs:    1D array of flow_node indices to look up.
        flow_to_host: edges[("flow_node","targets","host_node")] of shape (2, E)
                      row 0 = flow_idx, row 1 = host_idx
        host_to_flow: edges[("host_node","sources","flow_node")] of shape (2, E)
                      row 0 = host_idx, row 1 = flow_idx
    """
    flow_set = set(int(x) for x in flow_idxs)
    hosts: Set[int] = set()
    if flow_to_host.shape[1] > 0:
        f_arr = flow_to_host[0]
        h_arr = flow_to_host[1]
        for f, h in zip(f_arr, h_arr):
            if int(f) in flow_set:
                hosts.add(int(h))
    if host_to_flow.shape[1] > 0:
        h_arr = host_to_flow[0]
        f_arr = host_to_flow[1]
        for h, f in zip(h_arr, f_arr):
            if int(f) in flow_set:
                hosts.add(int(h))
    return hosts


def _format_overlap(name: str, split_set: Set[int], train_set: Set[int]) -> str:
    """One-line summary of split-vs-train overlap statistics."""
    if not split_set:
        return f"  {name:5}  (empty)"
    inter = split_set & train_set
    union = split_set | train_set
    mem = len(inter) / len(split_set)
    jac = len(inter) / len(union) if union else 0.0
    verdict = "⚠ HIGH" if mem > 0.70 else "? MID " if mem > 0.30 else "✓ LOW "
    return (
        f"  {name:5}  |split|={len(split_set):>5}  "
        f"|∩train|={len(inter):>5}  "
        f"mem={mem*100:5.1f}%  Jac={jac*100:5.1f}%  {verdict}"
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print(f"Loading MG24 (split_mode={args.split_mode}, "
          f"seed={args.seed})...")
    from utils.mg24_loader import (
        EDGE_FLOW_TARGETS_HOST,
        EDGE_HOST_SOURCES_FLOW,
        build_edges,
        load_mg24_data,
        to_pyg_hetero_data,
    )

    mg24 = load_mg24_data(
        root=os.path.join(args.data_root, "unsw_mg24"),
        subsample_ddos=args.mg24_subsample_ddos,
        seed=args.seed,
        min_host_flows=args.mg24_min_host_flows,
        verbose=True,
    )
    edges = build_edges(mg24)
    hd = to_pyg_hetero_data(
        mg24, edges, seed=args.seed, split_mode=args.split_mode,
    )

    y = hd["flow_node"].y.numpy()
    train_mask = hd["flow_node"].train_mask.numpy()
    val_mask = hd["flow_node"].val_mask.numpy()
    test_mask = hd["flow_node"].test_mask.numpy()

    flow_to_host = edges.get(EDGE_FLOW_TARGETS_HOST, np.zeros((2, 0), dtype=np.int64))
    host_to_flow = edges.get(EDGE_HOST_SOURCES_FLOW, np.zeros((2, 0), dtype=np.int64))

    splits: Dict[str, np.ndarray] = {
        "train": train_mask,
        "val":   val_mask,
        "test":  test_mask,
    }

    # ── Attack host overlap ──────────────────────────────────────────────────
    print("\n── Attack-host overlap (hosts touching MALICIOUS flows) ──")
    attack_sets: Dict[str, Set[int]] = {}
    for name, mask in splits.items():
        mal_flows = np.flatnonzero(mask & (y == 1))
        attack_sets[name] = _hosts_touching_flows(mal_flows, flow_to_host, host_to_flow)
    for name in ("train", "val", "test"):
        if name == "train":
            print(f"  {name:5}  |split|={len(attack_sets[name]):>5}  (baseline)")
        else:
            print(_format_overlap(name, attack_sets[name], attack_sets["train"]))

    # ── Benign host overlap (control) ────────────────────────────────────────
    print("\n── Benign-host overlap (hosts touching BENIGN flows; control) ──")
    benign_sets: Dict[str, Set[int]] = {}
    for name, mask in splits.items():
        ben_flows = np.flatnonzero(mask & (y == 0))
        benign_sets[name] = _hosts_touching_flows(ben_flows, flow_to_host, host_to_flow)
    for name in ("train", "val", "test"):
        if name == "train":
            print(f"  {name:5}  |split|={len(benign_sets[name]):>5}  (baseline)")
        else:
            print(_format_overlap(name, benign_sets[name], benign_sets["train"]))

    # ── Mixed-role hosts (host is both attack-touching AND benign-touching) ──
    print("\n── Dual-role hosts (touch both attack & benign flows) ──")
    for name in ("train", "val", "test"):
        dual = attack_sets[name] & benign_sets[name]
        total_attack = len(attack_sets[name])
        print(
            f"  {name:5}  attack hosts = {total_attack:>5}  "
            f"of which {len(dual):>5} also see benign  "
            f"({(len(dual)/total_attack*100 if total_attack else 0):.1f}%)"
        )

    # ── Verdict ──────────────────────────────────────────────────────────────
    test_mem = (
        len(attack_sets["test"] & attack_sets["train"]) / len(attack_sets["test"])
        if attack_sets["test"] else 0.0
    )
    val_mem = (
        len(attack_sets["val"] & attack_sets["train"]) / len(attack_sets["val"])
        if attack_sets["val"] else 0.0
    )
    print(
        f"\n── Verdict ──\n"
        f"  Attack-host memorisation: val={val_mem*100:.1f}%  test={test_mem*100:.1f}%\n"
    )
    if test_mem > 0.70:
        print(
            "  ⚠ H2 LIKELY TRUE: most test attack hosts were also in train.\n"
            "    CI-RCT's host_node embedding inevitably carries train labels\n"
            "    into test → 60 pp gap over baselines may be host memorisation.\n"
            "    Recommended fix: add --mask_host_label_history flag to train.py\n"
            "    so host_node features do not directly encode prior attack history."
        )
    elif test_mem > 0.30:
        print(
            "  ? AMBIGUOUS: substantial but partial host overlap.\n"
            "    Run an ablation (zero host_node features at test time) to\n"
            "    quantify how much of CI-RCT's gain is host-memorisation."
        )
    else:
        print(
            "  ✓ H1 LIKELY TRUE: low attack-host overlap.\n"
            "    CI-RCT's gain over baselines reflects genuine graph-based\n"
            "    cross-session generalisation — paper §4.2 narrative stands."
        )


if __name__ == "__main__":
    main()
