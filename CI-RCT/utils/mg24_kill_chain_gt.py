"""
MG24 Kill-Chain Ground Truth for Metric C (Explanation Quality).

For each fraud `flow_node` in the test split, derive the set of node IDs
that *should* appear in a correct explanation chain by walking the
DAG `device → process → host → flow` backwards from the target flow:

    fraud_flow (target)
       ← (host_sources_flow reverse)     ip_host
           ← (resolves_to_ip reverse)    audit_host       [DD-11 bridge]
               ← (process_runs_on rev.)  mal process(es)  [same incident]
                   ← (device_hosts rev.) device           [procmon only]

Cross-modality alignment uses the DD-14 incident key
``incident:<attack_type>:<core_stem>``, which is shared by the pcap-side
flow_node rows and the audit-side process_node rows of the same attack
execution. ``_normalize_incident_stem`` collapses ``audit_dos1`` ↔ ``dos1``
so both sides land in the same group.

API:
    compute_mg24_kill_chain_gt(mg24_data, type_offsets) → Dict[int, Set[int]]

    Returned dict maps each malicious flow's global node ID to the set
    of *expected* explanation node IDs (global). Empty for benign flows.
    The caller (``evaluate.py:eval_explanation_quality``) is responsible
    for further restricting to nodes that actually live inside the
    BFS-expanded ``TypedCausalGraph``.

Reference: unsw_mg24_plan.md § DD-16, § 6.5
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from utils.mg24_loader import (
    MG24Data,
    _attack_type_from_audit_host_ref,
    _normalize_incident_stem,
    _stem_for_attack_lookup,
)

# Prefix used by mg24_loader to tag malicious incidents (mirrors
# `_INCIDENT_PREFIX` in mg24_loader). Kept as a local constant rather
# than importing the private name, because the prefix is part of the
# data contract between loader and downstream consumers.
_INCIDENT_PREFIX = "incident:"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _flow_incident_key(
    source_file: str, attack_type: str, is_malicious: int,
) -> Optional[str]:
    """Build the incident group key for a single flow row, or None if benign."""
    if is_malicious != 1:
        return None
    stem = _stem_for_attack_lookup(Path(str(source_file)))
    core_id = _normalize_incident_stem(stem)
    return f"{_INCIDENT_PREFIX}{attack_type}:{core_id}"


def _process_incident_key(
    source: str, host_ref: str, is_malicious: int,
) -> Optional[str]:
    """
    Build the incident group key for a single audit-derived malicious
    process row. Returns None for benign rows or procmon-derived rows
    (procmon has no attack_type concept).
    """
    if source != "audit" or is_malicious != 1:
        return None
    attack = _attack_type_from_audit_host_ref(host_ref, 1)
    if attack is None:
        return None
    file_part = host_ref.split(":", 1)[1] if ":" in host_ref else host_ref
    stem = _stem_for_attack_lookup(Path(file_part))
    core_id = _normalize_incident_stem(stem)
    return f"{_INCIDENT_PREFIX}{attack}:{core_id}"


# ── Per-incident lookup tables ────────────────────────────────────────────────


def _build_incident_to_processes(
    processes: pd.DataFrame, process_offset: int,
) -> Dict[str, Set[int]]:
    """{incident_key: set of mal audit process global IDs}."""
    out: Dict[str, Set[int]] = defaultdict(set)
    if processes.empty:
        return out
    sources = processes["source"].astype(str).values
    host_refs = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    node_idx = processes["node_idx"].astype(int).values
    for i in range(len(processes)):
        key = _process_incident_key(sources[i], host_refs[i], is_mal[i])
        if key is not None:
            out[key].add(int(node_idx[i]) + process_offset)
    return out


def _build_incident_to_audit_hosts(
    processes: pd.DataFrame,
    hosts: pd.DataFrame,
    host_offset: int,
) -> Dict[str, Set[int]]:
    """{incident_key: set of audit_host_node global IDs}.

    The audit_host is the `audit:<source_file>` host row that the
    audit-derived process points at via its ``host_ref`` field.
    """
    out: Dict[str, Set[int]] = defaultdict(set)
    if processes.empty or hosts.empty:
        return out

    host_id_to_idx = dict(zip(
        hosts["host_id"].astype(str).values,
        hosts["node_idx"].astype(int).values,
    ))
    sources = processes["source"].astype(str).values
    host_refs = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    for i in range(len(processes)):
        key = _process_incident_key(sources[i], host_refs[i], is_mal[i])
        if key is None:
            continue
        host_ref = host_refs[i]
        local_idx = host_id_to_idx.get(host_ref)
        if local_idx is not None:
            out[key].add(int(local_idx) + host_offset)
    return out


def _build_audit_host_to_ip_hosts(
    audit_source_ips: Dict[str, Set[str]],
    hosts: pd.DataFrame,
    host_offset: int,
) -> Dict[int, Set[int]]:
    """
    {audit_host_global_id: set of ip_host_global_ids} from DD-11 bridge.

    Mirrors the edge construction in
    ``_build_host_resolves_to_ip(data)`` so the GT matches what the
    backbone actually sees in HeteroData.
    """
    out: Dict[int, Set[int]] = defaultdict(set)
    if hosts.empty or not audit_source_ips:
        return out

    host_id_to_idx = dict(zip(
        hosts["host_id"].astype(str).values,
        hosts["node_idx"].astype(int).values,
    ))
    for audit_src, ip_set in audit_source_ips.items():
        audit_host_id = f"audit:{audit_src}"
        audit_local = host_id_to_idx.get(audit_host_id)
        if audit_local is None:
            continue
        audit_global = int(audit_local) + host_offset
        for ip in ip_set:
            ip_host_id = f"ip:{ip}"
            ip_local = host_id_to_idx.get(ip_host_id)
            if ip_local is not None:
                out[audit_global].add(int(ip_local) + host_offset)
    return out


def _build_process_to_devices(
    processes: pd.DataFrame,
    devices: pd.DataFrame,
    process_offset: int,
    device_offset: int,
) -> Dict[int, Set[int]]:
    """
    {process_global_id: set of device_global_ids} for procmon-derived
    processes. Audit-derived processes have no device mapping in MG24,
    so their entries will simply be absent.
    """
    out: Dict[int, Set[int]] = defaultdict(set)
    if processes.empty or devices.empty:
        return out

    dev_id_to_idx = dict(zip(
        devices["device_id"].astype(str).values,
        devices["node_idx"].astype(int).values,
    ))
    procmon = processes[processes["source"] == "procmon"]
    if procmon.empty:
        return out
    for _, row in procmon.iterrows():
        host_id = str(row["host_ref"]).replace("host:", "")
        proc_global = int(row["node_idx"]) + process_offset
        if host_id == "central":
            for d in ("local1", "local2"):
                local = dev_id_to_idx.get(d)
                if local is not None:
                    out[proc_global].add(int(local) + device_offset)
        else:
            local = dev_id_to_idx.get(host_id)
            if local is not None:
                out[proc_global].add(int(local) + device_offset)
    return out


# ── Main GT entry point ───────────────────────────────────────────────────────


def compute_mg24_kill_chain_gt(
    mg24_data: MG24Data,
    type_offsets: Dict[str, int],
    *,
    test_mask: Optional[np.ndarray] = None,
    include_devices: bool = True,
    verbose: bool = True,
) -> Dict[int, Set[int]]:
    """
    Build ground-truth explanation chains for every malicious flow_node.

    Args:
        mg24_data:     ``load_mg24_data()`` output. Provides the dataframes
                       and the DD-11 ``audit_source_ips`` bridge map.
        type_offsets:  {node_type: global_id_offset}. Used to convert
                       per-type local indices into global IDs that match
                       what TypedCausalGraph / RootCauseTracer use.
        test_mask:     Optional boolean mask over flow_node rows. When
                       provided, GT is only built for malicious flows in
                       the test split (saves memory). When ``None``, GT
                       is built for *all* malicious flows.
        include_devices: Whether to include device_node IDs in GT chains.
                       For audit-derived attacks these are typically empty
                       (no device edge exists for audit processes).
        verbose:       Print coverage statistics.

    Returns:
        {flow_global_id: set of node_global_ids that should appear in the
        explanation chain}. Empty dict if the dataset has no labelled
        attack flows.
    """
    flows = mg24_data.flows
    processes = mg24_data.processes
    hosts = mg24_data.hosts
    devices = mg24_data.devices
    audit_source_ips = mg24_data.audit_source_ips

    if flows.empty:
        return {}

    flow_offset = type_offsets.get("flow_node", 0)
    host_offset = type_offsets.get("host_node", 0)
    process_offset = type_offsets.get("process_node", 0)
    device_offset = type_offsets.get("device_node", 0)

    # ── Per-incident lookup tables ────────────────────────────────
    incident_to_processes = _build_incident_to_processes(processes, process_offset)
    incident_to_audit_hosts = _build_incident_to_audit_hosts(
        processes, hosts, host_offset,
    )
    audit_host_to_ip_hosts = _build_audit_host_to_ip_hosts(
        audit_source_ips, hosts, host_offset,
    )
    process_to_devices = (
        _build_process_to_devices(processes, devices, process_offset, device_offset)
        if include_devices else {}
    )

    # ── Per-flow source-host lookup (host that sources the flow) ──
    # The actual edge construction in `_build_host_sources_flow` is based
    # on the flow's `src_ip` field mapping to a `host_id="ip:<ip>"`. We
    # mirror that here.
    host_id_to_idx = dict(zip(
        hosts["host_id"].astype(str).values,
        hosts["node_idx"].astype(int).values,
    )) if not hosts.empty else {}

    is_mal = flows["is_malicious"].astype(int).values
    source_files = flows.get(
        "source_file", pd.Series([""] * len(flows))
    ).fillna("unknown").astype(str).values
    attack_types = flows["attack_type"].fillna("unknown").astype(str).values
    # Column is named "Src IP" (CICFlowMeter convention), preserved by the loader.
    src_ip_col = (
        flows["Src IP"] if "Src IP" in flows.columns else pd.Series([""] * len(flows))
    )
    src_ips = src_ip_col.fillna("").astype(str).values
    node_idx = flows["node_idx"].astype(int).values

    # Filter to test malicious flows if mask provided.
    if test_mask is not None:
        if len(test_mask) != len(flows):
            raise ValueError(
                f"test_mask length {len(test_mask)} != n_flows {len(flows)}"
            )
        candidate_rows = np.flatnonzero(test_mask & (is_mal == 1))
    else:
        candidate_rows = np.flatnonzero(is_mal == 1)

    gt: Dict[int, Set[int]] = {}
    n_empty_chains = 0
    n_processes_total = 0
    n_audit_hosts_total = 0
    n_ip_hosts_total = 0
    n_devices_total = 0

    for i in candidate_rows:
        flow_global = int(node_idx[i]) + flow_offset
        key = _flow_incident_key(source_files[i], attack_types[i], is_mal[i])

        chain: Set[int] = {flow_global}

        # 1-hop: source IP host (from src_ip)
        if src_ips[i]:
            ip_host_id = f"ip:{src_ips[i]}"
            ip_local = host_id_to_idx.get(ip_host_id)
            if ip_local is not None:
                chain.add(int(ip_local) + host_offset)

        if key is not None:
            # 2-hop (cross-modal): audit hosts of the same incident
            audit_hosts = incident_to_audit_hosts.get(key, set())
            chain.update(audit_hosts)
            n_audit_hosts_total += len(audit_hosts)

            # 1-hop alt (via DD-11 bridge): IP hosts that the audit hosts resolve to
            for ah in audit_hosts:
                ip_hosts = audit_host_to_ip_hosts.get(ah, set())
                chain.update(ip_hosts)
                n_ip_hosts_total += len(ip_hosts)

            # 3-hop: malicious audit processes in the same incident
            procs = incident_to_processes.get(key, set())
            chain.update(procs)
            n_processes_total += len(procs)

            # 4-hop: device for those processes (procmon mapping; usually empty)
            if include_devices:
                for p in procs:
                    devs = process_to_devices.get(p, set())
                    chain.update(devs)
                    n_devices_total += len(devs)

        # Drop chains that only contain the target itself — explanation_recall
        # on those would just measure self-match, not meaningful.
        if len(chain) <= 1:
            n_empty_chains += 1
            continue
        gt[flow_global] = chain

    if verbose:
        scope = "test" if test_mask is not None else "all"
        print(f"[mg24_kill_chain_gt] scope={scope}, "
              f"n_mal_flows={len(candidate_rows):,}, "
              f"n_gt_built={len(gt):,}, "
              f"n_trivial_skipped={n_empty_chains:,}")
        print(f"  per-target avg chain size = "
              f"{sum(len(s) for s in gt.values()) / max(1, len(gt)):.2f}")
        print(f"  total nodes referenced: "
              f"audit_hosts={n_audit_hosts_total:,} "
              f"ip_hosts={n_ip_hosts_total:,} "
              f"processes={n_processes_total:,} "
              f"devices={n_devices_total:,}")
    return gt
