"""
UNSW-MG24 Multimodal Heterogeneous Graph Loader.

Stage 1: load 4 modalities (network flows, Linux audit logs, Windows Procmon
CSVs, power measurements) and construct node tables for the 5 node types
defined in unsw_mg24_plan.md § 4.1.

Stage 2 (separate commit): build edges + emit PyG HeteroData.

Design decisions referenced (see unsw_mg24_plan.md § 3.5):
    DD-1  Full-graph first; subsample ddos only on OOM fallback.
    DD-2  Modality-native timestamps preserved; cross-modal alignment in Stage 2.
    DD-3  Multi-task target heads; this loader produces all 5 node tables.
    DD-5  Audit type filter (handled by audit_parser).
    DD-6  File-level binary labels (assigned here via filename → attack_type).

Usage
─────
    from utils.mg24_loader import load_mg24_data

    data = load_mg24_data(root="data/unsw_mg24", subsample_ddos=1.0)
    print(data.flows.shape, data.hosts.shape, data.processes.shape)

Reference: unsw_mg24_plan.md § 6.4
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.audit_parser import parse_audit_dir, parse_audit_log
from utils.procmon_parser import parse_procmon_csv


def _safe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    pd.concat wrapper that silences a noisy pandas FutureWarning.

    When concatenating audit / Procmon log DataFrames, some files have all-NA
    columns (e.g. a log with no PATH records → `path` column is all NaN),
    which triggers a FutureWarning about pandas 3.0 changing how all-NA
    columns participate in dtype inference. The current behaviour produces
    the correct schema for our downstream code (which always re-coerces
    numerics), so we suppress just this specific warning.
    """
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation",
            category=FutureWarning,
        )
        return pd.concat(frames, ignore_index=True)


# ── Constants ─────────────────────────────────────────────────────────────────

# Department folder names → benign traffic CSV filename
_DEPT_TRAFFIC_FILES: List[Tuple[str, str]] = [
    # (subdir under "Benign network traffic", department label)
    ("admin department network traffic", "admin"),
    ("microgrid department network traffic", "microgrid"),
    ("research department network traffic", "research"),
    ("teaching department network traffic", "teaching"),
]

# Procmon file → (host_id, label_int) mapping. Includes "macilious" typo.
_PROCMON_FILES: List[Tuple[str, str, int]] = [
    ("Benign system call traces/Microgrid department/central_normal_system_calls.CSV", "central", 0),
    ("Benign system call traces/Microgrid department/local1_normal_system_calls.CSV", "local1", 0),
    ("Benign system call traces/Microgrid department/local2_normal_system_calls.CSV", "local2", 0),
    ("Malicious system call traces/central_malicious.CSV", "central", 1),
    ("Malicious system call traces/local1_macilious.CSV", "local1", 1),
    ("Malicious system call traces/local2_macilious.CSV", "local2", 1),
]

# Power CSV file → (device_id, label_int, schema_kind)
# Note: typos in original dataset ("macilious", "macilous").
_POWER_FILES: List[Tuple[str, str, int, str]] = [
    ("Power measurement/local1_normal.csv", "local1", 0, "mechanical"),
    ("Power measurement/local1_malicious.csv", "local1", 1, "mechanical"),
    ("Power measurement/local2_normal.csv", "local2", 0, "electrical"),
    ("Power measurement/local2_malicious.csv", "local2", 1, "electrical"),
]

# Filename stem (lowercased, no extension) → attack_type label.
# Covers both .pcap and .log naming variants. Unmapped malicious files
# default to attack_type="other_malicious".
_ATTACK_TYPE_MAP: Dict[str, str] = {
    # Network attacks
    "backdoor1": "backdoor",
    "backdoor2": "backdoor",
    "ddos1": "ddos",
    "dos1": "dos",
    "mitm1": "mitm",
    "scan1_nmap": "recon",
    "scan2_nikto": "recon",
    "shellshock": "shellshock",
    "sql_injection": "sql_injection",
    "samba_permission": "samba",
    "hydra_password": "credential",
    "ftp_password": "credential",
    "ransomware": "ransomware",
    # Pivoting kill chain network captures (pcapng or pcap)
    "ms17 traffic": "exploit",
    "ms17_010 traffic": "exploit",
    "ms17_010 traffic2": "exploit",
    "ms17_010_command": "exploit",
    "meterpreter set": "foothold",
    "nmap scan inside": "recon",
    "nmap scan vuln inside": "recon",
    "pivot attack scanning": "pivot",
    "pivot attack dos": "pivot",
    # Audit logs (Linux)
    "audit_dos1": "dos",
    "audit_dos2": "dos",
    "ddos1.log": "ddos",  # caught via stem == 'ddos1'; kept for clarity
    "audit_backdoor": "backdoor",
    "audit_backdoor2": "backdoor",
    "audit_backdoor_3": "backdoor",
    "audit_scan_nmap1": "recon",
    "audit_scan_nikto1": "recon",
    "audit_scan_nikto2": "recon",
    "audit_mitm1": "mitm",
    "audit_mitm2": "mitm",
    "audit_sql1": "sql_injection",
    "audit_sql2": "sql_injection",
    "audit_sql3": "sql_injection",
    "audit_password": "credential",
    "audit_hydra_passowrd_1": "credential",  # typo in dataset
    "audit_ftp_password": "credential",
    "audit_ransomware_1": "ransomware",
    "ransomware2": "ransomware",
    "audit_samaba_permission": "samba",  # typo in dataset
    "audit_reverse_shell": "reverse_shell",
    "shellshock1": "shellshock",
    "shellshock2": "shellshock",
    "shellshock3": "shellshock",
    "mimicry1": "mimicry",  # ★ GAN ground-truth
    "mimicry2": "mimicry",
}

# Common columns we expect from CICFlowMeter pcap_Flow.csv outputs.
_FLOW_KEY_COLS: List[str] = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
    "Protocol", "Timestamp",
]


# ── Data container ────────────────────────────────────────────────────────────


@dataclass
class MG24Data:
    """
    Container for loaded MG24 data.

    Modality DataFrames (one row per raw event) and node tables (one row per
    unique node) are kept as separate attributes; downstream Stage 2 code
    builds edge_index tensors against these tables.
    """
    flows: pd.DataFrame
    audit: pd.DataFrame
    procmon: pd.DataFrame
    power: pd.DataFrame

    hosts: pd.DataFrame
    processes: pd.DataFrame
    flow_nodes: pd.DataFrame
    devices: pd.DataFrame
    measurements: pd.DataFrame

    def summary(self) -> str:
        """Return a one-line summary of node counts per type."""
        return (
            f"hosts={len(self.hosts)} processes={len(self.processes)} "
            f"flows={len(self.flow_nodes)} devices={len(self.devices)} "
            f"measurements={len(self.measurements)}"
        )


# ── Top-level entry point ─────────────────────────────────────────────────────


def load_mg24_data(
    root: Union[str, Path] = "data/unsw_mg24",
    subsample_ddos: float = 1.0,
    seed: int = 42,
    prune_external_hosts: bool = True,
    min_host_flows: int = 5,
    verbose: bool = True,
) -> MG24Data:
    """
    Load all 4 modalities and construct the 5 node tables.

    Args:
        root:                  Path to the unsw_mg24 data directory.
        subsample_ddos:        Fraction of ddos1 flows to retain (DD-1). 1.0 = full,
                               0.1 = 10% (fallback for OOM).
        seed:                  Random seed used by the ddos subsampler.
        prune_external_hosts:  If True, drop external IPs that have no malicious
                               flows AND fewer than `min_host_flows` total flows
                               (Stage 2a). Internal subnets and procmon hostnames
                               are always retained.
        min_host_flows:        Minimum flow count for an external host to survive
                               pruning. Has no effect if prune_external_hosts=False.
        verbose:               Print progress messages.

    Returns:
        MG24Data with modality DataFrames and node tables populated.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MG24 root directory not found: {root}")

    if verbose:
        print(f"=== Loading MG24 from {root} ===")

    # ── 4 modality DataFrames ─────────────────────────────────────
    flows = _load_flows(root, subsample_ddos=subsample_ddos, seed=seed, verbose=verbose)
    audit = _load_audit(root, verbose=verbose)
    procmon = _load_procmon(root, verbose=verbose)
    power = _load_power(root, verbose=verbose)

    # ── 5 node tables ─────────────────────────────────────────────
    hosts = _build_host_table(flows, audit, procmon)
    hosts = _enrich_host_table(hosts, flows, procmon)
    if prune_external_hosts:
        hosts = _prune_hosts(hosts, min_flows=min_host_flows, verbose=verbose)
    processes = _build_process_table(audit, procmon)
    flow_nodes = _build_flow_node_table(flows)
    devices = _build_device_table()
    measurements = _build_measurement_table(power, devices)

    if verbose:
        data = MG24Data(
            flows=flows, audit=audit, procmon=procmon, power=power,
            hosts=hosts, processes=processes, flow_nodes=flow_nodes,
            devices=devices, measurements=measurements,
        )
        print(f"=== Done. Node counts: {data.summary()} ===")
        return data

    return MG24Data(
        flows=flows, audit=audit, procmon=procmon, power=power,
        hosts=hosts, processes=processes, flow_nodes=flow_nodes,
        devices=devices, measurements=measurements,
    )


# ── Modality loaders ──────────────────────────────────────────────────────────


def _load_flows(
    root: Path,
    subsample_ddos: float,
    seed: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Load benign + malicious network flow CSVs into a single DataFrame.

    Adds columns:
        attack_type     str   ('benign' for benign dept files, mapped name otherwise)
        is_malicious    int   (0 or 1, file-level binary label, DD-6)
        source_file     str   (basename of source CSV)
        dept            str   (department for benign rows; '' for malicious)

    Applies DDoS subsampling per DD-1.
    """
    if verbose:
        print("[flows] loading network flows...")

    benign_frames: List[pd.DataFrame] = []
    benign_dir = root / "Benign network traffic"
    if benign_dir.is_dir():
        for sub, dept in _DEPT_TRAFFIC_FILES:
            for csv_path in (benign_dir / sub).glob("*.csv"):
                df = _read_flow_csv(csv_path)
                if df.empty:
                    continue
                df["source_file"] = csv_path.name
                df["dept"] = dept
                df["attack_type"] = "benign"
                df["is_malicious"] = 0
                benign_frames.append(df)
                if verbose:
                    print(f"  benign  {dept:<10} {csv_path.name}: {len(df):,} rows")

    malicious_frames: List[pd.DataFrame] = []
    mal_dir = root / "Malicious network traffic"
    if mal_dir.is_dir():
        for csv_path in sorted(mal_dir.glob("*.csv")):
            df = _read_flow_csv(csv_path)
            if df.empty:
                continue
            stem = _stem_for_attack_lookup(csv_path)
            attack_type = _ATTACK_TYPE_MAP.get(stem, "other_malicious")
            df["source_file"] = csv_path.name
            df["dept"] = ""
            df["attack_type"] = attack_type
            df["is_malicious"] = 1
            malicious_frames.append(df)
            if verbose:
                print(f"  attack  {attack_type:<14} {csv_path.name}: {len(df):,} rows")

    all_frames = benign_frames + malicious_frames
    if not all_frames:
        return pd.DataFrame()

    flows = _safe_concat(all_frames)

    # DD-1: ddos subsample applied here, after concat, per attack_type.
    if subsample_ddos < 1.0:
        ddos_mask = flows["attack_type"] == "ddos"
        n_ddos = int(ddos_mask.sum())
        if n_ddos > 0:
            keep = max(1, int(n_ddos * subsample_ddos))
            ddos_idx = flows.index[ddos_mask]
            rng = np.random.default_rng(seed)
            sampled_idx = rng.choice(ddos_idx, size=keep, replace=False)
            non_ddos_idx = flows.index[~ddos_mask]
            flows = flows.loc[non_ddos_idx.union(pd.Index(sampled_idx))].reset_index(drop=True)
            if verbose:
                print(
                    f"  ddos subsample: {n_ddos:,} → {keep:,} "
                    f"({subsample_ddos:.0%}); seed={seed}"
                )

    if verbose:
        print(f"[flows] total: {len(flows):,} rows ({flows['is_malicious'].sum():,} malicious)")

    return flows


def _load_audit(root: Path, verbose: bool) -> pd.DataFrame:
    """
    Load Linux audit logs from benign dept dirs + malicious dir.

    Adds columns:
        attack_type   str
        is_malicious  int
        dept          str   ('' for malicious; benign dept name otherwise)
    """
    if verbose:
        print("[audit] loading Linux audit logs...")

    frames: List[pd.DataFrame] = []

    benign_root = root / "Benign system call traces"
    if benign_root.is_dir():
        for dept_dir in sorted(benign_root.iterdir()):
            if not dept_dir.is_dir():
                continue
            # Skip the Microgrid department subdir — that's Procmon CSVs, not audit logs.
            if dept_dir.name == "Microgrid department":
                continue
            df = parse_audit_dir(dept_dir)
            if df.empty:
                continue
            df["dept"] = dept_dir.name
            df["attack_type"] = "benign"
            df["is_malicious"] = 0
            frames.append(df)
            if verbose:
                print(f"  benign  {dept_dir.name}: {len(df):,} events")

    mal_dir = root / "Malicious system call traces"
    if mal_dir.is_dir():
        for log_path in sorted(mal_dir.glob("*.log")):
            df = parse_audit_log(log_path)
            if df.empty:
                continue
            stem = _stem_for_attack_lookup(log_path)
            attack_type = _ATTACK_TYPE_MAP.get(stem, "other_malicious")
            df["source_file"] = log_path.name
            df["dept"] = ""
            df["attack_type"] = attack_type
            df["is_malicious"] = 1
            frames.append(df)
            if verbose:
                print(f"  attack  {attack_type:<14} {log_path.name}: {len(df):,} events")

    if not frames:
        return pd.DataFrame()

    audit = _safe_concat(frames)
    if verbose:
        print(f"[audit] total: {len(audit):,} events ({audit['is_malicious'].sum():,} malicious)")
    return audit


def _load_procmon(root: Path, verbose: bool) -> pd.DataFrame:
    """
    Load Windows Procmon CSVs (3 benign + 3 malicious).

    Adds columns:
        attack_type   str
        is_malicious  int
    """
    if verbose:
        print("[procmon] loading Windows Procmon CSVs...")

    frames: List[pd.DataFrame] = []
    for rel_path, host_id, is_mal in _PROCMON_FILES:
        path = root / rel_path
        if not path.exists():
            if verbose:
                print(f"  MISSING: {rel_path}")
            continue
        df = parse_procmon_csv(path, host_id=host_id)
        if df.empty:
            continue
        df["source_file"] = path.name
        df["is_malicious"] = is_mal
        df["attack_type"] = "malicious" if is_mal else "benign"
        frames.append(df)
        if verbose:
            print(f"  {host_id:<8} {'mal' if is_mal else 'ben'} {path.name}: {len(df):,} events")

    if not frames:
        return pd.DataFrame()

    procmon = _safe_concat(frames)
    if verbose:
        print(f"[procmon] total: {len(procmon):,} events ({procmon['is_malicious'].sum():,} malicious)")
    return procmon


def _load_power(root: Path, verbose: bool) -> pd.DataFrame:
    """
    Load 4 power measurement CSVs into a single DataFrame.

    Schemas differ between local1 (mechanical) and local2 (electrical); we
    union all columns and leave non-applicable ones as NaN. Each row gets:
        device_id      str   (local1 / local2)
        is_malicious   int   (0 / 1)
        schema_kind    str   (mechanical / electrical)
        sample_index   int   (row index within source file; surrogate for time)
    """
    if verbose:
        print("[power] loading power measurements...")

    frames: List[pd.DataFrame] = []
    for rel_path, device_id, is_mal, schema in _POWER_FILES:
        path = root / rel_path
        if not path.exists():
            if verbose:
                print(f"  MISSING: {rel_path}")
            continue
        # Power CSVs have a header row, an empty separator row, and a units
        # row before the actual numeric data.
        df = pd.read_csv(path, skiprows=[1, 2], encoding="utf-8")
        # Drop rows that are entirely NA (file may have trailing blank rows).
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            continue
        df["device_id"] = device_id
        df["is_malicious"] = is_mal
        df["schema_kind"] = schema
        df["source_file"] = path.name
        df["sample_index"] = np.arange(len(df))
        frames.append(df)
        if verbose:
            print(
                f"  {device_id} {'mal' if is_mal else 'ben'} {path.name}: "
                f"{len(df):,} samples ({schema})"
            )

    if not frames:
        return pd.DataFrame()

    power = _safe_concat(frames)
    if verbose:
        print(f"[power] total: {len(power):,} samples")
    return power


# ── Node table builders ───────────────────────────────────────────────────────


def _build_host_table(
    flows: pd.DataFrame,
    audit: pd.DataFrame,
    procmon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build host_node table.

    Hosts come from two sources:
      - Network flows: every unique IP in (Src IP ∪ Dst IP) becomes a host.
        Identifier prefix: "ip:".
      - Procmon: every unique hostname in (net_src_host ∪ net_dst_host).
        Identifier prefix: "host:".

    Columns:
        host_id      str   (e.g. "ip:192.168.2.5" or "host:L-79GJ5Y2")
        host_kind    str   ("ip" or "hostname")
        raw_value    str   (the original IP / hostname)
        node_idx     int   (row index — used as PyG node index)
    """
    ips: set = set()
    if not flows.empty:
        for col in ("Src IP", "Dst IP"):
            if col in flows.columns:
                ips.update(flows[col].dropna().astype(str).unique())

    hostnames: set = set()
    if not procmon.empty:
        for col in ("net_src_host", "net_dst_host"):
            if col in procmon.columns:
                hostnames.update(procmon[col].dropna().astype(str).unique())
        # Also include the synthesised host_id column from Procmon (central / local1 / local2).
        if "host_id" in procmon.columns:
            hostnames.update(procmon["host_id"].dropna().astype(str).unique())

    # Synthetic audit-source hosts — one per audit log file. Without these,
    # all audit-derived processes would be detached from the host layer
    # (Phase 1 simplification documented in plan §6.4). The host_id format
    # `audit:<source_file>` matches the host_ref assigned in
    # _build_process_table, so the runs-edge construction joins them.
    audit_sources: set = set()
    if not audit.empty and "source_file" in audit.columns:
        audit_sources.update(audit["source_file"].dropna().astype(str).unique())

    rows = [
        {"host_id": f"ip:{ip}", "host_kind": "ip", "raw_value": ip}
        for ip in sorted(ips)
        if ip and ip != "nan"
    ] + [
        {"host_id": f"host:{name}", "host_kind": "hostname", "raw_value": name}
        for name in sorted(hostnames)
        if name and name != "nan"
    ] + [
        {"host_id": f"audit:{src}", "host_kind": "audit_source", "raw_value": src}
        for src in sorted(audit_sources)
        if src and src != "nan"
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["host_id", "host_kind", "raw_value", "node_idx"])
    else:
        df["node_idx"] = np.arange(len(df))
    return df


# Internal RFC1918 subnets (private IPv4) + IPv6 link-local prefix.
# Hosts on these are always retained during pruning.
_INTERNAL_SUBNETS: Tuple[str, ...] = (
    "10.",
    "192.168.",
    "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "fe80:", "fe80::",   # IPv6 link-local
)


def _enrich_host_table(
    hosts: pd.DataFrame,
    flows: pd.DataFrame,
    procmon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Augment host table with per-host statistics used by pruning and as
    initial node features.

    Adds columns:
        flow_count_total    int   total flows where host appears as src or dst
        flow_count_src      int   flows originated from this host
        flow_count_dst      int   flows targeted at this host
        mal_flow_count      int   flows that are labelled malicious
        is_internal_subnet  bool  RFC1918 / link-local
        procmon_event_count int   events from this host_id in procmon
                                  (only meaningful for procmon hostnames)
    """
    if hosts.empty:
        return hosts

    h = hosts.copy()
    h["flow_count_src"] = 0
    h["flow_count_dst"] = 0
    h["mal_flow_count"] = 0
    h["is_internal_subnet"] = False
    h["procmon_event_count"] = 0

    # Map raw_value (IP string) → counts. Only IP-kind hosts participate in flows.
    if not flows.empty:
        for direction, col in (("src", "Src IP"), ("dst", "Dst IP")):
            if col not in flows.columns:
                continue
            counts = flows[col].astype(str).value_counts()
            mal_counts = (
                flows[flows["is_malicious"] == 1][col]
                .astype(str)
                .value_counts()
            )
            ip_to_count = counts.to_dict()
            ip_to_mal = mal_counts.to_dict()

            ip_mask = h["host_kind"] == "ip"
            h.loc[ip_mask, f"flow_count_{direction}"] = (
                h.loc[ip_mask, "raw_value"].map(ip_to_count).fillna(0).astype(int)
            )
            # mal_flow_count accumulates over both directions.
            h.loc[ip_mask, "mal_flow_count"] = (
                h.loc[ip_mask, "mal_flow_count"]
                + h.loc[ip_mask, "raw_value"].map(ip_to_mal).fillna(0).astype(int)
            )

    h["flow_count_total"] = h["flow_count_src"] + h["flow_count_dst"]

    # Subnet classification.
    h["is_internal_subnet"] = h.apply(
        lambda r: r["host_kind"] == "ip"
        and any(r["raw_value"].startswith(p) for p in _INTERNAL_SUBNETS),
        axis=1,
    )

    # Procmon event count for hostname-kind hosts (central / local1 / local2 etc.)
    if not procmon.empty and "host_id" in procmon.columns:
        proc_counts = procmon["host_id"].astype(str).value_counts().to_dict()
        host_mask = h["host_kind"] == "hostname"
        h.loc[host_mask, "procmon_event_count"] = (
            h.loc[host_mask, "raw_value"].map(proc_counts).fillna(0).astype(int)
        )

    return h


def _prune_hosts(
    hosts: pd.DataFrame,
    *,
    min_flows: int,
    verbose: bool,
) -> pd.DataFrame:
    """
    Drop external-subnet IP hosts that touch no malicious flow AND have
    fewer than `min_flows` total connections (Stage 2a host pruning).

    Always retained:
        - Hostnames from Procmon (central / local1 / local2 / DESKTOP-* / L-*)
        - Audit-source synthetic hosts (one per audit log or benign dept)
        - Internal-subnet IPs (RFC1918 + IPv6 link-local)
        - Any IP touching at least one malicious flow (potential attacker / victim)
        - Any IP with flow_count_total >= min_flows (well-connected endpoint)

    Re-indexes node_idx after dropping rows.
    """
    if hosts.empty:
        return hosts

    keep_mask = (
        (hosts["host_kind"] == "hostname")
        | (hosts["host_kind"] == "audit_source")
        | hosts["is_internal_subnet"]
        | (hosts["mal_flow_count"] > 0)
        | (hosts["flow_count_total"] >= min_flows)
    )

    n_before = len(hosts)
    pruned = hosts[keep_mask].reset_index(drop=True).copy()
    pruned["node_idx"] = np.arange(len(pruned))

    if verbose:
        kind_counts = pruned["host_kind"].value_counts().to_dict()
        print(
            f"[hosts] pruned {n_before:,} → {len(pruned):,} "
            f"(retained: {kind_counts}; min_flows={min_flows})"
        )

    return pruned


def _build_process_table(
    audit: pd.DataFrame,
    procmon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build process_node table.

    A process is identified by (host_kind, host_value, comm_or_proc, pid).
    Processes from procmon use host_id (central/local1/local2). Processes
    from audit logs use the source_file as a host proxy (since audit logs
    don't carry IPs). Both produce a stable string ID.

    Columns:
        process_id   str   stable identifier
        host_ref     str   the host the process belongs to (matches host_id
                           from _build_host_table for procmon rows; an
                           audit-derived synthetic host token for audit rows)
        comm         str   process short name
        exe          str   executable path (audit only)
        pid          int
        source       str   "audit" or "procmon"
        node_idx     int
    """
    rows: List[Dict] = []

    if not audit.empty:
        # Audit log: (source_file or dept, comm, pid) → process_id.
        # We treat each audit-source host as "audit:<source_file>" since the
        # log file is the closest proxy we have to a host identity.
        sub = audit.copy()
        sub["pid"] = pd.to_numeric(sub["pid"], errors="coerce")
        sub = sub.dropna(subset=["pid"])
        # Use numpy's permissive cast (truncates) rather than pandas' Int64
        # safe cast, which rejects floats with sub-1 noise from malformed rows.
        sub["pid"] = sub["pid"].astype(np.int64)
        sub["host_ref"] = "audit:" + sub.get("source_file", pd.Series(dtype=str)).fillna("unknown")
        # Benign dept files have no source_file column; fall back to dept.
        if "source_file" not in sub.columns:
            sub["host_ref"] = "audit:" + sub.get("dept", "unknown").astype(str)

        keys = sub.groupby(["host_ref", "comm", "pid"], dropna=False).agg(
            event_count=("pid", "size"),
            is_malicious=("is_malicious", "max"),
        ).reset_index()
        for _, r in keys.iterrows():
            rows.append({
                "process_id": f"proc:audit:{r['host_ref']}:{r['comm']}:{r['pid']}",
                "host_ref": r["host_ref"],
                "comm": r["comm"],
                "exe": "",
                "pid": int(r["pid"]) if pd.notna(r["pid"]) else -1,
                "source": "audit",
                "event_count": int(r["event_count"]),
                "is_malicious": int(r["is_malicious"]) if pd.notna(r["is_malicious"]) else 0,
            })

    if not procmon.empty:
        sub = procmon.copy()
        # Procmon's on_bad_lines="skip" can still leave residual malformed PIDs;
        # coerce non-numeric PIDs to NaN and drop them defensively.
        sub["pid"] = pd.to_numeric(sub["pid"], errors="coerce")
        sub = sub.dropna(subset=["pid"])
        # Use numpy's permissive cast (truncates) rather than pandas' Int64
        # safe cast, which rejects floats with sub-1 noise from malformed rows.
        sub["pid"] = sub["pid"].astype(np.int64)
        keys = sub.groupby(["host_id", "process", "pid"], dropna=False).agg(
            event_count=("pid", "size"),
            is_malicious=("is_malicious", "max"),
        ).reset_index()
        for _, r in keys.iterrows():
            rows.append({
                "process_id": f"proc:procmon:{r['host_id']}:{r['process']}:{r['pid']}",
                "host_ref": f"host:{r['host_id']}",
                "comm": r["process"],
                "exe": "",
                "pid": int(r["pid"]) if pd.notna(r["pid"]) else -1,
                "source": "procmon",
                "event_count": int(r["event_count"]),
                "is_malicious": int(r["is_malicious"]) if pd.notna(r["is_malicious"]) else 0,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "process_id", "host_ref", "comm", "exe", "pid", "source",
            "event_count", "node_idx",
        ])

    # Drop garbage rows that leak in via Procmon's on_bad_lines="skip":
    #   - PID <= 0 (kernel idle / parser misalignment)
    #   - empty or whitespace-only comm
    #   - comm that is purely numeric (a column-shifted PID that landed in `comm`)
    df["comm"] = df["comm"].astype(str).str.strip()
    is_valid_comm = (df["comm"] != "") & (~df["comm"].str.fullmatch(r"\d+"))
    is_valid_pid = df["pid"] > 0
    df = df[is_valid_comm & is_valid_pid].reset_index(drop=True)

    df["node_idx"] = np.arange(len(df))
    return df


def _build_flow_node_table(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Build flow_node table.

    Each row in the flows DataFrame becomes one flow_node. We carry through
    the Flow ID as the unique identifier, plus a sequential node_idx.

    Columns:
        flow_node_id   str
        node_idx       int
    Other columns (84 CICFlowMeter features, label, etc.) are accessed via
    the same row index in the `flows` DataFrame.
    """
    if flows.empty:
        return pd.DataFrame(columns=["flow_node_id", "node_idx"])

    df = pd.DataFrame({
        "flow_node_id": flows.get("Flow ID", pd.Series(dtype=str)).fillna("").astype(str),
        "node_idx": np.arange(len(flows)),
    })
    # Disambiguate duplicate Flow IDs (CICFlowMeter sometimes emits the same
    # 5-tuple ID for parallel sub-flows) by appending node_idx.
    dup_mask = df.duplicated(subset=["flow_node_id"], keep=False)
    if dup_mask.any():
        df.loc[dup_mask, "flow_node_id"] = (
            df.loc[dup_mask, "flow_node_id"] + "#"
            + df.loc[dup_mask, "node_idx"].astype(str)
        )
    return df


def _build_device_table() -> pd.DataFrame:
    """
    Build device_node table. Two fixed devices:
        local1 → mechanical (synchronous generator + dynamometer)
        local2 → electrical (PV inverter)
    """
    return pd.DataFrame([
        {"device_id": "local1", "device_type": "generator",
         "schema_kind": "mechanical", "node_idx": 0},
        {"device_id": "local2", "device_type": "inverter",
         "schema_kind": "electrical", "node_idx": 1},
    ])


def _build_measurement_table(
    power: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build measurement_node table. Each row in the power DataFrame is one
    measurement_node, attached to its parent device.

    Columns:
        measurement_id  str   (e.g. "meas:local1:malicious:42")
        device_id       str
        sample_index    int   (within source file)
        is_malicious    int
        schema_kind     str
        node_idx        int
    """
    if power.empty:
        return pd.DataFrame(columns=[
            "measurement_id", "device_id", "sample_index", "is_malicious",
            "schema_kind", "node_idx",
        ])

    df = pd.DataFrame({
        "measurement_id": [
            f"meas:{r.device_id}:{'mal' if r.is_malicious else 'ben'}:{r.sample_index}"
            for r in power.itertuples()
        ],
        "device_id": power["device_id"].values,
        "sample_index": power["sample_index"].values,
        "is_malicious": power["is_malicious"].values,
        "schema_kind": power["schema_kind"].values,
    })
    df["node_idx"] = np.arange(len(df))
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────


def _read_flow_csv(path: Path) -> pd.DataFrame:
    """Read a CICFlowMeter CSV with permissive options for messy real data."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


# ── Stage 2b: Edge construction ───────────────────────────────────────────────


# Canonical edge type triples following PyG convention: (src_type, rel, dst_type)
EdgeKey = Tuple[str, str, str]

EDGE_HOST_RUNS_PROCESS: EdgeKey = ("host_node", "runs", "process_node")
EDGE_PROCESS_FORKS_PROCESS: EdgeKey = ("process_node", "forks", "process_node")
EDGE_FLOW_TARGETS_HOST: EdgeKey = ("flow_node", "targets", "host_node")
EDGE_HOST_SOURCES_FLOW: EdgeKey = ("host_node", "sources", "flow_node")
EDGE_DEVICE_REPORTS_MEASUREMENT: EdgeKey = ("device_node", "reports", "measurement_node")


def build_edges(data: MG24Data) -> Dict[EdgeKey, np.ndarray]:
    """
    Construct edge_index arrays for all edge types.

    Returns a dict keyed by (src_type, rel, dst_type) PyG triples; each value
    is an int64 array of shape (2, num_edges) where row 0 is source node
    indices and row 1 is destination node indices.

    Phase 1 implements 5 edge types listed in unsw_mg24_plan.md § 4.2.
    Deferred edges (mark in plan as TODO):
        - process -[generates]→ flow      (cross-modal, requires host↔IP map)
        - host -[pivots_to]→ host         (from pivot pcap analysis)
        - flow -[commands]→ device        (port-based, requires SCADA port catalog)
    """
    edges: Dict[EdgeKey, np.ndarray] = {}
    edges[EDGE_HOST_RUNS_PROCESS] = _build_host_runs_process(data)
    edges[EDGE_PROCESS_FORKS_PROCESS] = _build_process_forks_process(data)

    flow_targets = _build_flow_targets_host(data)
    edges[EDGE_FLOW_TARGETS_HOST] = flow_targets
    # Build the reverse `sources` edge from the same Src IP lookup.
    edges[EDGE_HOST_SOURCES_FLOW] = _build_host_sources_flow(data)

    edges[EDGE_DEVICE_REPORTS_MEASUREMENT] = _build_device_reports_measurement(data)
    return edges


def _build_host_runs_process(data: MG24Data) -> np.ndarray:
    """
    Edges from host_node to process_node.

    Procmon-derived processes carry host_ref like "host:central"; these match
    a host_node directly. Audit-derived processes carry host_ref like
    "audit:<filename>" which has no matching host node — those edges are
    skipped (Phase 1 limitation; documented in unsw_mg24_plan.md § 6.4).
    """
    if data.processes.empty or data.hosts.empty:
        return _empty_edge()

    host_id_to_idx = pd.Series(
        data.hosts["node_idx"].values, index=data.hosts["host_id"].values
    )
    proc_host_idx = data.processes["host_ref"].map(host_id_to_idx)
    valid = proc_host_idx.notna()
    if not valid.any():
        return _empty_edge()

    src = proc_host_idx[valid].astype(np.int64).values
    dst = data.processes.loc[valid, "node_idx"].astype(np.int64).values
    return np.stack([src, dst])


def _build_process_forks_process(data: MG24Data) -> np.ndarray:
    """
    Edges from parent process to child process (audit ppid → pid).

    For each audit row with valid pid and ppid, we look up the process_node
    indices for (host_ref, ppid) and (host_ref, pid). Edges are deduplicated.
    """
    if data.processes.empty or data.audit.empty:
        return _empty_edge()

    # Build (host_ref, pid) → process_node_idx lookup as a dict for O(1).
    proc_lookup: Dict[Tuple[str, int], int] = dict(
        zip(
            zip(data.processes["host_ref"].values, data.processes["pid"].astype(int).values),
            data.processes["node_idx"].astype(int).values,
        )
    )

    # Build host_ref column for audit table the same way _build_process_table does.
    a = data.audit.copy()
    a["pid"] = pd.to_numeric(a["pid"], errors="coerce")
    a["ppid"] = pd.to_numeric(a["ppid"], errors="coerce")
    a = a.dropna(subset=["pid", "ppid"]).copy()
    if a.empty:
        return _empty_edge()

    if "source_file" in a.columns:
        a["host_ref"] = "audit:" + a["source_file"].fillna("unknown").astype(str)
    else:
        a["host_ref"] = "audit:" + a.get("dept", "unknown").astype(str)

    # Drop self-edges: a process forks itself means parent==child.
    a = a[a["pid"] != a["ppid"]]
    if a.empty:
        return _empty_edge()

    # Vectorised lookup via dict.get; avoid iterrows for performance.
    parent_keys = list(zip(a["host_ref"].values, a["ppid"].astype(int).values))
    child_keys = list(zip(a["host_ref"].values, a["pid"].astype(int).values))

    pairs: List[Tuple[int, int]] = []
    seen: set = set()
    for pk, ck in zip(parent_keys, child_keys):
        p_idx = proc_lookup.get(pk)
        c_idx = proc_lookup.get(ck)
        if p_idx is None or c_idx is None:
            continue
        pair = (p_idx, c_idx)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if not pairs:
        return _empty_edge()

    arr = np.array(pairs, dtype=np.int64).T  # shape (2, num_edges)
    return arr


def _build_flow_targets_host(data: MG24Data) -> np.ndarray:
    """
    Edges flow_node → host_node, where host is the flow's Dst IP.

    Flows whose Dst IP was pruned in Stage 2a are silently dropped from the
    edge set (the flow_node still exists in the graph, just disconnected on
    this edge type).
    """
    if data.flow_nodes.empty or data.flows.empty or data.hosts.empty:
        return _empty_edge()

    ip_hosts = data.hosts[data.hosts["host_kind"] == "ip"]
    if ip_hosts.empty or "Dst IP" not in data.flows.columns:
        return _empty_edge()

    ip_to_idx = pd.Series(ip_hosts["node_idx"].values, index=ip_hosts["raw_value"].values)
    target_idx = data.flows["Dst IP"].astype(str).map(ip_to_idx)
    valid = target_idx.notna()
    if not valid.any():
        return _empty_edge()

    src = data.flow_nodes.loc[valid.values, "node_idx"].astype(np.int64).values
    dst = target_idx[valid].astype(np.int64).values
    return np.stack([src, dst])


def _build_host_sources_flow(data: MG24Data) -> np.ndarray:
    """
    Edges host_node → flow_node, where host is the flow's Src IP.

    Symmetric to _build_flow_targets_host; lets the GNN propagate signal
    in both directions across the flow/host boundary.
    """
    if data.flow_nodes.empty or data.flows.empty or data.hosts.empty:
        return _empty_edge()

    ip_hosts = data.hosts[data.hosts["host_kind"] == "ip"]
    if ip_hosts.empty or "Src IP" not in data.flows.columns:
        return _empty_edge()

    ip_to_idx = pd.Series(ip_hosts["node_idx"].values, index=ip_hosts["raw_value"].values)
    source_idx = data.flows["Src IP"].astype(str).map(ip_to_idx)
    valid = source_idx.notna()
    if not valid.any():
        return _empty_edge()

    src = source_idx[valid].astype(np.int64).values
    dst = data.flow_nodes.loc[valid.values, "node_idx"].astype(np.int64).values
    return np.stack([src, dst])


def _build_device_reports_measurement(data: MG24Data) -> np.ndarray:
    """
    Edges device_node → measurement_node. Trivial: each measurement_node
    carries its parent device_id.
    """
    if data.measurements.empty or data.devices.empty:
        return _empty_edge()

    dev_lookup = pd.Series(
        data.devices["node_idx"].values, index=data.devices["device_id"].values
    )
    src = data.measurements["device_id"].map(dev_lookup).astype(np.int64).values
    dst = data.measurements["node_idx"].astype(np.int64).values
    return np.stack([src, dst])


def _empty_edge() -> np.ndarray:
    """Empty PyG-format edge_index."""
    return np.zeros((2, 0), dtype=np.int64)


def _stem_for_attack_lookup(path: Path) -> str:
    """
    Normalise a filename into the lookup key used by _ATTACK_TYPE_MAP.

    Examples:
        "backdoor1.pcap_Flow.csv"     → "backdoor1"
        "ms17_010 traffic.pcapng"     → "ms17_010 traffic"
        "audit_MITM1.log"             → "audit_mitm1"
    """
    name = path.stem  # strip last extension
    # CICFlowMeter outputs are "<attack>.pcap_Flow.csv". `.stem` only strips ".csv",
    # leaving "<attack>.pcap_Flow". Strip that suffix too.
    name = re.sub(r"\.pcap_Flow$", "", name)
    name = re.sub(r"\.pcap$", "", name)
    return name.lower()


# ── Stage 2c: PyG HeteroData emission ─────────────────────────────────────────


# Numeric flow feature columns: anything not in this excluded set becomes a feature.
_FLOW_NON_FEATURE_COLS: set = {
    "Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "Protocol",
    "attack_type", "is_malicious", "source_file", "dept",
}

# Power CSV columns we extract (NaN for the alternate schema → filled with 0).
_POWER_FEATURE_COLS: List[str] = [
    "Speed", "Torque",                    # mechanical (local1)
    "Voltage", "Current",                 # electrical (local2)
    "Power", "Energy",                    # both schemas
    "Switching-Frequency-",               # local2-specific
    "Channel-1-RMS", "Channel-1-AVG", "Channel-1-Frequency",
]


def to_pyg_hetero_data(
    data: MG24Data,
    edges: Dict[EdgeKey, np.ndarray],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Convert MG24Data + edges into a PyG HeteroData object.

    Implements Stage 2c of unsw_mg24_plan.md § 6.4:
      - Per-type node feature tensors (DD-3 multi-task uses these)
      - Per-type label tensors (file-level binary, DD-6)
      - Stratified train/val/test masks for the three labelled node types
        (flow_node, process_node, measurement_node — see DD-3)

    Args:
        data:       MG24Data from load_mg24_data().
        edges:      Edge dict from build_edges().
        val_ratio:  Fraction of labelled nodes assigned to validation.
        test_ratio: Fraction assigned to test.
        seed:       Random seed for the stratified splits.

    Returns:
        torch_geometric.data.HeteroData
    """
    import torch
    from torch_geometric.data import HeteroData

    hd = HeteroData()

    # ── Node feature tensors ──────────────────────────────────────
    hd["host_node"].x = _host_features(data.hosts)
    hd["process_node"].x = _process_features(data.processes)
    hd["flow_node"].x = _flow_features(data.flows)
    hd["device_node"].x = _device_features(data.devices)
    hd["measurement_node"].x = _measurement_features(data.power)

    # ── Edge index tensors ────────────────────────────────────────
    for edge_key, edge_idx in edges.items():
        if edge_idx.shape[1] == 0:
            continue
        hd[edge_key].edge_index = torch.from_numpy(edge_idx).to(torch.long)

    # ── Multi-task labels (DD-3, DD-6) ────────────────────────────
    hd["flow_node"].y = torch.tensor(
        data.flows["is_malicious"].astype(int).values, dtype=torch.long
    )
    hd["process_node"].y = torch.tensor(
        data.processes["is_malicious"].astype(int).values, dtype=torch.long
    )
    hd["measurement_node"].y = torch.tensor(
        data.measurements["is_malicious"].astype(int).values, dtype=torch.long
    )

    # ── Train / val / test masks (stratified) ─────────────────────
    rng = np.random.default_rng(seed)
    for ntype in ("flow_node", "process_node", "measurement_node"):
        labels = hd[ntype].y.numpy()
        train_mask, val_mask, test_mask = _stratified_split(
            labels, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
        hd[ntype].train_mask = torch.from_numpy(train_mask)
        hd[ntype].val_mask = torch.from_numpy(val_mask)
        hd[ntype].test_mask = torch.from_numpy(test_mask)

    return hd


# ── Feature builders ──────────────────────────────────────────────────────────


def _log1p_zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Stabilise heavy-tailed numeric features.

    Pipeline:
      1. sign(x) * log1p(|x|): compresses [-1e9, 1e9] → roughly [-21, 21]
         (handles CICFlowMeter columns whose raw scale spans ~9 orders of
         magnitude — Flow Duration ≈ 1e7, byte counts ≈ 1e9, ratios ≈ 0-10).
      2. Per-column z-score: centre at 0, unit variance.
      3. Constant columns (std < eps) are zero-filled to avoid NaN.

    Why this matters:
        Without normalisation the first HGT projection produces enormous
        activations, the per-edge-type NCM MLPs saturate at sigmoid 0/1,
        BCE supervision plateaus, and the detection cross-entropy stays at
        large absolute values even when F1 is high. We observed that
        behaviour in the first MG24 pilot run (NCM loss stuck at 29.82,
        det loss ~ 400 at F1=0.99).

    Statistics are computed over the *current* dataset (no separate
    train-fit / val-transform), which is acceptable for a transductive
    full-graph training setting.
    """
    arr = np.asarray(arr, dtype=np.float32)
    sign = np.sign(arr)
    arr = sign * np.log1p(np.abs(arr))

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    const_mask = std < eps
    std = np.where(const_mask, 1.0, std)
    arr = (arr - mean) / std
    if const_mask.any():
        arr[:, const_mask[0]] = 0.0
    return arr.astype(np.float32, copy=False)


def _host_features(hosts: pd.DataFrame):
    """
    host_node features (9-dim): 6 numeric stats + 3-dim one-hot host_kind.
    """
    import torch

    if hosts.empty:
        return torch.zeros((0, 9), dtype=torch.float32)

    numeric = hosts[[
        "flow_count_total", "flow_count_src", "flow_count_dst",
        "mal_flow_count", "is_internal_subnet", "procmon_event_count",
    ]].fillna(0).astype(float).values

    # log1p the count columns (heavy-tailed), keep boolean as-is (column 4).
    log_cols = [0, 1, 2, 3, 5]
    numeric[:, log_cols] = np.log1p(np.maximum(numeric[:, log_cols], 0.0))

    # One-hot host_kind (ip / hostname / audit_source).
    kind = hosts["host_kind"].values
    one_hot = np.stack([
        (kind == "ip").astype(float),
        (kind == "hostname").astype(float),
        (kind == "audit_source").astype(float),
    ], axis=1)

    feats = np.concatenate([numeric, one_hot], axis=1).astype(np.float32)
    return torch.from_numpy(feats)


def _process_features(processes: pd.DataFrame):
    """
    process_node features (3-dim): log1p(event_count) + 2-dim one-hot source.
    """
    import torch

    if processes.empty:
        return torch.zeros((0, 3), dtype=torch.float32)

    event_count = np.log1p(processes["event_count"].fillna(0).astype(float).values)
    src = processes["source"].values
    one_hot = np.stack([
        (src == "audit").astype(float),
        (src == "procmon").astype(float),
    ], axis=1)

    feats = np.concatenate([event_count.reshape(-1, 1), one_hot], axis=1).astype(np.float32)
    return torch.from_numpy(feats)


def _flow_features(flows: pd.DataFrame):
    """
    flow_node features: all numeric CICFlowMeter columns minus identifiers,
    log1p-compressed and z-scored so heavy-tailed columns (durations, byte
    counts) do not dominate the HGT input. See `_log1p_zscore` for rationale.
    """
    import torch

    if flows.empty:
        return torch.zeros((0, 0), dtype=torch.float32)

    feature_cols = [
        c for c in flows.columns
        if c not in _FLOW_NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(flows[c])
    ]
    if not feature_cols:
        return torch.zeros((len(flows), 0), dtype=torch.float32)

    arr = flows[feature_cols].astype(np.float32).values
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
    arr = _log1p_zscore(arr)
    return torch.from_numpy(arr)


def _device_features(devices: pd.DataFrame):
    """device_node features (2-dim): one-hot device_type."""
    import torch

    if devices.empty:
        return torch.zeros((0, 2), dtype=torch.float32)

    dtype = devices["device_type"].values
    one_hot = np.stack([
        (dtype == "generator").astype(float),
        (dtype == "inverter").astype(float),
    ], axis=1).astype(np.float32)
    return torch.from_numpy(one_hot)


def _measurement_features(power: pd.DataFrame):
    """
    measurement_node features: extracted from the power DataFrame in the
    same row order. Missing columns (schema-specific) become 0; the union
    of mechanical (Speed/Torque) and electrical (Voltage/Current/Channel*)
    feature ranges is then log1p+z-scored together (see _log1p_zscore).
    """
    import torch

    if power.empty:
        return torch.zeros((0, len(_POWER_FEATURE_COLS)), dtype=torch.float32)

    feats = np.zeros((len(power), len(_POWER_FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(_POWER_FEATURE_COLS):
        if col in power.columns:
            vals = pd.to_numeric(power[col], errors="coerce").fillna(0.0).values
            feats[:, i] = vals.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e9, neginf=-1e9)
    feats = _log1p_zscore(feats)
    return torch.from_numpy(feats)


def _stratified_split(
    labels: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/val/test split returning three boolean masks of the
    same length as `labels`.

    Each label class is split independently in the requested proportions,
    so the resulting masks preserve the label distribution.
    """
    n = len(labels)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        n_val = int(round(len(idx) * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        train[train_idx] = True
        val[val_idx] = True
        test[test_idx] = True

    return train, val, test
