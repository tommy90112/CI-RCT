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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

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

    # DD-11: per-audit-log set of remote IPv4 addresses extracted from
    # SOCKADDR records (used to build the audit_source ↔ ip_host bridge
    # that connects the network island to the SCADA island).
    audit_source_ips: Dict[str, Set[str]] = field(default_factory=dict)

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

    # ── DD-11 bridge: per-log SOCKADDR IP extraction ──────────────
    # The bundled audit_parser drops SOCKADDR records (it only keeps
    # SYSCALL / PATH / SOCKETCALL), so we re-parse the raw .log files
    # here in a focused pass that only collects IPv4 destinations.
    audit_source_ips = _extract_audit_log_ips(root, verbose=verbose)

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
            audit_source_ips=audit_source_ips,
        )
        print(f"=== Done. Node counts: {data.summary()} ===")
        return data

    return MG24Data(
        flows=flows, audit=audit, procmon=procmon, power=power,
        hosts=hosts, processes=processes, flow_nodes=flow_nodes,
        devices=devices, measurements=measurements,
        audit_source_ips=audit_source_ips,
    )


# ── DD-11 bridge helper: audit log → remote IPv4 set ─────────────────────────


_SOCKADDR_AF_INET_PREFIX = "0200"   # AF_INET in little-endian hex
_SADDR_RE = re.compile(r"saddr=([0-9A-Fa-f]+)")


def _extract_audit_log_ips(
    root: Path,
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Extract per-audit-log unique remote IPv4 addresses from SOCKADDR records.

    DD-11 bridges the two disjoint islands of the kill-chain DAG (the
    network-side ip-kind hosts and the SCADA-side audit_source-kind hosts)
    by adding edges audit_source_host → ip_host wherever an audit log
    recorded a SOCKADDR with that destination IP.

    The bundled audit_parser silently drops SOCKADDR records (it keeps only
    SYSCALL / PATH / SOCKETCALL — see audit_parser._OUTPUT_COLUMNS), so this
    function does a focused single-pass scan of the raw .log files that
    only collects IPv4 sa_family records.

    SOCKADDR hex layout (Linux audit, AF_INET case):
        bytes 0-1 : sa_family (0x0002, little-endian → '0200')
        bytes 2-3 : port      (big-endian uint16)
        bytes 4-7 : IPv4 addr (big-endian, dotted)
        bytes 8-15: zero padding

    Non-IPv4 families (AF_UNIX 0x0100, AF_INET6 0x0A00, …) are skipped.
    Loopback (127.0.0.0/8) and the unspecified address (0.0.0.0) are
    filtered because they cannot meaningfully bridge to a flow's Src IP.

    Returns:
        Dict[str, Set[str]] keyed by log file basename
        (e.g. "audit_MITM1.log") — matches `source_file` column added by
        audit_parser.parse_audit_dir().
    """
    result: Dict[str, Set[str]] = {}

    candidates: List[Path] = []
    mal_dir = root / "Malicious system call traces"
    if mal_dir.is_dir():
        candidates.extend(sorted(mal_dir.glob("audit_*.log")))
    ben_dir = root / "Benign system call traces"
    if ben_dir.is_dir():
        for dept_dir in sorted(ben_dir.iterdir()):
            if not dept_dir.is_dir() or dept_dir.name == "Microgrid department":
                continue
            candidates.extend(sorted(dept_dir.glob("*.log")))

    total_records = 0
    total_ipv4 = 0
    for log_path in candidates:
        ips: Set[str] = set()
        try:
            with open(log_path, "r", errors="ignore") as fh:
                for line in fh:
                    if "type=SOCKADDR" not in line:
                        continue
                    total_records += 1
                    m = _SADDR_RE.search(line)
                    if not m:
                        continue
                    hexstr = m.group(1)
                    if len(hexstr) < 16:
                        continue
                    if hexstr[:4].upper() != _SOCKADDR_AF_INET_PREFIX.upper():
                        continue
                    try:
                        b = bytes.fromhex(hexstr[:16])
                    except ValueError:
                        continue
                    ip = ".".join(str(x) for x in b[4:8])
                    if ip.startswith("127.") or ip == "0.0.0.0":
                        continue
                    ips.add(ip)
                    total_ipv4 += 1
        except OSError as e:
            if verbose:
                print(f"  [bridge] failed to read {log_path.name}: {e}")
            continue
        if ips:
            result[log_path.name] = ips

    if verbose:
        n_pairs = sum(len(v) for v in result.values())
        print(f"[bridge] SOCKADDR scan: {total_records:,} records, "
              f"{total_ipv4:,} IPv4 decoded; {len(result)} logs contributed "
              f"{n_pairs} (log,IP) pairs")
    return result


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

EDGE_HOST_SOURCES_FLOW: EdgeKey = ("host_node", "sources", "flow_node")
EDGE_PROCESS_RUNS_ON_HOST: EdgeKey = ("process_node", "runs_on", "host_node")
EDGE_PROCESS_FORKS_PROCESS: EdgeKey = ("process_node", "forks", "process_node")
EDGE_DEVICE_HOSTS_PROCESS: EdgeKey = ("device_node", "hosts", "process_node")
EDGE_DEVICE_REPORTS_MEASUREMENT: EdgeKey = ("device_node", "reports", "measurement_node")
# DD-11: bridge edge linking the audit-source island to the network-IP island.
EDGE_HOST_RESOLVES_TO_IP: EdgeKey = ("host_node", "resolves_to_ip", "host_node")


def build_edges(data: MG24Data) -> Dict[EdgeKey, np.ndarray]:
    """
    Construct edge_index arrays for all edge types
    (DD-10 reversed kill-chain DAG + DD-11 audit↔ip bridge).

    Returns a dict keyed by (src_type, rel, dst_type) PyG triples; each value
    is an int64 array of shape (2, num_edges) where row 0 is source node
    indices and row 1 is destination node indices.

    Kill-chain DAG ordering (DD-10): edges point from upstream cause →
    downstream effect, so backward BFS from a labelled target (e.g. fraud
    flow_node) can reach its causal parents:

        device_node ──→ process_node ──→ host_node ──→ flow_node
                              │                   ↘
                              └→ process_forks      → (downstream)
        device_node ──→ measurement_node

    DD-11 bridge: audit_source-kind hosts → ip-kind hosts wherever the
    audit log recorded a SOCKADDR with that destination IP. Connects the
    two previously disjoint islands (network ip-hosts vs SCADA audit-source
    hosts) so backward BFS from a fraud flow can reach the audited host,
    its process, and from there the device/measurement subgraph.

    Why DD-10 reversed vs. DD-9: DD-9 used flow→host→process→device, making
    flow_node a DAG source with no parents (trace stuck at depth=0;
    see eval_dag_v1.log). DD-10 flips this to attacker-origin direction.

    Edge details:
        host → flow         uses Src IP (host as flow origin); replaces
                            DD-9's flow → host (Dst IP).
        process → host      process runs on host (DD-10).
        device → process    device hosts process (DD-10).
        device → measurement, process → process: unchanged.
        host → host         DD-11 audit_source → ip_host bridge
                            (type-level self-loop; src/dst are different
                            host nodes of different sub-kinds).

    Deferred edges (would require additional inference NOT present in raw
    dataset columns):
        - procmon hostname ↔ ip   (procmon CSV records local hostname
                                   `L-79GJ5Y2`, not remote IPs of its
                                   connections; would need IP-by-hostname
                                   lookup or testbed config)
        - host → host (lateral)   (would require pivot-pcap semantics)
    """
    edges: Dict[EdgeKey, np.ndarray] = {}
    edges[EDGE_HOST_SOURCES_FLOW] = _build_host_sources_flow(data)
    edges[EDGE_PROCESS_RUNS_ON_HOST] = _build_process_runs_on_host(data)
    edges[EDGE_PROCESS_FORKS_PROCESS] = _build_process_forks_process(data)
    edges[EDGE_DEVICE_HOSTS_PROCESS] = _build_device_hosts_process(data)
    edges[EDGE_DEVICE_REPORTS_MEASUREMENT] = _build_device_reports_measurement(data)
    edges[EDGE_HOST_RESOLVES_TO_IP] = _build_host_resolves_to_ip(data)
    return edges


def _build_process_runs_on_host(data: MG24Data) -> np.ndarray:
    """
    Edges from process_node to host_node (DD-10: reversed direction).

    Causal interpretation: a process is the active agent; running it on a
    host can compromise that host. The edge therefore points from the
    process (potential attack vector) to the host (affected resource).

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

    src = data.processes.loc[valid, "node_idx"].astype(np.int64).values
    dst = proc_host_idx[valid].astype(np.int64).values
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


def _build_host_sources_flow(data: MG24Data) -> np.ndarray:
    """
    Edges host_node → flow_node, where host is the flow's Src IP
    (DD-10: reversed direction + switched from Dst IP to Src IP).

    Causal interpretation: the host at the flow's Src IP is the originator
    of that flow — if the flow is malicious, the Src-IP host is the
    upstream cause. This makes flow_node have a non-empty parent set so
    backward BFS from a fraud flow can reach the responsible host.

    Flows whose Src IP was pruned in Stage 2a are silently dropped from the
    edge set (the flow_node still exists in the graph, just disconnected on
    this edge type).
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


def _build_device_hosts_process(data: MG24Data) -> np.ndarray:
    """
    Edges device_node → process_node (DD-10: reversed direction).

    Causal interpretation: the SCADA device is the physical substrate;
    processes run on top of it, so the device is the upstream cause of
    those processes existing. Backward BFS from a malicious process can
    reach the underlying device.

    Pure mapping from existing dataset columns — NO inferred information:
      - Procmon CSV records `host_id ∈ {central, local1, local2}`
      - Dataset has device_node entries `device_id ∈ {local1, local2}`
      - We connect:
          device:local1  → process on host:local1
          device:local2  → process on host:local2
          {device:local1, device:local2} → process on host:central
              (central is the SCADA aggregator — see _PROCMON_FILES)

    Linux-audit-derived processes carry host_ref="audit:<filename>" and are
    NOT connected here, because the dataset does not provide a mapping
    from audit-source filenames to physical SCADA devices.
    """
    if data.processes.empty or data.devices.empty:
        return _empty_edge()

    procmon_processes = data.processes[data.processes["source"] == "procmon"]
    if procmon_processes.empty:
        return _empty_edge()

    device_lookup = pd.Series(
        data.devices["node_idx"].values, index=data.devices["device_id"].values
    )

    src_list: List[int] = []
    dst_list: List[int] = []
    for _, row in procmon_processes.iterrows():
        host_id = row["host_ref"].replace("host:", "")  # "host:local1" → "local1"
        proc_idx = int(row["node_idx"])
        if host_id == "central":
            # Central SCADA aggregator: both local sub-devices host
            # central processes.
            for d in ("local1", "local2"):
                if d in device_lookup.index:
                    src_list.append(int(device_lookup[d]))
                    dst_list.append(proc_idx)
        elif host_id in device_lookup.index:
            src_list.append(int(device_lookup[host_id]))
            dst_list.append(proc_idx)

    if not src_list:
        return _empty_edge()
    return np.stack([np.array(src_list, dtype=np.int64),
                     np.array(dst_list, dtype=np.int64)])


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


def _build_host_resolves_to_ip(data: MG24Data) -> np.ndarray:
    """
    DD-11 bridge edges: audit_source host → ip host.

    For each audit log, the SOCKADDR records contain destination IPv4
    addresses that the audited host made connections to. We treat the
    audit_source host as the upstream cause for any ip-kind host whose
    IP appears in that log's SOCKADDR set. This adds a parent for
    ip-kind hosts so backward BFS from a fraud flow can cross from
    the network island into the SCADA / audit island.

    Edge direction (audit_source → ip_host) chosen so that
    parents(ip_host) ⊇ {audit_source}, enabling the depth ≥ 2
    backward trace fraud_flow → Src-IP-host → audit_source_host →
    process → … instead of bottoming out at the Src-IP-host.

    Silently skips any (log, ip) pair where either the audit_source
    host or the ip host is missing from the pruned host table.
    """
    if data.hosts.empty or not data.audit_source_ips:
        return _empty_edge()

    audit_hosts = data.hosts[data.hosts["host_kind"] == "audit_source"]
    ip_hosts = data.hosts[data.hosts["host_kind"] == "ip"]
    if audit_hosts.empty or ip_hosts.empty:
        return _empty_edge()

    audit_lookup = pd.Series(
        audit_hosts["node_idx"].values, index=audit_hosts["raw_value"].values
    )
    ip_lookup = pd.Series(
        ip_hosts["node_idx"].values, index=ip_hosts["raw_value"].values
    )

    src_list: List[int] = []
    dst_list: List[int] = []
    seen: set = set()
    for log_name, ips in data.audit_source_ips.items():
        # `raw_value` on audit_source hosts == log filename (set in
        # _build_host_table from audit.source_file). Both keys are str.
        if log_name not in audit_lookup.index:
            continue
        src_idx = int(audit_lookup[log_name])
        for ip in ips:
            if ip not in ip_lookup.index:
                continue
            dst_idx = int(ip_lookup[ip])
            pair = (src_idx, dst_idx)
            if pair in seen:
                continue
            seen.add(pair)
            src_list.append(src_idx)
            dst_list.append(dst_idx)

    if not src_list:
        return _empty_edge()
    return np.stack([np.array(src_list, dtype=np.int64),
                     np.array(dst_list, dtype=np.int64)])


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


SplitMode = Literal["row", "by_file", "hybrid", "by_incident"]


def to_pyg_hetero_data(
    data: MG24Data,
    edges: Dict[EdgeKey, np.ndarray],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_mode: SplitMode = "by_file",
    host_features_mode: HostFeaturesMode = "full",
    flow_features_exclude: Optional[List[str]] = None,
):
    """
    Convert MG24Data + edges into a PyG HeteroData object.

    Implements Stage 2c of unsw_mg24_plan.md § 6.4:
      - Per-type node feature tensors (DD-3 multi-task uses these)
      - Per-type label tensors (file-level binary, DD-6)
      - Stratified train/val/test masks for the three labelled node types
        (flow_node, process_node, measurement_node — see DD-3)

    Split modes (DD-8):
      "row":     Row-level random stratified split. Each row independently
                 assigned to train/val/test, stratified by binary label.
                 ⚠ Same-pcap rows leak across splits — overestimates F1.
      "by_file": File-level stratified split. Each source file goes wholly
                 to one of train/val/test, stratified by binary label.
                 Tests cross-session generalisation.
      "hybrid":  Benign rows split row-wise (stable baseline assumption);
                 malicious rows split by-file. Mirrors production IDS
                 deployment scenario.

    Args:
        data:       MG24Data from load_mg24_data().
        edges:      Edge dict from build_edges().
        val_ratio:  Fraction of labelled nodes assigned to validation.
        test_ratio: Fraction assigned to test.
        seed:       Random seed for the stratified splits.
        split_mode: One of "row", "by_file", "hybrid". See above.
        host_features_mode: DD-8 host-feature fairness mode. One of:
                    "full"         keep all 9 features incl. mal_flow_count
                    "no_mal_count" drop mal_flow_count (Fix 1)
                    "zeroed"       zero all host features (Fix 3)
                    For Fix 4 (host_node removed from detection entirely),
                    use CI_RCT(backbone_exclude_node_types=["host_node"]).
        flow_features_exclude: List of CICFlowMeter columns to drop from
                    flow_node features (DD-8 Fix 5). Used to ablate the
                    "Active Std/Max/Mean" timing-fingerprint trio that
                    fairness_audit.py flagged with univariate AUC > 0.95.
                    Columns absent from flows are silently ignored.

    Returns:
        torch_geometric.data.HeteroData
    """
    import torch
    from torch_geometric.data import HeteroData

    hd = HeteroData()

    # ── Node feature tensors ──────────────────────────────────────
    hd["host_node"].x = _host_features(data.hosts, mode=host_features_mode)
    hd["process_node"].x = _process_features(data.processes)
    hd["flow_node"].x = _flow_features(
        data.flows, exclude_columns=flow_features_exclude,
    )
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
    if split_mode == "by_incident":
        split_groups: Dict[str, np.ndarray] = {
            "flow_node": _incident_groups_for_flows(data.flows),
            "process_node": _incident_groups_for_processes(data.processes),
            "measurement_node": _incident_groups_for_measurements(data.measurements),
        }
        incident_split_map = _build_global_incident_split(
            split_groups,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
    else:
        split_groups = {
            "flow_node": _split_groups_for_flows(data.flows),
            "process_node": _split_groups_for_processes(data.processes),
            "measurement_node": _split_groups_for_measurements(data.measurements),
        }
        incident_split_map = None
    for ntype in ("flow_node", "process_node", "measurement_node"):
        labels = hd[ntype].y.numpy()
        groups = split_groups[ntype]
        train_mask, val_mask, test_mask = _build_split_masks(
            labels, groups,
            val_ratio=val_ratio, test_ratio=test_ratio,
            mode=split_mode, rng=rng,
            incident_split_map=incident_split_map,
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


HostFeaturesMode = Literal["full", "no_mal_count", "zeroed"]


def _host_features(
    hosts: pd.DataFrame,
    mode: HostFeaturesMode = "full",
):
    """
    host_node features, controlled by DD-8 Fix 1–3 fairness mode.

    Modes (input to ablation study quantifying host-level leakage):
      "full":          9-dim — 6 numeric stats (including the label-derived
                       `mal_flow_count`) + 3-dim one-hot host_kind. Baseline.
      "no_mal_count":  8-dim — drop `mal_flow_count`. Removes the direct
                       label leakage; keeps all other host statistics.
      "zeroed":        9-dim — same shape as "full" but every value is 0.
                       Eliminates all feature signal while preserving the
                       host_node as a topology placeholder in the graph
                       (HGT message passing can still flow through it).
    """
    import torch

    base_dim = 9 if mode in ("full", "zeroed") else 8
    if hosts.empty:
        return torch.zeros((0, base_dim), dtype=torch.float32)

    if mode == "zeroed":
        return torch.zeros((len(hosts), base_dim), dtype=torch.float32)

    if mode == "full":
        numeric_cols = [
            "flow_count_total", "flow_count_src", "flow_count_dst",
            "mal_flow_count", "is_internal_subnet", "procmon_event_count",
        ]
        # log1p the count columns; keep boolean is_internal_subnet (idx 4) as-is.
        log_cols = [0, 1, 2, 3, 5]
    elif mode == "no_mal_count":
        numeric_cols = [
            "flow_count_total", "flow_count_src", "flow_count_dst",
            "is_internal_subnet", "procmon_event_count",
        ]
        # log1p the count columns; keep boolean is_internal_subnet (idx 3) as-is.
        log_cols = [0, 1, 2, 4]
    else:
        raise ValueError(f"Unknown host_features mode: {mode!r}")

    numeric = hosts[numeric_cols].fillna(0).astype(float).values
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


def _flow_features(
    flows: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
):
    """
    flow_node features: all numeric CICFlowMeter columns minus identifiers,
    log1p-compressed and z-scored so heavy-tailed columns (durations, byte
    counts) do not dominate the HGT input. See `_log1p_zscore` for rationale.

    Args:
        flows:           Flows DataFrame.
        exclude_columns: CICFlowMeter columns to drop before z-scoring (DD-8
                         Fix 5 fairness ablation). Use this to disable
                         "attack-tool fingerprint" features such as the
                         Active Std/Max/Mean trio identified by
                         fairness_audit.py to have univariate AUC > 0.95.
                         Columns not present in `flows` are silently ignored.
    """
    import torch

    if flows.empty:
        return torch.zeros((0, 0), dtype=torch.float32)

    drop = set(exclude_columns or [])
    non_features = set(_FLOW_NON_FEATURE_COLS) | drop
    feature_cols = [
        c for c in flows.columns
        if c not in non_features
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


# ── DD-8: by-file / hybrid stratified splits ─────────────────────────────────


def _split_groups_for_flows(flows: pd.DataFrame) -> np.ndarray:
    """
    Per-row split-group identifier for flow_node = the source CSV filename.
    Each pcap is one independent attack/benign session.
    """
    if flows.empty or "source_file" not in flows.columns:
        return np.array([], dtype=object)
    return flows["source_file"].fillna("unknown").astype(str).values


def _split_groups_for_processes(processes: pd.DataFrame) -> np.ndarray:
    """
    Per-row split-group identifier for process_node.

    A process aggregates events from one source log/CSV:
      - audit-derived: host_ref already encodes the file ("audit:<filename>"
        or "audit:<dept>"), so use it directly.
      - procmon-derived: a single (host_id × is_malicious) pair maps 1:1 to
        one of the 6 procmon CSV files (e.g., host:central + is_malicious=1
        ⇒ central_malicious.CSV), so we synthesise that key.
    """
    if processes.empty:
        return np.array([], dtype=object)
    src = processes["source"].astype(str).values
    host_ref = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    out = np.empty(len(processes), dtype=object)
    for i in range(len(processes)):
        if src[i] == "audit":
            out[i] = host_ref[i]
        else:
            out[i] = f"{host_ref[i]}:mal{is_mal[i]}"
    return out


def _split_groups_for_measurements(measurements: pd.DataFrame) -> np.ndarray:
    """
    Per-row split-group identifier for measurement_node.

    Each (device_id × is_malicious) pair maps 1:1 to one of the 4 power CSV
    files (local1_normal, local1_malicious, local2_normal, local2_malicious).
    """
    if measurements.empty:
        return np.array([], dtype=object)
    dev = measurements["device_id"].astype(str).values
    is_mal = measurements["is_malicious"].astype(int).values
    return np.array([f"{dev[i]}:mal{is_mal[i]}" for i in range(len(measurements))], dtype=object)


# ── DD-13: incident-level (cross-modal) stratified split ────────────────────
#
# Rationale: by_file splits each node-type independently, so a malicious
# pcap can land in test while its paired audit log lands in train. The
# backbone then learns the label from one modality and predicts it on the
# other → F1 stays ~0.99 even after host_features_mode="zeroed".
#
# by_incident aligns split assignment across modalities by reusing the
# `attack_type` already present on flows/audit (mapped via _ATTACK_TYPE_MAP
# from filename stems). All rows tagged with the same attack_type — pcap,
# audit log, derived process_node — go into the same split. Benign rows
# fall back to per-file/per-host grouping (no cross-modal leakage risk
# since labels are uniformly 0).
#
# Procmon-derived processes have no attack_type concept (one CSV per host
# × is_malicious pair), and Power measurements live on an independent SCADA
# modality with no network/audit overlap; both keep their by_file group key
# under a namespace prefix that excludes them from incident alignment.


_INCIDENT_PREFIX = "incident:"
_BENIGN_PREFIX = "benign:"


def _attack_type_from_audit_host_ref(host_ref: str, is_malicious: int) -> Optional[str]:
    """
    Reverse-derive attack_type from a process_node's host_ref string.

    audit-derived host_ref takes the form "audit:<source_file>" (malicious)
    or "audit:<dept>" (benign). For malicious rows we strip the prefix and
    run the resulting filename through the same _stem_for_attack_lookup +
    _ATTACK_TYPE_MAP path that _load_flows / _load_audit use.

    Returns None when the row is benign or the host_ref does not resolve.
    """
    if is_malicious != 1:
        return None
    if not host_ref.startswith("audit:"):
        return None
    file_part = host_ref.split(":", 1)[1]
    stem = _stem_for_attack_lookup(Path(file_part))
    return _ATTACK_TYPE_MAP.get(stem, "other_malicious")


def _incident_groups_for_flows(flows: pd.DataFrame) -> np.ndarray:
    """
    Per-row incident-group identifier for flow_node.

    Malicious rows are keyed by `incident:<attack_type>` so they align with
    audit-derived process rows from the same attack. Benign rows keep a
    `benign:<source_file>` key — by_file behaviour, namespaced to avoid
    accidental collision with incident keys.
    """
    if flows.empty:
        return np.array([], dtype=object)
    is_mal = flows["is_malicious"].astype(int).values
    attack = flows["attack_type"].fillna("unknown").astype(str).values
    source = flows.get("source_file", pd.Series([""] * len(flows))).fillna("unknown").astype(str).values
    out = np.empty(len(flows), dtype=object)
    for i in range(len(flows)):
        if is_mal[i] == 1:
            out[i] = f"{_INCIDENT_PREFIX}{attack[i]}"
        else:
            out[i] = f"{_BENIGN_PREFIX}{source[i]}"
    return out


def _incident_groups_for_processes(processes: pd.DataFrame) -> np.ndarray:
    """
    Per-row incident-group identifier for process_node.

    Audit-derived malicious rows reverse-derive attack_type from host_ref
    so they align with the matching flow_node rows. Audit-derived benign
    rows keep the host_ref as a benign group (one per dept). Procmon-derived
    rows have no attack_type — they stay under a `procmon:` namespace so
    the incident-split path falls back to by_file behaviour for them.
    """
    if processes.empty:
        return np.array([], dtype=object)
    src = processes["source"].astype(str).values
    host_ref = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    out = np.empty(len(processes), dtype=object)
    for i in range(len(processes)):
        if src[i] == "audit":
            attack = _attack_type_from_audit_host_ref(host_ref[i], int(is_mal[i]))
            if attack is not None:
                out[i] = f"{_INCIDENT_PREFIX}{attack}"
            else:
                out[i] = f"{_BENIGN_PREFIX}{host_ref[i]}"
        else:
            out[i] = f"procmon:{host_ref[i]}:mal{is_mal[i]}"
    return out


def _incident_groups_for_measurements(measurements: pd.DataFrame) -> np.ndarray:
    """
    Per-row incident-group identifier for measurement_node.

    Power telemetry lives on a separate SCADA modality with no shared
    filename or attack_type with the network/audit side, so measurements
    do not participate in incident alignment — they stay under a
    `measure:` namespace and fall back to by_file behaviour.
    """
    if measurements.empty:
        return np.array([], dtype=object)
    dev = measurements["device_id"].astype(str).values
    is_mal = measurements["is_malicious"].astype(int).values
    return np.array(
        [f"measure:{dev[i]}:mal{is_mal[i]}" for i in range(len(measurements))],
        dtype=object,
    )


def _build_global_incident_split(
    incident_groups: Dict[str, np.ndarray],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, str]:
    """
    Build a deterministic group → split assignment shared across all node
    types. Only `incident:*` groups are aligned globally; `benign:*` /
    `procmon:*` / `measure:*` groups are assigned per-namespace using the
    same stratified-by-file routine so that within each namespace val/test
    receive a balanced size mix.

    Args:
        incident_groups: {node_type: per-row group array}
        val_ratio:       Fraction of (incident or per-namespace) groups
                         routed to val.
        test_ratio:      Fraction routed to test.
        rng:             Shared numpy Generator.

    Returns:
        {group_key: "train" | "val" | "test"}
    """
    # Aggregate group sizes across all node types (a group's "size" is the
    # union of rows tagged with it; this matches what the size-tier logic
    # in _stratified_split_by_file uses for balancing).
    group_size: Dict[str, int] = {}
    for arr in incident_groups.values():
        if len(arr) == 0:
            continue
        unique, counts = np.unique(arr, return_counts=True)
        for g, c in zip(unique, counts):
            group_size[g] = group_size.get(g, 0) + int(c)

    # Bucket groups by namespace.
    incidents: List[str] = sorted(g for g in group_size if g.startswith(_INCIDENT_PREFIX))
    benigns: List[str] = sorted(g for g in group_size if g.startswith(_BENIGN_PREFIX))
    procmons: List[str] = sorted(g for g in group_size if g.startswith("procmon:"))
    measures: List[str] = sorted(g for g in group_size if g.startswith("measure:"))

    assignment: Dict[str, str] = {}

    def _assign_bucket(groups: List[str]) -> None:
        if not groups:
            return
        # Sort by size descending then partition into 3 tiers so val/test
        # each get a balanced large/medium/small mix (mirrors DD-8 logic).
        groups_sorted = sorted(groups, key=lambda g: -group_size[g])
        n = len(groups_sorted)
        tiers = 3 if n >= 6 else 1
        tier_chunks = np.array_split(groups_sorted, tiers)
        for tier_arr in tier_chunks:
            tier = list(tier_arr)
            rng.shuffle(tier)
            n_tier = len(tier)
            n_test = max(1, int(round(n_tier * test_ratio))) if n_tier >= 2 else 0
            n_val = max(1, int(round(n_tier * val_ratio))) if n_tier >= 3 else 0
            if n_test + n_val >= n_tier:
                n_val = max(0, n_tier - n_test - 1)
            for g in tier[:n_test]:
                assignment[g] = "test"
            for g in tier[n_test:n_test + n_val]:
                assignment[g] = "val"
            for g in tier[n_test + n_val:]:
                assignment[g] = "train"

    _assign_bucket(incidents)
    _assign_bucket(benigns)
    _assign_bucket(procmons)
    _assign_bucket(measures)

    return assignment


def _stratified_split_by_incident(
    labels: np.ndarray,
    groups: np.ndarray,
    incident_split_map: Dict[str, str],
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a precomputed global group → split mapping to one node type.

    Rows whose group is in `incident_split_map` follow the global
    assignment (this is what aligns audit & pcap across modalities).
    Rows whose group is unknown fall back to _stratified_split_by_file
    using the original groups (defensive — should not happen when the
    mapping was built from these same groups).
    """
    n = len(labels)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    if len(groups) != n:
        return _stratified_split(
            labels, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )

    fallback_idx: List[int] = []
    for i in range(n):
        split = incident_split_map.get(groups[i])
        if split == "train":
            train[i] = True
        elif split == "val":
            val[i] = True
        elif split == "test":
            test[i] = True
        else:
            fallback_idx.append(i)

    if fallback_idx:
        sub_labels = labels[fallback_idx]
        sub_groups = groups[fallback_idx]
        sub_train, sub_val, sub_test = _stratified_split_by_file(
            sub_labels, sub_groups,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
        fb = np.array(fallback_idx, dtype=int)
        train[fb[sub_train]] = True
        val[fb[sub_val]] = True
        test[fb[sub_test]] = True

    return train, val, test


def _build_split_masks(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    mode: SplitMode,
    rng: np.random.Generator,
    incident_split_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dispatch to the requested split strategy.

    See DD-8 in unsw_mg24_plan.md for the rationale of each mode.
    """
    if mode == "row":
        return _stratified_split(
            labels, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
    if mode == "by_file":
        return _stratified_split_by_file(
            labels, groups,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
    if mode == "hybrid":
        return _stratified_split_hybrid(
            labels, groups,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
    if mode == "by_incident":
        if incident_split_map is None:
            raise ValueError(
                "split_mode='by_incident' requires a precomputed "
                "incident_split_map (built in to_pyg_hetero_data)."
            )
        return _stratified_split_by_incident(
            labels, groups, incident_split_map,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
    raise ValueError(f"Unknown split_mode: {mode!r}")


def _stratified_split_by_file(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
    size_tiers: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Size-stratified by-file split (DD-8).

    Each unique value in `groups` (source filename) is assigned wholly to
    train, val, or test — preventing same-pcap row leakage across splits.

    To avoid the val/test base-rate skew that pure random by-file shuffling
    causes when file sizes differ by orders of magnitude (MG24 ddos1=223k vs
    samba=118), files within each binary class are first sorted by row count
    and partitioned into `size_tiers` quantile-based tiers. Each tier is
    then shuffled and split 70/15/15 independently. This guarantees val and
    test each receive a balanced mix of large/medium/small files, so their
    malicious-to-benign ratios are comparable.

    Classes with < size_tiers*2 files fall back to single-tier shuffle
    (tiering would leave some tiers without enough files for a 3-way split).

    If `groups` has fewer entries than `labels` (missing source info), the
    affected rows fall back to the row-level _stratified_split.
    """
    n = len(labels)
    if len(groups) != n:
        return _stratified_split(
            labels, val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )

    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    # Each group inherits a single binary label (rounded mean — handles
    # rare mixed cases) and a size in rows.
    unique_groups = np.unique(groups)
    group_label: Dict[str, int] = {}
    group_size: Dict[str, int] = {}
    for g in unique_groups:
        mask = groups == g
        group_label[g] = int(round(float(labels[mask].mean())))
        group_size[g] = int(mask.sum())

    for cls in np.unique(list(group_label.values())):
        files_in_cls = [g for g, y in group_label.items() if y == cls]
        if not files_in_cls:
            continue

        # Sort by size descending; partition into tiers when class is large
        # enough that every tier can support a 3-way split.
        files_in_cls.sort(key=lambda g: -group_size[g])
        n_files = len(files_in_cls)
        actual_tiers = size_tiers if n_files >= size_tiers * 2 else 1
        tier_chunks = np.array_split(files_in_cls, actual_tiers)

        for tier_arr in tier_chunks:
            tier = list(tier_arr)
            rng.shuffle(tier)
            n_tier = len(tier)
            # max(1, ...) guarantees val/test get at least one file when possible.
            n_test = max(1, int(round(n_tier * test_ratio))) if n_tier >= 2 else 0
            n_val = max(1, int(round(n_tier * val_ratio))) if n_tier >= 3 else 0
            # Don't let val+test eat the whole tier.
            if n_test + n_val >= n_tier:
                n_val = max(0, n_tier - n_test - 1)
            test_files = tier[:n_test]
            val_files = tier[n_test:n_test + n_val]
            train_files = tier[n_test + n_val:]
            for f in test_files:
                test[groups == f] = True
            for f in val_files:
                val[groups == f] = True
            for f in train_files:
                train[groups == f] = True

    return train, val, test


def _stratified_split_hybrid(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hybrid split: benign rows split row-wise (assumes stable baseline),
    malicious rows split by-file (independent attack sessions).

    Mirrors a production IDS deployment scenario where the model has seen
    a long tail of normal traffic but must flag a *new* attack pcap.
    """
    n = len(labels)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    # Benign: row-level stratified split (single class within this slice).
    benign_idx = np.flatnonzero(labels == 0)
    if len(benign_idx) > 0:
        sub_train, sub_val, sub_test = _stratified_split(
            np.zeros(len(benign_idx), dtype=int),
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
        train[benign_idx[sub_train]] = True
        val[benign_idx[sub_val]] = True
        test[benign_idx[sub_test]] = True

    # Malicious: by-file split.
    mal_idx = np.flatnonzero(labels == 1)
    if len(mal_idx) > 0 and len(groups) == n:
        mal_groups = groups[mal_idx]
        mal_train, mal_val, mal_test = _stratified_split_by_file(
            np.ones(len(mal_idx), dtype=int),
            mal_groups,
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
        train[mal_idx[mal_train]] = True
        val[mal_idx[mal_val]] = True
        test[mal_idx[mal_test]] = True
    elif len(mal_idx) > 0:
        # Fallback: row-level if no group info.
        sub_train, sub_val, sub_test = _stratified_split(
            np.ones(len(mal_idx), dtype=int),
            val_ratio=val_ratio, test_ratio=test_ratio, rng=rng,
        )
        train[mal_idx[sub_train]] = True
        val[mal_idx[sub_val]] = True
        test[mal_idx[sub_test]] = True

    return train, val, test
