"""
Windows Procmon CSV Parser for UNSW-MG24.

Parses Procmon (Process Monitor) CSV exports collected from the SCADA central
workstation and the two local inverter controller hosts. Each CSV traces a
single critical microgrid process (ScadaApplication.exe / LVDacEms.exe).

Procmon's CSV columns:
    "Time of Day", "Process Name", "PID", "Operation", "Path", "Result", "Detail"

Operations are categorised into:
    - network   (TCP Send / Receive / Connect / Reconnect / TCPCopy / Retransmit / Disconnect)
    - file      (CreateFile / ReadFile / WriteFile / CloseFile / Query*File / FileSystemControl)
    - registry  (Reg* operations)
    - thread    (Thread Create / Thread Exit)
    - other     (everything else)

For network operations, the Path field has the form
    "<src_host>:<src_port> -> <dst_host>:<dst_port>"
which we split into structured columns. Hosts may be IPv4, IPv6, or hostnames;
IPv6 contains ':' so we always rsplit on the LAST colon to extract the port.

Usage
─────
    from utils.procmon_parser import parse_procmon_csv

    df = parse_procmon_csv(
        "data/unsw_mg24/Malicious system call traces/central_malicious.CSV",
        host_id="central",
    )

Output columns
──────────────
    time              str   (original "HH:MM:SS.fffffff", Procmon's clock)
    time_seconds      float (seconds since midnight; relative ordering only)
    process           str   (Process Name)
    pid               int
    operation         str   (raw Operation)
    op_category       str   ('network' | 'file' | 'registry' | 'thread' | 'other')
    path              str   (raw Path column; structured columns below carry parsed parts)
    result            str
    detail            str
    file_path         str   (only when op_category == 'file', else NaN)
    net_src_host      str   (only when op_category == 'network')
    net_src_port      int   (only when op_category == 'network')
    net_dst_host      str   (only when op_category == 'network')
    net_dst_port      int   (only when op_category == 'network')
    host_id           str   (caller-supplied identifier for the source host)

Reference: unsw_mg24_plan.md § 6.3
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd


# ── Operation categorisation ──────────────────────────────────────────────────

_NETWORK_PREFIXES: Tuple[str, ...] = ("TCP ", "UDP ")
_THREAD_PREFIXES: Tuple[str, ...] = ("Thread ",)
_REGISTRY_PREFIXES: Tuple[str, ...] = ("Reg",)
_FILE_KEYWORDS: Tuple[str, ...] = ("File", "Directory", "Volume")

_OUTPUT_COLUMNS = [
    "time", "time_seconds",
    "process", "pid",
    "operation", "op_category",
    "path", "result", "detail",
    "file_path",
    "net_src_host", "net_src_port", "net_dst_host", "net_dst_port",
    "host_id",
]


# ── Public API ────────────────────────────────────────────────────────────────


def parse_procmon_csv(
    path: Union[str, Path],
    host_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse a Procmon CSV file into a structured DataFrame.

    Args:
        path:    Path to the Procmon CSV.
        host_id: Optional identifier for the source host. When provided, the
                 returned DataFrame has a `host_id` column populated with this
                 value, used downstream for `host -[runs]→ process` edges.

    Returns:
        DataFrame with stable columns (see module docstring). Rows are sorted
        by ascending `time_seconds`.
    """
    path = Path(path)

    # Procmon CSV files are UTF-8 with BOM and may contain stray quotes; let
    # pandas handle encoding detection but strip the BOM if present.
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Normalise column names; Procmon's header includes spaces.
    raw = raw.rename(
        columns={
            "Time of Day": "time",
            "Process Name": "process",
            "PID": "pid",
            "Operation": "operation",
            "Path": "path",
            "Result": "result",
            "Detail": "detail",
        }
    )

    # Some columns may be missing on edge cases — ensure presence.
    for col in ("time", "process", "pid", "operation", "path", "result", "detail"):
        if col not in raw.columns:
            raw[col] = pd.NA

    raw["time_seconds"] = raw["time"].apply(_time_of_day_to_seconds)
    raw["op_category"] = raw["operation"].apply(_categorise_operation)

    # Network parsing: only fill src/dst columns for network ops, NaN otherwise.
    net_parsed = raw.apply(_parse_network_path, axis=1)
    raw[["net_src_host", "net_src_port", "net_dst_host", "net_dst_port"]] = net_parsed

    raw["file_path"] = raw.apply(
        lambda r: r["path"] if r["op_category"] == "file" else pd.NA,
        axis=1,
    )
    raw["host_id"] = host_id if host_id is not None else pd.NA

    df = raw[_OUTPUT_COLUMNS].copy()
    df = df.sort_values("time_seconds", kind="stable").reset_index(drop=True)
    return df


# ── Internal helpers ──────────────────────────────────────────────────────────


def _time_of_day_to_seconds(time_str: object) -> float:
    """Convert "HH:MM:SS.fffffff" to seconds since midnight (float)."""
    if not isinstance(time_str, str) or not time_str:
        return float("nan")
    try:
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return float("nan")


def _categorise_operation(op: object) -> str:
    """Map Procmon's Operation string to one of 5 categories."""
    if not isinstance(op, str):
        return "other"
    if op.startswith(_NETWORK_PREFIXES):
        return "network"
    if op.startswith(_THREAD_PREFIXES):
        return "thread"
    if op.startswith(_REGISTRY_PREFIXES):
        return "registry"
    if any(kw in op for kw in _FILE_KEYWORDS):
        return "file"
    return "other"


def _parse_network_path(row: pd.Series) -> pd.Series:
    """
    Parse `<src_host>:<src_port> -> <dst_host>:<dst_port>` from the Path field.

    Hosts may be IPv4, IPv6, or hostnames. We rsplit on the LAST colon to
    locate the port boundary, since IPv6 addresses contain colons internally.

    Returns a Series of (src_host, src_port, dst_host, dst_port). All four
    are NaN for non-network rows or unparseable paths.
    """
    if row["op_category"] != "network":
        return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])

    path = row["path"]
    if not isinstance(path, str) or " -> " not in path:
        return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])

    try:
        left, right = path.split(" -> ", 1)
        src_host, src_port = _split_host_port(left)
        dst_host, dst_port = _split_host_port(right)
        return pd.Series([src_host, src_port, dst_host, dst_port])
    except ValueError:
        return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])


def _split_host_port(endpoint: str) -> Tuple[str, Optional[int]]:
    """
    Split an "<host>:<port>" string at the LAST colon.

    Handles IPv6 addresses correctly (which contain internal colons).
    Returns (host, port) — port is None if the trailing token is non-numeric.
    """
    host, _, port_str = endpoint.rpartition(":")
    if not host:
        # No colon found — treat the entire string as the host.
        return endpoint, None
    try:
        return host, int(port_str)
    except ValueError:
        # Not a numeric port; keep the whole endpoint as host.
        return endpoint, None
