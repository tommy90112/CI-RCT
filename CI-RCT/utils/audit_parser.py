"""
Linux auditd Log Parser for UNSW-MG24.

Parses Linux auditd log files into a structured DataFrame, retaining only
the audit record types relevant to CI-RCT's heterogeneous graph construction:

    - SYSCALL    process behaviour (pid, ppid, uid, comm, exe, syscall_num)
    - PATH       file path arguments (companion to file-related syscalls)
    - SOCKETCALL network arguments (companion to socket-related syscalls)

Records sharing the same audit serial number are merged into a single event
row, since SYSCALL + PATH or SYSCALL + SOCKETCALL are emitted as separate
lines describing one logical event.

Usage
─────
    from utils.audit_parser import parse_audit_log, parse_audit_dir

    # Single file
    df = parse_audit_log("data/unsw_mg24/Malicious system call traces/audit_dos1.log")

    # Whole directory (e.g. all malicious logs)
    df = parse_audit_dir("data/unsw_mg24/Malicious system call traces")
    # df has an extra `source_file` column for downstream labelling

Output columns
──────────────
    timestamp     float (unix seconds, ms precision)
    serial        int   (audit event serial number)
    pid, ppid     int   (process / parent process IDs)
    uid, euid     int   (real / effective user IDs)
    comm          str   (process short name)
    exe           str   (process executable path)
    syscall       int   (Linux syscall number)
    success       bool  (syscall return success flag)
    exit_code     int   (syscall return value)
    path          str   (file path from PATH record, NaN if absent)
    socket_nargs  int   (socket call arg count, NaN if not a SOCKETCALL)
    socket_a0..a3 str   (socket call args, NaN if not a SOCKETCALL)
    source_file   str   (filename, only when using parse_audit_dir)

Reference: unsw_mg24_plan.md § 6.2
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


# ── Regex patterns (compiled once at module load) ─────────────────────────────

# Header: type=<TYPE> msg=audit(<unix_ts>:<serial>): <body>
_AUDIT_HEADER_RE = re.compile(
    r"^type=(\w+)\s+msg=audit\(([\d.]+):(\d+)\):\s*(.*)$"
)

# key=value pairs in the body. value is one of:
#   "quoted string"     → captured in group 2 (inner content)
#   bare token          → captured in group 1 (whole match)
# group 0 = key
_KV_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')

_DEFAULT_TYPES: Tuple[str, ...] = ("SYSCALL", "PATH", "SOCKETCALL")

# Output column order — kept stable for downstream loader contracts.
_OUTPUT_COLUMNS: List[str] = [
    "timestamp", "serial",
    "pid", "ppid", "uid", "euid",
    "comm", "exe",
    "syscall", "success", "exit_code",
    "path",
    "socket_nargs", "socket_a0", "socket_a1", "socket_a2", "socket_a3",
]


# ── Public API ────────────────────────────────────────────────────────────────


def parse_audit_log(
    path: Union[str, Path],
    types_to_keep: Iterable[str] = _DEFAULT_TYPES,
) -> pd.DataFrame:
    """
    Parse a single auditd log file.

    Args:
        path:          File path to the audit log.
        types_to_keep: Audit record types to retain. Other types are silently
                       skipped. Default: ("SYSCALL", "PATH", "SOCKETCALL").

    Returns:
        DataFrame with one row per merged audit event (keyed by serial number).
        Columns are stable (see module docstring); missing fields are NaN.
        Returned events are sorted by ascending timestamp.
    """
    types_set = set(types_to_keep)
    events: Dict[int, Dict[str, Any]] = {}

    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_line(line, types_set)
            if parsed is None:
                continue
            record_type, timestamp, serial, kv = parsed
            event = events.setdefault(
                serial,
                {"timestamp": timestamp, "serial": serial},
            )
            _merge_record_into_event(event, record_type, kv)

    if not events:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = pd.DataFrame(events.values())
    # Ensure stable column ordering even if some columns are absent.
    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_OUTPUT_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    return df


def parse_audit_dir(
    directory: Union[str, Path],
    types_to_keep: Iterable[str] = _DEFAULT_TYPES,
    pattern: str = "*.log",
) -> pd.DataFrame:
    """
    Parse every audit log file in a directory, concatenating results.

    Args:
        directory:     Directory containing audit log files.
        types_to_keep: Audit record types to retain (see parse_audit_log).
        pattern:       Glob pattern for matching log files. Default "*.log".

    Returns:
        Concatenated DataFrame across all matching files. An additional
        `source_file` column records the file each event came from, which is
        the basis for file-level attack-type labelling in the loader.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Audit directory not found: {directory}")

    frames: List[pd.DataFrame] = []
    for log_path in sorted(directory.glob(pattern)):
        df = parse_audit_log(log_path, types_to_keep=types_to_keep)
        if df.empty:
            continue
        df["source_file"] = log_path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS + ["source_file"])

    return pd.concat(frames, ignore_index=True)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_line(
    line: str,
    types_set: set,
) -> Optional[Tuple[str, float, int, Dict[str, str]]]:
    """
    Parse a single audit log line into (record_type, timestamp, serial, kv).

    Returns None if the line is malformed or its type is filtered out.
    """
    m = _AUDIT_HEADER_RE.match(line)
    if not m:
        return None

    record_type = m.group(1)
    if record_type not in types_set:
        return None

    try:
        timestamp = float(m.group(2))
        serial = int(m.group(3))
    except ValueError:
        return None

    body = m.group(4)
    kv: Dict[str, str] = {}
    for key, quoted, bare in _KV_RE.findall(body):
        # quoted captures inner content; bare captures the whole token.
        kv[key] = quoted if quoted else bare

    return record_type, timestamp, serial, kv


def _merge_record_into_event(
    event: Dict[str, Any],
    record_type: str,
    kv: Dict[str, str],
) -> None:
    """Mutate `event` by merging fields from a record of the given type."""
    if record_type == "SYSCALL":
        event["pid"] = _to_int(kv.get("pid"))
        event["ppid"] = _to_int(kv.get("ppid"))
        event["uid"] = _to_int(kv.get("uid"))
        event["euid"] = _to_int(kv.get("euid"))
        event["comm"] = kv.get("comm", "")
        event["exe"] = kv.get("exe", "")
        event["syscall"] = _to_int(kv.get("syscall"))
        event["success"] = kv.get("success") == "yes"
        event["exit_code"] = _to_int(kv.get("exit"))
    elif record_type == "PATH":
        # PATH records can repeat (item=0, item=1, ...). We keep the first
        # one, which is conventionally the primary target file.
        if "path" not in event:
            event["path"] = kv.get("name", "")
    elif record_type == "SOCKETCALL":
        event["socket_nargs"] = _to_int(kv.get("nargs"))
        event["socket_a0"] = kv.get("a0")
        event["socket_a1"] = kv.get("a1")
        event["socket_a2"] = kv.get("a2")
        event["socket_a3"] = kv.get("a3")


def _to_int(s: Optional[str]) -> Optional[int]:
    """Best-effort int conversion. Returns None for invalid inputs."""
    if s is None:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None
