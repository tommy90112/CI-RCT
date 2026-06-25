"""CSV export for traced root-cause chains.

A traced chain record (see ``utils.elliptic_identity.chain_to_record``) is a
nested dict carrying a per-node list.  For spreadsheet / pandas analysis we also
want a flat, one-row-per-chain table.  This module is the single source of truth
for that flattening, shared by ``evaluate.py`` (``--dump_csv``) and
``scripts/chains_json_to_csv.py``.

The full per-hop detail stays in the JSON dump; the CSV keeps a flat summary
plus the path / type / causal-effect sequences encoded with a ``|`` separator so
each chain stays reconstructable from a single row.

All functions are pure: they build new objects and never mutate the input
records.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

# One row per chain.  Order here is the column order written to disk.
CSV_FIELDNAMES: Tuple[str, ...] = (
    "target_txid",
    "depth",
    "root_type",
    "root_real_id",
    "root_is_fraud",
    "is_true_positive",
    "n_nodes",
    "chain_real_ids",
    "chain_types",
    "chain_ce",
)

# Separator for the path / type / CE sequences packed into a single cell.
PATH_SEP = "|"


def _format_ce(node: dict) -> str:
    """CE cell for one node: empty for the target, else the edge CE value."""
    if node.get("is_target"):
        return ""
    return f"{float(node.get('ce', 0.0)):.6g}"


def chain_record_to_row(record: dict) -> Dict[str, object]:
    """Flatten one nested chain record into a single flat CSV row (new dict).

    The ``chain_real_ids`` / ``chain_types`` / ``chain_ce`` columns are aligned
    position-by-position (downstream target → upstream root); the target's CE
    cell is empty because it has no incoming edge in the chain.
    """
    nodes = record.get("nodes", [])
    return {
        "target_txid": record.get("target_txid", ""),
        "depth": record.get("depth", 0),
        "root_type": record.get("root_type", ""),
        "root_real_id": record.get("root_real_id", ""),
        "root_is_fraud": bool(record.get("root_is_fraud", False)),
        "is_true_positive": bool(record.get("is_true_positive", False)),
        "n_nodes": len(nodes),
        "chain_real_ids": PATH_SEP.join(str(n.get("real_id", "")) for n in nodes),
        "chain_types": PATH_SEP.join(str(n.get("type", "")) for n in nodes),
        "chain_ce": PATH_SEP.join(_format_ce(n) for n in nodes),
    }


def chain_records_to_rows(records: List[dict]) -> List[Dict[str, object]]:
    """Return a new list of flat CSV rows, one per chain record."""
    return [chain_record_to_row(r) for r in records]


def write_chains_csv(records: List[dict], path: str) -> int:
    """Write chain records to ``path`` as a one-row-per-chain CSV.

    Creates parent directories as needed.  Returns the number of rows written.
    """
    rows = chain_records_to_rows(records)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
