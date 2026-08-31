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

# One row per chain.  Order here is the column order written to disk.  Optional
# φ columns (CSV_PHI_FIELDNAMES) are appended only when the records carry φ, so
# a no-φ dump stays byte-identical to before.
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

# Per-node φ sequences (added by utils.chain_phi). Each maps a node-record key to
# the flat CSV column name carrying that node's '|'-joined values.
CSV_PHI_FIELDNAMES: Tuple[Tuple[str, str], ...] = (
    ("phi_add", "chain_phi_add"),
    ("phi_asym", "chain_phi_asym"),
)

# Separator for the path / type / CE / φ sequences packed into a single cell.
PATH_SEP = "|"


def _format_value(node: dict, key: str) -> str:
    """One cell for ``node[key]``: empty for the target or a missing/None value."""
    if node.get("is_target") or node.get(key) is None:
        return ""
    return f"{float(node[key]):.6g}"


def _present_phi_keys(records: List[dict]) -> List[Tuple[str, str]]:
    """φ (node_key, column) pairs that have at least one real value to write."""
    return [
        (node_key, col)
        for node_key, col in CSV_PHI_FIELDNAMES
        if any(
            (not n.get("is_target")) and n.get(node_key) is not None
            for r in records for n in r.get("nodes", [])
        )
    ]


def chain_record_to_row(
    record: dict, phi_keys: List[Tuple[str, str]] = (),
) -> Dict[str, object]:
    """Flatten one nested chain record into a single flat CSV row (new dict).

    The ``chain_real_ids`` / ``chain_types`` / ``chain_ce`` (and any φ) columns
    are aligned position-by-position (downstream target → upstream root); the
    target's edge-valued cells are empty because it has no incoming edge.
    """
    nodes = record.get("nodes", [])
    row = {
        "target_txid": record.get("target_txid", ""),
        "depth": record.get("depth", 0),
        "root_type": record.get("root_type", ""),
        "root_real_id": record.get("root_real_id", ""),
        "root_is_fraud": bool(record.get("root_is_fraud", False)),
        "is_true_positive": bool(record.get("is_true_positive", False)),
        "n_nodes": len(nodes),
        "chain_real_ids": PATH_SEP.join(str(n.get("real_id", "")) for n in nodes),
        "chain_types": PATH_SEP.join(str(n.get("type", "")) for n in nodes),
        "chain_ce": PATH_SEP.join(_format_value(n, "ce") for n in nodes),
    }
    for node_key, col in phi_keys:
        row[col] = PATH_SEP.join(_format_value(n, node_key) for n in nodes)
    return row


def chain_records_to_rows(records: List[dict]) -> List[Dict[str, object]]:
    """Return a new list of flat CSV rows, one per chain record."""
    phi_keys = _present_phi_keys(records)
    return [chain_record_to_row(r, phi_keys) for r in records]


def csv_fieldnames(records: List[dict]) -> List[str]:
    """Column order for ``records`` (base columns + any present φ columns)."""
    return list(CSV_FIELDNAMES) + [col for _, col in _present_phi_keys(records)]


def write_chains_csv(records: List[dict], path: str) -> int:
    """Write chain records to ``path`` as a one-row-per-chain CSV.

    Creates parent directories as needed.  Returns the number of rows written.
    """
    rows = chain_records_to_rows(records)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames(records))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
