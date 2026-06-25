#!/usr/bin/env python3
"""Convert a crime-chain JSON dump into a flat one-row-per-chain CSV.

The JSON is the output of ``evaluate.py --dump_chains`` (or
``scripts/export_crime_chains.py``).  This converter lets you produce the CSV
from an existing dump without re-running evaluation.

Usage:
    python scripts/chains_json_to_csv.py viz/crime_chains.json
    python scripts/chains_json_to_csv.py in.json -o out.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.chain_export import write_chains_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("json_path", help="Path to the crime-chain JSON dump.")
    parser.add_argument("-o", "--out", default=None,
                        help="Output CSV path (default: input path with .csv).")
    args = parser.parse_args()

    if not os.path.isfile(args.json_path):
        parser.error(f"input not found: {args.json_path}")
    with open(args.json_path) as f:
        payload = json.load(f)

    records = payload.get("chains")
    if not isinstance(records, list):
        parser.error("malformed dump: expected a top-level 'chains' list.")

    out = args.out or os.path.splitext(args.json_path)[0] + ".csv"
    n = write_chains_csv(records, out)
    print(f"wrote {n} chains → {out}")


if __name__ == "__main__":
    main()
