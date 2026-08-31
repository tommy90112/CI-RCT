#!/usr/bin/env python3
"""
Aggregate Elliptic++ evaluate.py logs into one per-experiment results table.

Scans a directory of eval logs (default ``logs/elliptic``) and pulls, for EACH
log, the full set of thesis metrics across all four evaluation dimensions:

    A. Classification   wallet-F1, tx-F1, pooled fraud-F1, AUC
    B. Root cause        RCP, RCP-TruePos, Hit-rate, Chain-Validity, Mean-Depth
    C. Explanation       EA / ER / GT-Match  (one triple per Metric-C GT block)
    D. φ-stability       φ-std (sanity)

Output (both written to disk AND printed):
    <outdir>/results_summary.csv   — one row per log, every column (paper-ready)
    <outdir>/results_summary.md    — same as a Markdown table to paste in the thesis

This parser is PURE stdlib (no torch / pandas) so it runs on the laptop even
though train/eval must run on the server.

Usage (from the CI-RCT/ directory):
    python scripts/parse_elliptic_results.py                 # logs/elliptic/*.log
    python scripts/parse_elliptic_results.py logs/elliptic/ablation
    python scripts/parse_elliptic_results.py logs/elliptic --glob 'eval_*.log'
    python scripts/parse_elliptic_results.py logs/elliptic --out results/table

The companion ``parse_tracer_ablation.py`` stays focused on the Metric-B-only
tracer-algorithm ablation; this one is the wide per-experiment summary.
"""
import argparse
import csv
import glob
import os
import re

# ── print_section metrics: log label -> short column key ────────────────────────
# print_section() emits "  {key.replace('_',' ').title():40s}: {value:.4f}", so a
# dict key like ``root_cause_precision_true_pos`` shows up as the label below.
# Longer labels MUST be probed before their prefixes ("...True Pos" before
# "...Precision") — the trailing ``:`` anchor in the regex then keeps the short
# label from also matching the long line.
SECTION_FIELDS = {
    "Auc": "AUC",
    "Root Cause Precision True Pos": "RCP_TP",   # PRIMARY tracer judge
    "Root Cause Precision": "RCP",
    "Root Cause Hit Rate": "Hit",
    "Chain Validity": "ChainVal",
    "Mean Tracing Depth": "MTD",
    "Num Traced": "nTraced",
    "Num True Pos Traced": "nTP",
    "Phi Stability Std": "phiStd",
}
_SECTION_ORDER = sorted(SECTION_FIELDS, key=len, reverse=True)

# Per-type classification line, e.g.
#     · wallet      fraud_f1=0.7039  pred_rate=0.0123  thr=0.310
_PER_TYPE_RE = re.compile(
    r"·\s*(?P<type>\S+)\s+fraud_f1=(?P<f1>[-\d.]+)\s+"
    r"pred_rate=(?P<rate>[-\d.]+)\s+thr=(?P<thr>[-\d.]+)"
)
# Pooled headline (joint) / single-head headline (transaction variant).
_HEADLINE_RE = re.compile(r"Headline F1-score \(fraud class\)\s*=\s*([-\d.]+)")
# Metric-C block boundary: "[Metric C — GT = lfpn_forward]".
_METRIC_C_RE = re.compile(r"\[Metric C\s*[—-]\s*GT\s*=\s*([^\]]+)\]")
# EA / ER / GT inside a Metric-C block (print_section labels).
_EXPL_FIELDS = {
    "Explanation Accuracy": "EA",
    "Explanation Recall": "ER",
    "Gt Match Accuracy": "GT",
}


def _find_section_value(text, label):
    """Value of a print_section line ``  {label} : {float}`` or None."""
    m = re.search(rf"^\s*{re.escape(label)}\s*:\s*([-\d.]+)\s*$", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _parse_metric_c(text):
    """Return [{'gt': label, 'EA': .., 'ER': .., 'GT': ..}] per Metric-C block.

    The text after one ``[Metric C — GT = X]`` marker (up to the next marker)
    holds that block's EA/ER/GT print_section lines.
    """
    blocks = []
    matches = list(_METRIC_C_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end]
        row = {"gt": m.group(1).strip()}
        for label, key in _EXPL_FIELDS.items():
            v = _find_section_value(segment, label)
            if v is not None:
                row[key] = v
        blocks.append(row)
    return blocks


def parse_log(path):
    """Extract every metric from one eval log into a flat-ish dict."""
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}

    # A. per-type F1 (wallet / transaction) + pooled headline + AUC.
    for m in _PER_TYPE_RE.finditer(text):
        out[f"{m.group('type')}_F1"] = float(m.group("f1"))
    hm = _HEADLINE_RE.search(text)
    if hm:
        out["pooled"] = float(hm.group(1))

    # A/B/D print_section scalars (probe long labels first).
    for label in _SECTION_ORDER:
        v = _find_section_value(text, label)
        if v is not None:
            out[SECTION_FIELDS[label]] = v

    # C. explanation quality — one triple per GT block.
    out["metric_c"] = _parse_metric_c(text)
    return out


def _config_name(path):
    """Filename without eval_/train_ prefix or .log suffix."""
    base = os.path.basename(path)
    return re.sub(r"^(eval_|train_)|\.log$", "", base)


# Main wide table columns (label, dict-key, format). Metric C handled separately.
_COLS = [
    ("config", "config", "{}"),
    ("wallet-F1", "wallet_F1", "{:.4f}"),
    ("tx-F1", "transaction_F1", "{:.4f}"),
    ("pooled", "pooled", "{:.4f}"),
    ("AUC", "AUC", "{:.4f}"),
    ("RCP", "RCP", "{:.4f}"),
    ("RCP-TP★", "RCP_TP", "{:.4f}"),
    ("Hit", "Hit", "{:.4f}"),
    ("ChainVal", "ChainVal", "{:.4f}"),
    ("MTD", "MTD", "{:.3f}"),
    ("φ-std", "phiStd", "{:.4f}"),
]
# Metric-C triple appended as one combined "EA/ER/GT" column (first GT block,
# with the block label shown so multi-mode --lfpn_mode both stays unambiguous).


def _fmt(row, key, spec):
    v = row.get(key)
    return spec.format(v) if v is not None else "—"


def _metric_c_cell(row):
    blocks = row.get("metric_c") or []
    if not blocks:
        return "—"
    b = blocks[0]
    triple = "/".join(
        f"{b[k]:.3f}" if k in b else "—" for k in ("EA", "ER", "GT")
    )
    label = b.get("gt", "")
    suffix = f" ({label})" if len(blocks) == 1 else f" (+{len(blocks)-1} more)"
    return triple + suffix


def build_rows(logs):
    rows = []
    for p in logs:
        d = parse_log(p)
        d["config"] = _config_name(p)
        rows.append(d)
    # Order by the primary tracer judge RCP-TP desc (missing -> -1).
    rows.sort(key=lambda r: r.get("RCP_TP", -1), reverse=True)
    return rows


def write_markdown(rows, path):
    headers = [h for h, _, _ in _COLS] + ["EA/ER/GT"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        cells = [_fmt(r, k, s) for _, k, s in _COLS] + [_metric_c_cell(r)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("★ RCP-TruePos = 只算真正 illicit target 的根因精度（主判準）。")
    lines.append("EA/ER/GT = Metric C 解釋品質（accuracy / recall / gt-match），括號為 GT 模式。")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def write_csv(rows, path):
    # Flatten: fixed columns + EA/ER/GT for EVERY Metric-C block (gt-suffixed).
    gt_labels = []
    for r in rows:
        for b in r.get("metric_c") or []:
            if b["gt"] not in gt_labels:
                gt_labels.append(b["gt"])
    fixed = [k for _, k, _ in _COLS]
    expl_cols = [f"{m}_{g}" for g in gt_labels for m in ("EA", "ER", "GT")]
    fieldnames = fixed + expl_cols
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k) for k in fixed}
            for b in r.get("metric_c") or []:
                for m in ("EA", "ER", "GT"):
                    if m in b:
                        flat[f"{m}_{b['gt']}"] = b[m]
            w.writerow(flat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdir", nargs="?", default="logs/elliptic",
                    help="directory of eval logs (default: logs/elliptic)")
    ap.add_argument("--glob", default="*.log",
                    help="glob within logdir (default: *.log)")
    ap.add_argument("--out", default=None,
                    help="output path prefix (default: <logdir>/results_summary)")
    args = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(args.logdir, args.glob)))
    if not logs:
        ap.error(f"no logs matching {args.glob!r} under {args.logdir}")

    rows = build_rows(logs)
    out_prefix = args.out or os.path.join(args.logdir, "results_summary")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path, md_path = out_prefix + ".csv", out_prefix + ".md"

    write_csv(rows, csv_path)
    md = write_markdown(rows, md_path)

    print(md)
    print(f"[parsed {len(rows)} log(s)]")
    print(f"  CSV → {csv_path}")
    print(f"  MD  → {md_path}")


if __name__ == "__main__":
    main()
