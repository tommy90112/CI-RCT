#!/usr/bin/env python3
"""
Parse tracer-ablation eval logs into one comparison table.

Reads logs/elliptic/ablation/eval_<arm>.log (or a dir given on argv) and pulls
Metric B (root-cause tracing), the φ-stability sanity metric (D), and the
wall-clock line written by run_tracer_ablation.sh, then prints a Markdown table
ordered by the PRIMARY judge RCP-TruePos (root-cause precision on truly-illicit
targets — the cleanest tracer-quality metric; see tracer_ablation_plan.md §5).

Usage:
    python scripts/parse_tracer_ablation.py [logdir]
"""
import glob
import os
import re
import sys

# label in log  ->  short key
FIELDS = {
    "Root Cause Precision True Pos": "RCP_TP",   # PRIMARY
    "Root Cause Precision": "RCP",
    "Root Cause Hit Rate": "RHR",
    "Chain Validity": "CCV",
    "Mean Tracing Depth": "MTD",
    "Num True Pos Traced": "nTP",
    "Num Traced": "nTraced",
    "Phi Stability Std": "phiStd",
}
# Longer labels must be matched before their prefixes ("...True Pos" before "...Precision").
_ORDER = sorted(FIELDS, key=len, reverse=True)


def parse_log(path):
    # The trailing `\s*:` anchor makes "Root Cause Precision" match only its own
    # line and NOT "Root Cause Precision True Pos" (which has " True Pos" before
    # the colon), so the two map to distinct keys without any guard.
    out = {}
    text = open(path, encoding="utf-8", errors="replace").read()
    for label in _ORDER:
        m = re.search(rf"^\s*{re.escape(label)}\s*:\s*([-\d.]+)", text, re.MULTILINE)
        if m:
            out[FIELDS[label]] = float(m.group(1))
    m = re.search(r"WALLCLOCK_SECONDS=(\d+)", text)
    if m:
        out["sec"] = int(m.group(1))
    return out


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "logs/elliptic/ablation"
    logs = sorted(glob.glob(os.path.join(logdir, "eval_*.log")))
    if not logs:
        print(f"no eval_*.log under {logdir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in logs:
        arm = re.sub(r"^eval_|\.log$", "", os.path.basename(p))
        d = parse_log(p)
        d["arm"] = arm
        rows.append(d)

    # order by PRIMARY judge RCP_TP desc (missing -> -1)
    rows.sort(key=lambda r: r.get("RCP_TP", -1), reverse=True)

    cols = [("arm", "演算法臂", "{}"),
            ("RCP_TP", "RCP-TruePos★", "{:.4f}"),
            ("RCP", "RCP", "{:.4f}"),
            ("CCV", "CCV", "{:.4f}"),
            ("RHR", "RHR", "{:.4f}"),
            ("MTD", "MTD", "{:.3f}"),
            ("phiStd", "φ-std", "{:.4f}"),
            ("sec", "wall(s)", "{}")]

    def fmt(r, key, spec):
        v = r.get(key)
        return spec.format(v) if v is not None else "—"

    header = " | ".join(h for _, h, _ in cols)
    sep = " | ".join("---" for _ in cols)
    print(f"| {header} |")
    print(f"| {sep} |")
    for r in rows:
        line = " | ".join(fmt(r, k, s) for k, _, s in cols)
        print(f"| {line} |")

    print("\n★ 主判準 = RCP-TruePos(只算真正 illicit 的 target,排除偵測誤報)。")
    print("φ-std 各臂應幾乎一致(只跟 Shapley 有關、與搜尋無關),若差異大代表實作有 bug。")
    print("MTD 為描述量(鏈深),不排名;RHR/CCV 受鏈長度影響,需對照 MTD 一起看。")


if __name__ == "__main__":
    main()
