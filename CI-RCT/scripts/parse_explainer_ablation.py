#!/usr/bin/env python3
"""
Parse explainer-ablation eval logs into one Metric C comparison table.

Reads logs/elliptic/explainer_ablation/eval_<explainer>.log and pulls Metric C
(Explanation Accuracy / Recall) for BOTH ground-truth variants (LFPN-Strict and
LFPN-Extended), plus Metric B RCP-TruePos for context and the wall-clock line.
Metric C is the primary explainability judge; see tracer_ablation_writeup.md §1.

Usage: python scripts/parse_explainer_ablation.py [logdir]
"""
import glob
import os
import re
import sys


def parse_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    # Metric C "Explanation Accuracy/Recall" appear twice: 1st = LFPN-Strict,
    # 2nd = LFPN-Extended. Grab both in order.
    ea = re.findall(r"Explanation Accuracy\s*:\s*([-\d.]+)", text)
    er = re.findall(r"Explanation Recall\s*:\s*([-\d.]+)", text)
    rcp_tp = re.search(r"Root Cause Precision True Pos\s*:\s*([-\d.]+)", text)
    sec = re.search(r"WALLCLOCK_SECONDS=(\d+)", text)
    f = lambda m, i=0, default=None: float(m[i]) if (m and len(m) > i) else default
    return {
        "EA_strict": f(ea, 0), "ER_strict": f(er, 0),
        "EA_ext": f(ea, 1), "ER_ext": f(er, 1),
        "RCP_TP": float(rcp_tp.group(1)) if rcp_tp else None,
        "sec": int(sec.group(1)) if sec else None,
    }


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "logs/elliptic/explainer_ablation"
    logs = sorted(glob.glob(os.path.join(logdir, "eval_*.log")))
    if not logs:
        print(f"no eval_*.log under {logdir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in logs:
        d = parse_log(p)
        d["explainer"] = re.sub(r"^eval_|\.log$", "", os.path.basename(p))
        rows.append(d)
    # primary judge: LFPN-Strict explanation recall (coverage of the GT), desc
    rows.sort(key=lambda r: (r.get("ER_strict") or -1), reverse=True)

    cols = [("explainer", "Explainer", "{}"),
            ("EA_strict", "EA(strict)", "{:.4f}"),
            ("ER_strict", "ER(strict)★", "{:.4f}"),
            ("EA_ext", "EA(ext)", "{:.4f}"),
            ("ER_ext", "ER(ext)", "{:.4f}"),
            ("RCP_TP", "RCP-TP", "{:.4f}"),
            ("sec", "wall(s)", "{}")]
    fmt = lambda r, k, s: (s.format(r[k]) if r.get(k) is not None else "—")
    print("| " + " | ".join(h for _, h, _ in cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join(fmt(r, k, s) for k, _, s in cols) + " |")

    print("\n★ 主判準 = LFPN-Strict 解釋召回(ER);EA=精度、ER=覆蓋。")
    print("phi_asym=主方法;對照 saliency(無因果介入)、ce_only(無 Shapley)、phi_sym(無時序不對稱);cxgnn_ncm=外部 SOTA(比較非消融)。")
    print("註:各 explainer 輸出的節點集大小不同 → 公平比較建議再補 matched-top-k / PR 曲線(見 writeup §B)。")


if __name__ == "__main__":
    main()
