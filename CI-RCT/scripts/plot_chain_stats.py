#!/usr/bin/env python3
"""繪製追溯鏈統計圖（圖 4.X 深度分布、圖 4.Y 根因節點型別分布）。

資料來源為 evaluate.py 匯出之逐鏈 CSV（results/cc_fix_*_marginal.csv），
即 RootCauseTracer 於評估階段的輸出，而非原始資料集之統計。三檔分別對應
transaction / wallet / joint 三個訓練變體，鏈數（679 / 1737 / 2000）與平均
深度（1.6186 / 2.5492 / 1.7780）與論文表 4.6 一致。

用法：
    cd CI-RCT
    python scripts/plot_chain_stats.py            # 輸出至 figures/
    python scripts/plot_chain_stats.py --fmt pdf  # 另存向量版（png/pdf/svg）
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 路徑（以本檔位置推導專案根目錄，可攜）─────────────────────
ROOT = Path(__file__).resolve().parent.parent   # .../CI-RCT
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

# ── CJK 字型 ────────────────────────────────────────────────
matplotlib.rcParams["font.sans-serif"] = [
    "Heiti TC", "PingFang TC", "Arial Unicode MS", "STHeiti",
    "Noto Sans CJK TC", "Microsoft JhengHei", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 設計 token（dataviz reference palette, light；配色已通過
#    validate_palette.js：分類 CVD ΔE 96.7 ≫ 12 門檻）──────────
SURFACE = "#fcfcfb"
INK = "#000000"
INK_2 = "#000000"
MUTED = "#898781"
GRID = "#e1e0d9"       # 淺灰 hairline 格線
BASELINE = "#c3c2b7"   # 淺灰座標軸/刻度
# Okabe–Ito 色盲友善標準色（blue / vermillion / bluish-green），已過 validate_palette.js
BLUE, ORANGE, VIOLET = "#0072b2", "#d55e00", "#009e73"

# 變體 → (逐鏈 CSV, 顏色)
VARIANTS = [
    ("transaction", RESULTS / "cc_fix_transaction_marginal.csv", BLUE),
    ("wallet", RESULTS / "cc_fix_wallet_marginal.csv", ORANGE),
    ("joint", RESULTS / "cc_fix_marginal.csv", VIOLET),
]

BUCKETS = list(range(10)) + [10]          # 0..9，10 代表 "10+"
BUCKET_LABELS = [str(i) for i in range(10)] + ["10+"]


def load_data():
    """讀三個逐鏈 CSV，回傳每變體的深度、根因型別統計。"""
    data = {}
    for name, path, color in VARIANTS:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到逐鏈 CSV：{path}\n"
                f"請先執行 evaluate.py（--variant {name}）匯出 cc_fix_*_marginal.csv。"
            )
        rows = list(csv.DictReader(open(path)))
        if not rows:
            raise ValueError(f"逐鏈 CSV 為空：{path}")
        depths = [int(r["depth"]) for r in rows]
        rtypes = Counter(r["root_type"] for r in rows)
        data[name] = dict(
            color=color, n=len(rows), depths=depths,
            mean=sum(depths) / len(depths), rtypes=rtypes,
        )
    return data


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def _bucket_pct(depths):
    n = len(depths)
    c = Counter(min(d, 10) for d in depths)
    return [100.0 * c.get(b, 0) / n for b in BUCKETS]


def plot_depth_dist(data, out_path):
    """圖 4.X — 追溯鏈深度分布（三變體並列，佔比正規化）。"""
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)

    x = range(len(BUCKETS))
    w = 0.26
    offsets = [-w - 0.03, 0.0, w + 0.03]
    for (name, _, _), off in zip(VARIANTS, offsets):
        d = data[name]
        ax.bar([i + off for i in x], _bucket_pct(d["depths"]), width=w,
               color=d["color"], edgecolor="none",
               label=f"{name}", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Traceability Chain Depth（hops）", color=INK_2, fontsize=11, labelpad=8)
    ax.set_ylabel("Proportion of Traceability Chains（%）", color=INK_2, fontsize=11, labelpad=8)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=True, fontsize=10, loc="upper right", labelcolor=INK_2,
                    facecolor=SURFACE, edgecolor=BASELINE, framealpha=1.0,
                    borderpad=0.8, labelspacing=0.7)
    leg.get_frame().set_linewidth(1.0)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_root_type_dist(data, out_path):
    """圖 4.Y — 根因節點型別分布（100% 堆疊，錢包 vs 交易）。"""
    order = ["joint", "wallet", "transaction"]   # 由下而上；錢包占比遞減
    fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax)

    ys = range(len(order))
    for y, name in zip(ys, order):
        d = data[name]
        n = d["n"]
        w_pct = 100.0 * d["rtypes"].get("wallet", 0) / n
        t_pct = 100.0 * d["rtypes"].get("transaction", 0) / n
        ax.barh(y, w_pct, color=BLUE, edgecolor="none", zorder=3)
        ax.barh(y, t_pct, left=w_pct, color=ORANGE, edgecolor="none", zorder=3)
        ax.text(w_pct / 2, y, f"Wallet\n{w_pct:.1f}%", ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
        ax.text(w_pct + t_pct / 2, y, f"Transaction\n{t_pct:.1f}%", ha="center",
                va="center", color="white", fontsize=10.5, fontweight="bold")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{n}" for n in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Root Node Type Proportion of Traceability Chain（%）", color=INK_2, fontsize=11, labelpad=8)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=BLUE, label="Type: Wallet"),
               Patch(facecolor=ORANGE, label="Type: Transaction")]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right",
              bbox_to_anchor=(1.0, -0.42), ncol=2, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def dump_values(data, out_path):
    """輸出底層計算數值供覆核。"""
    with open(out_path, "w") as f:
        for name, _, _ in VARIANTS:
            d = data[name]
            c = Counter(min(x, 10) for x in d["depths"])
            f.write(f"[{name}] n={d['n']} mean_depth={d['mean']:.4f}\n")
            f.write(f"  root_type: wallet={d['rtypes'].get('wallet', 0)} "
                    f"transaction={d['rtypes'].get('transaction', 0)}\n")
            f.write(f"  depth buckets 0..10+: {[c.get(b, 0) for b in BUCKETS]}\n\n")


def main():
    ap = argparse.ArgumentParser(description="繪製追溯鏈統計圖（圖 4.X / 4.Y）")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"],
                    help="輸出圖檔格式（預設 png；LaTeX 建議 pdf 或 svg）")
    ap.add_argument("--outdir", default=str(FIGDIR),
                    help="輸出目錄（預設 CI-RCT/figures/）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    plot_depth_dist(data, outdir / f"fig_4X_chain_depth_dist.{args.fmt}")
    plot_root_type_dist(data, outdir / f"fig_4Y_root_type_dist.{args.fmt}")
    dump_values(data, outdir / "fig_4XY_data.txt")

    print("Wrote:")
    for p in sorted(outdir.glob("fig_4*")):
        print(" ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
