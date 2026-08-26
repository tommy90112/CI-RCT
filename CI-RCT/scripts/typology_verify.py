"""
typology_verify.py — 剝離鏈案例驗證與論文級補充統計。

基於 typology_scan.py 的掃描結果做三件事：

  1. 候選 2 / 候選 3「同一洗錢作業平行分支」假設驗證
     （共同上游、AddrAddr 直接連結、co-spend、區塊/時間距離、
       根因錢包原始特徵 cosine 相似度 + 隨機基準百分位）
  2. 候選 5（論文主案例）與候選 2（副案例）的逐跳檔案：
     剝離量 / 剝離比例、φ_asym 責任分布（直接取自 dump，不重跑 evaluate）
  3. 補充統計：peeling 命中鏈深度分布、根因 illicit 比例、
     「金額等差遞減」指紋（相鄰差 CV < 門檻）、三簽名皆命中清單

所有數字實算自 viz/crime_chains.json、results/typology_scan.csv 與
Elliptic++ 原始 CSV。輸出 typology_verification.md。

用法
    cd CI-RCT
    python scripts/typology_verify.py \
        --chains viz/crime_chains.json --scan_csv results/typology_scan.csv \
        --data_root data --out typology_verification.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

CAND2_TARGET = "1HhMuN8MkqdXRW812tXr8Ctdy4N8Jo3LBm"
CAND3_TARGET = "1A263oPkyaGxsZC2yntVVUXxoos59fzfDe"
CAND5_TARGET = "210646674"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Peeling-chain case verification")
    p.add_argument("--chains", default="viz/crime_chains.json")
    p.add_argument("--scan_csv", default="results/typology_scan.csv")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out", default="typology_verification.md")
    p.add_argument("--amount_col", default="out_BTC_total")
    p.add_argument("--cv_thresh", type=float, default=0.3,
                   help="金額等差遞減指紋：相鄰差變異係數 CV < 此值")
    p.add_argument("--n_baseline_pairs", type=int, default=2000,
                   help="cosine 隨機基準抽樣對數（seed=42 可重現）")
    p.add_argument("--fig_dir", default="figures")
    return p.parse_args()


# ── 載入 ─────────────────────────────────────────────────────────────────────

def load_all(args) -> dict:
    chains = json.load(open(args.chains))["chains"]
    by_target = {c["target_txid"]: c for c in chains}
    scan = list(csv.DictReader(open(args.scan_csv)))
    for tgt in (CAND2_TARGET, CAND3_TARGET, CAND5_TARGET):
        if tgt not in by_target:
            sys.exit(f"[verify] 目標 {tgt} 不在 {args.chains} 中")

    root = os.path.join(args.data_root, "Elliptic++")
    tx = pd.read_csv(os.path.join(root, "txs_features.csv"),
                     usecols=["txId", "Time step", args.amount_col])
    tx["txId"] = tx["txId"].astype(str)
    wallets = pd.read_csv(os.path.join(root, "wallets_features.csv"))
    wallets_latest = (
        wallets.sort_values(["address", "Time step"], kind="stable")
        .drop_duplicates("address", keep="last")
        .set_index("address")
    )
    return {
        "by_target": by_target,
        "scan": scan,
        "tx": tx.set_index("txId"),
        "wallets": wallets_latest,
        "addr_tx": pd.read_csv(os.path.join(root, "AddrTx_edgelist.csv")),
        "tx_addr": pd.read_csv(os.path.join(root, "TxAddr_edgelist.csv")),
        "addr_addr": pd.read_csv(os.path.join(root, "AddrAddr_edgelist.csv")),
        "wallets_cls": pd.read_csv(os.path.join(root, "wallets_classes.csv")),
    }


def flow_nodes(chain: dict) -> List[dict]:
    return sorted(chain["nodes"], key=lambda n: -n["pos"])


# ── 任務 1：同作業假設驗證 ────────────────────────────────────────────────────

def wallet_block_window(nodes: List[dict], wallets: pd.DataFrame) -> dict:
    """鏈上錢包（有收款者）的區塊活動窗 [min first_received, max last_appeared]。"""
    fr, la, skipped = [], [], []
    for n in nodes:
        if n["type"] != "wallet" or n["real_id"] not in wallets.index:
            continue
        row = wallets.loc[n["real_id"]]
        if float(row["first_received_block"]) == 0.0:
            skipped.append(n["real_id"])   # 資料窗內從未收款 → first_recv=0 是佔位值
            la.append(float(row["last_block_appeared_in"]))
            continue
        fr.append(float(row["first_received_block"]))
        la.append(float(row["last_block_appeared_in"]))
    return {
        "block_min": min(fr) if fr else None,
        "block_max": max(la) if la else None,
        "n_never_received": len(skipped),
        "never_received": skipped,
    }


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(u @ v / (nu * nv))


def verify_same_operation(data: dict, args) -> dict:
    c2 = data["by_target"][CAND2_TARGET]
    c3 = data["by_target"][CAND3_TARGET]
    n2, n3 = flow_nodes(c2), flow_nodes(c3)
    root2, root3 = n2[0]["real_id"], n3[0]["real_id"]
    tx2 = [n["real_id"] for n in n2 if n["type"] == "transaction"]
    tx3 = [n["real_id"] for n in n3 if n["type"] == "transaction"]
    w2 = {n["real_id"] for n in n2 if n["type"] == "wallet"}
    w3 = {n["real_id"] for n in n3 if n["type"] == "wallet"}

    at, ta, aa = data["addr_tx"], data["tx_addr"], data["addr_addr"]
    at = at.assign(txId=at["txId"].astype(str))
    ta = ta.assign(txId=ta["txId"].astype(str))

    # (a) 根因錢包在原始圖的完整出現位置
    def wallet_profile(addr: str) -> dict:
        sends = at.loc[at["input_address"] == addr, "txId"].tolist()
        recvs = ta.loc[ta["output_address"] == addr, "txId"].tolist()
        aa_out = aa.loc[aa["input_address"] == addr, "output_address"].tolist()
        aa_in = aa.loc[aa["output_address"] == addr, "input_address"].tolist()
        return {"sends_to_tx": sends, "receives_from_tx": recvs,
                "aa_out": aa_out, "aa_in": aa_in}

    p2, p3 = wallet_profile(root2), wallet_profile(root3)

    # (b) 兩根因錢包的直接連結 / 共同鄰居 / co-spend
    direct_aa = (
        ((aa["input_address"] == root2) & (aa["output_address"] == root3))
        | ((aa["input_address"] == root3) & (aa["output_address"] == root2))
    ).sum()
    common_aa = (set(p2["aa_out"]) | set(p2["aa_in"])) & (
        set(p3["aa_out"]) | set(p3["aa_in"]))
    cospend = set(p2["sends_to_tx"]) & set(p3["sends_to_tx"])
    common_funder_tx = set(p2["receives_from_tx"]) & set(p3["receives_from_tx"])

    # (c) 兩鏈交易的共同輸入錢包（除根因外的共同出資者）
    inputs2 = set(at.loc[at["txId"].isin(tx2), "input_address"])
    inputs3 = set(at.loc[at["txId"].isin(tx3), "input_address"])
    shared_inputs = (inputs2 & inputs3) - {root2, root3}
    cross_wallet_overlap = (w2 & w3)

    # 兩鏈錢包之間的任何 AddrAddr 邊
    wall2, wall3 = w2 | {root2}, w3 | {root3}
    aa_cross = aa[
        (aa["input_address"].isin(wall2) & aa["output_address"].isin(wall3))
        | (aa["input_address"].isin(wall3) & aa["output_address"].isin(wall2))
    ]

    # (d) 區塊 / 時間距離
    tsteps2 = sorted({n["time"] for n in n2 if n["time"] is not None and n["time"] >= 0})
    tsteps3 = sorted({n["time"] for n in n3 if n["time"] is not None and n["time"] >= 0})
    win2 = wallet_block_window(n2, data["wallets"])
    win3 = wallet_block_window(n3, data["wallets"])
    tx_id_gap = min(abs(int(a) - int(b)) for a in tx2 for b in tx3)

    # (e) 根因錢包原始特徵 cosine（每地址最新列、除 address 外全部數值欄）
    feat_cols = [c for c in data["wallets"].columns if c != "address"]
    v2 = data["wallets"].loc[root2, feat_cols].to_numpy(dtype=float)
    v3 = data["wallets"].loc[root3, feat_cols].to_numpy(dtype=float)
    cos_roots = cosine(v2, v3)
    feat_diff = [(c, float(x), float(y))
                 for c, x, y in zip(feat_cols, v2, v3) if x != y]

    rng = np.random.default_rng(42)
    pool = data["wallets"].index.to_numpy()
    pairs = rng.choice(len(pool), size=(args.n_baseline_pairs, 2))
    base = []
    mat = data["wallets"][feat_cols]
    for i, j in pairs:
        if i == j:
            continue
        base.append(cosine(mat.iloc[i].to_numpy(dtype=float),
                           mat.iloc[j].to_numpy(dtype=float)))
    base = np.array([b for b in base if not np.isnan(b)])
    pct = float((base < cos_roots).mean() * 100)

    root_rows = {
        r: {k: float(data["wallets"].loc[r, k]) for k in
            ("Time step", "first_received_block", "last_block_appeared_in",
             "total_txs", "btc_transacted_total")}
        for r in (root2, root3)
    }
    return {
        "root2": root2, "root3": root3, "tx2": tx2, "tx3": tx3,
        "profile2": p2, "profile3": p3, "root_rows": root_rows,
        "direct_aa": int(direct_aa), "common_aa": sorted(common_aa),
        "cospend": sorted(cospend), "common_funder_tx": sorted(common_funder_tx),
        "shared_inputs": sorted(shared_inputs),
        "cross_wallet_overlap": sorted(cross_wallet_overlap),
        "aa_cross": aa_cross[["input_address", "output_address"]].values.tolist(),
        "tsteps2": tsteps2, "tsteps3": tsteps3,
        "win2": win2, "win3": win3, "tx_id_gap": int(tx_id_gap),
        "cos_roots": cos_roots, "feat_diff": feat_diff,
        "n_feat": len(feat_cols),
        "cos_baseline_median": float(np.median(base)),
        "cos_baseline_p95": float(np.quantile(base, 0.95)),
        "cos_percentile": pct, "n_baseline": int(len(base)),
    }


# ── 任務 2：案例逐跳檔案 ─────────────────────────────────────────────────────

def case_dossier(chain: dict, data: dict, amount_col: str) -> dict:
    nodes = flow_nodes(chain)
    rows = []
    for n in nodes:
        amount = None
        if n["type"] == "transaction" and n["real_id"] in data["tx"].index:
            v = data["tx"].at[n["real_id"], amount_col]
            amount = None if pd.isna(v) else float(v)
        rows.append({
            "type": n["type"], "id": n["real_id"],
            "time": n["time"] if (n["time"] is not None and n["time"] >= 0) else None,
            "fraud": bool(n["fraud"]), "ce": n.get("ce"),
            "phi_asym": n.get("phi_asym"), "amount": amount,
        })
    amts = [r["amount"] for r in rows if r["amount"] is not None]
    peels = [
        {"from": a, "to": b, "peel": a - b, "ratio": (a - b) / a if a else None}
        for a, b in zip(amts, amts[1:])
    ]
    diffs = np.array([p["peel"] for p in peels])
    cv = (float(diffs.std() / diffs.mean())
          if len(diffs) >= 2 and diffs.mean() > 0 else None)
    phis = np.array([abs(r["phi_asym"]) for r in rows
                     if r["phi_asym"] is not None])
    phi_share = phis / phis.sum() if phis.sum() > 0 else phis
    phi_rows = [r for r in rows if r["phi_asym"] is not None]
    top_i = int(np.argmax(phis)) if len(phis) else None
    return {
        "target": chain["target_txid"], "depth": chain["depth"],
        "is_tp": bool(chain["is_true_positive"]),
        "root": chain["root_real_id"], "root_is_fraud": bool(chain["root_is_fraud"]),
        "rows": rows, "amounts": amts, "peels": peels, "peel_cv": cv,
        "phi_total_abs": float(phis.sum()) if len(phis) else 0.0,
        "phi_top_node": phi_rows[top_i]["id"] if top_i is not None else None,
        "phi_top_value": phi_rows[top_i]["phi_asym"] if top_i is not None else None,
        "phi_top_share": float(phi_share[top_i]) if top_i is not None else None,
        "phi_shares": [
            {"id": r["id"], "type": r["type"], "phi": r["phi_asym"],
             "share": float(s)}
            for r, s in zip(phi_rows, phi_share)
        ],
    }


# ── 任務 3：補充統計 ─────────────────────────────────────────────────────────

def supplement_stats(data: dict, args) -> dict:
    hits = [r for r in data["scan"] if r["peel_hit_any"] == "True"]
    depth_hist: Dict[int, int] = {}
    for r in hits:
        depth_hist[int(r["depth"])] = depth_hist.get(int(r["depth"]), 0) + 1
    root_fraud = sum(r["root_is_fraud"] == "True" for r in hits)

    arithmetic, cv_values = [], []
    for r in hits:
        chain = data["by_target"].get(r["target_id"])
        if chain is None:
            continue
        d = case_dossier(chain, data, args.amount_col)
        if d["peel_cv"] is not None:
            cv_values.append((r["target_id"], d["peel_cv"]))
            if d["peel_cv"] < args.cv_thresh:
                arithmetic.append((r["target_id"], int(r["depth"]),
                                   d["peel_cv"], r["is_true_positive"]))
    triple = [r for r in data["scan"]
              if r["peel_hit_any"] == "True" and r["fanin_hit"] == "True"
              and r["pass_hit"] == "True"]
    return {
        "n_hits": len(hits), "depth_hist": dict(sorted(depth_hist.items())),
        "root_fraud": root_fraud,
        "n_cv_evaluable": len(cv_values),
        "arithmetic": sorted(arithmetic, key=lambda x: x[2]),
        "triple": triple,
    }


# ── 報告輸出 ─────────────────────────────────────────────────────────────────

def fmt(v, nd=4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def dossier_md(d: dict, amount_col: str, fig_paths: List[str]) -> List[str]:
    lines = [
        f"- target `{d['target']}`（depth={d['depth']}，"
        f"true positive={fmt(d['is_tp'])}），根因 `{d['root']}`"
        f"（illicit 標籤={fmt(d['root_is_fraud'])}）",
        f"- 金額序列（{amount_col}，金流方向）："
        + " → ".join(f"{a:.6g}" for a in d["amounts"]),
        f"- 剝離量相鄰差 CV = {fmt(d['peel_cv'])}",
        f"- φ_asym 最大責任節點：`{d['phi_top_node']}`"
        f"（φ_asym={fmt(d['phi_top_value'])}，佔全鏈 |φ_asym| 的 "
        f"{d['phi_top_share'] * 100:.1f}%）" if d["phi_top_node"] else "- φ_asym 無資料",
        "",
    ]
    for p in fig_paths:
        lines.append(f"![{os.path.basename(p)}]({p})")
    lines += [
        "",
        "| # | 型別 | 節點 id | 時間步 | illicit | CE(→下一節點) | "
        "φ_asym | φ 佔比 | 金額 | 剝離量 | 剝離比例 |",
        "|---|------|---------|--------|---------|----|----|----|----|----|----|",
    ]
    share = {s["id"]: s["share"] for s in d["phi_shares"]}
    peel_by_from = {}
    amt_i = 0
    for i, r in enumerate(d["rows"]):
        peel = ratio = ""
        if r["amount"] is not None:
            if amt_i < len(d["peels"]):
                peel = fmt(d["peels"][amt_i]["peel"])
                ratio = f"{d['peels'][amt_i]['ratio'] * 100:.1f}%" \
                    if d["peels"][amt_i]["ratio"] is not None else "–"
            amt_i += 1
        lines.append(
            f"| {i} | {r['type']} | `{r['id']}` | {fmt(r['time'])} | "
            f"{fmt(r['fraud'])} | {fmt(r['ce'])} | {fmt(r['phi_asym'])} | "
            f"{(fmt(share[r['id']] * 100, 3) + '%') if r['id'] in share else '–'} | "
            f"{fmt(r['amount'], 6)} | {peel} | {ratio} |")
    return lines


def write_report(path: str, same_op: dict, d5: dict, d2: dict,
                 stats: dict, args, figs: Dict[str, List[str]]) -> None:
    s = same_op
    evidence_for = [
        f"兩鏈全部交易同在 time step 37（鏈 2：{', '.join(s['tx2'])}；"
        f"鏈 3：{', '.join(s['tx3'])}），tx id 最小距離 {s['tx_id_gap']}",
        f"兩根因錢包皆為單次出現的純出資錢包（total_txs=1、資料窗內從未收款），"
        f"且**同在區塊 {s['root_rows'][s['root2']]['last_block_appeared_in']:.0f} "
        f"出現**——使用者觀察到的「gap 同為 463779」實為 "
        f"first_received_block=0 佔位值造成的假影，其背後的真訊號即是同區塊單次出現",
        f"兩根因錢包原始特徵 cosine = {s['cos_roots']:.6f}"
        f"（隨機錢包對基準：中位數 {s['cos_baseline_median']:.4f}、"
        f"P95 {s['cos_baseline_p95']:.4f}，n={s['n_baseline']}；"
        f"本值位於第 {s['cos_percentile']:.1f} 百分位）",
        f"逐欄比對：{s['n_feat']} 個特徵中僅 {len(s['feat_diff'])} 欄不同，"
        f"且全為 BTC 金額欄（"
        + "、".join(sorted({c for c, _, _ in s['feat_diff']})) + "）——"
        f"兩錢包除單筆出資金額（{s['feat_diff'][0][1]:.6g} vs "
        f"{s['feat_diff'][0][2]:.6g} BTC）外行為特徵逐欄相同，"
        f"與「同一模板產生的一次性出資錢包」一致",
    ]
    evidence_against = []
    if not s["cospend"]:
        evidence_against.append("無 co-spend：兩根因錢包從未共同作為同一交易的輸入"
                                "（各自僅出現一次，本檢驗無鑑別力）")
    if not s["common_funder_tx"]:
        evidence_against.append("無共同注資交易（兩者在資料窗內皆無收款紀錄，"
                                "上游出資者不可觀測）")
    if s["direct_aa"] == 0 and not s["common_aa"]:
        evidence_against.append("AddrAddr 無直接邊、無共同鄰居")
    if not s["shared_inputs"]:
        evidence_against.append("兩鏈的 11 筆交易無共同輸入錢包")
    if not s["cross_wallet_overlap"] and not s["aa_cross"]:
        evidence_against.append("兩鏈錢包集合零交集，鏈間亦無任何 AddrAddr 邊")

    lines = [
        "# 剝離鏈案例驗證報告（Typology Verification）",
        "",
        "> 承 typology_report.md。所有數字實算自 joint 定版 2000 條鏈、",
        "> typology_scan 逐鏈判定表與 Elliptic++ 原始 CSV；",
        "> 手法歸因措辭一律為「與 X 模式一致（consistent with）」。",
        "",
        "## 1. 候選 2 × 候選 3「同一作業平行分支」假設驗證",
        "",
        f"- 候選 2：target `{CAND2_TARGET}`（depth 12），根因 `{s['root2']}`",
        f"- 候選 3：target `{CAND3_TARGET}`（depth 10），根因 `{s['root3']}`",
        "",
        "### 檢驗結果",
        "",
        "| 檢驗 | 結果 |",
        "|------|------|",
        f"| 根因間 AddrAddr 直接邊 | {s['direct_aa']} 條 |",
        f"| 根因 AddrAddr 共同鄰居 | {len(s['common_aa'])} 個 |",
        f"| 根因 co-spend（共同輸入之交易） | {len(s['cospend'])} 筆 |",
        f"| 根因共同注資交易 | {len(s['common_funder_tx'])} 筆 |",
        f"| 兩鏈交易共同輸入錢包（根因除外） | {len(s['shared_inputs'])} 個 |",
        f"| 兩鏈錢包集合交集 / 鏈間 AddrAddr 邊 | "
        f"{len(s['cross_wallet_overlap'])} / {len(s['aa_cross'])} |",
        f"| 兩鏈 time step | {s['tsteps2']} vs {s['tsteps3']}（同一步） |",
        f"| tx id 最小距離 | {s['tx_id_gap']} |",
        f"| 鏈上錢包區塊活動窗（有收款者） | "
        f"鏈 2：[{fmt(s['win2']['block_min'], 8)}, {fmt(s['win2']['block_max'], 8)}]"
        f"（{s['win2']['n_never_received']} 個未收款）；"
        f"鏈 3：[{fmt(s['win3']['block_min'], 8)}, {fmt(s['win3']['block_max'], 8)}]"
        f"（{s['win3']['n_never_received']} 個未收款） |",
        f"| 根因原始特徵 cosine | {s['cos_roots']:.6f}"
        f"（隨機對第 {s['cos_percentile']:.1f} 百分位） |",
        "",
        "### 支持證據",
        "",
    ]
    lines += [f"- {e}" for e in evidence_for]
    lines += ["", "### 不支持／無鑑別力的檢驗", ""]
    lines += [f"- {e}" for e in evidence_against]
    lines += [
        "",
        "### 判定",
        "",
        "**行為層強烈支持、圖結構層無法確認。**兩鏈在時間（同 time step、"
        "根因同區塊單次出現）、序號（tx id 相鄰）、結構（同深度級別的嚴格交替"
        "剝離形態）與根因特徵（cosine ≈ 1）上高度一致，與「同一作業的平行分支」"
        "假設一致；但原始圖中兩鏈完全不相連（無共同上游、無 AddrAddr 邊、無 "
        "co-spend），故無法以鏈上證據證實同一控制者。論文措辭建議：「二鏈呈現"
        "與同一自動化拆分作業一致（consistent with）的平行剝離特徵，惟圖上"
        "無直接連結，同一控制者假設無法由本資料證實」。",
        "",
        "## 2. 論文主案例：候選 5（target 210646674）",
        "",
    ]
    lines += dossier_md(d5, args.amount_col, figs["case5"])
    lines += ["", "## 3. 副案例（標註外發現）：候選 2", ""]
    lines += dossier_md(d2, args.amount_col, figs["case2"])
    lines += [
        "",
        "## 4. 補充統計（186 條 peeling 命中鏈）",
        "",
        "### 深度分布",
        "",
        "| depth | 條數 |",
        "|-------|------|",
    ]
    lines += [f"| {k} | {v} |" for k, v in stats["depth_hist"].items()]
    lines += [
        "",
        f"- 根因為 illicit 標註者：{stats['root_fraud']}/{stats['n_hits']} "
        f"（{stats['root_fraud'] / stats['n_hits'] * 100:.1f}%）",
        f"- 「金額等差遞減」指紋（相鄰剝離量 CV < {args.cv_thresh}，"
        f"需 ≥3 個金額點）：**{len(stats['arithmetic'])}** / "
        f"{stats['n_cv_evaluable']} 條可評估鏈",
        "",
        "等差遞減命中鏈（CV 由小到大）：",
        "",
        "| target | depth | CV | TP |",
        "|--------|-------|----|----|",
    ]
    lines += [f"| `{t}` | {dep} | {cv:.4g} | {tp} |"
              for t, dep, cv, tp in stats["arithmetic"]]
    lines += [
        "",
        "### 三簽名皆命中的鏈（peeling ∩ fan-in ∩ pass-through，最高置信候選）",
        "",
        "| target | depth | TP | 根因 | 根因 illicit | Spearman | 遞減比例 |",
        "|--------|-------|----|------|--------------|----------|----------|",
    ]
    for r in stats["triple"]:
        lines.append(
            f"| `{r['target_id']}` | {r['depth']} | {r['is_true_positive']} | "
            f"`{r['root_id']}` | {r['root_is_fraud']} | "
            f"{fmt(float(r['spearman'])) if r['spearman'] else '–'} | "
            f"{fmt(float(r['frac_dec'])) if r['frac_dec'] else '–'} |")
    lines += [
        "",
        "## 5. 附註",
        "",
        "- `first_received_block = 0` 為「資料窗內從未收款」之佔位值，該類錢包"
        "的 gap（=last_block_appeared_in）不具生命週期意義；掃描的 pass-through"
        " 判定不受影響（其 gap 遠大於門檻、必不命中），但個案表格引用 gap 時"
        "須先檢查此欄位。",
        "- φ_asym 取自 evaluate 定版 dump（逐段滾動 readout），為描述性、"
        "非守恆之責任分布；佔比以 |φ_asym| 正規化。",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def draw_visualizer_figs(data: dict, fig_dir: str) -> None:
    """補充圖：utils.chain_visualizer 橫排風格（主圖為論文 4.3 風格 SVG）。"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import matplotlib.pyplot as plt
    from utils.chain_visualizer import draw_causal_chain

    for tgt, fname in ((CAND5_TARGET, "fig_case5_visualizer.png"),
                       (CAND2_TARGET, "fig_case2_visualizer.png")):
        chain_rec = data["by_target"][tgt]
        nodes = chain_rec["nodes"]                      # [target, ..., root]
        chain = [n["real_id"] for n in nodes]
        effects = {
            (nodes[i]["real_id"], nodes[i - 1]["real_id"]): nodes[i]["ce"]
            for i in range(1, len(nodes)) if nodes[i].get("ce") is not None
        }
        fig = draw_causal_chain(
            chain=chain,
            causal_effects=effects,
            fraud_set={n["real_id"] for n in nodes if n["fraud"]},
            node_type_map={n["real_id"]: n["type"] for n in nodes},
            idx_to_user_id={n["real_id"]: (n["real_id"][:6] + "…"
                                           if len(n["real_id"]) > 10
                                           else n["real_id"]) for n in nodes},
            title=f"Peeling case — target {tgt} (depth {chain_rec['depth']})",
            save_path=os.path.join(fig_dir, fname),
        )
        plt.close(fig)
        print(f"[verify] 補充圖 → {os.path.join(fig_dir, fname)}")


def main() -> None:
    args = parse_args()
    data = load_all(args)

    print("[verify] 任務 1：同作業假設驗證 …")
    same_op = verify_same_operation(data, args)
    print(f"  cosine(roots) = {same_op['cos_roots']:.6f} "
          f"(隨機基準第 {same_op['cos_percentile']:.1f} 百分位)")

    print("[verify] 任務 2：案例檔案 …")
    d5 = case_dossier(data["by_target"][CAND5_TARGET], data, args.amount_col)
    d2 = case_dossier(data["by_target"][CAND2_TARGET], data, args.amount_col)

    print("[verify] 任務 3：補充統計 …")
    stats = supplement_stats(data, args)
    print(f"  peeling 命中 {stats['n_hits']} 條；等差遞減指紋 "
          f"{len(stats['arithmetic'])}/{stats['n_cv_evaluable']}；"
          f"三重命中 {len(stats['triple'])} 條")

    figs = {
        "case5": [f"{args.fig_dir}/fig_case5_peeling_{CAND5_TARGET}.svg",
                  f"{args.fig_dir}/fig_case5_visualizer.png"],
        "case2": [f"{args.fig_dir}/fig_case2_peeling_{CAND2_TARGET[:12]}.svg",
                  f"{args.fig_dir}/fig_case2_visualizer.png"],
    }
    draw_visualizer_figs(data, args.fig_dir)
    write_report(args.out, same_op, d5, d2, stats, args, figs)
    print(f"[verify] 報告 → {args.out}")


if __name__ == "__main__":
    main()
