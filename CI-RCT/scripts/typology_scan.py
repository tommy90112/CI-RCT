"""
typology_scan.py — 洗錢手法結構簽名掃描（typology attribution scan）。

對 evaluate.py --dump_chains 匯出的追溯鏈（crime_chains.json）逐鏈比對三種
文獻已命名手法的「結構簽名」（Elliptic2 / MIT-IBM Watson 2024）：

    1. Peeling chain（剝離鏈）  — 長交替鏈 + 時間單調 + 金額遞減
    2. Fan-in / structuring     — 鏈上存在高入度樞紐且上游時間步集中
    3. Pass-through wallet      — 鏈上錢包快進快出（區塊區間 < illicit P25）

Elliptic++ 沒有手法標註，因此輸出結論一律是「與 X 模式一致
（consistent with）」，不是「證實為 X」。

輸入
    --chains     evaluate.py --dump_chains 產出的 JSON（逐節點含 time/ce/fraud）
    --data_root  Elliptic++ 原始 CSV 目錄的上層（data/Elliptic++/...）

輸出
    --out_csv    逐鏈判定表（每鏈一列，含所有簽名子條件）
    --out_report Markdown 報告（統計總表 + top-N peeling 候選 + 指定案例明細）

所有門檻皆為 CLI 參數；所有數字皆由實際資料計算，不推估。

用法
    cd CI-RCT
    python scripts/typology_scan.py \
        --chains viz/crime_chains.json --data_root data \
        --out_csv results/typology_scan.csv --out_report typology_report.md \
        --case_tx 54824221
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# loader 的時間 sentinel：圖內未被注資的錢包 / 無時間戳節點
NO_TIME = -1

TX_REQUIRED_COLS = ("txId", "Time step")
WALLET_REQUIRED_COLS = (
    "address", "Time step", "first_received_block", "last_block_appeared_in",
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Typology structural-signature scan")
    p.add_argument("--chains", default="viz/crime_chains.json")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_csv", default="results/typology_scan.csv")
    p.add_argument("--out_report", default="typology_report.md")

    # Peeling chain 門檻
    p.add_argument("--peel_min_depth", type=int, default=4)
    p.add_argument("--spearman_thresh", type=float, default=-0.5,
                   help="金額遞減判定（寬鬆版 A）：Spearman < 此值")
    p.add_argument("--frac_dec_thresh", type=float, default=0.7,
                   help="金額遞減判定（寬鬆版 B）：遞減跳數比例 > 此值")
    p.add_argument("--amount_col", default="out_BTC_total",
                   help="沿鏈交易金額欄位（txs_features.csv 的 17 統計量之一）")
    p.add_argument("--min_amounts_frac", type=int, default=2,
                   help="計算遞減跳數比例所需的最少可用金額點數")
    p.add_argument("--min_amounts_spearman", type=int, default=3,
                   help="計算 Spearman 所需的最少可用金額點數")
    p.add_argument("--min_known_times", type=int, default=2,
                   help="時間單調檢查所需的最少已知時間步節點數")

    # Fan-in 門檻
    p.add_argument("--fanin_indegree_quantile", type=float, default=0.95,
                   help="樞紐入度需 >= 全體同型別節點入度的此分位數")
    p.add_argument("--fanin_time_range", type=int, default=3,
                   help="樞紐上游來源時間步極差上限（time steps）")
    p.add_argument("--fanin_min_upstream_known", type=int, default=2,
                   help="評估上游時間集中所需的最少已知時間上游數")
    p.add_argument("--fanin_all_edges", action="store_true",
                   help="入度與上游額外納入 tx→tx 與 addr→addr 邊"
                        "（預設僅金流邊 wallet→tx / tx→wallet）")

    # Pass-through 門檻
    p.add_argument("--passthrough_quantile", type=float, default=0.25,
                   help="錢包 (last_block_appeared_in - first_received_block) "
                        "需小於全體 illicit 錢包該區間分布的此分位數")

    # 選例
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--case_tx", default="54824221",
                   help="必查案例的 target id（論文案例一）；空字串停用")
    return p.parse_args()


# ── 資料載入（fail-fast 驗證，不自行假設欄位） ────────────────────────────────

def load_chains(path: str) -> Tuple[dict, List[dict]]:
    if not os.path.exists(path):
        sys.exit(f"[typology_scan] chains 檔不存在: {path}")
    with open(path) as f:
        payload = json.load(f)
    if "chains" not in payload:
        sys.exit(f"[typology_scan] {path} 缺少 'chains' 欄位（需為 "
                 f"evaluate.py --dump_chains 產出的 JSON）")
    chains = payload["chains"]
    for key in ("target_txid", "depth", "is_true_positive", "nodes"):
        if chains and key not in chains[0]:
            sys.exit(f"[typology_scan] chain record 缺少欄位 '{key}'")
    for key in ("type", "real_id", "time", "fraud", "pos"):
        if chains and key not in chains[0]["nodes"][0]:
            sys.exit(f"[typology_scan] node record 缺少欄位 '{key}'")
    return payload.get("meta", {}), chains


def load_tx_table(data_root: str, amount_col: str) -> pd.DataFrame:
    """txId(str) 為 index：Time step + 指定金額欄位（原始未標準化值）。"""
    path = os.path.join(data_root, "Elliptic++", "txs_features.csv")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in (*TX_REQUIRED_COLS, amount_col) if c not in header]
    if missing:
        sys.exit(f"[typology_scan] txs_features.csv 缺少欄位: {missing}")
    df = pd.read_csv(path, usecols=["txId", "Time step", amount_col])
    df["txId"] = df["txId"].astype(str)
    return df.set_index("txId")


def load_wallet_table(data_root: str) -> pd.DataFrame:
    """
    address 為 index 的 per-address 錢包表（與 joint 變體同款 dedup：
    每地址保留最新 Time step 那一列的累積快照）。附 gap 欄位 =
    last_block_appeared_in - first_received_block。
    """
    path = os.path.join(data_root, "Elliptic++", "wallets_features.csv")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in WALLET_REQUIRED_COLS if c not in header]
    if missing:
        sys.exit(f"[typology_scan] wallets_features.csv 缺少欄位: {missing}")
    df = pd.read_csv(path, usecols=list(WALLET_REQUIRED_COLS))
    latest = (
        df.sort_values(["address", "Time step"], kind="stable")
        .drop_duplicates("address", keep="last")
        .reset_index(drop=True)
    )
    gap = latest["last_block_appeared_in"] - latest["first_received_block"]
    return latest.assign(gap=gap).set_index("address")


def load_illicit_wallets(data_root: str) -> set:
    path = os.path.join(data_root, "Elliptic++", "wallets_classes.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "class" not in df.columns or "address" not in df.columns:
        sys.exit("[typology_scan] wallets_classes.csv 缺少 address/class 欄位")
    return set(df.loc[df["class"] == 1, "address"])


# ── 入度 / 上游結構（Fan-in 用） ──────────────────────────────────────────────

def build_indegree_tables(
    data_root: str,
    tx_ids: List[str],
    wallet_ids: List[str],
    all_edges: bool,
    quantile: float,
) -> dict:
    """
    以原始邊表計算全體節點入度與同型別 P-quantile 門檻。

    預設僅金流邊：tx 入度 = AddrTx（輸入錢包數）、wallet 入度 = TxAddr
    （付款交易數）。--fanin_all_edges 時 tx 另加 tx→tx、wallet 另加
    addr→addr 的入邊。母體含入度為 0 的節點。
    """
    root = os.path.join(data_root, "Elliptic++")
    addr_tx = pd.read_csv(os.path.join(root, "AddrTx_edgelist.csv"))
    tx_addr = pd.read_csv(os.path.join(root, "TxAddr_edgelist.csv"))

    tx_counts = addr_tx["txId"].astype(str).value_counts()
    wallet_counts = tx_addr["output_address"].value_counts()
    if all_edges:
        tt = pd.read_csv(os.path.join(root, "txs_edgelist.csv"))
        aa = pd.read_csv(os.path.join(root, "AddrAddr_edgelist.csv"))
        tx_counts = tx_counts.add(tt["txId2"].astype(str).value_counts(),
                                  fill_value=0)
        wallet_counts = wallet_counts.add(
            aa["output_address"].value_counts(), fill_value=0)

    tx_pop = tx_counts.reindex(tx_ids).fillna(0).to_numpy()
    wallet_pop = wallet_counts.reindex(wallet_ids).fillna(0).to_numpy()
    return {
        "tx_indeg": tx_counts.to_dict(),
        "wallet_indeg": wallet_counts.to_dict(),
        "tx_p": float(np.quantile(tx_pop, quantile)),
        "wallet_p": float(np.quantile(wallet_pop, quantile)),
        "addr_tx": addr_tx,
        "tx_addr": tx_addr,
        "aa": (aa if all_edges else None),
        "tt": (tt if all_edges else None),
    }


def build_upstream_time_maps(
    tables: dict,
    tx_time: Dict[str, int],
    chain_tx_ids: set,
    chain_wallet_ids: set,
    all_edges: bool,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    回傳 (upstream_times, wallet_fund_time)。

    upstream_times[node_id] = 該（鏈上）節點所有上游來源的已知時間步：
      tx 樞紐上游 = 輸入錢包（AddrTx）的注資時間（min over TxAddr 付款 tx 時間步）
      wallet 樞紐上游 = 付款交易（TxAddr）的 Time step
      --fanin_all_edges 時 tx 另加上游 tx、wallet 另加上游 wallet（注資時間）。
    """
    tx_addr, addr_tx = tables["tx_addr"], tables["addr_tx"]

    ta = tx_addr.assign(_t=tx_addr["txId"].astype(str).map(tx_time)).dropna(
        subset=["_t"])
    wallet_fund = ta.groupby("output_address")["_t"].min().astype(int).to_dict()

    upstream: Dict[str, List[int]] = {}
    at_hit = addr_tx[addr_tx["txId"].astype(str).isin(chain_tx_ids)]
    for txid, grp in at_hit.groupby(at_hit["txId"].astype(str)):
        times = [wallet_fund[a] for a in grp["input_address"] if a in wallet_fund]
        upstream[txid] = upstream.get(txid, []) + times

    ta_hit = tx_addr[tx_addr["output_address"].isin(chain_wallet_ids)]
    for addr, grp in ta_hit.groupby("output_address"):
        times = [tx_time[t] for t in grp["txId"].astype(str) if t in tx_time]
        upstream[addr] = upstream.get(addr, []) + times

    if all_edges:
        tt_hit = tables["tt"][tables["tt"]["txId2"].astype(str).isin(chain_tx_ids)]
        for txid, grp in tt_hit.groupby(tt_hit["txId2"].astype(str)):
            times = [tx_time[t] for t in grp["txId1"].astype(str) if t in tx_time]
            upstream[txid] = upstream.get(txid, []) + times
        aa_hit = tables["aa"][tables["aa"]["output_address"].isin(chain_wallet_ids)]
        for addr, grp in aa_hit.groupby("output_address"):
            times = [wallet_fund[a] for a in grp["input_address"]
                     if a in wallet_fund]
            upstream[addr] = upstream.get(addr, []) + times

    return upstream, wallet_fund


# ── 簽名計算（逐鏈） ──────────────────────────────────────────────────────────

def flow_order(chain: dict) -> List[dict]:
    """節點按金流方向排序：root（最上游）在前、target（最下游）在後。"""
    return sorted(chain["nodes"], key=lambda n: -n["pos"])


def check_alternation(nodes_flow: List[dict]) -> bool:
    types = [n["type"] for n in nodes_flow]
    return all(a != b for a, b in zip(types, types[1:]))


def check_time_monotone(
    nodes_flow: List[dict], min_known: int,
) -> Tuple[Optional[bool], int, int]:
    """
    金流方向時間步單調不減（只看已知時間；sentinel -1 視為未知並計數）。
    回傳 (是否單調 / None=不可評估, 已知數, 未知數)。
    """
    known = [n["time"] for n in nodes_flow
             if n["time"] is not None and n["time"] >= 0]
    n_missing = len(nodes_flow) - len(known)
    if len(known) < min_known:
        return None, len(known), n_missing
    ok = all(a <= b for a, b in zip(known, known[1:]))
    return ok, len(known), n_missing


def amount_sequence(
    nodes_flow: List[dict], tx_table: pd.DataFrame, amount_col: str,
) -> Tuple[List[float], int]:
    """沿金流方向的交易金額序列（NaN 略過並計數）。"""
    vals, n_nan = [], 0
    for n in nodes_flow:
        if n["type"] != "transaction":
            continue
        if n["real_id"] not in tx_table.index:
            n_nan += 1
            continue
        v = tx_table.at[n["real_id"], amount_col]
        if pd.isna(v):
            n_nan += 1
        else:
            vals.append(float(v))
    return vals, n_nan


def amount_trend(
    vals: List[float], min_spearman: int, min_frac: int,
) -> Tuple[Optional[float], Optional[float]]:
    """回傳 (spearman, 遞減跳數比例)；點數不足或序列常數時為 None。"""
    rho = None
    if len(vals) >= min_spearman and len(set(vals)) > 1:
        rho = float(spearmanr(range(len(vals)), vals).statistic)
    frac = None
    if len(vals) >= min_frac:
        steps = list(zip(vals, vals[1:]))
        frac = sum(b < a for a, b in steps) / len(steps)
    return rho, frac


def scan_peeling(chain: dict, tx_table: pd.DataFrame, args) -> dict:
    nodes_flow = flow_order(chain)
    alt_ok = check_alternation(nodes_flow)
    time_ok, n_known, n_missing = check_time_monotone(
        nodes_flow, args.min_known_times)
    vals, n_nan = amount_sequence(nodes_flow, tx_table, args.amount_col)
    rho, frac = amount_trend(
        vals, args.min_amounts_spearman, args.min_amounts_frac)

    struct_ok = (chain["depth"] >= args.peel_min_depth
                 and alt_ok and time_ok is True)
    hit_rho = struct_ok and rho is not None and rho < args.spearman_thresh
    hit_frac = struct_ok and frac is not None and frac > args.frac_dec_thresh
    strength_parts = [x for x in (max(0.0, -rho) if rho is not None else None,
                                  frac) if x is not None]
    return {
        "alt_ok": alt_ok,
        "time_ok": time_ok,
        "n_time_known": n_known,
        "n_time_missing": n_missing,
        "n_amounts": len(vals),
        "n_amount_missing": n_nan,
        "amount_seq": vals,
        "spearman": rho,
        "frac_dec": frac,
        "peel_struct_ok": struct_ok,
        "peel_hit_spearman": hit_rho,
        "peel_hit_frac": hit_frac,
        "peel_hit_any": hit_rho or hit_frac,
        "peel_strength": max(strength_parts) if strength_parts else 0.0,
    }


def scan_fanin(chain: dict, tables: dict,
               upstream_times: Dict[str, List[int]], args) -> dict:
    """鏈上是否存在高入度樞紐且其上游來源時間步集中。"""
    hits, best = [], None
    for n in chain["nodes"]:
        if n["type"] == "transaction":
            indeg = float(tables["tx_indeg"].get(n["real_id"], 0))
            thresh = tables["tx_p"]
        else:
            indeg = float(tables["wallet_indeg"].get(n["real_id"], 0))
            thresh = tables["wallet_p"]
        if indeg < thresh or thresh <= 0:
            continue
        times = upstream_times.get(n["real_id"], [])
        t_range = (max(times) - min(times)) if times else None
        concentrated = (len(times) >= args.fanin_min_upstream_known
                        and t_range is not None
                        and t_range <= args.fanin_time_range)
        cand = {
            "id": n["real_id"], "type": n["type"], "indeg": indeg,
            "indeg_thresh": thresh, "n_upstream_known": len(times),
            "time_range": t_range, "hit": concentrated,
        }
        if concentrated:
            hits.append(cand)
        if best is None or indeg / thresh > best["indeg"] / best["indeg_thresh"]:
            best = cand
    top = max(hits, key=lambda h: h["indeg"]) if hits else best
    return {
        "fanin_hit": bool(hits),
        "fanin_n_hub_hits": len(hits),
        "fanin_hub_id": top["id"] if top else "",
        "fanin_hub_type": top["type"] if top else "",
        "fanin_hub_indeg": top["indeg"] if top else None,
        "fanin_hub_time_range": top["time_range"] if top else None,
        "fanin_hub_n_upstream": top["n_upstream_known"] if top else None,
    }


def scan_passthrough(chain: dict, wallet_table: pd.DataFrame,
                     gap_thresh: float) -> dict:
    """鏈上錢包的 (last_block_appeared_in − first_received_block) < 門檻。"""
    hit_ids, n_wallets, n_unknown = [], 0, 0
    for n in chain["nodes"]:
        if n["type"] != "wallet":
            continue
        n_wallets += 1
        if n["real_id"] not in wallet_table.index:
            n_unknown += 1
            continue
        if float(wallet_table.at[n["real_id"], "gap"]) < gap_thresh:
            hit_ids.append(n["real_id"])
    return {
        "pass_hit": bool(hit_ids),
        "pass_n_wallets_hit": len(hit_ids),
        "pass_n_wallets": n_wallets,
        "pass_n_wallet_unknown": n_unknown,
        "pass_wallet_ids": ";".join(hit_ids),
    }


def scan_chain(chain: dict, tx_table: pd.DataFrame,
               wallet_table: pd.DataFrame, tables: dict,
               upstream_times: Dict[str, List[int]],
               gap_thresh: float, args) -> dict:
    nodes = chain["nodes"]
    base = {
        "target_id": chain["target_txid"],
        "target_type": nodes[0]["type"],
        "depth": chain["depth"],
        "n_nodes": len(nodes),
        "is_true_positive": bool(chain["is_true_positive"]),
        "root_id": chain["root_real_id"],
        "root_type": chain["root_type"],
        "root_is_fraud": bool(chain["root_is_fraud"]),
    }
    peel = scan_peeling(chain, tx_table, args)
    fanin = scan_fanin(chain, tables, upstream_times, args)
    pt = scan_passthrough(chain, wallet_table, gap_thresh)
    score = (base["depth"] * peel["peel_strength"]
             * (1 if peel["peel_hit_any"] else 0))
    return {**base, **peel, **fanin, **pt, "peel_score": score}


# ── 統計與報告 ────────────────────────────────────────────────────────────────

def rate(n: int, d: int) -> str:
    return f"{n} ({n / d * 100:.1f}%)" if d else "0 (–)"


def summarize(rows: List[dict]) -> List[Tuple[str, str, str]]:
    """(簽名, 全體命中, TP 鏈命中) 統計列。"""
    tp_rows = [r for r in rows if r["is_true_positive"]]
    out = []
    for label, key in (
        ("Peeling — 結構條件（深度+交替+時間單調）", "peel_struct_ok"),
        ("Peeling — 命中（Spearman 版）", "peel_hit_spearman"),
        ("Peeling — 命中（遞減跳數比例版）", "peel_hit_frac"),
        ("Peeling — 命中（兩版任一）", "peel_hit_any"),
        ("Fan-in / structuring", "fanin_hit"),
        ("Pass-through wallet", "pass_hit"),
    ):
        out.append((label,
                    rate(sum(bool(r[key]) for r in rows), len(rows)),
                    rate(sum(bool(r[key]) for r in tp_rows), len(tp_rows))))
    return out


def overlap_table(rows: List[dict]) -> List[Tuple[str, int]]:
    combos = {}
    for r in rows:
        key = (bool(r["peel_hit_any"]), bool(r["fanin_hit"]), bool(r["pass_hit"]))
        combos[key] = combos.get(key, 0) + 1
    names = {
        (True, False, False): "僅 Peeling",
        (False, True, False): "僅 Fan-in",
        (False, False, True): "僅 Pass-through",
        (True, True, False): "Peeling ∩ Fan-in",
        (True, False, True): "Peeling ∩ Pass-through",
        (False, True, True): "Fan-in ∩ Pass-through",
        (True, True, True): "三者皆命中",
        (False, False, False): "皆未命中",
    }
    return [(names[k], combos.get(k, 0)) for k in names]


def fmt(v, nd=4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def chain_detail_md(chain: dict, row: dict, tx_table: pd.DataFrame,
                    wallet_table: pd.DataFrame, amount_col: str) -> List[str]:
    """單一鏈的逐跳明細表（金流方向 root→target）。"""
    lines = [
        f"- 節點數 {row['n_nodes']}（depth={row['depth']}），"
        f"true positive={fmt(row['is_true_positive'])}，"
        f"根因節點 `{row['root_id']}`（{row['root_type']}，"
        f"illicit 標籤={fmt(row['root_is_fraud'])}）",
        f"- 型別嚴格交替={fmt(row['alt_ok'])}，時間單調不減={fmt(row['time_ok'])}"
        f"（已知時間節點 {row['n_time_known']}、未知 {row['n_time_missing']}），"
        f"金額點數 {row['n_amounts']}",
        f"- Spearman={fmt(row['spearman'])}，遞減跳數比例={fmt(row['frac_dec'])}，"
        f"Fan-in 命中={fmt(row['fanin_hit'])}，Pass-through 命中={fmt(row['pass_hit'])}",
        "",
        "| # | 型別 | 節點 id | 時間步 | illicit | CE(→下一節點) | "
        f"{amount_col} | 錢包 gap(blocks) |",
        "|---|------|---------|--------|---------|---------------|------|------|",
    ]
    for i, n in enumerate(flow_order(chain)):
        amount = ""
        if n["type"] == "transaction" and n["real_id"] in tx_table.index:
            v = tx_table.at[n["real_id"], amount_col]
            amount = fmt(float(v)) if not pd.isna(v) else "NaN"
        gap = ""
        if n["type"] == "wallet" and n["real_id"] in wallet_table.index:
            gap = fmt(float(wallet_table.at[n["real_id"], "gap"]), nd=6)
        t = n["time"] if n["time"] is not None and n["time"] >= 0 else "未知"
        ce = fmt(n.get("ce")) if not n.get("is_target") else "–"
        lines.append(
            f"| {i} | {n['type']} | `{n['real_id']}` | {t} | "
            f"{fmt(bool(n['fraud']))} | {ce} | {amount} | {gap} |")
    return lines


def write_report(path: str, meta: dict, rows: List[dict], chains: List[dict],
                 args, context: dict) -> None:
    by_target = {c["target_txid"]: c for c in chains}
    row_by_target = {r["target_id"]: r for r in rows}
    peel_rows = sorted(
        (r for r in rows if r["peel_hit_any"]),
        key=lambda r: (r["is_true_positive"], r["peel_score"]), reverse=True)

    lines = [
        "# 洗錢手法結構簽名掃描報告（Typology Attribution Scan）",
        "",
        "> Elliptic++ 無手法類型標註，本報告所有歸因均為**結構簽名比對**，",
        "> 結論措辭一律為「與 X 模式一致（consistent with X）」，非「證實為 X」。",
        "",
        "## 1. 掃描設定",
        "",
        f"- 輸入鏈檔：`{args.chains}`（checkpoint："
        f"`{meta.get('checkpoint', '?')}`，共 {len(rows)} 條追溯鏈，"
        f"true positive {sum(r['is_true_positive'] for r in rows)} 條）",
        f"- Peeling：depth ≥ {args.peel_min_depth}、型別嚴格交替、"
        f"時間步沿金流方向單調不減（僅檢查已知時間步，未知以 -1 sentinel 標記）、"
        f"金額（`{args.amount_col}`，原始未標準化值）遞減 —— "
        f"Spearman < {args.spearman_thresh} 或 遞減跳數比例 > {args.frac_dec_thresh}",
        f"- Fan-in：鏈上存在節點入度 ≥ 同型別 P{args.fanin_indegree_quantile * 100:.0f}"
        f"（tx 門檻 = {context['tx_p']:.6g}、wallet 門檻 = {context['wallet_p']:.6g}；"
        f"{'含 tx→tx / addr→addr 邊' if args.fanin_all_edges else '僅金流邊 wallet→tx / tx→wallet'}），"
        f"且其上游來源時間步極差 ≤ {args.fanin_time_range}",
        f"- Pass-through：錢包 last_block_appeared_in − first_received_block < "
        f"illicit 錢包分布 P{args.passthrough_quantile * 100:.0f} = "
        f"{context['gap_thresh']:.6g} blocks"
        f"（illicit 錢包 n={context['n_illicit_gap']}，"
        f"median={context['gap_median']:.6g}）",
        "",
        "## 2. 統計總表",
        "",
        "| 簽名 | 全體命中（n=%d） | true-positive 鏈命中（n=%d） |" % (
            len(rows), sum(r["is_true_positive"] for r in rows)),
        "|------|------------------|------------------------------|",
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in summarize(rows)]
    lines += ["", "### 簽名重疊情形", "", "| 組合 | 鏈數 |", "|------|------|"]
    lines += [f"| {name} | {n} |" for name, n in overlap_table(rows)]

    lines += [
        "",
        f"## 3. Peeling chain 候選案例（前 {args.top_n} 條）",
        "",
        "排序鍵：true positive 優先，其次 peel_score = depth × 金額遞減強度"
        "（強度 = max(−Spearman, 遞減跳數比例)）。",
        "",
    ]
    if not peel_rows:
        lines.append("（無命中鏈）")
    for i, r in enumerate(peel_rows[: args.top_n], 1):
        lines += [f"### 候選 {i}：target `{r['target_id']}`"
                  f"（depth={r['depth']}，peel_score={fmt(r['peel_score'])}）", ""]
        lines += chain_detail_md(by_target[r["target_id"]], r, context["tx_table"],
                                 context["wallet_table"], args.amount_col)
        lines.append("")

    if args.case_tx:
        lines += ["", f"## 4. 論文案例一比對：target `{args.case_tx}`", ""]
        if args.case_tx not in row_by_target:
            lines.append(f"（`{args.case_tx}` 不在輸入鏈檔中）")
        else:
            r = row_by_target[args.case_tx]
            verdict = ("**命中 Peeling chain 簽名**" if r["peel_hit_any"]
                       else "未命中 Peeling chain 簽名")
            lines += [
                f"{verdict}（結構條件={fmt(r['peel_struct_ok'])}，"
                f"Spearman 版={fmt(r['peel_hit_spearman'])}，"
                f"遞減跳數比例版={fmt(r['peel_hit_frac'])}）。", "",
            ]
            lines += chain_detail_md(by_target[args.case_tx], r,
                                     context["tx_table"],
                                     context["wallet_table"], args.amount_col)

    lines += [
        "", "## 5. 方法限制", "",
        "- 判定僅依結構簽名，無手法 ground truth；同一鏈可與多種模式一致。",
        "- 錢包時間步為圖內最早注資時間（loader 定義），圖內未注資錢包時間未知，"
        "已知時間不足的鏈其時間單調條件視為不可評估（不計命中）。",
        "- 金額為交易層級統計量（非逐邊 UTXO 金額），遞減趨勢是近似判準。",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


CSV_COLS = (
    "target_id", "target_type", "depth", "n_nodes", "is_true_positive",
    "root_id", "root_type", "root_is_fraud",
    "alt_ok", "time_ok", "n_time_known", "n_time_missing",
    "n_amounts", "n_amount_missing", "spearman", "frac_dec",
    "peel_struct_ok", "peel_hit_spearman", "peel_hit_frac", "peel_hit_any",
    "peel_strength", "peel_score",
    "fanin_hit", "fanin_n_hub_hits", "fanin_hub_id", "fanin_hub_type",
    "fanin_hub_indeg", "fanin_hub_time_range", "fanin_hub_n_upstream",
    "pass_hit", "pass_n_wallets_hit", "pass_n_wallets",
    "pass_n_wallet_unknown", "pass_wallet_ids",
)


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_COLS})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    meta, chains = load_chains(args.chains)
    print(f"[typology_scan] {len(chains)} chains（{args.chains}）")

    tx_table = load_tx_table(args.data_root, args.amount_col)
    wallet_table = load_wallet_table(args.data_root)
    illicit = load_illicit_wallets(args.data_root)
    illicit_gaps = wallet_table.loc[
        wallet_table.index.isin(illicit), "gap"].dropna()
    if illicit_gaps.empty:
        sys.exit("[typology_scan] 找不到任何 illicit 錢包的 gap 分布")
    gap_thresh = float(illicit_gaps.quantile(args.passthrough_quantile))

    tables = build_indegree_tables(
        args.data_root,
        tx_ids=tx_table.index.tolist(),
        wallet_ids=wallet_table.index.tolist(),
        all_edges=args.fanin_all_edges,
        quantile=args.fanin_indegree_quantile,
    )
    print(f"[typology_scan] 入度 P{args.fanin_indegree_quantile * 100:.0f}: "
          f"tx={tables['tx_p']:.6g}, wallet={tables['wallet_p']:.6g}; "
          f"pass-through gap 門檻={gap_thresh:.6g}")

    chain_tx_ids = {n["real_id"] for c in chains for n in c["nodes"]
                    if n["type"] == "transaction"}
    chain_wallet_ids = {n["real_id"] for c in chains for n in c["nodes"]
                        if n["type"] == "wallet"}
    tx_time = tx_table["Time step"].astype(int).to_dict()
    upstream_times, _ = build_upstream_time_maps(
        tables, tx_time, chain_tx_ids, chain_wallet_ids, args.fanin_all_edges)

    rows = [scan_chain(c, tx_table, wallet_table, tables, upstream_times,
                       gap_thresh, args) for c in chains]

    write_csv(args.out_csv, rows)
    print(f"[typology_scan] 逐鏈判定表 → {args.out_csv}")

    context = {
        "tx_p": tables["tx_p"], "wallet_p": tables["wallet_p"],
        "gap_thresh": gap_thresh, "n_illicit_gap": int(len(illicit_gaps)),
        "gap_median": float(illicit_gaps.median()),
        "tx_table": tx_table, "wallet_table": wallet_table,
    }
    write_report(args.out_report, meta, rows, chains, args, context)
    print(f"[typology_scan] 報告 → {args.out_report}")

    for label, all_hit, tp_hit in summarize(rows):
        print(f"  {label}: 全體 {all_hit} / TP {tp_hit}")


if __name__ == "__main__":
    main()
