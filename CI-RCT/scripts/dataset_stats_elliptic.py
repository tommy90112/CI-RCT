"""Elliptic++ 資料集基本敘述統計（涵蓋 transaction / wallet / joint 三個變體）。

標籤編碼（與 utils/elliptic_plus_loader.py 一致）：
    class 1 = illicit (詐欺)  → 正類 y=1
    class 2 = licit   (正常)  → 負類 y=0
    class 3 = unknown (未知)  → 不參與訓練/評估，僅作圖結構上下文

三個變體的節點口徑（重點）：
    - transaction 變體：以交易節點為監督目標。
    - wallet 變體 & joint 變體：錢包節點採 ``wallet_per_address=True``，
      即每個「位址」去重成「一個」節點（保留最新 time step），做 address-level
      的分層切分，避免同一位址跨 time step 造成 train/test 洩漏。
      ⚠️ 注意：這與舊版 per-(address, timestep) 逐列設計數字不同。舊版逐列會把
      同一位址算多次（labeled=367,472）；目前去重後 labeled=265,354。

分層切分：70/15/15，per-class 各自切分，n_train=int(n*0.70)、n_val=int(n*0.15)、
其餘為 test（完全複現 elliptic_plus_loader._stratified_masks 的 int 截斷）。

用法：
    cd CI-RCT
    python scripts/dataset_stats_elliptic.py
    python scripts/dataset_stats_elliptic.py --data_root data/Elliptic++
"""

import argparse
from pathlib import Path

import pandas as pd

# 標籤編碼 → 可讀名稱
CLASS_ILLICIT, CLASS_LICIT, CLASS_UNKNOWN = 1, 2, 3
CLASS_NAMES = {
    CLASS_ILLICIT: "illicit (詐欺)",
    CLASS_LICIT: "licit (正常)",
    CLASS_UNKNOWN: "unknown (未知)",
}

# 分層切分比例（與 elliptic_plus_loader.TRAIN_RATIO / VAL_RATIO 一致）
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


# ── 通用小工具 ────────────────────────────────────────────────────────────────

def _class_counts(classes: pd.Series) -> dict:
    """回傳 {1: n, 2: n, 3: n}。"""
    vc = classes.value_counts()
    return {c: int(vc.get(c, 0)) for c in (CLASS_ILLICIT, CLASS_LICIT, CLASS_UNKNOWN)}


def _print_dist(title: str, counts: dict) -> None:
    total = sum(counts.values())
    print(f"\n--- {title} ---")
    print(f"{'class':<6}{'名稱':<18}{'數量':>12}{'佔比':>10}")
    print("-" * 46)
    for c in (CLASS_ILLICIT, CLASS_LICIT, CLASS_UNKNOWN):
        n = counts[c]
        pct = n / total * 100 if total else 0.0
        print(f"{c:<6}{CLASS_NAMES[c]:<18}{n:>12,}{pct:>9.2f}%")
    print("-" * 46)
    labeled = counts[CLASS_ILLICIT] + counts[CLASS_LICIT]
    print(f"{'':<6}{'合計':<18}{total:>12,}")
    print(f"{'':<6}{'labeled (1+2)':<18}{labeled:>12,}")


def _stratified_split_counts(n_illicit: int, n_licit: int) -> dict:
    """複現 _stratified_masks 的 per-class int 截斷切分，回傳各集合筆數。"""
    def split(n):
        n_tr = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        return n_tr, n_val, n - n_tr - n_val

    i_tr, i_val, i_te = split(n_illicit)
    l_tr, l_val, l_te = split(n_licit)
    return {
        "train": (i_tr + l_tr, i_tr, l_tr),
        "val": (i_val + l_val, i_val, l_val),
        "test": (i_te + l_te, i_te, l_te),
    }


def _print_split(split: dict) -> None:
    print(f"\n  分層切分 70/15/15（labeled 節點；illicit / licit 各自切分）")
    print(f"  {'集合':<8}{'總數':>10}{'illicit':>10}{'licit':>10}")
    print("  " + "-" * 38)
    for name, key in (("train", "train"), ("val", "val"), ("test", "test")):
        tot, ill, lic = split[key]
        print(f"  {name:<8}{tot:>10,}{ill:>10,}{lic:>10,}")
    total = sum(split[k][0] for k in ("train", "val", "test"))
    print("  " + "-" * 38)
    print(f"  {'合計':<8}{total:>10,}")


# ── 資料載入（輕量：只讀必要欄位，免 torch/pyg）────────────────────────────────

def _load_connected_wallet_universe(root: Path, include_addr_addr: bool) -> pd.DataFrame:
    """複現 loader 的 connected-set 過濾 + 每位址去重（保留最新 time step）。

    回傳一個 DataFrame，每列一個 wallet 節點（address 唯一），含其 class。
    """
    # wallets_features.csv 前兩欄 = address / Time step（per-(addr, timestep) 逐列）
    wf = pd.read_csv(root / "wallets_features.csv", usecols=[0, 1])
    wf.columns = ["address", "Time step"]

    # connected 集合：出現在 AddrTx(input) 或 TxAddr(output) 的位址；
    # include_addr_addr 時再併入 AddrAddr 兩端。
    at = pd.read_csv(root / "AddrTx_edgelist.csv")
    at.columns = [c.strip() for c in at.columns]
    ta = pd.read_csv(root / "TxAddr_edgelist.csv")
    ta.columns = [c.strip() for c in ta.columns]
    connected = set(at["input_address"].dropna()) | set(ta["output_address"].dropna())
    if include_addr_addr:
        aa = pd.read_csv(root / "AddrAddr_edgelist.csv")
        aa.columns = [c.strip() for c in aa.columns]
        connected |= set(aa["input_address"].dropna()) | set(aa["output_address"].dropna())

    filt = wf[wf["address"].isin(connected)].copy()
    # 每位址去重（保留最新 Time step，等同 _dedup_wallets_per_address 的計數）
    dedup = filt.sort_values("Time step").drop_duplicates("address", keep="last")

    wcls = pd.read_csv(root / "wallets_classes.csv")
    wcls.columns = [c.strip() for c in wcls.columns]
    cls_map = dict(zip(wcls["address"], wcls["class"]))
    dedup["class"] = dedup["address"].map(lambda a: cls_map.get(a, CLASS_UNKNOWN))
    return dedup.reset_index(drop=True)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Elliptic++ 敘述統計（三變體）")
    parser.add_argument("--data_root", type=str, default="data/Elliptic++")
    parser.add_argument(
        "--include_addr_addr", type=lambda x: x.lower() == "true", default=True,
        help="wallet connected 集合是否併入 AddrAddr 兩端（預設 true，對齊 Exp-05 主結果）",
    )
    args = parser.parse_args()
    root = Path(args.data_root)

    print("=" * 46)
    print("Elliptic++ 資料集基本敘述統計")
    print(f"資料夾：{root.resolve()}")
    print(f"include_addr_addr = {args.include_addr_addr}")
    print("=" * 46)

    # ── 0. 原始資料集（*_classes.csv 唯一節點）──────────────────────────────
    txs_cls = pd.read_csv(root / "txs_classes.csv")
    txs_cls.columns = [c.strip() for c in txs_cls.columns]
    tx_counts = _class_counts(txs_cls["class"])

    wallets_cls = pd.read_csv(root / "wallets_classes.csv")
    wallets_cls.columns = [c.strip() for c in wallets_cls.columns]
    wallet_raw_counts = _class_counts(wallets_cls["class"])

    print("\n########## 0. 原始資料集（唯一節點，未過濾）##########")
    _print_dist("交易 transaction（txs_classes.csv）", tx_counts)
    _print_dist("錢包 wallet（wallets_classes.csv，唯一位址）", wallet_raw_counts)

    # ── 1. transaction 變體 ──────────────────────────────────────────────────
    print("\n\n########## 1. transaction 變體（監督目標：交易）##########")
    _print_dist("交易節點標籤分布", tx_counts)
    tx_split = _stratified_split_counts(tx_counts[CLASS_ILLICIT], tx_counts[CLASS_LICIT])
    _print_split(tx_split)

    # ── 2. wallet 變體（= joint 的 wallet 端；per-address 去重）───────────────
    print("\n\n########## 2. wallet 變體（監督目標：錢包；per-address 去重）##########")
    wallet_univ = _load_connected_wallet_universe(root, args.include_addr_addr)
    wallet_counts = _class_counts(wallet_univ["class"])
    _print_dist("connected + 去重後的錢包節點標籤分布", wallet_counts)
    w_split = _stratified_split_counts(wallet_counts[CLASS_ILLICIT], wallet_counts[CLASS_LICIT])
    _print_split(w_split)

    # ── 3. joint 變體（交易 primary + 錢包 clean overlay）─────────────────────
    print("\n\n########## 3. joint 變體（交易 + 錢包 同圖聯合監督）##########")
    print("  交易端 = transaction 變體；錢包端 = wallet 變體（同上，per-address 去重）。")
    _print_dist("交易節點", tx_counts)
    _print_dist("錢包節點（去重）", wallet_counts)

    # ── 總結對照表 ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 46)
    print("總結對照（labeled = class 1 + class 2）")
    print("=" * 46)
    tx_lab = tx_counts[CLASS_ILLICIT] + tx_counts[CLASS_LICIT]
    w_lab = wallet_counts[CLASS_ILLICIT] + wallet_counts[CLASS_LICIT]
    print(f"交易節點：總 {sum(tx_counts.values()):,}｜"
          f"illicit {tx_counts[CLASS_ILLICIT]:,}｜licit {tx_counts[CLASS_LICIT]:,}｜"
          f"unknown {tx_counts[CLASS_UNKNOWN]:,}｜labeled {tx_lab:,}")
    print(f"錢包節點（去重）：總 {sum(wallet_counts.values()):,}｜"
          f"illicit {wallet_counts[CLASS_ILLICIT]:,}｜licit {wallet_counts[CLASS_LICIT]:,}｜"
          f"unknown {wallet_counts[CLASS_UNKNOWN]:,}｜labeled {w_lab:,}")
    print(f"\n⚠️ 舊版 per-(address, timestep) 逐列口徑的 labeled wallet 為 367,472"
          f"（illicit 28,601 / licit 338,871），與目前去重口徑不同，請勿混用。")


if __name__ == "__main__":
    main()
