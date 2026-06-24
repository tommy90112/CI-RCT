"""
diag_wallet_tx_ce_sign.py — 一次性診斷腳本（可刪除，不在 git 追蹤）

目的
────
回答論文裡的問題：「Elliptic++ 的 wallet→transaction 邊，CE 為何多半是負的
（均值 ≈ −0.35）？」本腳本提供兩條互補證據，把「現象」變成「為什麼」。

CE 定義回顧（model/hetero_ncm.py:compute_causal_effect）：
    CE(wallet→tx) = p_actual − p_null
    p_actual = MLP_τ([ h_wallet           ‖ type_emb_wallet ])
    p_null   = MLP_τ([ 0（同維度零向量）  ‖ type_emb_wallet ])
所以 CE < 0  ⇔  p_null > p_actual（把錢包特徵歸零後，讀出的詐騙分數反而更高）。

兩層診斷
────────
A（成分拆解，免標籤）：對每條 wallet→tx 邊，分別算出 p_actual 與 p_null 的
    分佈。假說：p_null 系統性高於 p_actual ——零基線是分布外(OOD)的退化點，
    沒有被「這看起來像正常錢包」的特徵往下壓，所以基線偏高、CE 偏負。

B（依錢包真實標籤拆 CE）：把 wallet→tx 的 CE 依「來源錢包的真實 class」
    （1=illicit / 2=licit / 3=unknown）分組。假說：
      - licit 錢包 → 大量負 CE（合法性訊號把分數往下壓）；
      - illicit 錢包 → CE 明顯較高甚至為正（真實特徵把分數往上推）。
    若成立，這同時是「模型確實學到 illicit→推高、licit→壓低」的漂亮佐證。

用法（在 server，跟你慣用的 evaluate 指令同參數）
────────────────────────────────────────────────
  python scripts/diag_wallet_tx_ce_sign.py \
    --dataset elliptic++ --variant joint --data_root data \
    --checkpoint checkpoints/ci_rct_elliptic++_best.pt \
    --include_addr_addr true --hidden_dim 128 \
    --node_limit 20000 --num_seeds 20 --device cuda

備註
────
* 直接重用 evaluate.parse_args() —— 旗標與正式 evaluate 完全一致，因果圖以
  相同方式建立，wallet→tx 邊集合與你 `--debug` 的 DIAGNOSTIC 1 可互相對照。
* 錢包標籤直接讀 wallets_classes.csv（不依賴 loader 是否掛 .y），並用
  lfpn_utils._rebuild_wallet_to_idx 還原成與圖一致的錢包排序。
"""
import os
import sys
from collections import Counter, defaultdict

import torch

# 讓 scripts/ 下能 import 專案根的 evaluate / model / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from utils.data_utils import (
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)

# 直接重用 evaluate.py 的載入 / 解析邏輯，確保跟正式 evaluate 完全同路徑
import evaluate as ev

WALLET_TYPE = "wallet"
TX_TYPE = "transaction"
CLASS_NAME = {1: "illicit", 2: "licit", 3: "unknown"}


def build_pipeline(args, device):
    """複製 evaluate.main() 的 load → graph → model，回傳診斷需要的物件。

    含 arch_get（讀回 checkpoint 內嵌架構，避免 num_hgt_layers 不符導致權重錯位）
    與 joint variant 分支，使本腳本對 transaction / wallet / joint 皆正確。
    """
    # ── 從 checkpoint 還原架構（v2 格式優先於 CLI 旗標）─────────────────────
    ckpt_arch = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_arch = CI_RCT.read_arch_metadata(args.checkpoint, device=args.device)
    if ckpt_arch is None and args.checkpoint:
        print("  [arch] checkpoint 無內嵌架構（legacy）— 改用 CLI 旗標；"
              "請確認 --num_hgt_layers 等與訓練配方一致，否則 F1 會默默崩。")

    def arch_get(key, cli_value):
        if ckpt_arch and ckpt_arch.get(key) is not None:
            stored = ckpt_arch[key]
            if stored != cli_value:
                print(f"  [arch] {key}: checkpoint={stored}（覆蓋 CLI={cli_value}）")
            return stored
        return cli_value

    print(f"Loading dataset: {args.dataset} (variant={args.variant})")
    data, target_type = ev._load_variant_dataset(args)
    data = data.to(device)
    labels = data[target_type].y
    test_mask = data[target_type].test_mask

    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [offset + i for i in test_indices if labels[i].item() == 1]

    fraud_wallet_ids = []
    if (args.variant == "joint" and WALLET_TYPE in data.node_types
            and hasattr(data[WALLET_TYPE], "y")):
        w_off = type_offsets[WALLET_TYPE]
        fraud_wallet_ids = [
            w_off + i
            for i in (data[WALLET_TYPE].y == 1).nonzero(as_tuple=True)[0].tolist()
        ]

    gt_list = ev.build_gt_list(args, data, type_offsets)
    gt_tx_ids = []
    for _, gt in gt_list:
        gt_tx_ids.extend(gt.keys())
    seed_ids = list(dict.fromkeys(
        fraud_global_ids[:args.num_seeds]
        + fraud_wallet_ids[:args.num_seeds]
        + gt_tx_ids
    ))

    # rare / blocked edge types — 與 evaluate.main 相同的解析
    raw = args.rare_edge_types.strip()
    if raw.lower() == "none":
        rare_edge_types = set()
    elif raw:
        rare_edge_types = {t.strip() for t in raw.split(",") if t.strip()}
    else:
        rare_edge_types = default_rare_edge_types(args.dataset)

    raw_b = args.blocked_edge_types.strip()
    if raw_b.lower() == "none":
        blocked_edge_types = set()
    elif raw_b:
        blocked_edge_types = {t.strip() for t in raw_b.split(",") if t.strip()}
    else:
        blocked_edge_types = default_blocked_edge_types(args.dataset)

    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=seed_ids if seed_ids else None,
        hop_limit=args.hop_limit,
        node_limit=args.node_limit,
        blocked_edge_types=blocked_edge_types if blocked_edge_types else None,
        rare_edge_types=rare_edge_types if rare_edge_types else None,
        rare_reserve=args.rare_edge_reserve,
        rare_max_hops=args.rare_edge_max_hops,
    )
    print(f"  Causal graph: {len(causal_graph.v)} nodes, "
          f"{len(causal_graph.edge_type_map)} directed edges")

    config = CI_RCT_Config(
        dataset=args.dataset,
        target_node_type=target_type,
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        top_k_paths=args.top_k,
        hidden_dim=arch_get("hidden_dim", args.hidden_dim),
        num_hgt_layers=arch_get("num_hgt_layers", args.num_hgt_layers),
        num_heads=arch_get("num_heads", args.num_heads),
        dropout=arch_get("dropout", args.dropout),
        node_type_emb_dim=arch_get("node_type_emb_dim", args.type_emb_dim),
    )
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }
    backbone_exclude_node_types = arch_get("backbone_exclude_node_types", [])

    if args.variant == "joint":
        from model.ci_rct_joint import CI_RCT_Joint
        aux_node_types = arch_get("aux_node_types", [WALLET_TYPE])
        aux_num_classes = arch_get("aux_num_classes", {}) or {
            t: int(data[t].y.max().item()) + 1 for t in aux_node_types
        }
        model = CI_RCT_Joint(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            use_gan=False,
            backbone_exclude_node_types=backbone_exclude_node_types,
            aux_node_types=list(aux_node_types),
            aux_num_classes=aux_num_classes,
        ).to(device)
    else:
        model = CI_RCT(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            use_gan=False,
            backbone_exclude_node_types=backbone_exclude_node_types,
        ).to(device)

    model.eval()
    with torch.no_grad():
        model.forward(data)  # warm-up（lazy HGTConv 權重）
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — 用隨機初始化模型（baseline，僅供 sanity check）。")

    return model, data, causal_graph, type_offsets, target_type


def load_wallet_class_map(args, wallet_offset):
    """{wallet_global_id: class∈{1,2,3}}，以與 loader 一致的錢包排序建立。

    讀 wallets_classes.csv，並用 lfpn_utils._rebuild_wallet_to_idx 還原
    address→local_idx（與圖內錢包 offset 對齊）。失敗時回傳空 dict（DIAG B 略過）。
    """
    try:
        import pandas as pd
        from pathlib import Path
        from utils.lfpn_utils import _rebuild_wallet_to_idx
    except Exception as e:
        print(f"  [wallet_class] 無法載入 helper：{e}")
        return {}

    root = Path(os.path.join(args.data_root, "Elliptic++"))
    try:
        wallets_cls = pd.read_csv(root / "wallets_classes.csv")
        wallets_cls.columns = [c.strip() for c in wallets_cls.columns]
        wallets = pd.read_csv(root / "wallets_features.csv", usecols=[0])
        wallets.columns = ["address"]
        txs_feat = pd.read_csv(root / "txs_features.csv", usecols=[0])
        txs_feat.columns = ["txId"]
        txs_cls_df = pd.read_csv(root / "txs_classes.csv")
        txs_cls_df.columns = [c.strip() for c in txs_cls_df.columns]
        addr_tx = pd.read_csv(root / "AddrTx_edgelist.csv")
        addr_tx.columns = [c.strip() for c in addr_tx.columns]
        tx_addr = pd.read_csv(root / "TxAddr_edgelist.csv")
        tx_addr.columns = [c.strip() for c in tx_addr.columns]
        addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
        addr_addr.columns = [c.strip() for c in addr_addr.columns]

        tx_to_idx = {tid: i for i, tid in enumerate(txs_feat["txId"].tolist())}
        wallet_to_idx = _rebuild_wallet_to_idx(
            wallets=wallets,
            wallets_cls=wallets_cls,
            txs_cls=txs_cls_df,
            tx_to_idx=tx_to_idx,
            addr_tx=addr_tx,
            tx_addr=tx_addr,
            addr_addr=addr_addr,
            include_addr_addr=args.include_addr_addr,
            fraud_subgraph=args.fraud_subgraph,
            fraud_subgraph_hops=args.fraud_subgraph_hops,
            verbose=False,
            wallet_per_address=args.variant in ("wallet", "joint"),
        )
        cls_by_addr = {
            str(a): int(c)
            for a, c in zip(wallets_cls["address"].astype(str), wallets_cls["class"])
        }
        return {
            wallet_offset + local: cls_by_addr[str(a)]
            for a, local in wallet_to_idx.items()
            if str(a) in cls_by_addr
        }
    except Exception as e:
        print(f"  [wallet_class] 建立錢包標籤對照失敗：{e}")
        return {}


def _summary(name, arr):
    """印一行分佈摘要。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        print(f"  {name:<10s}: (空)")
        return
    print(f"  {name:<10s}: n={arr.size:>7d}  mean={arr.mean():>+8.4f}  "
          f"median={np.median(arr):>+8.4f}  std={arr.std():>7.4f}  "
          f"min={arr.min():>+7.4f}  max={arr.max():>+7.4f}")


def _text_hist(ce, lo=-1.0, hi=1.0, bins=20, width=40):
    """CE 的文字直方圖。"""
    ce = np.asarray(ce, dtype=float)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(ce, lo, hi), bins=edges)
    n = max(1, counts.max())
    print(f"\n  CE 直方圖（{ce.size} 條 wallet→tx 邊；│標記 CE=0）")
    for i in range(bins):
        l, r = edges[i], edges[i + 1]
        bar = "█" * int(width * counts[i] / n)
        mark = " 0" if l <= 0.0 < r else ""
        print(f"    [{l:>+5.2f},{r:>+5.2f}) {counts[i]:>7d} {bar}{mark}")


def diagnose(model, data, causal_graph, type_offsets, args):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        _, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)

    if not hasattr(model, "hetero_ncm"):
        print("  [error] model 沒有 hetero_ncm 屬性，無法拆 p_actual/p_null。")
        return
    ncm = model.hetero_ncm

    # ── 蒐集 wallet→tx 邊（依 node_type 判定，邊型別字串取自 edge_type_map）──
    by_etype = defaultdict(list)  # etype -> [(src, dst), ...]
    for (src, dst), etype in causal_graph.edge_type_map.items():
        if (causal_graph.node_type.get(src) == WALLET_TYPE
                and causal_graph.node_type.get(dst) == TX_TYPE
                and src in flat_h):
            by_etype[etype].append((src, dst))

    n_edges = sum(len(v) for v in by_etype.values())
    if n_edges == 0:
        print("\n沒有 wallet→transaction 邊在因果圖內。"
              "檢查 --variant / --include_addr_addr / 圖預算（--node_limit 等）。")
        return
    print(f"\n蒐集到 {n_edges} 條 wallet→tx 邊，邊型別：{sorted(by_etype)}")

    # ── 向量化拆出 p_actual / p_null（每個邊型別各跑一次其專屬 MLP）─────────
    rows = []  # (src, dst, etype, p_actual, p_null, ce)
    with torch.no_grad():
        wallet_type_idx = torch.tensor(
            ncm.node_type_to_idx.get(WALLET_TYPE, 0), dtype=torch.long, device=device
        )
        type_emb = ncm.type_embeddings(wallet_type_idx)  # [T]
        for etype, elist in by_etype.items():
            if etype not in ncm.edge_type_models:
                print(f"  [warn] 邊型別 '{etype}' 不在 NCM，略過 {len(elist)} 條。")
                continue
            mlp = ncm.edge_type_models[etype]
            H = torch.stack([flat_h[s] for (s, _) in elist]).to(device)  # [N, D]
            T = type_emb.unsqueeze(0).expand(H.size(0), -1)               # [N, T]
            p_actual = mlp(torch.cat([H, T], dim=-1)).squeeze(-1)         # [N]
            p_null = mlp(torch.cat([torch.zeros_like(H), T], dim=-1)).squeeze(-1)
            ce = p_actual - p_null
            for (s, d), pa, pn, c in zip(
                elist, p_actual.tolist(), p_null.tolist(), ce.tolist()
            ):
                rows.append((s, d, etype, pa, pn, c))

    pa = np.array([r[3] for r in rows])
    pn = np.array([r[4] for r in rows])
    ce = np.array([r[5] for r in rows])

    # ── DIAG A：p_actual vs p_null vs CE ───────────────────────────────────
    print("\n" + "═" * 70)
    print("  [DIAG A] wallet→tx 的 CE 成分拆解（CE = p_actual − p_null）")
    print("═" * 70)
    _summary("p_actual", pa)
    _summary("p_null", pn)
    _summary("CE", ce)
    print(f"\n  CE < 0 的比例            : {float((ce < 0).mean()) * 100:6.2f}%")
    print(f"  p_null > p_actual 的比例 : {float((pn > pa).mean()) * 100:6.2f}%")
    print(f"  → 若兩者都高，代表『零基線系統性偏高』就是 CE 偏負的直接成因。")
    _text_hist(ce)

    # ── DIAG B：依來源錢包真實 class 拆 CE ─────────────────────────────────
    print("\n" + "═" * 70)
    print("  [DIAG B] wallet→tx 的 CE 依『來源錢包真實 class』分組")
    print("═" * 70)
    cls_map = load_wallet_class_map(args, type_offsets[WALLET_TYPE])
    if not cls_map:
        print("  錢包標籤對照為空，DIAG B 略過。")
    else:
        by_class = defaultdict(lambda: {"ce": [], "pa": [], "pn": []})
        for (s, d, et, p_a, p_n, c) in rows:
            cl = cls_map.get(s)  # None = 不在標籤表
            key = CLASS_NAME.get(cl, "no_label")
            by_class[key]["ce"].append(c)
            by_class[key]["pa"].append(p_a)
            by_class[key]["pn"].append(p_n)

        print(f"  {'class':<10s} {'n':>7s} {'mean_CE':>9s} {'med_CE':>9s} "
              f"{'%CE<0':>7s} {'mean_pa':>9s} {'mean_pn':>9s}")
        order = ["illicit", "licit", "unknown", "no_label"]
        for key in order + [k for k in by_class if k not in order]:
            if key not in by_class:
                continue
            d = by_class[key]
            c = np.asarray(d["ce"]); a = np.asarray(d["pa"]); p = np.asarray(d["pn"])
            print(f"  {key:<10s} {c.size:>7d} {c.mean():>+9.4f} "
                  f"{np.median(c):>+9.4f} {float((c < 0).mean()) * 100:>6.1f}% "
                  f"{a.mean():>+9.4f} {p.mean():>+9.4f}")
        print("\n  假說預測：illicit 的 mean_CE 明顯高於 licit（甚至為正），"
              "licit 以負 CE 為主 → 證實『illicit 推高、licit 壓低』。")

    # ── DIAG C：聚焦「dst 為 fraud tx」的邊 —— 追溯首跳真正關心的子集 ────────
    # tx 標籤：class 1(illicit)→y=1；class 2/3 →y=0（見 elliptic_plus_loader）。
    print("\n" + "═" * 70)
    print("  [DIAG C] 只看 dst=fraud tx 的 wallet→tx 邊：|CE| 會把首跳帶向誰？")
    print("═" * 70)
    tx_off = type_offsets[TX_TYPE]
    tx_y_cpu = data[TX_TYPE].y.detach().cpu()

    def dst_is_fraud(dst):
        local = dst - tx_off
        return 0 <= local < tx_y_cpu.numel() and int(tx_y_cpu[local].item()) == 1

    if not cls_map:
        print("  錢包標籤對照為空，DIAG C 略過（C1/C2 需要錢包 class）。")
    else:
        # C1：依 (dst 是否 fraud) × (來源錢包 class) 聚合 CE / |CE|
        agg = defaultdict(lambda: {"ce": [], "abs": []})
        for (s, d, et, p_a, p_n, c) in rows:
            scl = CLASS_NAME.get(cls_map.get(s), "no_label")
            dkey = "fraud_tx" if dst_is_fraud(d) else "non_fraud_tx"
            agg[(dkey, scl)]["ce"].append(c)
            agg[(dkey, scl)]["abs"].append(abs(c))
        print("  [C1] (dst 是否 fraud) × (來源錢包 class) 的 CE / |CE| 聚合")
        print(f"  {'dst':<13s} {'src_class':<10s} {'n':>7s} "
              f"{'mean_CE':>9s} {'mean|CE|':>9s}")
        for dkey in ("fraud_tx", "non_fraud_tx"):
            for scl in ("illicit", "licit", "unknown", "no_label"):
                if (dkey, scl) not in agg:
                    continue
                g = agg[(dkey, scl)]
                ce_a = np.asarray(g["ce"]); ab = np.asarray(g["abs"])
                print(f"  {dkey:<13s} {scl:<10s} {ce_a.size:>7d} "
                      f"{ce_a.mean():>+9.4f} {ab.mean():>9.4f}")

        # C2：對每個 fraud tx，比較「|CE|-max 的錢包父」與「signed-CE-max 的錢包父」
        # 各是什麼 class —— 直接量化『現用 |CE| 選擇』vs『帶號選擇』把首跳帶向誰。
        parents_by_ftx = defaultdict(list)  # dst -> [(src, ce)]
        for (s, d, et, p_a, p_n, c) in rows:
            if dst_is_fraud(d):
                parents_by_ftx[d].append((s, c))
        absmax_tally, signedmax_tally = Counter(), Counter()
        for d, plist in parents_by_ftx.items():
            bs_abs = max(plist, key=lambda t: abs(t[1]))[0]   # |CE|-max（現用）
            bs_sgn = max(plist, key=lambda t: t[1])[0]        # signed-CE-max
            absmax_tally[CLASS_NAME.get(cls_map.get(bs_abs), "no_label")] += 1
            signedmax_tally[CLASS_NAME.get(cls_map.get(bs_sgn), "no_label")] += 1
        n_ftx = len(parents_by_ftx)
        cols = ("illicit", "licit", "unknown", "no_label")
        print(f"\n  [C2] {n_ftx} 個 fraud tx（≥1 錢包父）首跳『會選到的錢包父』class 分佈")
        print(f"  {'rule':<16s} " + "".join(f"{k:>10s}" for k in cols))
        for name, tal in (("|CE|-max(now)", absmax_tally),
                          ("signed-CE-max", signedmax_tally)):
            cells = "".join(
                f"{(100 * tal.get(k, 0) / max(1, n_ftx)):>9.1f}%" for k in cols
            )
            print(f"  {name:<16s} {cells}")
        print("\n  判讀：")
        print("   · |CE|-max(now) 的 illicit% 高 → 紅旗 B 在實務上無害，可放心寫『|CE| 設計正當』。")
        print("   · 若其 illicit% 偏低、licit% 高，且 signed-CE-max 的 illicit% 明顯較高")
        print("     → |CE| 確實把首跳帶向 licit，論文須誠實寫成 limitation（並靠 tie-break 補救）。")

    # ── 輸出逐邊 CSV（供畫圖）── cls_map 在 DIAG B 已建立（可能為空 dict）──
    ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint or "nockpt"))[0]
    out_dir = os.path.join("logs", "diag_wallet_ce")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{ckpt_tag}_wallet_tx_ce.csv")
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_global", "dst_global", "edge_type",
                    "src_wallet_class", "src_wallet_class_name", "dst_is_fraud_tx",
                    "p_actual", "p_null", "ce"])
        for (s, d, et, p_a, p_n, c) in rows:
            cl = cls_map.get(s)
            w.writerow([s, d, et, cl if cl is not None else "",
                        CLASS_NAME.get(cl, ""), int(dst_is_fraud(d)),
                        f"{p_a:.6f}", f"{p_n:.6f}", f"{c:.6f}"])
    print(f"\n[csv] 逐邊明細已寫出 → {out_csv}（{len(rows)} 列，可用於畫分佈圖）")


def main():
    args = ev.parse_args()
    device = torch.device(args.device)
    if args.dataset != "elliptic++":
        print(f"  [warn] 本診斷針對 Elliptic++ 的 wallet→tx 邊；"
              f"目前 --dataset={args.dataset}，DIAG B 的標籤對照可能不適用。")
    model, data, causal_graph, type_offsets, target_type = build_pipeline(args, device)
    diagnose(model, data, causal_graph, type_offsets, args)
    print(f"\n{'─' * 55}\n  診斷完成。\n{'─' * 55}\n")


if __name__ == "__main__":
    main()
