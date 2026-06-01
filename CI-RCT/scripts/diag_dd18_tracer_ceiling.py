"""
diag_dd18_tracer_ceiling.py — 一次性診斷腳本（可刪除，不在 git 追蹤）

目的
────
dd18 (GAN, high-AUC) 在 evaluate 加了 --prefer_root_types process_node 後
RCP 仍只有 0.195（447 條 trace 停在 host、116 條爬到 process）。
本腳本判定：tie-break 是「圖結構天花板（host 祖先裡根本沒 process）」
還是「邏輯沒生效（明明有 process 候選卻沒選）」。

兩層診斷
────────
L1（純結構，免 CE）：對每個 fraud flow，反向 BFS 看「k 跳內能否碰到 process_node」。
    若多數 host-stopped target 的祖先裡根本沒有 process → 結構天花板鐵證。
L2（要 CE，重跑 trace）：對每條實際 trace 記錄
    - 停止原因（no_parents / weak_ce / cycle / max_hops）
    - 停在 host 時，當前節點上游的「型別分佈」與「process 候選是否通過 CE 門檻」
    → 直接回答「tie-break 為何沒把它推向 process」。

用法（在 server，跟你慣用的 evaluate 指令同參數）
────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python scripts/diag_dd18_tracer_ceiling.py \
  --dataset unsw_mg24 --data_root data \
  --checkpoint checkpoints/dd18/ci_rct_unsw_mg24_best.pt \
  --mg24_split_mode by_incident --mg24_host_role zeroed \
  --mg24_subsample_ddos 0.1 --num_hgt_layers 2 \
  --max_explain 500 --node_limit 100000 --ce_threshold 0.0001 \
  --prefer_root_types process_node --debug
"""
import os
import sys
from collections import Counter, defaultdict, deque

import torch

# 讓 scripts/ 下能 import 專案根的 evaluate / model / utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from model.root_cause_tracer import RootCauseTracer
from utils.data_utils import (
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
)

# 直接重用 evaluate.py 的載入 / 解析邏輯，確保跟正式 evaluate 完全同路徑
import evaluate as ev

PREFER_BFS_MAX = 6  # L1 結構 BFS 的最大反向跳數（比 tracer max_hops 寬，看天花板）


def build_pipeline(args, device):
    """複製 evaluate.main() 的 load → graph → model → CE，回傳診斷需要的物件。"""
    data, target_type = ev.load_dataset(
        args.dataset, args.data_root,
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
        max_flows=args.max_flows,
        mg24_subsample_ddos=args.mg24_subsample_ddos,
        mg24_min_host_flows=args.mg24_min_host_flows,
        mg24_prune_external=args.mg24_prune_external,
        mg24_split_mode=args.mg24_split_mode,
        mg24_host_role=args.mg24_host_role,
        mg24_drop_features=args.mg24_drop_features,
        seed=args.seed,
    )
    data = data.to(device)
    labels = data[target_type].y
    test_mask = data[target_type].test_mask

    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [offset + i for i in test_indices if labels[i].item() == 1]

    gt_list = ev.build_gt_list(args, data, type_offsets)
    gt_tx_ids = []
    for _, gt in gt_list:
        gt_tx_ids.extend(gt.keys())
    seed_ids = list(dict.fromkeys(fraud_global_ids[:args.num_seeds] + gt_tx_ids))

    # rare / blocked edge types — 與 evaluate.main 相同的解析
    raw = args.rare_edge_types.strip()
    if raw.lower() == "none":
        rare_edge_types = set()
    elif raw:
        rare_edge_types = {t.strip() for t in raw.split(",") if t.strip()}
    else:
        rare_edge_types = ev.default_rare_edge_types(args.dataset)

    raw_b = args.blocked_edge_types.strip()
    if raw_b.lower() == "none":
        blocked_edge_types = set()
    elif raw_b:
        blocked_edge_types = {t.strip() for t in raw_b.split(",") if t.strip()}
    else:
        blocked_edge_types = ev.default_blocked_edge_types(args.dataset)

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
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_type_emb_dim=args.type_emb_dim,
    )
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }
    model = CI_RCT(
        config=config,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        use_gan=False,
    ).to(device)
    model.eval()
    with torch.no_grad():
        model.forward(data)  # warm-up（lazy HGTConv）
    model.load_checkpoint(args.checkpoint, device=args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)
    causal_effects = model.compute_causal_effects(flat_h, causal_graph)

    fraud_predicted_global = [
        offset + idx for idx in test_indices
        if logits[idx].argmax().item() == 1 and (offset + idx) in causal_graph.set_v
    ][: args.max_explain]

    # fraud_label_set（RCP 命中判定用）— 完全對齊 evaluate.py line 406-440：
    # flow fraud（test）+ 所有 process_node / measurement_node 標記惡意者（global id）
    fraud_label_set = set(fraud_global_ids)
    if args.dataset == "unsw_mg24":
        for ntype in ("process_node", "measurement_node"):
            if ntype not in data.node_types or not hasattr(data[ntype], "y"):
                continue
            ntype_offset = type_offsets.get(ntype, 0)
            ntype_labels = data[ntype].y
            fraud_label_set |= {
                ntype_offset + i for i in range(ntype_labels.size(0))
                if int(ntype_labels[i].item()) == 1
            }

    return {
        "causal_graph": causal_graph,
        "causal_effects": causal_effects,
        "fraud_predicted_global": fraud_predicted_global,
        "fraud_label_set": fraud_label_set,
        "prefer_set": ev._parse_prefer_root_types(args.prefer_root_types),
        "ce_threshold": args.ce_threshold,
        "max_hops": args.max_hops,
    }


def l1_structural_reachability(g, targets, prefer_types):
    """純結構：每個 target 反向 BFS，記錄『最近的 prefer-type 祖先在第幾跳』。"""
    print("\n" + "=" * 60)
    print("L1 — 純結構可達性（免 CE）：fraud flow 的祖先裡有沒有 process")
    print("=" * 60)
    hop_to_prefer = Counter()   # k 跳碰到 prefer / "none"=完全碰不到
    unreachable = 0
    for t in targets:
        found_hop = None
        visited = {t}
        frontier = [t]
        for hop in range(1, PREFER_BFS_MAX + 1):
            nxt = []
            for node in frontier:
                for p in g.parents(node):
                    if p in visited:
                        continue
                    visited.add(p)
                    if g.node_type.get(p) in prefer_types:
                        found_hop = hop
                        break
                    nxt.append(p)
                if found_hop is not None:
                    break
            if found_hop is not None:
                break
            frontier = nxt
            if not frontier:
                break
        if found_hop is None:
            unreachable += 1
            hop_to_prefer["unreachable"] += 1
        else:
            hop_to_prefer[found_hop] += 1

    n = max(1, len(targets))
    print(f"  targets={len(targets)}, prefer_types={sorted(prefer_types)}, "
          f"BFS max={PREFER_BFS_MAX} hops")
    for key in sorted(hop_to_prefer.keys(), key=lambda x: (x == 'unreachable', x)):
        c = hop_to_prefer[key]
        label = f"reach @ hop {key}" if key != "unreachable" else "UNREACHABLE (no process ancestor)"
        print(f"    {label:<42s}: {c:>4d}  ({100*c/n:.1f}%)")
    print(f"\n  >>> {unreachable}/{len(targets)} ({100*unreachable/n:.1f}%) targets "
          f"有 process 祖先在 {PREFER_BFS_MAX} 跳內【完全碰不到】")
    print("      若此比例高 → tie-break 救不回是【結構天花板】，非邏輯 bug。")
    return hop_to_prefer


def l2_trace_dissection(g, ce, targets, fraud_label_set, prefer_set, threshold, max_hops):
    """重跑 trace（開 tie-break），逐跳記錄停止原因 + host-stopped 時上游型別。"""
    print("\n" + "=" * 60)
    print("L2 — trace 拆解（已開 tie-break）：停在 host 的為什麼沒爬到 process")
    print("=" * 60)
    tracer = RootCauseTracer(
        causal_graph=g, max_hops=max_hops, threshold=threshold,
        prefer_root_types=prefer_set,
    )

    def abs_ce(u, cur):
        return abs(ce.get((u, cur), 0.0))

    # ── L1.7 trace replay：對前 5 條停在 host 的 flow，逐跳印 tracer 真實決策 ──
    print("\n  [L1.7 trace replay] 前 5 條停在 host 的 flow，逐跳真實決策")
    replayed = 0
    for t in targets:
        root, chain = tracer.trace_root_cause(t, ce)
        if g.node_type.get(root) in prefer_set:
            continue  # 只看停在 host 的
        replayed += 1
        print(f"\n    ── flow {t} (chain len={len(chain)}, root={root} "
              f"type={g.node_type.get(root)}) ──")
        cur = t
        visited = {t}
        for hop in range(1, 7):
            ups = list(g.parents(cur))
            if not ups:
                print(f"      hop{hop}: current={cur}({g.node_type.get(cur)}) "
                      f"→ NO PARENTS, stop")
                break
            # 印每個 parent 的型別 + CE
            desc = []
            for u in ups[:6]:
                et = g.get_edge_type(u, cur)
                desc.append(f"{u}({g.node_type.get(u)},|CE|={abs_ce(u,cur):.5f},et={et})")
            print(f"      hop{hop}: current={cur}({g.node_type.get(cur)}) "
                  f"parents[{len(ups)}]: " + " | ".join(desc))
            best = max(ups, key=lambda u: abs_ce(u, cur))
            if abs_ce(best, cur) < threshold:
                print(f"             → best={best} |CE|={abs_ce(best,cur):.5f} "
                      f"< thresh {threshold} → STOP (weak CE)")
                break
            if best in visited:
                print(f"             → best={best} already visited → STOP (cycle)")
                break
            print(f"             → pick {best}({g.node_type.get(best)})")
            visited.add(best)
            cur = best
        if replayed >= 5:
            break

    stop_reason = Counter()
    root_type = Counter()
    # 針對停在 host 的：當前節點上游有沒有「通過門檻的 process 候選」
    host_stop_upstream = Counter()  # has_process_above_thresh / only_host / no_parents / all_below_thresh
    host_stop_examples = []

    # ── L1.5 探針：解開 L1(全展開 BFS) vs L2(單一貪婪 trace) 的矛盾 ──
    # 假設：flow 有多個 host parent，trace 貪婪選到「孤立 host」(no parent)死路，
    # 但 BFS 從另一個 host parent 走到了 process。驗證只需這幾個結構數字。
    multi_host_parent = Counter()     # flow 的 host parent 個數分佈
    root_host_parent_cnt = Counter()  # trace 停下的 root host 實際 parent 數
    flow_has_proc_via_other = 0       # flow 的「非 trace 選中」host parent 裡有人連到 process

    # ── L1.6：直接拆穿 L1(BFS走到process) vs L2(trace死在host) 的矛盾 ──
    flow_parent_typecombo = Counter()   # flow 全部 parent 的型別組合（不只 host）
    first_hop_type = Counter()          # trace 第一跳實際走到的型別
    # L1-style BFS 路徑 vs L2-style 單路徑：對同一 flow，記錄 BFS 走到 process 的
    # 第一步是什麼型別（看它跟 trace 第一跳是否同一個節點）
    bfs_first_step_type = Counter()

    for t in targets:
        root, chain = tracer.trace_root_cause(t, ce)
        rtype = g.node_type.get(root, "unknown")
        root_type[rtype] += 1

        # L1.6：flow 全部 parent 的型別（不分 host）+ trace 第一跳型別
        all_parents = list(g.parents(t))
        combo = tuple(sorted(Counter(g.node_type.get(p, "?") for p in all_parents).items()))
        flow_parent_typecombo[combo] += 1
        if len(chain) > 1:
            first_hop_type[g.node_type.get(chain[1], "?")] += 1
        else:
            first_hop_type["(no hop / depth0)"] += 1
        # BFS 一步：flow 的 parent 裡，哪個型別「往上還有 parent」（能繼續走）
        for p in all_parents:
            if g.parents(p):  # 這個 parent 自己還有 parent → BFS 走得動
                bfs_first_step_type[g.node_type.get(p, "?")] += 1
                break

        # L1.5：flow(target t) 的直接 parent 結構
        t_parents = all_parents
        host_parents = [p for p in t_parents if g.node_type.get(p) == "host_node"]
        multi_host_parent[min(len(host_parents), 5)] += 1  # 0,1,2,3,4,5+
        if rtype not in prefer_set:
            # trace 停在 host：印它真實 parent 數 + 型別
            rp = list(g.parents(root))
            root_host_parent_cnt[min(len(rp), 5)] += 1
            # flow 的「其他」host parent（非 trace 第一跳選中）裡，有沒有人連到 process？
            chosen_first = chain[1] if len(chain) > 1 else None
            for hp in host_parents:
                if hp == chosen_first:
                    continue
                if any(g.node_type.get(pp) in prefer_set for pp in g.parents(hp)):
                    flow_has_proc_via_other += 1
                    break

        # 重判此條的停止原因（複製 tracer 的停止邏輯）
        current = chain[-1]
        upstream = g.get_upstream_neighbors(current)
        if not upstream:
            stop_reason["no_parents"] += 1
        else:
            best = max(upstream, key=lambda u: abs_ce(u, current))
            if abs_ce(best, current) < threshold:
                stop_reason["weak_ce"] += 1
            elif best in set(chain):
                stop_reason["cycle_visited"] += 1
            else:
                stop_reason["max_hops"] += 1

        # host-stopped 解剖
        if rtype not in prefer_set:  # 停在非 root-capable 型別（通常是 host）
            ups = g.get_upstream_neighbors(current)
            if not ups:
                host_stop_upstream["no_parents"] += 1
            else:
                proc_pass = [
                    u for u in ups
                    if g.node_type.get(u) in prefer_set and abs_ce(u, current) >= threshold
                ]
                proc_any = [u for u in ups if g.node_type.get(u) in prefer_set]
                passing = [u for u in ups if abs_ce(u, current) >= threshold]
                if proc_pass:
                    host_stop_upstream["HAS_process_passing_thresh"] += 1
                    if len(host_stop_examples) < 8:
                        host_stop_examples.append(
                            (current, rtype, len(ups), len(proc_any),
                             len(proc_pass), len(passing))
                        )
                elif proc_any:
                    host_stop_upstream["has_process_but_below_thresh"] += 1
                elif not passing:
                    host_stop_upstream["all_upstream_below_thresh"] += 1
                else:
                    host_stop_upstream["only_nonprocess_above_thresh"] += 1

    n = max(1, len(targets))
    print(f"  traced={len(targets)}")
    print("\n  [root type 分佈]")
    for rt in sorted(root_type):
        c = root_type[rt]
        nf = "✓ in fraud_label_set" if rt in prefer_set else "✗ NOT labelable"
        print(f"    {rt:<18s}: {c:>4d}  ({100*c/n:.1f}%)  {nf}")

    print("\n  [停止原因]")
    for r in sorted(stop_reason):
        print(f"    {r:<22s}: {stop_reason[r]:>4d}  ({100*stop_reason[r]/n:.1f}%)")

    print("\n  [L1.6 ★直接拆穿矛盾] flow 全部 parent 的型別組合（top 6）")
    for combo, c in flow_parent_typecombo.most_common(6):
        combo_str = ", ".join(f"{t}×{n}" for t, n in combo) or "(無 parent)"
        print(f"    {c:>4d}×  parent={{{combo_str}}}")
    print("\n  [L1.6] trace 第一跳實際走到的型別")
    for ty in sorted(first_hop_type, key=lambda k: -first_hop_type[k]):
        print(f"    第一跳 → {ty:<20s}: {first_hop_type[ty]:>4d}")
    print("\n  [L1.6] flow 的 parent 裡，第一個『自己還有 parent（BFS走得動）』的型別")
    if bfs_first_step_type:
        for ty in sorted(bfs_first_step_type, key=lambda k: -bfs_first_step_type[k]):
            print(f"    可續走的 parent 型別 = {ty:<18s}: {bfs_first_step_type[ty]:>4d}")
    else:
        print("    （沒有任何 flow 的 parent 還有 parent → BFS 也走不動，L1 數字存疑）")

    print("\n  [L1.5 矛盾拆解] flow 的 host-parent 個數分佈")
    for k in sorted(multi_host_parent):
        lbl = f"{k}+" if k == 5 else str(k)
        print(f"    flow 有 {lbl:>2s} 個 host parent : {multi_host_parent[k]:>4d}")
    print("\n  [L1.5] trace 停下的 root host 的【真實 parent 數】")
    for k in sorted(root_host_parent_cnt):
        lbl = f"{k}+" if k == 5 else str(k)
        print(f"    root host 有 {lbl:>2s} 個 parent : {root_host_parent_cnt[k]:>4d}")
    print(f"\n  [L1.5 ★關鍵] trace 停在 host 的條目中，flow 還有【另一個】"
          f"host parent 能通到 process：{flow_has_proc_via_other}")
    print("      >0 → trace 貪婪選錯岔路（活路存在但沒走）→ 可改 tracer 救回 RCP")
    print("      =0 → flow 的所有 host parent 都是死路 → 真結構天花板")

    print("\n  [停在 host 的上游解剖]（關鍵：tie-break 為何沒選 process）")
    hs_total = max(1, sum(host_stop_upstream.values()))
    for k in sorted(host_stop_upstream):
        c = host_stop_upstream[k]
        print(f"    {k:<32s}: {c:>4d}  ({100*c/hs_total:.1f}%)")

    if host_stop_examples:
        print("\n  [⚠ 異常樣本] 停在 host 但上游有通過門檻的 process 候選")
        print("    （若此類 >0 → tie-break 邏輯沒生效，是 BUG，非結構天花板）")
        print(f"    {'node':>10s} {'type':>8s} {'#up':>5s} {'#proc':>6s} "
              f"{'#proc≥th':>9s} {'#≥th':>6s}")
        for row in host_stop_examples:
            print(f"    {row[0]:>10d} {row[1]:>8s} {row[2]:>5d} {row[3]:>6d} "
                  f"{row[4]:>9d} {row[5]:>6d}")
    else:
        print("\n  [✓] 沒有任何『停在 host 卻有通過門檻的 process 上游』樣本")
        print("      → tie-break 邏輯正確；停在 host 是因為上游沒有合格 process。")

    return root_type, stop_reason, host_stop_upstream


def main():
    args = ev.parse_args()
    device = torch.device(args.device)
    if args.prefer_root_types.strip() == "":
        print("⚠ 警告：--prefer_root_types 為空，診斷會以 legacy 模式跑（建議帶 process_node）")

    print(f"Loading dataset: {args.dataset}")
    P = build_pipeline(args, device)

    g = P["causal_graph"]
    targets = P["fraud_predicted_global"]
    prefer_set = P["prefer_set"] or {"process_node"}

    print(f"\n診斷 targets（fraud-predicted test flows in graph）= {len(targets)}")
    print(f"prefer_set = {sorted(prefer_set)}  |  ce_threshold = {P['ce_threshold']}")

    l1_structural_reachability(g, targets, prefer_set)
    l2_trace_dissection(
        g, P["causal_effects"], targets, P["fraud_label_set"],
        prefer_set, P["ce_threshold"], P["max_hops"],
    )

    print("\n" + "=" * 60)
    print("判讀指引")
    print("=" * 60)
    print("  ① L1 UNREACHABLE 比例高 + L2 無異常樣本 → 結構天花板，RCP 已是上限。")
    print("     論文寫法：dd18 高 AUC / dd14 高 RCP 兩配置並陳 trade-off。")
    print("  ② L2 出現『停在 host 卻有通過門檻 process 上游』樣本 → tie-break BUG，")
    print("     需修 _select_best_upstream 或 evaluate wiring，修完 RCP 可再救。")


if __name__ == "__main__":
    main()
