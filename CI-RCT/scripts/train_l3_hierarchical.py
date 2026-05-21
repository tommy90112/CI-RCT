"""
L3 — Stage 2 hierarchical attack-type fine-tune (frozen-backbone partial).

Background:
  DD-14 backbone is trained on the binary fraud/benign task with
  target_node_type=flow_node (Stage 1). L2 probe (logs/diag_l2_post_fix/)
  showed that the frozen embedding has attack-type signal under a
  random row split (flow_node by_file macro-F1 = 0.66) but collapses
  under the main by_incident split (0.14) — the gap is driven by
  incident-level class drift, not absence of signal. L3 closes that
  gap by fine-tuning only the last HGT layer + a fresh per-node-type
  attack-type head, leaving the binary head and lower layers frozen
  so the Stage 1 RCP/CCV trace numbers stay intact.

Architecture:
  HeteroData
      ↓
  input_proj                [FROZEN]
      ↓
  HGT layers [0 .. L-2]     [FROZEN]
      ↓
  HGT layer  [L-1]          [TRAINABLE] — last layer adapts to attack-type
      ↓
      ├──→ classifier       [FROZEN]   — Stage 1 fraud/benign head, untouched
      └──→ attack heads     [NEW + TRAINABLE]
              flow_attack_head    : Linear(H, H) → ELU → Dropout → Linear(H, C_flow)
              process_attack_head : same with C_process

Loss:
  L_l3 = α · CE_weighted(flow_attack_head | flow.is_malicious=1)
       + β · CE_weighted(process_attack_head | process.is_malicious=1)
  - benign nodes carry no attack_type label → masked out
  - class weights from inverse-frequency on the train slice

Output:
  - <output_dir>/l3.log                    : stdout snapshot
  - <output_dir>/checkpoint_l3.pt          : full model (backbone+heads)
  - <output_dir>/l3_{flow,process}_test_per_class.csv
  - <output_dir>/l3_{flow,process}_test_confusion.csv
  - <output_dir>/l3_summary.json           : macro-F1 / accuracy / config snapshot

Usage:
  python scripts/train_l3_hierarchical.py \
      --checkpoint checkpoints/ci_rct_unsw_mg24_best.pt \
      --output_dir logs/l3_post_fix \
      --device cuda \
      --subsample_ddos 0.1 \
      --mg24_host_role zeroed \
      --epochs 30
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import CI_RCT_Config  # noqa: E402
from model.ci_rct import CI_RCT  # noqa: E402
from utils.mg24_loader import (  # noqa: E402
    _attack_type_from_audit_host_ref,
    build_edges,
    load_mg24_data,
    to_pyg_hetero_data,
)


# ── Attack-type label derivation (shared with L2 probe) ───────────────────────


def attack_labels_for_flows(flows: pd.DataFrame) -> np.ndarray:
    """One label per flow_node row. Uses the dataframe's attack_type directly."""
    return flows["attack_type"].astype(str).fillna("unknown").values


def attack_labels_for_processes(processes: pd.DataFrame) -> np.ndarray:
    """One label per process_node row (audit + procmon paths)."""
    if processes.empty:
        return np.array([], dtype=object)
    src = processes["source"].astype(str).values
    host_ref = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    out = np.empty(len(processes), dtype=object)
    for i in range(len(processes)):
        if src[i] == "audit":
            if is_mal[i] == 1:
                at = _attack_type_from_audit_host_ref(host_ref[i], 1)
                out[i] = at if at else "other_malicious"
            else:
                out[i] = "benign"
        else:
            out[i] = "procmon_mal" if is_mal[i] == 1 else "procmon_benign"
    return out


# ── Stage 2 fraud-row stratified split (by_file) ─────────────────────────────


def stratified_fraud_split(
    raw_labels: np.ndarray,
    is_malicious: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-level random split over fraud rows only, **stratified by
    attack_type** so every class with ≥ 4 samples appears in
    train/val/test proportionally.

    Why this exists:
      DD-14's by_incident split was designed for Stage-1 fraud/benign
      generalisation testing. When inherited by Stage 2 attack-type
      classification it shatters the class distribution — concretely,
      val ended up 100% backdoor while ddos/dos/recon went entirely
      into train, so the head had no class-aligned validation signal
      and best-val selection picked the noisiest pre-convergence epoch.

      L2 by_file probe showed flow_node macro-F1 = 0.66 under row-level
      random split (vs 0.14 by_incident) — so the embedding *does*
      carry attack-type signal, the by_incident split just doesn't let
      a classifier read it. This function reproduces the L2-probe-style
      split for Stage 2 training.

    Classes with < 4 samples are placed entirely in train (cannot be
    stratified into 3 buckets). Their val/test support becomes 0; the
    caller's metrics naturally exclude them.
    """
    n_total = len(raw_labels)
    fraud_idx = np.flatnonzero(is_malicious.astype(bool))
    if fraud_idx.size == 0:
        empty = np.zeros(n_total, dtype=bool)
        return empty, empty.copy(), empty.copy()

    fraud_labels = raw_labels[fraud_idx]
    classes = sorted(set(fraud_labels.tolist()))

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for c in classes:
        cls_mask = fraud_labels == c
        cls_idx = fraud_idx[cls_mask].copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n < 4:
            train_parts.append(cls_idx)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_parts.append(cls_idx[:n_test])
        val_parts.append(cls_idx[n_test:n_test + n_val])
        train_parts.append(cls_idx[n_test + n_val:])

    def _to_mask(parts: List[np.ndarray]) -> np.ndarray:
        m = np.zeros(n_total, dtype=bool)
        if parts:
            m[np.concatenate(parts)] = True
        return m

    return _to_mask(train_parts), _to_mask(val_parts), _to_mask(test_parts)


# ── Stage-2 attack-type head ─────────────────────────────────────────────────


class AttackTypeHead(nn.Module):
    """Two-layer MLP head: hidden → hidden → num_classes."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.act(self.proj(h))))


# ── Freezing logic ───────────────────────────────────────────────────────────


def apply_l3_freeze(model: CI_RCT, n_unfrozen_hgt_layers: int) -> Tuple[int, int]:
    """
    Freeze everything except (a) the last `n_unfrozen_hgt_layers` HGT layers
    of the backbone, and (b) the Stage-1 binary classifier is always frozen
    so its DD-14 calibration is preserved.

    NCM / GAN / type_embeddings are all frozen as well — Stage 2 attack-type
    fine-tune should not perturb Module 2/3 trace machinery.

    Returns (n_trainable_params, n_total_params).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = len(model.backbone.hgt_layers)
    unfreeze_from = max(0, n_layers - n_unfrozen_hgt_layers)
    for i in range(unfreeze_from, n_layers):
        for p in model.backbone.hgt_layers[i].parameters():
            p.requires_grad_(True)
    # Backbone classifier (Stage 1) stays frozen — protects 0.92 RCP.
    # input_proj stays frozen — feature embedding from DD-14 is reused.

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ── Class-weighted loss helper ───────────────────────────────────────────────


def compute_class_weights(
    labels: np.ndarray,
    label_to_idx: Dict[str, int],
    device: torch.device,
    mode: str = "sqrt",
) -> torch.Tensor:
    """
    Class weights normalised to mean=1, with a tunable severity.

    The first L3 run used ``mode='inverse'`` (1/N_class) which crushed
    macro-F1 below random: with a 1000-vs-3 class ratio, the minority
    weight ends up ~67× the majority and the optimiser pushes every
    prediction toward the rarest classes. The default here is
    ``mode='sqrt'`` (1/sqrt(N_class)), which keeps imbalance pressure
    without that runaway.

    mode:
      - ``'none'``    : uniform weight 1.0 (vanilla CE; majority baseline check)
      - ``'sqrt'``    : 1/sqrt(N_class)            ← default (recommended)
      - ``'log'``     : 1/log(N_class + e)         ← even gentler
      - ``'inverse'`` : 1/N_class                  ← legacy, do not use
    """
    num_classes = len(label_to_idx)
    counts = np.zeros(num_classes, dtype=np.float64)
    for lab in labels:
        idx = label_to_idx.get(str(lab))
        if idx is not None:
            counts[idx] += 1

    weights = np.ones(num_classes, dtype=np.float64)
    observed = counts > 0
    if mode == "none" or not observed.any():
        return torch.tensor(weights, dtype=torch.float32, device=device)

    obs_counts = counts[observed]
    if mode == "inverse":
        raw = 1.0 / obs_counts
    elif mode == "log":
        raw = 1.0 / np.log(obs_counts + np.e)
    elif mode == "sqrt":
        raw = 1.0 / np.sqrt(obs_counts)
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode!r}")

    raw = raw * (raw.size / raw.sum())  # normalise so observed weights mean=1
    weights[observed] = raw
    return torch.tensor(weights, dtype=torch.float32, device=device)


def majority_baseline_accuracy(
    label_idx: torch.Tensor, mask: torch.Tensor, num_classes: int,
) -> Tuple[float, int, int]:
    """
    Majority-class baseline accuracy on the masked subset.

    Returns (accuracy_if_predicting_majority, majority_class_idx, n_majority).
    Useful as a sanity floor: any model worth training must beat this.
    """
    if int(mask.sum().item()) == 0:
        return 0.0, -1, 0
    sub = label_idx[mask].cpu().numpy()
    if len(sub) == 0:
        return 0.0, -1, 0
    counts = np.bincount(sub, minlength=num_classes)
    maj_idx = int(np.argmax(counts))
    maj_n = int(counts[maj_idx])
    return maj_n / len(sub), maj_idx, maj_n


# ── Per-type Stage-2 supervision tensors ─────────────────────────────────────


def build_attack_label_tensors(
    raw_labels: np.ndarray,
    is_malicious: np.ndarray,
    train_mask_np: np.ndarray,
    val_mask_np: np.ndarray,
    test_mask_np: np.ndarray,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, int], List[str],
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a tensor of integer attack-type indices restricted to malicious
    nodes that also appear in the {train, val, test} splits.

    Returns:
        label_idx       : LongTensor [N] — -1 for nodes outside the
                          mal+split universe; valid integer otherwise.
        label_to_idx    : {label_str: int}
        idx_to_label    : [label_str]
        mask_train_atk  : BoolTensor [N] — fraud train nodes only
        mask_val_atk    : BoolTensor [N] — fraud val nodes only
        mask_test_atk   : BoolTensor [N] — fraud test nodes only

    `label_to_idx` is built from the **train** slice only (so the head's
    output dimension matches what it was actually supervised on). Val/test
    nodes whose label is unseen-in-train become invalid and are excluded
    from the corresponding mask; their counts are reported separately.
    """
    n = len(raw_labels)
    mal_bool = is_malicious.astype(bool)

    train_fraud = mal_bool & train_mask_np
    val_fraud = mal_bool & val_mask_np
    test_fraud = mal_bool & test_mask_np

    train_labels = raw_labels[train_fraud]
    train_classes = sorted(set(train_labels.tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(train_classes)}
    idx_to_label = list(train_classes)

    label_idx = np.full(n, -1, dtype=np.int64)
    for i in np.flatnonzero(mal_bool):
        idx = label_to_idx.get(str(raw_labels[i]))
        if idx is not None:
            label_idx[i] = idx

    has_valid_label = label_idx >= 0
    mask_train_atk = train_fraud & has_valid_label
    mask_val_atk = val_fraud & has_valid_label
    mask_test_atk = test_fraud & has_valid_label

    return (
        torch.from_numpy(label_idx).to(device),
        label_to_idx,
        idx_to_label,
        torch.from_numpy(mask_train_atk).to(device),
        torch.from_numpy(mask_val_atk).to(device),
        torch.from_numpy(mask_test_atk).to(device),
    )


# ── Stage-2 forward + loss ──────────────────────────────────────────────────


def stage2_loss(
    h_dict: Dict[str, torch.Tensor],
    heads: Dict[str, AttackTypeHead],
    label_tensors: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    class_weights: Dict[str, torch.Tensor],
    head_weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-head class-weighted CE, masked to fraud nodes only."""
    total = None
    by_head: Dict[str, float] = {}
    for ntype, head in heads.items():
        if ntype not in h_dict:
            continue
        mask = masks[ntype]
        if int(mask.sum().item()) == 0:
            by_head[ntype] = 0.0
            continue
        logits = head(h_dict[ntype][mask])
        targets = label_tensors[ntype][mask]
        loss = F.cross_entropy(logits, targets, weight=class_weights[ntype])
        weighted = head_weights[ntype] * loss
        total = weighted if total is None else total + weighted
        by_head[ntype] = float(loss.detach().item())
    if total is None:
        # No supervision available — return a zero scalar that still
        # carries a graph dependency so the optimiser does not crash.
        any_h = next(iter(h_dict.values()))
        total = (any_h * 0.0).sum()
    return total, by_head


# ── Evaluation ───────────────────────────────────────────────────────────────


def eval_head(
    h: torch.Tensor, head: AttackTypeHead,
    label_idx: torch.Tensor, mask: torch.Tensor,
    idx_to_label: List[str],
) -> Dict[str, object]:
    """Return macro-F1, accuracy, per-class report, confusion matrix."""
    from sklearn.metrics import (
        f1_score, accuracy_score, classification_report, confusion_matrix,
    )

    if int(mask.sum().item()) == 0:
        return {"error": "no samples in mask"}

    with torch.no_grad():
        logits = head(h[mask])
        pred = logits.argmax(dim=-1).cpu().numpy()
    target = label_idx[mask].cpu().numpy()
    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    report = classification_report(
        target, pred,
        labels=list(range(len(idx_to_label))),
        target_names=label_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(target, pred, labels=list(range(len(idx_to_label))))
    return {
        "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(target, pred, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(target, pred)),
        "n": int(mask.sum().item()),
        "per_class": pd.DataFrame(report).T,
        "confusion_matrix": pd.DataFrame(cm, index=label_names, columns=label_names),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DD-14 (or compatible) CI_RCT checkpoint.")
    parser.add_argument("--data_root", default="data/unsw_mg24")
    parser.add_argument("--output_dir", default="logs/l3_post_fix")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_ddos", type=float, default=0.1,
                        help="Match DD-14 training config.")
    parser.add_argument("--mg24_host_role", default="zeroed",
                        choices=("full", "no_mal_count", "zeroed"))
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Must match the checkpoint's hidden_dim.")
    parser.add_argument("--num_hgt_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    parser.add_argument("--n_unfrozen_hgt_layers", type=int, default=1,
                        help="Number of trailing HGT layers to fine-tune.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_head", type=float, default=1e-3,
                        help="Learning rate for the new attack-type heads.")
    parser.add_argument("--lr_backbone", type=float, default=1e-4,
                        help="Learning rate for unfrozen HGT layers (lower).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_weight_flow", type=float, default=1.0)
    parser.add_argument("--head_weight_process", type=float, default=1.0)
    parser.add_argument("--class_weight_mode", default="sqrt",
                        choices=("none", "sqrt", "log", "inverse"),
                        help="Class-weight severity for the CE loss. "
                             "'sqrt' (default) avoids the minority-runaway "
                             "seen with 'inverse' in the v1 run.")
    parser.add_argument("--stage2_split_mode", default="by_file",
                        choices=("by_file", "by_incident"),
                        help="Train/val/test split for the Stage-2 "
                             "attack-type head. Default 'by_file' (row-level "
                             "stratified random over fraud rows) avoids the "
                             "class-shatter that 'by_incident' caused in v2. "
                             "Stage-1 fraud/benign generalisation is already "
                             "established via DD-14 by_incident — no need to "
                             "re-test it at Stage 2.")
    parser.add_argument("--stage2_val_ratio", type=float, default=0.15)
    parser.add_argument("--stage2_test_ratio", type=float, default=0.15)
    parser.add_argument("--smoke", action="store_true",
                        help="Single-epoch dry run (skip checkpoint save).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[L3] device: {device}")

    # ── Data ──
    print(f"[L3] loading MG24 (subsample_ddos={args.subsample_ddos}, "
          f"split_mode=by_incident, host_role={args.mg24_host_role})")
    mg24 = load_mg24_data(
        Path(args.data_root),
        subsample_ddos=args.subsample_ddos,
        seed=args.seed, verbose=False,
    )
    edges = build_edges(mg24)
    data = to_pyg_hetero_data(
        mg24, edges,
        seed=args.seed,
        split_mode="by_incident",
        host_features_mode=args.mg24_host_role,
    ).to(device)
    print(
        "[L3] node counts: "
        + ", ".join(f"{nt}={data[nt].x.size(0):,}"
                    for nt in sorted(data.node_types)
                    if data[nt].x is not None)
    )

    # ── Model + checkpoint (Stage 1 already trained) ──
    config = CI_RCT_Config(
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_type_emb_dim=args.type_emb_dim,
        target_node_type="flow_node",
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
    model.load_checkpoint(args.checkpoint, device=str(device))
    print(f"[L3] loaded Stage-1 checkpoint: {args.checkpoint}")

    # ── Freeze policy ──
    trainable, total = apply_l3_freeze(model, args.n_unfrozen_hgt_layers)
    print(f"[L3] backbone partial freeze: "
          f"unfrozen_hgt_layers={args.n_unfrozen_hgt_layers}, "
          f"trainable_backbone_params={trainable:,}/{total:,} "
          f"({100*trainable/max(total,1):.2f}%)")

    # ── Attack-type label tensors per node type ──
    target_types = ("flow_node", "process_node")
    raw_label_builders = {
        "flow_node": lambda: attack_labels_for_flows(mg24.flows),
        "process_node": lambda: attack_labels_for_processes(mg24.processes),
    }

    label_tensors: Dict[str, torch.Tensor] = {}
    label_to_idx: Dict[str, Dict[str, int]] = {}
    idx_to_label: Dict[str, List[str]] = {}
    masks_train: Dict[str, torch.Tensor] = {}
    masks_val: Dict[str, torch.Tensor] = {}
    masks_test: Dict[str, torch.Tensor] = {}
    class_weights: Dict[str, torch.Tensor] = {}
    majority_baseline_val: Dict[str, float] = {}
    majority_baseline_test: Dict[str, float] = {}

    stage2_split_rng = np.random.default_rng(args.seed)
    print(f"[L3] stage2_split_mode = {args.stage2_split_mode}")

    for ntype in target_types:
        raw = raw_label_builders[ntype]()
        is_mal = data[ntype].y.cpu().numpy().astype(int)

        if args.stage2_split_mode == "by_incident":
            train_np = data[ntype].train_mask.cpu().numpy()
            val_np = data[ntype].val_mask.cpu().numpy()
            test_np = data[ntype].test_mask.cpu().numpy()
        else:  # by_file — row-level stratified over fraud rows
            train_np, val_np, test_np = stratified_fraud_split(
                raw, is_mal,
                val_ratio=args.stage2_val_ratio,
                test_ratio=args.stage2_test_ratio,
                rng=stage2_split_rng,
            )

        (
            idx_tensor, l2i, i2l,
            m_tr, m_va, m_te,
        ) = build_attack_label_tensors(
            raw, is_mal, train_np, val_np, test_np, device=device,
        )
        label_tensors[ntype] = idx_tensor
        label_to_idx[ntype] = l2i
        idx_to_label[ntype] = i2l
        masks_train[ntype] = m_tr
        masks_val[ntype] = m_va
        masks_test[ntype] = m_te

        train_raw = raw[(is_mal.astype(bool)) & train_np]
        class_weights[ntype] = compute_class_weights(
            train_raw, l2i, device, mode=args.class_weight_mode,
        )

        unseen_val = int(((data[ntype].val_mask.cpu().numpy()
                           & is_mal.astype(bool))
                          & (~m_va.cpu().numpy())).sum())
        unseen_test = int(((data[ntype].test_mask.cpu().numpy()
                            & is_mal.astype(bool))
                           & (~m_te.cpu().numpy())).sum())
        train_dist = dict(Counter(train_raw.tolist()))
        print(
            f"[L3] {ntype}: train_classes={len(i2l)}, "
            f"n_train_fraud={int(m_tr.sum().item())}, "
            f"n_val_fraud={int(m_va.sum().item())} (+unseen={unseen_val}), "
            f"n_test_fraud={int(m_te.sum().item())} (+unseen={unseen_test})"
        )
        print(f"       train dist (top 10): "
              f"{dict(sorted(train_dist.items(), key=lambda kv: -kv[1])[:10])}")

        n_cls = max(1, len(i2l))
        cw_vec = class_weights[ntype].cpu().numpy()
        cw_max = float(cw_vec.max()) if cw_vec.size else 0.0
        cw_min = float(cw_vec[cw_vec > 0].min()) if (cw_vec > 0).any() else 0.0
        cw_ratio = cw_max / cw_min if cw_min > 0 else float("inf")
        maj_acc_va, maj_idx_va, maj_n_va = majority_baseline_accuracy(
            idx_tensor, m_va, n_cls,
        )
        maj_acc_te, maj_idx_te, maj_n_te = majority_baseline_accuracy(
            idx_tensor, m_te, n_cls,
        )
        majority_baseline_val[ntype] = maj_acc_va
        majority_baseline_test[ntype] = maj_acc_te
        maj_lbl_va = i2l[maj_idx_va] if 0 <= maj_idx_va < len(i2l) else "n/a"
        maj_lbl_te = i2l[maj_idx_te] if 0 <= maj_idx_te < len(i2l) else "n/a"
        print(
            f"       class_weight[{args.class_weight_mode}] max/min ratio = "
            f"{cw_ratio:.2f} (max={cw_max:.3f}, min={cw_min:.3f})"
        )
        print(
            f"       MAJORITY-BASELINE: val acc={maj_acc_va:.4f} "
            f"(class={maj_lbl_va!r}, n={maj_n_va})  "
            f"test acc={maj_acc_te:.4f} (class={maj_lbl_te!r}, n={maj_n_te})  "
            f"random = {1.0/max(1,n_cls):.4f}"
        )

    # ── Attack-type heads ──
    heads: Dict[str, AttackTypeHead] = {
        ntype: AttackTypeHead(
            hidden_dim=config.hidden_dim,
            num_classes=max(1, len(idx_to_label[ntype])),
            dropout=args.head_dropout,
        ).to(device)
        for ntype in target_types
    }
    head_module_dict = nn.ModuleDict(heads)

    head_weights = {
        "flow_node": args.head_weight_flow,
        "process_node": args.head_weight_process,
    }

    # ── Optimiser: head params at lr_head, unfrozen HGT layers at lr_backbone ──
    head_params = list(head_module_dict.parameters())
    backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    param_groups = [
        {"params": head_params, "lr": args.lr_head},
    ]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": args.lr_backbone}
        )
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    # ── Training loop ──
    epochs = 1 if args.smoke else args.epochs
    best_val_f1 = -1.0
    best_state: Optional[Dict[str, object]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        head_module_dict.train()
        optimizer.zero_grad()

        # Backbone forward is dataset-wide (single graph); h_dict has all types.
        _, h_dict = model.forward(data)
        loss, by_head = stage2_loss(
            h_dict, heads, label_tensors, masks_train,
            class_weights, head_weights,
        )
        loss.backward()
        optimizer.step()

        # ── Eval (val) ──
        model.eval()
        head_module_dict.eval()
        with torch.no_grad():
            _, h_eval = model.forward(data)
        val_f1_avg = []
        eval_strs = []
        for ntype in target_types:
            if ntype not in h_eval:
                continue
            r = eval_head(
                h_eval[ntype], heads[ntype],
                label_tensors[ntype], masks_val[ntype],
                idx_to_label[ntype],
            )
            if "error" in r:
                eval_strs.append(f"{ntype}: skip ({r['error']})")
                continue
            val_f1_avg.append(r["macro_f1"])
            maj = majority_baseline_val.get(ntype, 0.0)
            beats_str = "+" if r["accuracy"] > maj else "-"
            eval_strs.append(
                f"{ntype}: macro-F1={r['macro_f1']:.4f} "
                f"acc={r['accuracy']:.4f}{beats_str}maj={maj:.4f} "
                f"n={r['n']}"
            )

        val_avg = float(np.mean(val_f1_avg)) if val_f1_avg else 0.0
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in by_head.items())
        print(
            f"[L3] epoch {epoch:>3}/{epochs}  "
            f"L_total={float(loss.item()):.4f}  ({loss_str})  "
            f"| val: " + " | ".join(eval_strs) + f"  avg_val_macro_f1={val_avg:.4f}"
        )

        if val_avg > best_val_f1:
            best_val_f1 = val_avg
            best_state = {
                "epoch": epoch,
                "backbone": copy.deepcopy(model.backbone.state_dict()),
                "heads": copy.deepcopy(head_module_dict.state_dict()),
                "label_to_idx": {k: dict(v) for k, v in label_to_idx.items()},
                "idx_to_label": {k: list(v) for k, v in idx_to_label.items()},
            }

    # ── Reload best, final test eval ──
    if best_state is not None:
        model.backbone.load_state_dict(best_state["backbone"])
        head_module_dict.load_state_dict(best_state["heads"])
        print(f"[L3] restored best val checkpoint from epoch {best_state['epoch']} "
              f"(val_macro_f1_avg={best_val_f1:.4f})")

    model.eval()
    head_module_dict.eval()
    with torch.no_grad():
        _, h_final = model.forward(data)

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "epochs": epochs,
            "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone,
            "n_unfrozen_hgt_layers": args.n_unfrozen_hgt_layers,
            "head_weights": head_weights,
            "class_weight_mode": args.class_weight_mode,
            "stage2_split_mode": args.stage2_split_mode,
            "stage2_val_ratio": args.stage2_val_ratio,
            "stage2_test_ratio": args.stage2_test_ratio,
            "host_role": args.mg24_host_role,
            "subsample_ddos": args.subsample_ddos,
            "best_val_epoch": (best_state or {}).get("epoch"),
            "best_val_macro_f1_avg": best_val_f1,
        },
        "majority_baseline": {
            "val": majority_baseline_val,
            "test": majority_baseline_test,
        },
        "results": {},
    }

    for ntype in target_types:
        if ntype not in h_final:
            continue
        r = eval_head(
            h_final[ntype], heads[ntype],
            label_tensors[ntype], masks_test[ntype],
            idx_to_label[ntype],
        )
        if "error" in r:
            print(f"\n[L3 TEST] {ntype}: skip ({r['error']})")
            summary["results"][ntype] = {"error": r["error"]}
            continue
        print(f"\n[L3 TEST] {ntype}: "
              f"macro-F1={r['macro_f1']:.4f}  micro-F1={r['micro_f1']:.4f}  "
              f"acc={r['accuracy']:.4f}  n={r['n']}")
        r["per_class"].to_csv(
            out_dir / f"l3_{ntype}_test_per_class.csv"
        )
        r["confusion_matrix"].to_csv(
            out_dir / f"l3_{ntype}_test_confusion.csv"
        )
        summary["results"][ntype] = {
            "macro_f1": r["macro_f1"],
            "micro_f1": r["micro_f1"],
            "accuracy": r["accuracy"],
            "n": r["n"],
            "label_to_idx": label_to_idx[ntype],
        }

    if not args.smoke:
        ckpt_path = out_dir / "checkpoint_l3.pt"
        torch.save({
            "model_state": model.state_dict(),
            "heads_state": head_module_dict.state_dict(),
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "config": summary["config"],
        }, ckpt_path)
        print(f"[L3] saved checkpoint: {ckpt_path}")

    with (out_dir / "l3_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[L3] summary written: {out_dir / 'l3_summary.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
