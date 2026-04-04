"""
Causal chain visualizer for CI-RCT fraud tracing.

Produces figures like:

    [ROOT: user_1234 FRAUD] --CE=0.71--> [wallet_abc] --CE=0.82--> [user_5678 FRAUD]

Usage:
    from utils.chain_visualizer import draw_causal_chain, draw_case_studies

    draw_causal_chain(
        chain=[target_id, ..., root_id],      # RootCauseTracer output (target first)
        causal_effects={(src, dst): ce_score},
        fraud_set={set of fraud global node IDs},
        node_type_map={global_id: "user" | "wallet"},
        title="Case #1",
        save_path="figures/case_01.png",
    )
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Colour scheme ──────────────────────────────────────────────────────────────

_COLOUR = {
    "fraud_user":   "#e74c3c",   # red    — blacklisted user
    "normal_user":  "#2ecc71",   # green  — clean user
    "wallet":       "#95a5a6",   # grey   — external wallet
    "root":         "#e67e22",   # orange — root cause border accent
}

_TEXT_COLOUR = {
    "fraud_user":  "white",
    "normal_user": "white",
    "wallet":      "white",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def draw_causal_chain(
    chain: List,
    causal_effects: Dict[Tuple, float],
    fraud_set: Set,
    node_type_map: Dict,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
    idx_to_user_id: Optional[Dict] = None,
) -> plt.Figure:
    """
    Draw one causal chain as a horizontal node-arrow diagram.

    Parameters
    ----------
    chain           RootCauseTracer output — [target, ..., root_cause] order
    causal_effects  {(src, dst): CE_score}  (upstream causes downstream)
    fraud_set       Set of global node IDs known to be fraudulent
    node_type_map   {global_id: "user" | "wallet"}
    title           Figure title (e.g. "Case #1: user_5678")
    save_path       If given, save figure to this path (PNG/PDF/SVG)
    show            If True, call plt.show() — use only in notebook contexts
    idx_to_user_id  Optional {global_id: original_user_id} for readable labels

    Returns
    -------
    matplotlib Figure
    """
    # Display order: root → ... → target  (reverse of RootCauseTracer chain)
    display_chain = list(reversed(chain))
    n = len(display_chain)

    if n == 0:
        raise ValueError("chain must not be empty.")

    fig_w = max(6, n * 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    # ── Node positions (evenly spaced horizontally) ──────────────────────────
    xs = np.linspace(0.0, 1.0, n)
    y = 0.5

    box_w, box_h = 0.14, 0.30   # in axes fraction units

    for pos, gid in enumerate(display_chain):
        x = xs[pos]
        ntype = node_type_map.get(gid, "user")
        is_fraud = gid in fraud_set
        is_root = (pos == 0)
        is_target = (pos == n - 1)

        colour_key = "wallet" if ntype == "wallet" else ("fraud_user" if is_fraud else "normal_user")
        face_colour = _COLOUR[colour_key]
        text_colour = _TEXT_COLOUR[colour_key]
        edge_colour = _COLOUR["root"] if is_root else "none"
        line_w = 2.5 if is_root else 0

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=face_colour,
            edgecolor=edge_colour,
            linewidth=line_w,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(rect)

        # Node label
        label_lines = [_node_label(gid, ntype, idx_to_user_id)]
        if is_fraud:
            label_lines.append("FRAUD")
        if is_root:
            label_lines.append("▲ ROOT")
        if is_target:
            label_lines.append("▼ TARGET")

        ax.text(
            x, y,
            "\n".join(label_lines),
            ha="center", va="center",
            fontsize=7.5, color=text_colour, fontweight="bold",
            transform=ax.transAxes,
            zorder=3,
        )

    # ── Arrows between consecutive nodes ────────────────────────────────────
    for i in range(n - 1):
        src_gid = display_chain[i]
        dst_gid = display_chain[i + 1]

        # CE score: display_chain is reversed, so src_gid causes dst_gid
        # In original chain order: dst_gid is upstream of src_gid
        # causal_effects key is (upstream → downstream) = (dst_gid → src_gid)? No.
        # causal_effects[(upstream, downstream)] where upstream = chain[i+1]
        # In display order: display_chain = reversed(chain)
        # display_chain[i] = chain[n-1-i], display_chain[i+1] = chain[n-2-i]
        # The edge in causal_effects: (chain[n-2-i], chain[n-1-i])
        #                           = (display_chain[i+1], display_chain[i])
        # But we want CE score of the arrow display[i] → display[i+1],
        # which is the causal influence of src on dst in the fraud flow.
        # For display purposes, use the CE of the upstream node causing the
        # downstream node: key = (display_chain[i], display_chain[i+1])
        # But in RootCauseTracer, chain is [target, ..., root], and
        # causal_effects[(u, current)] is the CE of upstream u on current.
        # So: causal_effects[(chain[hop+1], chain[hop])]
        # In display order: causal_effects[(display[i+1], display[i])]  -- wait
        # Let me be precise:
        #   original chain: [c0=target, c1, c2, ..., ck=root]
        #   edge meaning: c(i+1) is upstream of c(i), CE = effects[(c(i+1), c(i))]
        #   display: [ck, ..., c1, c0] → display[i] = chain[k-i]
        #   arrow display[i] → display[i+1]:  chain[k-i] → chain[k-i-1]
        #   CE key: (chain[k-i-1+1], chain[k-i-1]) = (chain[k-i], chain[k-i-1])
        #         = (display[i], display[i+1])  ✓
        ce = causal_effects.get((src_gid, dst_gid), 0.0)

        x_start = xs[i] + box_w / 2
        x_end   = xs[i + 1] - box_w / 2
        ax.annotate(
            "",
            xy=(x_end, y),
            xytext=(x_start, y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#2c3e50",
                lw=1.5,
            ),
            zorder=1,
        )
        # CE label above arrow
        mid_x = (x_start + x_end) / 2
        ax.text(
            mid_x, y + 0.22,
            f"CE={ce:.2f}",
            ha="center", va="bottom",
            fontsize=7, color="#2c3e50",
            transform=ax.transAxes,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=_COLOUR["fraud_user"],  label="Blacklisted user"),
        mpatches.Patch(color=_COLOUR["normal_user"], label="Normal user"),
        mpatches.Patch(color=_COLOUR["wallet"],      label="External wallet"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def draw_case_studies(
    cases: List[Dict],
    save_dir: str = "figures/case_studies",
    prefix: str = "case",
) -> List[str]:
    """
    Batch-draw multiple case studies and save to files.

    Parameters
    ----------
    cases     List of dicts, each with keys:
                chain, causal_effects, fraud_set, node_type_map
                Optional: title, idx_to_user_id
    save_dir  Output directory
    prefix    File name prefix (e.g. "case" → case_01.png, case_02.png)

    Returns
    -------
    List of saved file paths
    """
    saved = []
    for i, case in enumerate(cases, start=1):
        path = os.path.join(save_dir, f"{prefix}_{i:02d}.png")
        fig = draw_causal_chain(
            chain=case["chain"],
            causal_effects=case["causal_effects"],
            fraud_set=case["fraud_set"],
            node_type_map=case["node_type_map"],
            title=case.get("title", f"Case #{i}"),
            save_path=path,
            idx_to_user_id=case.get("idx_to_user_id"),
        )
        plt.close(fig)
        saved.append(path)
    return saved


# ── Private helpers ────────────────────────────────────────────────────────────

def _node_label(
    gid: int,
    ntype: str,
    idx_to_user_id: Optional[Dict],
) -> str:
    """Format a short node label for display."""
    if idx_to_user_id and gid in idx_to_user_id:
        original_id = idx_to_user_id[gid]
        return f"{ntype[:1].upper()} {original_id}"
    return f"{ntype[:1].upper()} #{gid}"
