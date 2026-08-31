"""
Tests for utils.elliptic_plus_wallet_loader.

Verifies the wallet-target loader produces CLEAN binary wallet labels
(class1->1, class2->0, class3 excluded) and a valid stratified 70/15/15 split,
that its per-node ordering stays aligned with the base transaction loader, and
that it collapses the real-data quirk of duplicate wallet rows (one row per
wallet-timestep) to ONE node per unique address (canonical, leak-free universe).

A small synthetic Elliptic++ fixture is built on a tmp dir (no real data).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils.elliptic_plus_loader import (
    _dedup_wallets_per_address,
    load_elliptic_plus_dataset,
)
from utils.elliptic_plus_wallet_loader import (
    _build_clean_wallet_labels,
    load_elliptic_plus_wallet_dataset,
)

# Fixture composition: 10 illicit + 10 licit + 4 unknown wallets, all connected.
N_ILLICIT, N_LICIT, N_UNKNOWN = 10, 10, 4
N_WALLETS = N_ILLICIT + N_LICIT + N_UNKNOWN
N_TX = 8
N_TX_FEAT_COLS = 184  # base loader uses positional cols up to index 183
N_WALLET_FEAT = 54


def _addr(i: int) -> str:
    return f"w{i:03d}"


def _wallet_class(i: int) -> int:
    if i < N_ILLICIT:
        return 1
    if i < N_ILLICIT + N_LICIT:
        return 2
    return 3


def _build_fixture(tmp_path, duplicate: bool = False) -> str:
    """Write a minimal-but-valid synthetic Elliptic++ CSV set; return its dir.

    duplicate=True appends extra wallets_features rows for w000 (illicit, +2) and
    w010 (licit, +1) to emulate the per-(wallet, timestep) duplicate rows of the
    real dataset.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "Elliptic++"
    root.mkdir()

    tx_ids = [101 + i for i in range(N_TX)]

    # txs_features.csv — 184 columns (txId, Time step, then 182 features).
    cols = {"txId": tx_ids, "Time step": [1 + (i % 5) for i in range(N_TX)]}
    for c in range(2, N_TX_FEAT_COLS):
        cols[f"f{c}"] = rng.standard_normal(N_TX).astype(np.float32)
    pd.DataFrame(cols).to_csv(root / "txs_features.csv", index=False)

    # txs_classes.csv
    tx_classes = [1, 2, 1, 2, 3, 2, 1, 2][:N_TX]
    pd.DataFrame({"txId": tx_ids, "class": tx_classes}).to_csv(
        root / "txs_classes.csv", index=False
    )

    # txs_edgelist.csv
    pd.DataFrame(
        {"txId1": [101, 102, 103, 104], "txId2": [102, 103, 104, 105]}
    ).to_csv(root / "txs_edgelist.csv", index=False)

    # wallets_features.csv — address + 54 numeric features (one row per address,
    # plus optional duplicate rows to emulate per-timestep snapshots).
    waddr = [_addr(i) for i in range(N_WALLETS)]
    if duplicate:
        waddr = waddr + [_addr(0), _addr(0), _addr(10)]  # +2 illicit, +1 licit
    n_rows = len(waddr)
    wcols = {"address": waddr}
    for c in range(N_WALLET_FEAT):
        wcols[f"wf{c}"] = rng.standard_normal(n_rows).astype(np.float32)
    pd.DataFrame(wcols).to_csv(root / "wallets_features.csv", index=False)

    # wallets_classes.csv — one row per unique address.
    pd.DataFrame(
        {"address": [_addr(i) for i in range(N_WALLETS)],
         "class": [_wallet_class(i) for i in range(N_WALLETS)]}
    ).to_csv(root / "wallets_classes.csv", index=False)

    # AddrTx / TxAddr — connect every (unique) wallet so all land in the graph.
    uniq = [_addr(i) for i in range(N_WALLETS)]
    at = pd.DataFrame(
        {"input_address": uniq, "txId": [tx_ids[i % N_TX] for i in range(N_WALLETS)]}
    )
    at.to_csv(root / "AddrTx_edgelist.csv", index=False)
    ta = pd.DataFrame(
        {"txId": [tx_ids[i % N_TX] for i in range(N_WALLETS)], "output_address": uniq}
    )
    ta.to_csv(root / "TxAddr_edgelist.csv", index=False)

    # AddrAddr — a couple of wallet→wallet edges.
    pd.DataFrame(
        {"input_address": [_addr(0), _addr(1)], "output_address": [_addr(1), _addr(2)]}
    ).to_csv(root / "AddrAddr_edgelist.csv", index=False)

    return str(root)


@pytest.fixture
def elliptic_dir(tmp_path):
    return _build_fixture(tmp_path, duplicate=False)


@pytest.fixture
def elliptic_dir_dup(tmp_path):
    return _build_fixture(tmp_path, duplicate=True)


# ── _build_clean_wallet_labels (core logic, light) ───────────────────────────


def test_clean_labels_mapping(elliptic_dir):
    all_wallet_ids, y_cls, _ = _build_clean_wallet_labels(
        Path(elliptic_dir),
        include_addr_addr=False, fraud_subgraph=False, fraud_subgraph_hops=2,
    )
    assert len(all_wallet_ids) == N_WALLETS
    pos = {addr: i for i, addr in enumerate(all_wallet_ids)}
    for i in range(N_WALLETS):
        idx = pos[_addr(i)]
        expected = 1 if _wallet_class(i) == 1 else 0
        assert int(y_cls[idx]) == expected
    assert int(y_cls.sum()) == N_ILLICIT  # positives present


def test_labeled_idx_excludes_unknown(elliptic_dir):
    all_wallet_ids, _, labeled_idx = _build_clean_wallet_labels(
        Path(elliptic_dir),
        include_addr_addr=False, fraud_subgraph=False, fraud_subgraph_hops=2,
    )
    pos = {addr: i for i, addr in enumerate(all_wallet_ids)}
    labeled = set(labeled_idx)
    assert len(labeled) == N_ILLICIT + N_LICIT
    unknown_idx = {pos[_addr(i)] for i in range(N_WALLETS) if _wallet_class(i) == 3}
    assert labeled.isdisjoint(unknown_idx)


# ── Full loader integration ──────────────────────────────────────────────────


def test_target_is_wallet(elliptic_dir):
    data, target = load_elliptic_plus_wallet_dataset(elliptic_dir)
    assert target == "wallet"
    w = data["wallet"]
    for attr in ("y", "train_mask", "val_mask", "test_mask"):
        assert hasattr(w, attr), f"missing data['wallet'].{attr}"
    assert w.y.dtype == torch.long
    assert int(w.y.sum()) == N_ILLICIT


def test_masks_disjoint_and_cover_labeled(elliptic_dir):
    data, _ = load_elliptic_plus_wallet_dataset(elliptic_dir)
    w = data["wallet"]
    tr, va, te = w.train_mask, w.val_mask, w.test_mask
    # Pairwise disjoint.
    assert not (tr & va).any()
    assert not (tr & te).any()
    assert not (va & te).any()
    # Union == labeled wallets (class 1 or 2); unknown excluded.
    n_labeled = N_ILLICIT + N_LICIT
    assert int((tr | va | te).sum()) == n_labeled
    # Roughly 70/15/15 (per-class flooring tolerated): 7 train per class -> 14.
    assert int(tr.sum()) == 2 * int(n_labeled / 2 * 0.70)
    assert int(tr.sum()) > int(va.sum())
    assert int(te.sum()) >= int(va.sum())


def test_index_alignment_with_base_loader(elliptic_dir):
    """Clean illicit positions must match the base loader's polluted y==1 set."""
    base_data, base_target = load_elliptic_plus_dataset(elliptic_dir)
    assert base_target == "transaction"
    wallet_data, _ = load_elliptic_plus_wallet_dataset(elliptic_dir)

    assert wallet_data["wallet"].num_nodes == base_data["wallet"].num_nodes
    base_pos = set(base_data["wallet"].y.eq(1).nonzero(as_tuple=True)[0].tolist())
    clean_pos = set(wallet_data["wallet"].y.eq(1).nonzero(as_tuple=True)[0].tolist())
    assert clean_pos == base_pos


def test_duplicate_address_rows_collapsed_to_per_address(elliptic_dir_dup):
    """Per-(wallet, timestep) duplicate rows must COLLAPSE to one node per address.

    The dup fixture adds 2 extra w000 (illicit) rows and 1 extra w010 (licit) row.
    With per-address dedup the node universe stays at N_WALLETS (the duplicates
    vanish) and the illicit count stays N_ILLICIT — no per-row inflation, no
    cross-split entity leakage.
    """
    data, _ = load_elliptic_plus_wallet_dataset(elliptic_dir_dup)
    w = data["wallet"]
    assert int(w.num_nodes) == N_WALLETS          # collapsed, not N_WALLETS + 3
    assert w.y.numel() == w.num_nodes
    assert int(w.y.sum()) == N_ILLICIT            # 10 unique illicit, not 12 rows


def test_dedup_keeps_last_timestep_row():
    """_dedup_wallets_per_address keeps exactly one row per address — the latest
    `Time step` — using only original row values (no aggregation)."""
    df = pd.DataFrame(
        {
            "address": ["a", "a", "a", "b", "c", "c"],
            "Time step": [3, 1, 2, 5, 7, 4],
            "wf0": [30.0, 10.0, 20.0, 50.0, 70.0, 40.0],
        }
    )
    out = _dedup_wallets_per_address(df)
    assert len(out) == 3                                  # one node per address
    assert (out.groupby("address").size() == 1).all()
    kept = dict(zip(out["address"], out["Time step"]))
    assert kept == {"a": 3, "b": 5, "c": 7}              # latest timestep wins
    # retained values are real original rows (a@ts3 → wf0=30, c@ts7 → wf0=70)
    vals = dict(zip(out["address"], out["wf0"]))
    assert vals["a"] == 30.0 and vals["c"] == 70.0


def test_missing_csv_raises(tmp_path):
    empty = tmp_path / "Elliptic++"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_elliptic_plus_wallet_dataset(str(empty))
