"""
Tests for utils.elliptic_plus_joint_loader.

Verifies the joint loader returns target "transaction" while BOTH transaction
and wallet carry labels + train/val/test masks, and the wallet side uses the
clean (class3-excluded) labels. Reuses the synthetic Elliptic++ fixture from
the wallet-loader tests.
"""
import torch

from tests.test_elliptic_plus_wallet_loader import (
    N_ILLICIT,
    N_LICIT,
    N_WALLETS,
    _build_fixture,
)
from utils.elliptic_plus_joint_loader import load_elliptic_plus_joint_dataset


def test_joint_returns_transaction_target(tmp_path):
    root = _build_fixture(tmp_path)
    data, target = load_elliptic_plus_joint_dataset(root)
    assert target == "transaction"


def test_both_types_supervised(tmp_path):
    root = _build_fixture(tmp_path)
    data, _ = load_elliptic_plus_joint_dataset(root)
    for nt in ("transaction", "wallet"):
        for attr in ("y", "train_mask", "val_mask", "test_mask"):
            assert hasattr(data[nt], attr), f"missing data['{nt}'].{attr}"


def test_wallet_side_is_clean(tmp_path):
    root = _build_fixture(tmp_path)
    data, _ = load_elliptic_plus_joint_dataset(root)
    w = data["wallet"]
    assert w.y.dtype == torch.long
    assert int(w.y.sum()) == N_ILLICIT
    # Masks cover only labeled wallets (class 1 or 2), unknown excluded.
    n_labeled = N_ILLICIT + N_LICIT
    covered = int((w.train_mask | w.val_mask | w.test_mask).sum())
    assert covered == n_labeled
    assert covered < N_WALLETS  # unknown wallets left out


def test_masks_disjoint(tmp_path):
    root = _build_fixture(tmp_path)
    data, _ = load_elliptic_plus_joint_dataset(root)
    w = data["wallet"]
    assert not (w.train_mask & w.val_mask).any()
    assert not (w.train_mask & w.test_mask).any()
    assert not (w.val_mask & w.test_mask).any()
