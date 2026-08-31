"""Feature-name alignment for L3 attribution (utils.feature_names)."""
import os

import pytest

from utils.feature_names import get_feature_names, is_anonymous_feature

_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "Elliptic++")
_HAVE_DATA = os.path.exists(os.path.join(_ROOT, "txs_features.csv")) and os.path.exists(
    os.path.join(_ROOT, "wallets_features.csv")
)
pytestmark = pytest.mark.skipif(not _HAVE_DATA, reason="Elliptic++ CSVs not present")


def test_transaction_names_match_loader_selection():
    names = get_feature_names(_ROOT)["transaction"]
    # 93 Local_feature_* + 17 named stats = 110 (Aggregate_* dropped).
    assert len(names) == 110
    assert names[0] == "Local_feature_1"
    assert names[92] == "Local_feature_93"
    # The 17 stat columns are NOT Local_feature_* (they are named).
    assert not any(n.startswith("Local_feature_") for n in names[93:])


def test_wallet_names_are_named_and_exclude_keys():
    names = get_feature_names(_ROOT)["wallet"]
    assert "address" not in names
    assert "Time step" not in names  # dropped by default (non per-address variant)
    # Wallet features are human-readable, none anonymised.
    assert not any(is_anonymous_feature(n) for n in names)
    assert "btc_transacted_total" in names


def test_per_address_keeps_timestep():
    default = get_feature_names(_ROOT)["wallet"]
    per_addr = get_feature_names(_ROOT, wallet_per_address=True)["wallet"]
    assert "Time step" in per_addr
    assert len(per_addr) == len(default) + 1


def test_is_anonymous_feature():
    assert is_anonymous_feature("Local_feature_1")
    assert is_anonymous_feature("Aggregate_feature_5")
    assert not is_anonymous_feature("btc_transacted_total")
