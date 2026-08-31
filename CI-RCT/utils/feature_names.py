"""Human-readable feature names per node type, aligned to the model's x columns.

L3 feature attribution (spec §12) needs to map a node's feature *index* back to a
*name*. The Elliptic++ loader (``utils.elliptic_plus_loader``) builds the feature
tensors by selecting fixed column ranges; this module reproduces that exact
selection from the CSV headers (header-only read — cheap, no full load), so the
returned names line up 1-to-1 with ``data[ntype].x`` columns.

Naming reality (drives L3 interpretability, see spec §12.5):
  - wallet      : every column is named & interpretable (btc_transacted_total …).
  - transaction : 93 anonymised ``Local_feature_*`` + 17 named stats.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Mirror utils.elliptic_plus_loader._TX_FEAT_COLS exactly (0-based CSV columns):
#   cols 2..94   = Local_feature_1..93   (KEEP, anonymised)
#   cols 167..183 = 17 named tx-level stats (KEEP)
#   cols 95..166 = Aggregate_feature_* (DROPPED to avoid message-passing leakage)
_LOCAL_FEAT_COLS = list(range(2, 95))
_STATS_FEAT_COLS = list(range(167, 184))
_TX_FEAT_COLS = _LOCAL_FEAT_COLS + _STATS_FEAT_COLS

# Mirror _build_wallet_features: drop 'address' (+ 'Time step' unless per-address).
_WALLET_DROP_DEFAULT = ("address", "Time step")
_WALLET_DROP_PER_ADDRESS = ("address",)

# A feature is "anonymised" (not human-meaningful) when its name matches these.
_ANON_PREFIXES = ("Local_feature_", "Aggregate_feature_")


def is_anonymous_feature(name: str) -> bool:
    """True for opaque Elliptic++ feature names (Local_/Aggregate_*)."""
    return name.startswith(_ANON_PREFIXES)


def _header(path: Path) -> List[str]:
    """Read just the CSV header row (nrows=0) → ordered column names."""
    return list(pd.read_csv(path, nrows=0).columns)


def get_feature_names(
    root: str | Path,
    *,
    wallet_per_address: bool = False,
) -> Dict[str, List[str]]:
    """Return ``{'transaction': [...], 'wallet': [...]}`` aligned to model x columns.

    ``root`` is the Elliptic++ data dir (holds ``txs_features.csv`` /
    ``wallets_features.csv``). ``wallet_per_address`` must match the loader flag
    used to build the graph, since it changes whether 'Time step' is a feature.

    Raises ``FileNotFoundError`` with a clear message when a CSV is missing.
    """
    root = Path(root)
    tx_path = root / "txs_features.csv"
    wallet_path = root / "wallets_features.csv"
    for p in (tx_path, wallet_path):
        if not p.exists():
            raise FileNotFoundError(f"feature CSV not found: {p}")

    tx_cols = _header(tx_path)
    tx_names = [tx_cols[i] for i in _TX_FEAT_COLS if i < len(tx_cols)]

    wallet_cols = _header(wallet_path)
    drop = _WALLET_DROP_PER_ADDRESS if wallet_per_address else _WALLET_DROP_DEFAULT
    wallet_names = [c for c in wallet_cols if c not in drop]

    return {"transaction": tx_names, "wallet": wallet_names}
