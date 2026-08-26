"""
Tests for the GraphBEAN-style reconstruction self-supervision (Step 1).

Covers:
  * decoders are built only when use_reconstruction=True (feature decoder per
    node type + one bipartite cross-type edge decoder);
  * reconstruction_loss trains over ALL nodes and back-propagates;
  * compute_total_loss now returns a 6-tuple ending in recon_loss;
  * with use_reconstruction=False there are NO decoders and recon is exactly 0
    (transaction variant stays byte-identical).

Requires torch / torch_geometric — run on the training box (the dev laptop's
torch import segfaults), e.g.  pytest tests/test_reconstruction.py -q
"""
import torch
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT

HID = 16


def _tiny_data() -> HeteroData:
    """A minimal Elliptic++-shaped hetero graph: wallet + transaction + the
    four edge types (two cross-type wallet↔tx, two same-type)."""
    d = HeteroData()
    d["transaction"].x = torch.randn(5, 7)
    d["wallet"].x = torch.randn(4, 3)
    d["wallet", "sends", "transaction"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 3]])
    d["transaction", "pays", "wallet"].edge_index = torch.tensor([[0, 1], [0, 2]])
    d["transaction", "flows_to", "transaction"].edge_index = torch.tensor([[0, 1], [1, 2]])
    d["wallet", "connects", "wallet"].edge_index = torch.tensor([[0], [1]])
    return d


def _config() -> CI_RCT_Config:
    return CI_RCT_Config(
        target_node_type="transaction",
        hidden_dim=HID,
        num_hgt_layers=1,
        num_heads=2,
        lambda_recon=1.0,
    )


def _model(use_reconstruction: bool) -> CI_RCT:
    d = _tiny_data()
    return CI_RCT(
        _config(),
        d.metadata(),
        in_channels_dict={"transaction": 7, "wallet": 3},
        use_gan=False,
        use_reconstruction=use_reconstruction,
    )


def test_decoders_built_when_enabled():
    m = _model(use_reconstruction=True)
    # one feature decoder per node type (incl. the unlabeled-heavy wallet type)
    assert set(m.feature_decoders.keys()) == {"transaction", "wallet"}
    # exactly one bipartite (cross-type) edge decoder
    assert m.edge_decoder is not None
    assert m._recon_edge_type is not None
    assert m._recon_edge_type[0] != m._recon_edge_type[2]  # cross-type = wallet↔tx


def test_reconstruction_loss_trains_all_nodes():
    m = _model(use_reconstruction=True)
    d = _tiny_data()
    # synthetic embeddings → isolates the decoder/loss logic from the backbone
    h = {
        "transaction": torch.randn(5, HID, requires_grad=True),
        "wallet": torch.randn(4, HID, requires_grad=True),
    }
    loss = m.reconstruction_loss(d, h)
    assert loss.dim() == 0 or loss.numel() == 1
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    # gradient flows back to BOTH types' embeddings (all nodes get signal)
    assert h["wallet"].grad is not None and h["transaction"].grad is not None


def test_compute_total_loss_returns_six_tuple_with_recon():
    m = _model(use_reconstruction=True)
    d = _tiny_data()
    labels = torch.tensor([0, 1, 0, 1, 0])
    out = m.compute_total_loss(
        data=d, labels=labels, train_mask=torch.ones(5, dtype=torch.bool)
    )
    assert len(out) == 6
    total, det, adv, stab, ncm, recon = out
    assert torch.isfinite(recon) and recon.item() > 0
    # total includes λ_recon · recon
    assert torch.isfinite(total)


def test_disabled_has_no_decoders_and_zero_recon():
    m = _model(use_reconstruction=False)
    assert m.feature_decoders is None
    assert m.edge_decoder is None
    d = _tiny_data()
    h = {"transaction": torch.randn(5, HID), "wallet": torch.randn(4, HID)}
    assert float(m.reconstruction_loss(d, h)) == 0.0
    # and compute_total_loss still returns 6 items, recon == 0
    out = m.compute_total_loss(
        data=d, labels=torch.zeros(5, dtype=torch.long),
        train_mask=torch.ones(5, dtype=torch.bool),
    )
    assert len(out) == 6 and float(out[5]) == 0.0
