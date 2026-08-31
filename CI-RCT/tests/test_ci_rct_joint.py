"""
Unit tests for model.ci_rct_joint.CI_RCT_Joint (joint transaction + wallet).

Verifies the auxiliary wallet head is built correctly, all_logits returns both
heads' logits, the auxiliary loss is differentiable, arch metadata round-trips
(so evaluate_joint can rebuild the wallet head), and the primary (transaction)
head / backbone are left structurally untouched.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.ci_rct_joint import CI_RCT_Joint

N_TX, N_WALLET, HID = 6, 8, 16


def _toy_joint() -> HeteroData:
    """Tiny transaction+wallet graph mirroring the Elliptic++ schema."""
    hd = HeteroData()
    hd["transaction"].x = torch.randn(N_TX, 5)
    hd["wallet"].x = torch.randn(N_WALLET, 4)
    hd["wallet", "sends", "transaction"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    hd["transaction", "pays", "wallet"].edge_index = torch.tensor(
        [[0, 1, 2], [4, 5, 6]], dtype=torch.long
    )
    hd["transaction", "flows_to", "transaction"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    hd["wallet", "connects", "wallet"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    hd["transaction"].y = torch.tensor([0, 1, 0, 1, 0, 1])
    hd["wallet"].y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    tx_train = torch.zeros(N_TX, dtype=torch.bool); tx_train[:4] = True
    w_train = torch.zeros(N_WALLET, dtype=torch.bool); w_train[:6] = True
    hd["transaction"].train_mask = tx_train
    hd["wallet"].train_mask = w_train
    return hd


def _build_model(hd, aux=("wallet",)):
    config = CI_RCT_Config(
        dataset="elliptic++", target_node_type="transaction",
        hidden_dim=HID, num_hgt_layers=1, num_heads=2, dropout=0.0,
        node_type_emb_dim=8,
    )
    in_ch = {"transaction": 5, "wallet": 4}
    return CI_RCT_Joint(
        config=config, metadata=hd.metadata(), in_channels_dict=in_ch,
        use_gan=False, num_classes=2,
        aux_node_types=list(aux), aux_num_classes={t: 2 for t in aux},
    )


def test_aux_head_built():
    hd = _toy_joint()
    model = _build_model(hd)
    assert "wallet" in model.aux_classifiers
    assert model.aux_classifiers["wallet"].in_features == HID
    assert model.aux_classifiers["wallet"].out_features == 2
    assert model.aux_node_types == ["wallet"]


def test_primary_head_untouched():
    """The transaction head stays backbone.classifier; aux is separate."""
    hd = _toy_joint()
    model = _build_model(hd)
    assert isinstance(model.backbone.classifier, torch.nn.Linear)
    assert model.backbone.classifier.out_features == 2
    # aux params must not live under backbone.classifier
    assert "wallet" not in dict(model.backbone.named_children())


def test_all_logits_shapes():
    hd = _toy_joint()
    model = _build_model(hd)
    model.eval()
    logits_by_type, h_dict = model.all_logits(hd)
    assert set(logits_by_type.keys()) == {"transaction", "wallet"}
    assert logits_by_type["transaction"].shape == (N_TX, 2)
    assert logits_by_type["wallet"].shape == (N_WALLET, 2)
    # Primary head logits equal the inherited forward's logits.
    primary, _ = model.forward(hd)
    assert torch.allclose(logits_by_type["transaction"], primary)


def test_aux_detection_loss_differentiable():
    hd = _toy_joint()
    model = _build_model(hd)
    model.train()
    loss = model.aux_detection_loss(
        hd,
        aux_labels={"wallet": hd["wallet"].y},
        aux_masks={"wallet": hd["wallet"].train_mask},
        aux_class_weights={"wallet": None},
    )
    assert loss.requires_grad
    assert float(loss.item()) >= 0.0
    loss.backward()  # must not raise
    # gradient reaches the wallet head
    assert model.aux_classifiers["wallet"].weight.grad is not None


def test_arch_metadata_roundtrip(tmp_path):
    hd = _toy_joint()
    model = _build_model(hd)
    model.eval()
    with torch.no_grad():
        model.forward(hd)  # materialise lazy HGT weights before save
    arch = model.arch_metadata()
    assert arch["aux_node_types"] == ["wallet"]
    assert arch["aux_num_classes"] == {"wallet": 2}

    ckpt = tmp_path / "joint.pt"
    model.save_checkpoint(str(ckpt))
    read = CI_RCT_Joint.read_arch_metadata(str(ckpt))
    assert read["aux_node_types"] == ["wallet"]
    assert read["aux_num_classes"] == {"wallet": 2}

    # Rebuild from arch → wallet head exists and the checkpoint loads.
    rebuilt = _build_model(hd)
    rebuilt.eval()
    with torch.no_grad():
        rebuilt.forward(hd)
    rebuilt.load_checkpoint(str(ckpt))
    assert "wallet" in rebuilt.aux_classifiers


def test_invalid_aux_types_raise():
    hd = _toy_joint()
    with pytest.raises(ValueError):
        _build_model(hd, aux=("transaction",))  # same as primary target
