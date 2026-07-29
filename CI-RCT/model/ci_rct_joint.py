"""
CI_RCT_Joint — a multi-task subclass of CI_RCT that classifies the primary
target type AND one or more auxiliary node types (e.g. wallet) with the SAME
shared backbone.

Design goal: a single joint model that detects fraud on both transaction and
wallet nodes, so a pooled detection F1 and a root-cause trace that seeds from
both node types become possible — WITHOUT modifying the base CI_RCT / backbone.

This is pure addition (subclassing): the primary classification head
(``backbone.classifier``), the GAN discriminator path, and ``forward`` are all
inherited unchanged, so the transaction-only behaviour of CI_RCT is byte-for-byte
preserved.  The auxiliary head(s) live here in ``self.aux_classifiers`` and read
the same per-node embeddings ``h_dict`` the backbone already produces.

Training (see train_joint.py): the existing train.py steps train the primary
head + NCM + GAN unchanged; an extra "aux step" calls ``aux_detection_loss`` so
the auxiliary cross-entropy back-propagates into the shared backbone and the aux
head — making the two tasks genuinely joint.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT


class CI_RCT_Joint(CI_RCT):
    """CI_RCT with extra classification heads for auxiliary node types.

    Args (in addition to CI_RCT's):
        aux_node_types:   node types to classify besides ``config.target_node_type``
                          (e.g. ``["wallet"]``). Empty/None → behaves like CI_RCT.
        aux_num_classes:  {node_type: num_classes}; defaults to ``num_classes``.
    """

    def __init__(
        self,
        config: CI_RCT_Config,
        metadata: tuple,
        in_channels_dict: Optional[Dict[str, int]] = None,
        node_feature_dim: Optional[int] = None,
        use_gan: bool = True,
        num_classes: int = 2,
        backbone_exclude_node_types: Optional[List[str]] = None,
        aux_node_types: Optional[List[str]] = None,
        aux_num_classes: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(
            config=config,
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            node_feature_dim=node_feature_dim,
            use_gan=use_gan,
            num_classes=num_classes,
            backbone_exclude_node_types=backbone_exclude_node_types,
        )

        self.aux_node_types: List[str] = sorted(aux_node_types or [])

        target = config.target_node_type
        excluded = set(self.backbone_exclude_node_types)
        for t in self.aux_node_types:
            if t == target:
                raise ValueError(
                    f"aux node type {t!r} must differ from target_node_type {target!r}."
                )
            if t in excluded:
                raise ValueError(
                    f"aux node type {t!r} cannot be in "
                    f"backbone_exclude_node_types {sorted(excluded)}."
                )

        anc = aux_num_classes or {}
        self.aux_classifiers = nn.ModuleDict(
            {
                t: nn.Linear(config.hidden_dim, int(anc.get(t, num_classes)))
                for t in self.aux_node_types
            }
        )
        # Per-type class count, persisted via arch_metadata for eval-side rebuild.
        self._aux_num_classes: Dict[str, int] = {
            t: self.aux_classifiers[t].out_features for t in self.aux_node_types
        }

    # ── Multi-head inference ────────────────────────────────────────────────────

    def all_logits(
        self, data: HeteroData
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Return ``(logits_by_type, h_dict)`` where ``logits_by_type`` maps the
        primary target type → its (inherited) logits and each aux type → its
        own head's logits, all from a SINGLE backbone forward.
        """
        primary_logits, h_dict = self.forward(data)
        logits_by_type: Dict[str, Tensor] = {
            self.config.target_node_type: primary_logits
        }
        for t in self.aux_node_types:
            if t in h_dict:
                logits_by_type[t] = self.aux_classifiers[t](h_dict[t])
        return logits_by_type, h_dict

    # ── Auxiliary detection loss ────────────────────────────────────────────────

    def aux_detection_loss(
        self,
        data: HeteroData,
        aux_labels: Dict[str, Tensor],
        aux_masks: Dict[str, Tensor],
        aux_class_weights: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Summed cross-entropy over the aux types' masked nodes. One backbone
        forward; gradients reach the shared backbone and the aux heads.
        """
        device = next(self.parameters()).device
        if not self.aux_node_types:
            return torch.zeros(1, device=device)

        _, h_dict = self.forward(data)
        cw = aux_class_weights or {}
        total: Optional[Tensor] = None
        for t in self.aux_node_types:
            if t not in aux_labels or t not in aux_masks or t not in h_dict:
                continue
            mask = aux_masks[t]
            logits_t = self.aux_classifiers[t](h_dict[t])[mask]
            loss_t = F.cross_entropy(logits_t, aux_labels[t][mask], weight=cw.get(t))
            total = loss_t if total is None else total + loss_t

        return total if total is not None else torch.zeros(1, device=device)

    # ── Checkpoint metadata ─────────────────────────────────────────────────────

    def arch_metadata(self) -> Dict[str, object]:
        d = super().arch_metadata()
        d["aux_node_types"] = list(self.aux_node_types)
        d["aux_num_classes"] = dict(self._aux_num_classes)
        return d
