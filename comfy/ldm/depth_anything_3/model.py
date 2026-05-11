# DepthAnything3Net: top-level wrapper that combines backbone + head.
#
# This wrapper covers the monocular forward path only (single image -> depth).
# Camera encoder/decoder, ray-pose head, 3D Gaussians and the Nested
# architecture are intentionally omitted. The HF state dict for those
# components is filtered out before loading -- see
# ``comfy.supported_models.DepthAnything3.process_unet_state_dict``.
#
# The class signature mirrors the upstream YAML config so a single dit_config
# detected from the state dict in ``comfy/model_detection.py`` is sufficient
# to construct the right variant.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .dinov2 import DinoV2
from .dpt import DPT, DualDPT


_HEAD_REGISTRY = {
    "dpt": DPT,
    "dualdpt": DualDPT,
}


class DepthAnything3Net(nn.Module):
    """ComfyUI-side DepthAnything3 network (monocular path only).

    Parameters mirror the variant YAML configs from the upstream repo.
    Values are auto-detected by ``comfy/model_detection.py`` from the state
    dict. The kwargs ``device``, ``dtype`` and ``operations`` are injected by
    ``BaseModel``.
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        # --- Backbone ---
        backbone_name: str = "vitl",
        out_layers: Sequence[int] = (4, 11, 17, 23),
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = False,
        # --- Head ---
        head_type: str = "dpt",                # "dpt" or "dualdpt"
        head_dim_in: int = 1024,
        head_output_dim: int = 1,              # 1 = depth only, 2 = depth+conf
        head_features: int = 256,
        head_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        head_use_sky_head: bool = True,        # ignored by DualDPT
        head_pos_embed: Optional[bool] = None, # default: True for DualDPT, False for DPT
        # ComfyUI plumbing
        device=None, dtype=None, operations=None,
        **_ignored,
    ):
        super().__init__()
        head_cls = _HEAD_REGISTRY[head_type.lower()]
        self.head_type = head_type.lower()
        self.has_sky = (self.head_type == "dpt") and head_use_sky_head
        self.has_conf = head_output_dim > 1

        self.backbone = DinoV2(
            name=backbone_name,
            out_layers=list(out_layers),
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            device=device, dtype=dtype, operations=operations,
        )

        head_kwargs = dict(
            dim_in=head_dim_in,
            patch_size=self.PATCH_SIZE,
            output_dim=head_output_dim,
            features=head_features,
            out_channels=tuple(head_out_channels),
            device=device, dtype=dtype, operations=operations,
        )
        if self.head_type == "dpt":
            head_kwargs.update(
                use_sky_head=head_use_sky_head,
                pos_embed=(False if head_pos_embed is None else head_pos_embed),
            )
        else:  # dualdpt
            head_kwargs.update(
                pos_embed=(True if head_pos_embed is None else head_pos_embed),
            )
        self.head = head_cls(**head_kwargs)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, **_unused) -> Dict[str, torch.Tensor]:
        """Run monocular forward.

        Args:
            image: ``(B, 3, H, W)`` ImageNet-normalised image tensor, or
                   ``(B, S, 3, H, W)`` if a fake "views" axis is supplied.
                   H and W must be multiples of 14.

        Returns:
            Dict with:
              - ``depth``:      ``(B, H, W)`` raw depth values.
              - ``depth_conf``: ``(B, H, W)`` confidence (DualDPT variants only).
              - ``sky``:        ``(B, H, W)`` sky probability/logit
                                (DPT variants only).
        """
        if image.ndim == 4:
            image = image.unsqueeze(1)  # (B, 1, 3, H, W)
        assert image.ndim == 5 and image.shape[2] == 3, \
            f"image must be (B,3,H,W) or (B,S,3,H,W); got {tuple(image.shape)}"

        B, S, _, H, W = image.shape
        assert H % self.PATCH_SIZE == 0 and W % self.PATCH_SIZE == 0, \
            f"image H,W must be multiples of {self.PATCH_SIZE}; got {(H, W)}"

        feats = self.backbone(image)
        head_out = self.head(feats, H=H, W=W, patch_start_idx=0)

        # Flatten the views axis (S=1 in mono inference path).
        out: Dict[str, torch.Tensor] = {}
        for k, v in head_out.items():
            if v.ndim >= 3 and v.shape[0] == B and v.shape[1] == S:
                out[k] = v.reshape(B * S, *v.shape[2:])
            else:
                out[k] = v
        return out
