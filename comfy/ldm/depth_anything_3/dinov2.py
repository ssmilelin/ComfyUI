# DINOv2 backbone for Depth Anything 3 (monocular inference path).
#
# Why not reuse ``comfy/image_encoders/dino2.py``?
#   The existing ``Dinov2Model`` is a vanilla HuggingFace-style DINOv2 with a
#   different state-dict layout (separate Q/K/V, ``embeddings.*`` /
#   ``encoder.layer.*`` keys, ``layer_scale1.lambda1``) and no support for the
#   architectural extensions DA3 adds on top of DINOv2 (RoPE, QK-norm,
#   alternating local/global attention, concatenated camera token). Loading
#   raw DA3 HF safetensors into ``Dinov2Model`` would require splitting
#   ``attn.qkv`` weights and a large rename map for every block, and we'd
#   still need to write the DA3 extensions separately. Keeping the upstream
#   ``pretrained.*`` key layout here means HF weights load directly with no
#   conversion step.
#
# Ported from the upstream repo at:
#   src/depth_anything_3/model/dinov2/{dinov2,vision_transformer}.py
#   src/depth_anything_3/model/dinov2/layers/*
#
# DA3 extensions on top of vanilla DINOv2 (only used by Small/Base variants):
#   - 2D Rotary Position Embedding starting at ``rope_start``
#   - QK-norm starting at ``qknorm_start``
#   - Alternating local/global attention blocks starting at ``alt_start``
#   - Camera-conditioning token concatenated to features (``cat_token=True``),
#     with a learned parameter ``camera_token`` injected at block
#     ``alt_start`` when no external camera token is supplied.
#
# For the Mono/Metric variants the configuration disables all of the above
# (alt_start/qknorm_start/rope_start = -1, cat_token=False) so this module
# collapses to a vanilla DINOv2-ViT encoder.

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------------------------
# 2D rotary position embedding
# -----------------------------------------------------------------------------


class PositionGetter:
    def __init__(self):
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device) -> torch.Tensor:
        key = (height, width)
        if key not in self._cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self._cache[key] = torch.cartesian_prod(y, x)
        cached = self._cache[key]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = frequency
        self._freq_cache: dict = {}

    def _components(self, dim: int, seq_len: int, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self._freq_cache:
            exp = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exp)
            pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            ang = torch.einsum("i,j->ij", pos, inv_freq)
            ang = ang.to(dtype)
            ang = torch.cat((ang, ang), dim=-1)
            self._freq_cache[key] = (ang.cos().to(dtype), ang.sin().to(dtype))
        return self._freq_cache[key]

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d(self, tokens, positions, cos_c, sin_c):
        cos = F.embedding(positions, cos_c)[:, None, :, :]
        sin = F.embedding(positions, sin_c)[:, None, :, :]
        return (tokens * cos) + (self._rotate(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        feature_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos_c, sin_c = self._components(feature_dim, max_pos, tokens.device, tokens.dtype)
        v, h = tokens.chunk(2, dim=-1)
        v = self._apply_1d(v, positions[..., 0], cos_c, sin_c)
        h = self._apply_1d(h, positions[..., 1], cos_c, sin_c)
        return torch.cat((v, h), dim=-1)


# -----------------------------------------------------------------------------
# Patch embed / MLP / SwiGLU / LayerScale
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, in_chans=3, embed_dim=384,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = operations.Conv2d(
            in_chans, embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size,
            device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=True,
                 act_layer=nn.GELU, device=None, dtype=None, operations=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, bias=bias, device=device, dtype=dtype)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, in_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwiGLUFFNFused(nn.Module):
    """SwiGLU FFN matching upstream xformers.ops.SwiGLU layout (used for vitg)."""

    def __init__(self, in_features, hidden_features=None, bias=True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        # NOTE: xformers SwiGLU stores w12 as a single fused Linear (in, 2*hidden);
        # split-by-half at forward time. We don't currently need this for the
        # Apache-2.0 variants but keep it for parity with the upstream key names.
        self.w12 = operations.Linear(in_features, 2 * hidden_features, bias=bias, device=device, dtype=dtype)
        self.w3 = operations.Linear(hidden_features, in_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x * comfy_cast(self.gamma, x)


def comfy_cast(p: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast a parameter to match the reference tensor's device/dtype."""
    if p.device != ref.device or p.dtype != ref.dtype:
        return p.to(device=ref.device, dtype=ref.dtype)
    return p


# -----------------------------------------------------------------------------
# Attention + Block
# -----------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True,
                 qk_norm: bool = False, rope: Optional[RotaryPositionEmbedding2D] = None,
                 device=None, dtype=None, operations=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.q_norm = operations.LayerNorm(self.head_dim, device=device, dtype=dtype) if qk_norm else nn.Identity()
        self.k_norm = operations.LayerNorm(self.head_dim, device=device, dtype=dtype) if qk_norm else nn.Identity()
        self.proj = operations.Linear(dim, dim, bias=proj_bias, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=(attn_mask[:, None].repeat(1, self.num_heads, 1, 1)
                       if attn_mask is not None else None),
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True,
                 init_values: Optional[float] = 1.0,
                 norm_layer=nn.LayerNorm,
                 ffn_layer=Mlp,
                 qk_norm: bool = False,
                 rope: Optional[RotaryPositionEmbedding2D] = None,
                 ln_eps: float = 1e-6,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=ln_eps, device=device, dtype=dtype)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            qk_norm=qk_norm, rope=rope, device=device, dtype=dtype, operations=operations,
        )
        self.ls1 = (LayerScale(dim, init_values=init_values, device=device, dtype=dtype)
                    if init_values else nn.Identity())

        self.norm2 = operations.LayerNorm(dim, eps=ln_eps, device=device, dtype=dtype)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden,
            bias=ffn_bias, device=device, dtype=dtype, operations=operations,
        )
        self.ls2 = (LayerScale(dim, init_values=init_values, device=device, dtype=dtype)
                    if init_values else nn.Identity())

    def forward(self, x, pos=None, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# DINOv2 vision transformer
# -----------------------------------------------------------------------------


_BACKBONE_PRESETS = {
    "vits": dict(embed_dim=384,  depth=12, num_heads=6,  ffn_layer="mlp"),
    "vitb": dict(embed_dim=768,  depth=12, num_heads=12, ffn_layer="mlp"),
    "vitl": dict(embed_dim=1024, depth=24, num_heads=16, ffn_layer="mlp"),
    "vitg": dict(embed_dim=1536, depth=40, num_heads=24, ffn_layer="swiglufused"),
}


class DinoVisionTransformer(nn.Module):
    PATCH_SIZE = 14

    def __init__(self,
                 embed_dim: int,
                 depth: int,
                 num_heads: int,
                 ffn_layer: str = "mlp",
                 mlp_ratio: float = 4.0,
                 init_values: float = 1.0,
                 alt_start: int = -1,
                 qknorm_start: int = -1,
                 rope_start: int = -1,
                 rope_freq: float = 100.0,
                 cat_token: bool = True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.patch_size = self.PATCH_SIZE
        self.num_register_tokens = 0
        self.patch_start_idx = 1
        self.num_tokens = 1

        self.patch_embed = PatchEmbed(
            patch_size=self.PATCH_SIZE, in_chans=3, embed_dim=embed_dim,
            device=device, dtype=dtype, operations=operations,
        )
        # Number of patch positions for the historical 518x518 reference grid.
        ref_grid = 518 // self.PATCH_SIZE
        num_patches = ref_grid * ref_grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=device, dtype=dtype))
        if alt_start != -1:
            self.camera_token = nn.Parameter(torch.zeros(1, 2, embed_dim, device=device, dtype=dtype))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim,
                                                  device=device, dtype=dtype))

        if rope_start != -1 and rope_freq > 0:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
            self.position_getter = PositionGetter()
        else:
            self.rope = None
            self.position_getter = None

        if ffn_layer == "mlp":
            ffn = Mlp
        elif ffn_layer in ("swiglu", "swiglufused"):
            ffn = SwiGLUFFNFused
        else:
            raise NotImplementedError(f"Unsupported ffn_layer: {ffn_layer}")

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=init_values,
                norm_layer=norm_layer,
                ffn_layer=ffn,
                qk_norm=(qknorm_start != -1 and i >= qknorm_start),
                rope=(self.rope if (rope_start != -1 and i >= rope_start) else None),
                device=device, dtype=dtype, operations=operations,
            )
            for i in range(depth)
        ])
        self.norm = operations.LayerNorm(embed_dim, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        pos_embed = comfy_cast(self.pos_embed, x).float()
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        # Historical 0.1 offset preserves bicubic resample compatibility with
        # the original DINOv2 release; see the upstream PR for context.
        sx = float(w0 + 0.1) / M
        sy = float(h0 + 0.1) / M
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy), mode="bicubic", antialias=False,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, 3, H, W) -> tokens (B, S, 1+N, C)
        B, S, _, H, W = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        cls_token = comfy_cast(self.cls_token, x).expand(B, S, -1).reshape(B * S, 1, self.embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        if self.rope is None:
            return None, None
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
        pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
        pos_nodiff = torch.zeros_like(pos)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
            pos = torch.cat([pos_special, pos], dim=2)
            pos_nodiff = pos_nodiff + 1
            pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _attn(self, x, blk, attn_type, pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        else:  # "global"
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        x = blk(x, pos=pos, attn_mask=attn_mask)
        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        else:
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def get_intermediate_layers(self, x: torch.Tensor, out_layers: List[int],
                                cam_token: Optional[torch.Tensor] = None):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens(x)
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        outputs = []
        local_x = x

        for i, blk in enumerate(self.blocks):
            if self.rope is not None and i >= self.rope_start:
                g_pos, l_pos = pos_nodiff, pos
            else:
                g_pos, l_pos = None, None

            if self.alt_start != -1 and i == self.alt_start:
                # Inject camera token at the alt-start boundary.
                if cam_token is not None:
                    inj = cam_token
                else:
                    ct = comfy_cast(self.camera_token, x)
                    ref_token = ct[:, :1].expand(B, -1, -1)
                    src_token = ct[:, 1:].expand(B, max(S - 1, 0), -1)
                    inj = torch.cat([ref_token, src_token], dim=1)
                x = x.clone()
                x[:, :, 0] = inj

            if self.alt_start != -1 and i >= self.alt_start and (i % 2 == 1):
                x = self._attn(x, blk, "global", pos=g_pos)
            else:
                x = self._attn(x, blk, "local", pos=l_pos)
                local_x = x

            if i in out_layers:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                outputs.append(out_x)

        # Apply final norm. Upstream norms only the "global" half when cat_token.
        normed: List[torch.Tensor] = []
        camera_tokens: List[torch.Tensor] = []
        for out_x in outputs:
            # Camera/cls token slot is index 0 *before* register-token stripping.
            camera_tokens.append(out_x[:, :, 0])
            if out_x.shape[-1] == self.embed_dim:
                normed.append(self.norm(out_x))
            elif out_x.shape[-1] == self.embed_dim * 2:
                left = out_x[..., : self.embed_dim]
                right = self.norm(out_x[..., self.embed_dim :])
                normed.append(torch.cat([left, right], dim=-1))
            else:
                raise ValueError(f"Unexpected token width: {out_x.shape[-1]}")
        # Drop cls/cam token + register tokens from patch sequence.
        normed = [o[..., 1 + self.num_register_tokens:, :] for o in normed]
        # Match upstream signature consumed by DA3 heads:
        #   feats[i][0] = normed patch tokens, feats[i][1] = camera/cls token.
        return list(zip(normed, camera_tokens))


class DinoV2(nn.Module):
    """Top-level DINOv2 wrapper matching upstream key layout (``self.pretrained``)."""

    def __init__(self, name: str = "vits",
                 out_layers: Optional[List[int]] = None,
                 alt_start: int = -1,
                 qknorm_start: int = -1,
                 rope_start: int = -1,
                 cat_token: bool = True,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        if name not in _BACKBONE_PRESETS:
            raise ValueError(f"Unknown DINOv2 backbone variant: {name!r}")
        preset = _BACKBONE_PRESETS[name]
        self.name = name
        self.out_layers = list(out_layers) if out_layers is not None else [5, 7, 9, 11]
        self.cat_token = cat_token
        self.pretrained = DinoVisionTransformer(
            embed_dim=preset["embed_dim"],
            depth=preset["depth"],
            num_heads=preset["num_heads"],
            ffn_layer=preset["ffn_layer"],
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            device=device, dtype=dtype, operations=operations,
        )

    def forward(self, x, cam_token=None, **_unused):
        return self.pretrained.get_intermediate_layers(x, self.out_layers, cam_token=cam_token)
