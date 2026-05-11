"""ComfyUI nodes for Depth Anything 3.

Adds three nodes:

* ``LoadDepthAnything3`` -- load a DA3 ``.safetensors`` file from the
  ``models/depth_estimation/`` folder. Falls back to ``models/diffusion_models/``
  so existing installations keep working.
* ``DepthAnything3Depth`` -- run depth estimation and return a normalised
  depth map as a ComfyUI ``IMAGE`` (visualisation / ControlNet input).
* ``DepthAnything3DepthRaw`` -- run depth estimation and return the raw depth,
  confidence and sky channels as ``MASK`` outputs.
"""

from __future__ import annotations

from typing_extensions import override

import torch

import comfy.model_management as mm
import comfy.sd
import folder_paths
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess
from comfy_api.latest import ComfyExtension, io


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------


class LoadDepthAnything3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadDepthAnything3",
            display_name="Load Depth Anything 3",
            category="loaders/depth_estimation",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("depth_estimation"),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp16", "bf16", "fp32"],
                    default="default",
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model_name, weight_dtype) -> io.NodeOutput:
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        path = folder_paths.get_full_path_or_raise("depth_estimation", model_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        return io.NodeOutput(model)


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------


def _run_da3(model_patcher, image: torch.Tensor, process_res: int,
             method: str = "upper_bound_resize"):
    """Run the DA3 network on a (B, H, W, 3) ``IMAGE`` batch.

    Returns ``(depth, confidence, sky)`` tensors with the original image
    resolution. Any of ``confidence`` / ``sky`` may be ``None`` depending on
    the variant.
    """
    assert image.ndim == 4 and image.shape[-1] == 3, \
        f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"

    B, H, W, _ = image.shape
    mm.load_model_gpu(model_patcher)
    diffusion = model_patcher.model.diffusion_model
    device = mm.get_torch_device()
    dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

    depths, confs, skies = [], [], []
    # Process one image at a time to keep peak memory predictable; DA3 is
    # an inference-only model and per-sample latency dominates anyway.
    for i in range(B):
        single = image[i:i + 1].to(device)
        x = da3_preprocess.preprocess_image(single, process_res=process_res, method=method)
        x = x.to(dtype=dtype)
        with torch.no_grad():
            out = diffusion(x)

        depth_lr = out["depth"]
        # Resize back to the original (H, W).
        depth_full = torch.nn.functional.interpolate(
            depth_lr.unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()
        depths.append(depth_full)

        if "depth_conf" in out:
            conf_full = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            confs.append(conf_full)
        if "sky" in out:
            sky_full = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            skies.append(sky_full)

    depth = torch.cat(depths, dim=0)
    confidence = torch.cat(confs, dim=0) if confs else None
    sky = torch.cat(skies, dim=0) if skies else None
    return depth, confidence, sky


# -----------------------------------------------------------------------------
# Depth -> visualisation IMAGE
# -----------------------------------------------------------------------------


class DepthAnything3Depth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3Depth",
            display_name="Depth Anything 3 (Depth)",
            category="image/depth",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
                io.Int.Input("process_res", default=504, min=140, max=2520, step=14,
                             tooltip="Longest-side target resolution (multiple of 14)."),
                io.Combo.Input("resize_method",
                               options=["upper_bound_resize", "lower_bound_resize"],
                               default="upper_bound_resize"),
                io.Combo.Input("normalization",
                               options=["v2_style", "min_max", "raw"],
                               default="v2_style",
                               tooltip="How to map raw depth -> [0, 1] image."),
                io.Boolean.Input("apply_sky_clip", default=True,
                                 tooltip="(Mono/Metric only) clip sky depth to 99th percentile."),
            ],
            outputs=[
                io.Image.Output("depth_image"),
                io.Mask.Output("sky_mask",
                               tooltip="Sky probability (Mono/Metric variants), else zeros."),
                io.Mask.Output("confidence",
                               tooltip="Depth confidence (Small/Base/DualDPT variants), else zeros."),
            ],
        )

    @classmethod
    def execute(cls, model, image, process_res, resize_method, normalization,
                apply_sky_clip) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, process_res, method=resize_method)

        if apply_sky_clip and sky is not None:
            depth = torch.stack([
                da3_preprocess.apply_sky_aware_clip(depth[i], sky[i])
                for i in range(depth.shape[0])
            ], dim=0)

        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(depth[i],
                                                       sky[i] if sky is not None else None)
                for i in range(depth.shape[0])
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        # (B, H, W) -> (B, H, W, 3) grayscale IMAGE.
        out_image = norm.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0).contiguous()
        sky_mask = sky if sky is not None else torch.zeros_like(depth)
        conf_mask = confidence if confidence is not None else torch.zeros_like(depth)
        return io.NodeOutput(out_image, sky_mask.contiguous(), conf_mask.contiguous())


# -----------------------------------------------------------------------------
# Raw depth output (useful for downstream metric work)
# -----------------------------------------------------------------------------


class DepthAnything3DepthRaw(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3DepthRaw",
            display_name="Depth Anything 3 (Raw Depth)",
            category="image/depth",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
                io.Int.Input("process_res", default=504, min=140, max=2520, step=14),
                io.Combo.Input("resize_method",
                               options=["upper_bound_resize", "lower_bound_resize"],
                               default="upper_bound_resize"),
            ],
            outputs=[
                io.Mask.Output("depth",
                               tooltip="Raw depth values (no normalisation, no clipping)."),
                io.Mask.Output("confidence"),
                io.Mask.Output("sky"),
            ],
        )

    @classmethod
    def execute(cls, model, image, process_res, resize_method) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, process_res, method=resize_method)
        zeros = torch.zeros_like(depth)
        return io.NodeOutput(
            depth.contiguous(),
            (confidence if confidence is not None else zeros).contiguous(),
            (sky if sky is not None else zeros).contiguous(),
        )


# -----------------------------------------------------------------------------
# Extension registration
# -----------------------------------------------------------------------------


class DepthAnything3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadDepthAnything3,
            DepthAnything3Depth,
            DepthAnything3DepthRaw,
        ]


async def comfy_entrypoint() -> DepthAnything3Extension:
    return DepthAnything3Extension()
