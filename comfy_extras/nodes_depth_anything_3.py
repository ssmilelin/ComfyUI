"""ComfyUI nodes for Depth Anything 3.

Adds these nodes:

* ``LoadDepthAnything3`` -- load a DA3 ``.safetensors`` file from the
  ``models/depth_estimation/`` folder. Falls back to ``models/diffusion_models/``
  so existing installations keep working.
* ``DepthAnything3Depth`` -- run depth estimation and return a normalised
  depth map as a ComfyUI ``IMAGE`` (visualisation / ControlNet input).
* ``DepthAnything3DepthRaw`` -- run depth estimation and return the raw depth,
  confidence and sky channels as ``MASK`` outputs.
* ``DepthAnything3MultiView`` -- multi-view path: depth + per-view extrinsics
  + intrinsics. Pose is decoded either from the camera-decoder MLP (default)
  or from the auxiliary ray output via RANSAC (DA3-Small/Base only).
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


class DepthAnything3MultiView(io.ComfyNode):
    """Multi-view depth + pose estimation for DA3-Small / DA3-Base / DA3-Large.

    Treats each batch element of the input ``IMAGE`` as a separate view of
    the same scene. The selected reference view is auto-chosen by the
    backbone via ``ref_view_strategy`` (when at least 3 views are
    supplied), unless camera extrinsics are provided -- in which case the
    geometry is pinned by the user and no reordering is done.

    Output structure:
      * ``depth_image`` -- per-view normalised depth as a stacked ``IMAGE``
        batch (one frame per view, original input order).
      * ``confidence`` / ``sky`` -- per-view masks (zero when the variant
        does not produce them).
      * ``camera`` -- ``LATENT`` dict with keys::
            samples:    (1, S, 1, h_p, w_p)  -- raw depth packed as latent
            type:       "da3_multiview"
            extrinsics: (1, S, 4, 4)          world-to-camera matrices
            intrinsics: (1, S, 3, 3)          pixel-space intrinsics
            depth_raw:  (S, H, W)             un-normalised depth
            confidence: (S, H, W)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3MultiView",
            display_name="Depth Anything 3 (Multi-View)",
            category="image/depth",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image",
                               tooltip="Image batch where each frame is a view of the same scene."),
                io.Int.Input("process_res", default=504, min=140, max=2520, step=14,
                             tooltip="Longest-side target resolution (multiple of 14)."),
                io.Combo.Input("resize_method",
                               options=["upper_bound_resize", "lower_bound_resize"],
                               default="upper_bound_resize"),
                io.Combo.Input("ref_view_strategy",
                               options=["saddle_balanced", "saddle_sim_range", "first", "middle"],
                               default="saddle_balanced",
                               tooltip="Reference view selection (only applied when "
                                       "S>=3 and no extrinsics are provided)."),
                io.Combo.Input("pose_method",
                               options=["cam_dec", "ray_pose"],
                               default="cam_dec",
                               tooltip="cam_dec: small MLP on the final cam token (works for "
                                       "all variants with cam_dec). ray_pose: RANSAC over the "
                                       "DualDPT auxiliary ray output (DA3-Small/Base only)."),
                io.Combo.Input("normalization",
                               options=["v2_style", "min_max", "raw"],
                               default="v2_style"),
            ],
            outputs=[
                io.Image.Output("depth_image"),
                io.Mask.Output("confidence"),
                io.Mask.Output("sky_mask"),
                io.Latent.Output("camera",
                                 tooltip="Per-view extrinsics + intrinsics + raw depth."),
            ],
        )

    @classmethod
    def execute(cls, model, image, process_res, resize_method, ref_view_strategy,
                pose_method, normalization) -> io.NodeOutput:
        assert image.ndim == 4 and image.shape[-1] == 3, \
            f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
        S, H, W, _ = image.shape

        mm.load_model_gpu(model)
        diffusion = model.model.diffusion_model
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

        # Stack all views as a single batch element with views axis = S.
        x = image.to(device)
        x = da3_preprocess.preprocess_image(x, process_res=process_res, method=resize_method)
        x = x.to(dtype=dtype).unsqueeze(0)  # (1, S, 3, H', W')

        use_ray_pose = (pose_method == "ray_pose")
        with torch.no_grad():
            out = diffusion(x, use_ray_pose=use_ray_pose,
                            ref_view_strategy=ref_view_strategy)

        # ``out["depth"]`` is (S, h_p, w_p); resize back to (S, H, W).
        depth_lr = out["depth"].float()
        depth = torch.nn.functional.interpolate(
            depth_lr.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()

        if "depth_conf" in out:
            conf = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
        else:
            conf = torch.zeros_like(depth)

        if "sky" in out:
            sky = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
        else:
            sky = torch.zeros_like(depth)

        # Pose. Defaults to identity when neither cam_dec nor ray_pose is wired up.
        if "extrinsics" in out and "intrinsics" in out:
            extrinsics = out["extrinsics"].float().cpu()
            intrinsics = out["intrinsics"].float().cpu()
        else:
            extrinsics = torch.eye(4)[None, None].expand(1, S, 4, 4).clone()
            intrinsics = torch.eye(3)[None, None].expand(1, S, 3, 3).clone()

        # Normalised depth viz per view (same path as the mono node).
        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(depth[i],
                                                       sky[i] if "sky" in out else None)
                for i in range(S)
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        depth_image = norm.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0).contiguous()

        camera_latent = {
            # The Latent contract requires a ``samples`` field; pack the raw
            # depth there so a downstream node still has a tensor to chain on.
            "samples": depth.unsqueeze(0).unsqueeze(2).contiguous(),  # (1, S, 1, H, W)
            "type": "da3_multiview",
            "extrinsics": extrinsics.contiguous(),
            "intrinsics": intrinsics.contiguous(),
            "depth_raw": depth.contiguous(),
            "confidence": conf.contiguous(),
        }
        return io.NodeOutput(
            depth_image,
            conf.contiguous(),
            sky.contiguous(),
            camera_latent,
        )


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
            DepthAnything3MultiView,
        ]


async def comfy_entrypoint() -> DepthAnything3Extension:
    return DepthAnything3Extension()
