"""ComfyUI nodes for Depth Anything 3.

Adds these nodes:

* ``LoadDepthAnything3`` -- load a DA3 ``.safetensors`` file from the
  ``models/depth_estimation/`` folder. Falls back to ``models/diffusion_models/``
  so existing installations keep working.
* ``DepthAnything3`` -- unified depth estimation node supporting both mono and
  multi-view modes via a DynamicCombo selector. In mono mode, returns a
  normalised depth image plus sky/confidence masks. In multi-view mode,
  additionally returns per-view extrinsics, intrinsics and raw depth packed
  as a LATENT.

Model capability matrix
-----------------------
  Variant               head_type  has_sky  has_conf  cam_dec
  DA3-Small             dualdpt    False    True      yes
  DA3-Base              dualdpt    False    True      yes
  DA3-Mono-Large        dpt        True     False     no
  DA3-Metric-Large      dpt        True     False     no  (raw output is metres)

The node raises a ``ValueError`` at execution time when the selected
parameters conflict with the loaded model's capabilities (e.g.
``apply_sky_clip=True`` on a model with no sky head).
"""

from __future__ import annotations

from typing_extensions import override

import torch

import comfy.model_management as mm
import comfy.sd
import folder_paths
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess
from comfy_api.latest import ComfyExtension, io


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

        path = folder_paths.get_full_path_or_raise("geometry_estimation", model_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        return io.NodeOutput(model)


def _normalize_confidence(conf: torch.Tensor) -> torch.Tensor:
    """Map raw confidence (expp1 activaton, range [1, ∞)) to [0, 1] per image.

    The model uses ``exp(x) + 1`` so every pixel is guaranteed to be ≥ 1.
    Min-max normalization per image preserves the spatial pattern (high
    confidence = brighter) while producing a valid mask in [0, 1].
    """
    B = conf.shape[0]
    out = []
    for i in range(B):
        c = conf[i]
        c_min = c.min()
        c_max = c.max()
        if c_max > c_min:
            out.append((c - c_min) / (c_max - c_min))
        else:
            out.append(torch.ones_like(c))
    return torch.stack(out, dim=0)


def _run_da3(model_patcher, image: torch.Tensor, process_res: int,
             method: str = "upper_bound_resize"):
    """Run DA3 on ``(B,H,W,3)`` IMAGE; returns depth/conf/sky at original resolution (or None)."""
    assert image.ndim == 4 and image.shape[-1] == 3, \
        f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"

    B, H, W, _ = image.shape
    mm.load_model_gpu(model_patcher)
    diffusion = model_patcher.model.diffusion_model
    device = mm.get_torch_device()
    dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

    depths, confs, skies = [], [], []
    for i in range(B):
        single = image[i:i + 1].to(device)
        x = da3_preprocess.preprocess_image(single, process_res=process_res, method=method)
        x = x.to(dtype=dtype)
        with torch.no_grad():
            out = diffusion(x)

        depth_lr = out["depth"]
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


class DepthAnything3(io.ComfyNode):
    """Unified Depth Anything 3 node.

    Mono mode
    ---------
    Runs the model on each batch element independently and returns a
    normalised depth image together with sky and confidence masks.

    Multi-view mode
    ---------------
    Treats every batch element as a separate view of the same scene.
    Runs all views in a single forward pass so cross-view attention can
    establish geometric consistency. Additionally returns a ``LATENT``
    dict with per-view camera extrinsics, intrinsics and raw depth.

    Capability errors
    -----------------
    A ``ValueError`` is raised immediately when a parameter requires a
    model feature that is absent in the loaded checkpoint (e.g.
    ``apply_sky_clip=True`` on DA3-Small/Base which has no sky head,
    or ``pose_method='cam_dec'`` on a monocular model).

    Camera LATENT structure (multi-view only)
    -----------------------------------------
      samples:    (1, S, 1, H, W)  -- raw depth packed as latent samples
      type:       "da3_multiview"
      extrinsics: (1, S, 4, 4)     -- world-to-camera matrices
      intrinsics: (1, S, 3, 3)     -- pixel-space intrinsics
      depth_raw:  (S, H, W)        -- un-normalised depth
      confidence: (S, H, W)        -- per-pixel confidence (zeros if N/A)
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3",
            display_name="Depth Anything 3",
            category="image/depth",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image",
                               tooltip="Single image or image batch. "
                                       "In multi-view mode each frame is treated as "
                                       "a separate view of the same scene."),
                io.Int.Input("process_res", default=504, min=140, max=2520, step=14,
                             tooltip="Resolution the model runs at (longest side, multiple of 14). "
                                     "Lower = faster / less VRAM; higher = more detail. "
                                     "Output is upsampled back to the original size."),
                io.Combo.Input("resize_method",
                               options=["upper_bound_resize", "lower_bound_resize"],
                               default="upper_bound_resize",
                               tooltip="upper_bound_resize: scale so the longest side = process_res "
                                       "(caps memory, default). "
                                       "lower_bound_resize: scale so the shortest side = process_res "
                                       "(preserves more detail on tall/wide images, uses more memory)."),
                io.Combo.Input("normalization",
                               options=["v2_style", "min_max", "raw"],
                               default="v2_style",
                               tooltip="How to map raw depth to [0, 1] for the output image. "
                                       "'raw' preserves absolute values — use this to keep "
                                       "metric units when running DA3-Metric-Large."),
                io.Boolean.Input("apply_sky_clip", default=True,
                                 tooltip="Clip sky-region depth to the 99th percentile before "
                                         "normalisation. Requires a sky segmentation head "
                                         "(DA3-Mono-Large or DA3-Metric-Large). "
                                         "Raises an error on DA3-Small/Base."),
                io.DynamicCombo.Input("mode",
                                      tooltip="mono: single image or independent batch — "
                                              "use with any model. "
                                              "multiview: all frames processed together with "
                                              "cross-view attention for geometric consistency; "
                                              "also outputs camera pose — requires DA3-Small or DA3-Base.",
                                      options=[
                    io.DynamicCombo.Option("mono", []),
                    io.DynamicCombo.Option("multiview", [
                        io.Combo.Input("ref_view_strategy",
                                       options=["saddle_balanced", "saddle_sim_range",
                                                "first", "middle"],
                                       default="saddle_balanced",
                                       tooltip="Which view to use as the geometric anchor "
                                               "(only applied when S >= 3 and no extrinsics "
                                               "are provided). "
                                               "saddle_balanced: picks the view whose CLS-token "
                                               "features are closest to the median across "
                                               "similarity, norm and variance — best general "
                                               "choice. "
                                               "saddle_sim_range: picks the view with the widest "
                                               "similarity spread to other views — favours "
                                               "the most distinct viewpoint. "
                                               "first / middle: deterministic positional fallbacks."),
                        io.Combo.Input("pose_method",
                                       options=["cam_dec", "ray_pose"],
                                       default="cam_dec",
                                       tooltip="cam_dec: small MLP on the final camera token "
                                               "(DA3-Small/Base). "
                                               "ray_pose: RANSAC over the DualDPT ray output "
                                               "(DA3-Small/Base only)."),
                    ]),
                ]),
            ],
            outputs=[
                io.Image.Output("depth_image"),
                io.Mask.Output("sky_mask",
                               tooltip="Sky probability mask (Mono/Metric variants). "
                                       "Zeros for Small/Base."),
                io.Mask.Output("confidence",
                               tooltip="Depth confidence (Small/Base variants). "
                                       "Zeros for Mono/Metric."),
                io.Latent.Output("camera",
                                 tooltip="Multi-view: per-view extrinsics + intrinsics + raw depth. "
                                         "In mono mode this is an empty placeholder."),
            ],
        )

    @classmethod
    def execute(cls, model, image, process_res, resize_method, normalization,
                apply_sky_clip, mode) -> io.NodeOutput:
        diffusion = model.model.diffusion_model
        mode_val = mode["mode"]  # "mono" or "multiview"

        # Capability check for sky clip — fires in both modes.
        if apply_sky_clip and not diffusion.has_sky:
            raise ValueError(
                "apply_sky_clip=True requires a sky segmentation head, but the loaded "
                "model does not have one. Set apply_sky_clip=False, or load a model "
                "that includes a sky head (e.g. DA3-Mono-Large or DA3-Metric-Large)."
            )

        if mode_val == "mono":
            return cls._execute_mono(
                model, image, process_res, resize_method,
                normalization, apply_sky_clip,
            )

        # Capability checks for multi-view pose.
        pose_method = mode["pose_method"]
        ref_view_strategy = mode["ref_view_strategy"]

        if pose_method == "cam_dec" and diffusion.cam_dec is None:
            raise ValueError(
                "pose_method='cam_dec' requires a camera decoder, but the loaded "
                "model does not have one. Load a model with a camera decoder "
                "(e.g. DA3-Small or DA3-Base), or set pose_method='ray_pose'."
            )
        if pose_method == "ray_pose" and diffusion.head_type != "dualdpt":
            raise ValueError(
                "pose_method='ray_pose' requires a DualDPT head, but the loaded "
                "model has a DPT head. Load a model with a DualDPT head "
                "(e.g. DA3-Small or DA3-Base), or set pose_method='cam_dec'."
            )

        return cls._execute_multiview(
            model, image, process_res, resize_method,
            normalization, apply_sky_clip,
            ref_view_strategy, pose_method,
        )

    @classmethod
    def _execute_mono(cls, model, image, process_res, resize_method,
                      normalization, apply_sky_clip) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, process_res, method=resize_method)

        if apply_sky_clip and sky is not None:
            depth = torch.stack([
                da3_preprocess.apply_sky_aware_clip(depth[i], sky[i])
                for i in range(depth.shape[0])
            ], dim=0)

        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(
                    depth[i], sky[i] if sky is not None else None)
                for i in range(depth.shape[0])
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        out_image = norm.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0).contiguous()
        sky_mask = sky if sky is not None else torch.zeros_like(depth)
        conf_mask = (_normalize_confidence(confidence)
                     if confidence is not None else torch.zeros_like(depth))
        camera = {"samples": torch.zeros(1, 1, 1, 1, 1), "type": "mono"}
        return io.NodeOutput(
            out_image,
            sky_mask.contiguous(),
            conf_mask.contiguous(),
            camera,
        )

    @classmethod
    def _execute_multiview(cls, model, image, process_res, resize_method,
                           normalization, apply_sky_clip,
                           ref_view_strategy, pose_method) -> io.NodeOutput:
        assert image.ndim == 4 and image.shape[-1] == 3, \
            f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
        S, H, W, _ = image.shape

        mm.load_model_gpu(model)
        diffusion = model.model.diffusion_model
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

        # All views in a single forward pass: (1, S, 3, H', W').
        x = image.to(device)
        x = da3_preprocess.preprocess_image(x, process_res=process_res, method=resize_method)
        x = x.to(dtype=dtype).unsqueeze(0)

        use_ray_pose = (pose_method == "ray_pose")
        with torch.no_grad():
            out = diffusion(x, use_ray_pose=use_ray_pose,
                            ref_view_strategy=ref_view_strategy)

        depth = torch.nn.functional.interpolate(
            out["depth"].float().unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()

        conf_raw = torch.zeros_like(depth)
        if "depth_conf" in out:
            conf_raw = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

        conf_mask = _normalize_confidence(conf_raw) if conf_raw.any() else conf_raw

        sky = torch.zeros_like(depth)
        if "sky" in out:
            sky = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

        if apply_sky_clip and "sky" in out:
            depth = torch.stack([
                da3_preprocess.apply_sky_aware_clip(depth[i], sky[i])
                for i in range(S)
            ], dim=0)

        if "extrinsics" in out and "intrinsics" in out:
            extrinsics = out["extrinsics"].float().cpu()
            intrinsics = out["intrinsics"].float().cpu()
        else:
            extrinsics = torch.eye(4)[None, None].expand(1, S, 4, 4).clone()
            intrinsics = torch.eye(3)[None, None].expand(1, S, 3, 3).clone()

        sky_for_norm = sky if diffusion.has_sky else None
        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(
                    depth[i], sky_for_norm[i] if sky_for_norm is not None else None)
                for i in range(S)
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        depth_image = norm.unsqueeze(-1).repeat(1, 1, 1, 3).clamp(0.0, 1.0).contiguous()

        camera_latent = {
            "samples": depth.unsqueeze(0).unsqueeze(2).contiguous(),  # (1, S, 1, H, W)
            "type": "da3_multiview",
            "extrinsics": extrinsics.contiguous(),
            "intrinsics": intrinsics.contiguous(),
            "depth_raw": depth.contiguous(),
            "confidence": conf_raw.contiguous(),
        }
        return io.NodeOutput(
            depth_image,
            sky.contiguous(),
            conf_mask.contiguous(),
            camera_latent,
        )


class DepthAnything3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadDepthAnything3,
            DepthAnything3,
        ]


async def comfy_entrypoint() -> DepthAnything3Extension:
    return DepthAnything3Extension()
