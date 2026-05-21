"""ComfyUI nodes for Depth Anything 3.

Adds these nodes:

* ``LoadDepthAnything3`` -- load a DA3 ``.safetensors`` file from the
  ``models/geometry_estimation/`` folder.
* ``DepthAnything3`` -- unified depth estimation node supporting both mono and
  multi-view modes via a DynamicCombo selector. Returns a DA3_GEOMETRY dict of
  raw tensors (depth, sky, confidence, camera). Feed into ``DepthAnything3Render``
  to produce display images, or directly into ``MoGeRender`` for depth / mask views.
* ``DepthAnything3Render`` -- post-processes a DA3_GEOMETRY dict: applies optional
  sky clipping, normalises depth and confidence, and returns display images.

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
from comfy.ldm.colormap import turbo as _turbo
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess
from comfy_api.latest import ComfyExtension, io

DA3ModelType = io.Custom("DA3_MODEL")
DA3Geometry = io.Custom("DA3_GEOMETRY")

# DA3_GEOMETRY is a dict with these optional keys (absent when the upstream model didn't produce them):
#
# Per-frame tensors — B = batch size in mono mode; B = S (number of views) in multi-view mode.
#   "depth":       torch.Tensor (B, H, W)         -- raw model depth (always present; matches MoGe convention)
#   "image":       torch.Tensor (B, H, W, 3)      -- source image in [0, 1], CPU (always present)
#   "mode":        str                            -- "mono" or "multiview" (always present)
#   "sky":         torch.Tensor (B, H, W)         -- sky probability in [0, 1] (Mono/Metric variants only)
#   "confidence":  torch.Tensor (B, H, W)         -- raw model confidence output (Small/Base variants only)
#
# Multi-view only — S = number of views; the leading 1 is the scene dimension from the model.
#   "extrinsics":  torch.Tensor (1, S, 4, 4)      -- world-to-camera matrices
#   "intrinsics":  torch.Tensor (1, S, 3, 3)      -- pixel-space intrinsics


class LoadDepthAnything3Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadDepthAnything3Model",
            display_name="Load Depth Anything 3",
            category="loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("geometry_estimation"),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp16", "bf16", "fp32"],
                    default="default",
                ),
            ],
            outputs=[DA3ModelType.Output()],
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


class DepthAnything3Inference(io.ComfyNode):
    """Raw Depth Anything 3 inference node.

    Outputs a DA3_GEOMETRY dict of raw tensors. All display normalization
    (sky clipping, depth scaling, confidence normalisation) is handled by
    the companion ``DepthAnything3Render`` node.

    Mono mode: each batch element is processed independently.
    Multi-view mode: all frames share a single forward pass with cross-view
    attention; adds ``extrinsics`` and ``intrinsics`` to the geometry dict.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3Inference",
            search_aliases=["depth", "geometry", "da3", "depth anything", "monocular", "pointmap", "sky", "3d", "metric depth", "disparity"],
            display_name="Run Depth Anything 3",
            category="image/geometry_estimation",
            description="Run Depth Anything 3 on an image or image batch. In multi-view mode each frame is treated as a separate view of the same scene.",
            inputs=[
                DA3ModelType.Input("da3_model"),
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
                               tooltip="- upper_bound_resize: scale so the longest side = process_res (caps memory, default).\n"
                                       "- lower_bound_resize: scale so the shortest side = process_res (preserves more detail on tall/wide images, uses more memory)."),
                io.DynamicCombo.Input("mode",
                                      tooltip="- mono: single image or independent batch — use with any model.\n"
                                              "- multiview: all frames processed together with cross-view attention for geometric consistency; also outputs camera pose — requires DA3-Small or DA3-Base.",
                                      options=[
                    io.DynamicCombo.Option("mono", []),
                    io.DynamicCombo.Option("multiview", [
                        io.Combo.Input("ref_view_strategy",
                                       options=["saddle_balanced", "saddle_sim_range",
                                                "first", "middle"],
                                       default="saddle_balanced",
                                               tooltip="Which view to use as the geometric anchor (only applied when S >= 3 and no extrinsics are provided).\n"
                                               "- saddle_balanced: picks the view whose CLS-token features are closest to the median across similarity, norm and variance — best general choice.\n"
                                               "- saddle_sim_range: picks the view with the widest similarity spread to other views — favours the most distinct viewpoint.\n"
                                               "- first / middle: deterministic positional fallbacks."),
                        io.Combo.Input("pose_method",
                                       options=["cam_dec", "ray_pose"],
                                       default="cam_dec",
                                               tooltip="- cam_dec: small MLP on the final camera token (DA3-Small/Base).\n"
                                               "- ray_pose: RANSAC over the DualDPT ray output (DA3-Small/Base only)."),
                    ]),
                ]),
            ],
            outputs=[
                DA3Geometry.Output("geometry",
                                   tooltip="DA3_GEOMETRY dict of raw tensors.\n"
                                           "- Always: 'depth' (B,H,W), 'image', 'mode'.\n"
                                           "- Optional: 'sky' + 'mask' (Mono/Metric), 'confidence' raw (Small/Base), 'extrinsics' + 'intrinsics' (multi-view)."),
            ],
        )

    @classmethod
    def execute(cls, da3_model, image, process_res, resize_method, mode) -> io.NodeOutput:
        mode_val = mode["mode"]  # "mono" or "multiview"

        if mode_val == "mono":
            return cls._execute_mono(da3_model, image, process_res, resize_method)

        # Capability checks for multi-view pose.
        diffusion = da3_model.model.diffusion_model
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
            da3_model, image, process_res, resize_method,
            ref_view_strategy, pose_method,
        )

    @classmethod
    def _execute_mono(cls, model, image, process_res, resize_method) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, process_res, method=resize_method)

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "mono",
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if confidence is not None:
            geometry["confidence"] = confidence.contiguous()
        return io.NodeOutput(geometry)

    @classmethod
    def _execute_multiview(cls, model, image, process_res, resize_method,
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

        sky = None
        if "sky" in out:
            sky = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

        if "extrinsics" in out and "intrinsics" in out:
            extrinsics = out["extrinsics"].float().cpu()
            intrinsics = out["intrinsics"].float().cpu()
        else:
            extrinsics = torch.eye(4)[None, None].expand(1, S, 4, 4).clone()
            intrinsics = torch.eye(3)[None, None].expand(1, S, 3, 3).clone()

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "multiview",
            "extrinsics": extrinsics.contiguous(),
            "intrinsics": intrinsics.contiguous(),
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if "depth_conf" in out:
            conf = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            geometry["confidence"] = conf.contiguous()
        return io.NodeOutput(geometry)




class DepthAnything3Render(io.ComfyNode):
    """Visualise a DA3_GEOMETRY packet as a single image.

    Mirrors the MoGeRender interface: one ``output`` selector, one IMAGE out.
    Use multiple nodes in parallel to get depth + sky + confidence simultaneously.
    """

    _DEPTH_RENDER_INPUTS = [
        io.Combo.Input("normalization",
                    options=["v2_style", "min_max", "raw"],
                    default="v2_style",
                    tooltip="- v2_style: mean/std normalisation for perceptually balanced results (default).\n"
                            "- min_max: stretches the full depth range to [0, 1] for maximum contrast.\n"
                            "- raw: no scaling — preserves metric units for DA3-Metric-Large."),
        io.Boolean.Input("apply_sky_clip", default=False,
                        tooltip="Clip sky-region depth to the 99th percentile of foreground depth before "
                                "normalisation. Requires a 'sky' tensor in the geometry "
                                "(DA3-Mono-Large or DA3-Metric-Large); raises an error otherwise."),
    ]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything3Render",
            display_name="Depth Anything 3 Render",
            category="image/geometry_estimation",
            description="Visualise a DA3_GEOMETRY packet. Drop multiple nodes to get different views simultaneously.",
            inputs=[
                DA3Geometry.Input("geometry"),
                io.DynamicCombo.Input("output",
                                      tooltip="- depth: normalised greyscale depth image.\n"
                                              "- depth_colored: depth mapped through the Turbo colormap.\n"
                                              "- sky_mask: sky probability in [0, 1] (Mono/Metric variants only).\n"
                                              "- confidence: normalised depth confidence (Small/Base variants only).",
                                      options=[
                    io.DynamicCombo.Option("depth", cls._DEPTH_RENDER_INPUTS),
                    io.DynamicCombo.Option("depth_colored", cls._DEPTH_RENDER_INPUTS),
                    io.DynamicCombo.Option("sky_mask", []),
                    io.DynamicCombo.Option("confidence", []),
                ]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, geometry, output) -> io.NodeOutput:
        output_val = output["output"]

        if output_val in ("depth", "depth_colored"):
            normalization = output["normalization"]
            apply_sky_clip = output["apply_sky_clip"]
            if apply_sky_clip and "sky" not in geometry:
                raise ValueError(
                    "apply_sky_clip=True requires a sky tensor in the geometry, but none is present. "
                    "Run with DA3-Mono-Large or DA3-Metric-Large, or set apply_sky_clip=False."
                )
            depth = geometry["depth"]
            sky = geometry.get("sky")
            if apply_sky_clip and sky is not None:
                depth = torch.stack([
                    da3_preprocess.apply_sky_aware_clip(depth[i], sky[i])
                    for i in range(depth.shape[0])
                ], dim=0)
            grey = cls._depth_to_image(depth, sky, normalization)  # (B,H,W,3) greyscale
            result = _turbo(grey[..., 0]) if output_val == "depth_colored" else grey

        elif output_val == "sky_mask":
            if "sky" not in geometry:
                raise ValueError("geometry has no sky output; run with DA3-Mono-Large or DA3-Metric-Large.")
            sky = geometry["sky"]
            result = sky.unsqueeze(-1).expand(*sky.shape, 3).contiguous()

        elif output_val == "confidence":
            if "confidence" not in geometry:
                raise ValueError("geometry has no confidence output; run with DA3-Small or DA3-Base.")
            result = cls._normalize_confidence(geometry["confidence"])
            result = result.unsqueeze(-1).expand(*result.shape, 3).contiguous()

        else:
            raise ValueError(f"Unknown output mode: {output_val}")

        return io.NodeOutput(result.float())

    @staticmethod
    def _depth_to_image(depth: torch.Tensor, sky_for_norm: torch.Tensor | None,
                        normalization: str) -> torch.Tensor:
        """Normalise depth and pack as an (B,H,W,3) image tensor."""
        N = depth.shape[0]
        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(
                    depth[i], sky_for_norm[i] if sky_for_norm is not None else None)
                for i in range(N)
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        out = norm.unsqueeze(-1).repeat(1, 1, 1, 3)
        if normalization != "raw":
            out = out.clamp(0.0, 1.0)
        return out.contiguous()

    @staticmethod
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


class DepthAnything3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadDepthAnything3Model,
            DepthAnything3Inference,
            DepthAnything3Render,
        ]


async def comfy_entrypoint() -> DepthAnything3Extension:
    return DepthAnything3Extension()
