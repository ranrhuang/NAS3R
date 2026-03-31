from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .backbone.vggt.heads.dpt_gs_head import DPTGSHead
from .backbone.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from .backbone.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map_batch
from ...misc.intrinsics_utils import normalize_intrinsics


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNAS3RCfg:
    name: Literal["nas3r"]
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pretrained_weights: str = ""
    input_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    input_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True

    estimating_focal: bool = False
    estimating_pose: bool = True

    equal_fxfy: bool = True
    equal_view_intrinsics: bool = True

    near_plane: float = 1.0
    far_plane: float = 100.0


class EncoderNAS3R(Encoder[EncoderNAS3RCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNAS3RCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = 14
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.embed_dim = 1024
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )
        elif 'dpt' in head_type:
            self.gaussian_param_head = DPTGSHead(dim_in=2 * self.embed_dim, output_dim=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
    ):
        device = context["image"].device
        b, v_cxt, _, h, w = context["image"].shape

        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = dict()
            context_target["image"] = torch.cat([context["image"], target["image"]], dim=1)
            context_target["intrinsics"] = torch.cat([context["intrinsics"], target["intrinsics"]], dim=1)
            aggregated_tokens_list, ps_idx = self.backbone(context_target, target_num_views=v_tgt)
        else:
            v_tgt = 0
            aggregated_tokens_list, ps_idx = self.backbone(context, target_num_views=0)

        if self.cfg.estimating_pose or self.cfg.estimating_focal:
            
            pose_enc = self.backbone.model.camera_head(
                aggregated_tokens_list,
                num_iterations=self.backbone.cfg.num_iterations
            )
            pose_enc = pose_enc[-1]
            extri, intri = pose_encoding_to_extri_intri(
                pose_enc, context["image"].shape[-2:],
                equal_fxfy=self.cfg.equal_fxfy,
                equal_view_intrinsics=self.cfg.equal_view_intrinsics
            )
            extri = self.process_pose(extri, v_cxt)
            intri = normalize_intrinsics(intri, w, h)
            # print("pred from camera head", extri.shape, intri.shape, extri, intri)

            if self.cfg.estimating_pose:
                pred_extrinsics = extri

            if self.cfg.estimating_focal:
                pred_intrinsics = intri

        context_aggregated_tokens_list = []
        for aggregated_tokens in aggregated_tokens_list:
            context_aggregated_tokens_list.append(aggregated_tokens[:, :v_cxt].contiguous())

        # Predict Depth Maps
        depth_map, depth_conf = self.backbone.model.depth_head(context_aggregated_tokens_list, context["image"], ps_idx)
        if self.backbone.cfg.depth_activation == 'sigmoid':
            depth_map = (1 - depth_map) * self.cfg.near_plane + depth_map * self.cfg.far_plane

        depths_per_view = depth_map.squeeze(-1)
        # print("depths_per_view", self.backbone.cfg.depth_activation, depths_per_view.shape, depths_per_view.max(), depths_per_view.min())

        context_extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        context_intrinsics = pred_intrinsics[:, :v_cxt] if self.cfg.estimating_focal else context["intrinsics"]

        point_map_from_depth = unproject_depth_map_to_point_map_batch(
            rearrange(depth_map, "b v ... -> (b v) ..."),
            rearrange(context_extrinsics, "b v ... -> (b v) ..."),
            rearrange(context_intrinsics, "b v ... -> (b v) ..."),
        )
        depth_to_pts_all = rearrange(point_map_from_depth, "(b v) ... -> b v ...", b=b, v=v_cxt)
        depth_to_pts_all = rearrange(depth_to_pts_all, "b v h w xyz -> b v (h w) xyz")

        # Predict gaussians
        gs_map = self.gaussian_param_head(context_aggregated_tokens_list, context["image"], ps_idx)
        gaussians = rearrange(gs_map, "b v h w c -> b v (h w) c")
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        depth_to_pts_all = depth_to_pts_all.unsqueeze(-2)

        if self.pose_free:
            gaussians = self.gaussian_adapter.forward(
                depth_to_pts_all.unsqueeze(-2),
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v_cxt, -1, -1, -1)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                rearrange(depths_per_view, "b v h w -> b v (h w) () ()").contiguous(),
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            visualization_dump["means"] = rearrange(
                gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
            ) # (b, v, h, w, 1, 3)
            visualization_dump['opacities'] = rearrange(
                gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ) # (b, v, h, w, 1, 1)

        encoder_output = dict()

        encoder_output["gaussians"] = Gaussians(
            rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
            rearrange(gaussians.rotations, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.scales, "b v r srf spp i  -> b (v r srf spp) i "),
            rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
        )

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = dict()
            encoder_output['extrinsics']['c'] = pred_extrinsics[:, :v_cxt]
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics

        if self.cfg.estimating_focal:
            encoder_output['intrinsics'] = dict()
            encoder_output['intrinsics']['c'] = pred_intrinsics[:, :v_cxt]
            if target is not None:
                encoder_output['intrinsics']['cwt'] = pred_intrinsics

        return encoder_output

    def process_pose(self, poses, context_views):
        b, v = poses.shape[:2]

        poses = closed_form_inverse_se3(rearrange(poses, "b v ... -> (b v) ...")) # world to cam -> cam to world 
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)

        if self.cfg.pose_make_baseline_1:
            a = poses[:, 0, :3, 3]  # [b, 3]
            b = poses[:, context_views - 1, :3, 3]  #  [b, 3]

            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]

            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:,0] # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses # [b,1,4,4] @ [b,v,4,4]

        return poses


        

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )
            return batch

        return data_shim
