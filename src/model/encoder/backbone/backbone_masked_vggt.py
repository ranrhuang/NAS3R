from dataclasses import dataclass
from typing import Literal
from jaxtyping import Float

import torch
from torch import Tensor, nn

from .vggt.models.vggt import VGGT
from .vggt.utils.load_fn import load_and_preprocess_images
from .vggt.utils.pose_enc import pose_encoding_to_extri_intri


from ....dataset.types import BatchedViews
from .backbone import Backbone



@dataclass
class BackboneMaskedVGGTCfg:
    name: Literal["masked_vggt"]
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    pred_intrinsics_encoder: bool = False
    pretrained: bool = True

    init_intrinsics: bool = False
    init_poses: bool = False
    num_iterations: int = 1
    depth_activation: str = 'exp'


class BackboneMaskedVGGT(Backbone[BackboneMaskedVGGTCfg]):
    def __init__(self, cfg: BackboneMaskedVGGTCfg, d_in: int) -> None:
        super().__init__(cfg)
        if self.cfg.pretrained:
            self.model = VGGT.from_pretrained("facebook/VGGT-1B", intrinsics_embed_loc=self.cfg.intrinsics_embed_loc, depth_activation=self.cfg.depth_activation)
        else:
            self.model = VGGT(intrinsics_embed_loc=self.cfg.intrinsics_embed_loc, depth_activation=self.cfg.depth_activation)

        
        if self.cfg.init_intrinsics:
            nn.init.constant_(self.model.camera_head.pose_branch.fc2.weight[-2:], 0.0)
            nn.init.constant_(self.model.camera_head.pose_branch.fc2.bias[-2:], 2 * torch.atan(torch.tensor(0.5)) / self.cfg.num_iterations)

        if self.cfg.init_poses:
            nn.init.constant_(self.model.camera_head.pose_branch.fc2.weight[:7], 0.0)
            nn.init.constant_(self.model.camera_head.pose_branch.fc2.bias[:6], 0 / self.cfg.num_iterations)
            nn.init.constant_(self.model.camera_head.pose_branch.fc2.bias[6:7], 1 / self.cfg.num_iterations)

                
    def forward(
        self,
        context: dict,
        target_num_views: int = 0,
    ):
        b, v, _, h, w = context["image"].shape
        # Compute features from the DINO-pretrained resnet50.
        patch_tokens = self.model.aggregator._encoder(context["image"], context["intrinsics"].clone())

        # context + target
        aggregated_tokens_list, ps_idx = self.model.aggregator._decoder_with_mask(patch_tokens, h, w, context["intrinsics"], num_target=target_num_views)

        return aggregated_tokens_list, ps_idx



    @property
    def patch_size(self) -> int:
        return 14

    @property
    def d_out(self) -> int:
        return 1024

