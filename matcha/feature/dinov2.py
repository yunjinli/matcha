import torch
import torch.nn as nn
import torch.nn.functional as F

from matcha.third_party.dinov2 import vit_large
from .base_feature import BaseFeature


class DINOv2Feature(BaseFeature):
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 14,
        "keypoint_method": None,
        "max_length": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, config=None):
        super().__init__(config={**self.default_config, **config} if config else self.default_config, name="DINOv2")
        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )
        self.model = vit_large(**vit_kwargs).eval()

        dinov2_weights = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            map_location="cpu",
        )
        self.model.load_state_dict(dinov2_weights)

    def describe(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        desc = self.forward(img=img)
        return desc

    def detect_and_describe(self, img: torch.Tensor, **kwargs):
        B, _, _H1, _W1 = img.shape
        feat = self.describe(img, **kwargs)
        kpts = self.detect_keypoints(img)
        descs = self.sample_descriptor(kpts=kpts, desc=feat, w=_W1, h=_H1)
        return kpts, descs

    def forward(self, img, size_factor=14):
        B, _, H, W = img.shape
        if H % size_factor == 0:
            nH = H
        else:
            nH = (H // size_factor + 1) * size_factor

        if W % size_factor == 0:
            nW = W
        else:
            nW = (W // size_factor + 1) * size_factor

        if nH != H or nW != W:
            nimg = F.interpolate(
                img, size=(nH, nW), mode="bilinear", align_corners=True
            )
        else:
            nimg = img
        feat_dino = self.model.forward_features(nimg)["x_norm_patchtokens"]
        feat_dino = feat_dino.permute(0, 2, 1).reshape(B, 1024, nH // 14, nW // 14)
        feat_dino = F.normalize(feat_dino, dim=1)
        return feat_dino
