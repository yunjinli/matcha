from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from DeDoDe import dedode_descriptor_G
from .base_feature import BaseFeature


class DeDoDe_GFeature(BaseFeature):
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 1,
        "keypoint_method": None,
        "max_length": None,
    }

    def __init__(self, config=None):
        super().__init__(
            config={**self.default_config, **config} if config else self.default_config,
            name="DeDoDe_G")
        self.descriptor = dedode_descriptor_G()
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def describe(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass to compute the descriptor for the input image.
        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs: Additional keyword arguments (not used).
        Returns:
            torch.Tensor: Descriptor
        """

        batch = {"image": self.normalizer(img)}
        desc = self.descriptor.forward(batch)["description_grid"]
        return desc

    def detect_and_describe(self, img: torch.Tensor, **kwargs):
        B, _, _H1, _W1 = img.shape
        feat = self.describe(img, **kwargs)
        kpts = self.detect_keypoints(img)
        descs = self.sample_descriptor(kpts=kpts, desc=feat, w=_W1, h=_H1)
        return kpts, descs
