from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from third_party.superpoint.superpoint import SuperPoint
from matcha.utils.image_processor import ImageProcessor


class BaseFeature(nn.Module):
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 1,
        "keypoint_method": None,
        "max_length": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, config=None, name=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = SimpleNamespace(**{**self.default_config, **config})  # the order
        self.device = self.config.device
        self.topK = self.config.topK
        self.simple_method = self.config.keypoint_method
        self.upsampling = self.config.upsampling

        self.name = name
        self.detector_tag = "None"
        if self.config.keypoint_method:
            self.init_detector()
        self.init_image_processor()
        print(f"BaseFeature - config:{self.config}")

    def init_image_processor(self):
        self.image_processor = ImageProcessor(
            image_size=self.config.image_size,
            gray_scale=False,
            scale_factor=self.config.scale_factor,
            max_length=self.config.max_length,
        )

        imsize = self.config.image_size
        if imsize:
            self.img_process_tag = f"imsize{imsize[0]}x{imsize[1]}"
        elif self.config.max_length:
            self.img_process_tag = (
                f"imsize{self.config.max_length}sf{self.config.scale_factor}"
            )
        elif self.config.scale_factor:
            self.img_process_tag = f"imsize_sf{self.config.scale_factor}"

    def init_detector(self):
        keypoint_method = self.config.keypoint_method
        if keypoint_method == "superpoint":
            detector = SuperPoint(config={"max_keypoints": self.config.topK})
        elif keypoint_method == "disk":
            detector = kornia.feature.DISK.from_pretrained("depth")
        else:
            raise ValueError(f"{keypoint_method} not supported!")

        self.detector = detector.eval().to(self.device)
        self.detector_tag = f"{keypoint_method}_topk{self.config.topK}"

    def load_image(self, img_path: str):
        return self.image_processor.load(img_path)

    def detect_keypoints(self, img: torch.Tensor):
        if self.config.keypoint_method == "superpoint":
            sp_out = self.detector({"image": img.mean(1, keepdim=True)})
            kpts = sp_out["keypoints"][0][None]
        elif self.config.keypoint_method == "disk":
            disk_out = self.detector(
                img,
                n=self.config.topK,
                window_size=5,
                score_threshold=0.0,
                pad_if_not_divisible=True,
            )
            kpts = disk_out[0].keypoints[None]
        return kpts

    def sample_descriptor(self, kpts: torch.Tensor, desc: torch.Tensor, w: int, h: int) -> torch.Tensor:
        desc = desc.float()
        with torch.no_grad():
            c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).to(kpts.device).float()
            kpts_norm = (kpts - c) / c
            sample_desc = F.grid_sample(
                desc, kpts_norm.unsqueeze(2), align_corners=True
            ).squeeze(-1)
        return sample_desc

    def describe(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def detect_and_describe(self, img: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def concat_feature(self, feat_c: torch.Tensor, feat_f: torch.Tensor) -> torch.Tensor:
        """
        Concatenate coarse and fine features, ensuring they have the same spatial dimensions.
        Args:
            feat_c (torch.Tensor): Coarse feature tensor of shape (B, C, H, W).
            feat_f (torch.Tensor): Fine feature tensor of shape (B, C', H', W').
        Returns:
            torch.Tensor: Concatenated feature tensor of shape (B, C'', H, W).
        """

        if feat_f.shape[2] != feat_c.shape[2] or feat_f.shape[3] != feat_c.shape[3]:
            feat_c = F.interpolate(
                feat_c, size=(feat_f.shape[2], feat_f.shape[3]), mode="bilinear"
            )
            feat_c = F.normalize(feat_c, dim=1)

        stride = feat_c.shape[1] // feat_f.shape[1]
        feat = torch.concat([feat_f, feat_c[:, ::stride, :, :]], dim=1)
        feat = F.normalize(feat, dim=1)
        return feat
