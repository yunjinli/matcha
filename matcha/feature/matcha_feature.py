from typing import Mapping, Any

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from matcha.model.matcha import Matcha
from matcha.feature.base_feature import BaseFeature


class MatchaFeature(BaseFeature):
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 32,
        "keypoint_method": "disk",
        "max_length": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, config=None):
        super().__init__(config={**self.default_config, **config, } if config else self.default_config,
                         name="MatchaFeature")
        self.model = Matcha(config=self.config).to(self.device)
        self.model.eval()
        print(self.config)

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        self.model.load_state_dict(state_dict, strict=strict)

    def describe(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.model(
            img=img,
            feat_c=kwargs.get("feat_c", None),
            feat_f=kwargs.get("feat_f", None),
            cat=kwargs.get("cat", None),
            semantic_mode=kwargs.get("semantic_mode", False),
        )
        return feat

    def detect_and_describe(self, img: torch.Tensor, **kwargs):
        B, _, H1, W1 = img.shape
        feat_c = kwargs.get("feat_c", None)
        feat_f = kwargs.get("feat_f", None)
        cat = kwargs.get("cat", None)
        feat = self.model(img, feat_c=feat_c, feat_f=feat_f, cat=cat)

        kpts = self.detect_keypoints(img)
        sampled_descs = self.sample_descriptor(kpts=kpts, desc=feat, w=W1, h=H1)
        sampled_descs = F.normalize(sampled_descs, dim=1)
        return kpts, sampled_descs
