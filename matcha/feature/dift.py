from typing import Tuple

import torch

from third_party.dift.dift_sd import SDFeaturizer4Eval
from .base_feature import BaseFeature


class DIFTFeature(BaseFeature):
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 32,
        "keypoint_method": None,
        "max_length": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "feature_type": "feat_c",  # feat_c, feat_f, or feat_uni
    }

    def __init__(self, cat_list: list = None, config=None):
        super().__init__(config={**self.default_config, **config} if config else self.default_config, name="DIFT")
        self.model = SDFeaturizer4Eval(cat_list=cat_list if cat_list is not None else [])

    def describe(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        cat = kwargs.get("cat", None)
        prompt_embed = kwargs.get("prompt_embed", None)
        ensemble_size = kwargs.get("ensemble_size", 8)
        feature_type = self.config.feature_type
        if feature_type in ["feat_c", "feat_f"]:
            return self.forward(
                img=img,
                cat=cat,
                prompt_embed=prompt_embed,
                ensemble_size=ensemble_size,
                feature_type=feature_type)
        elif feature_type == "feat_uni":
            feat_c = self.forward(img=img, cat=cat, prompt_embed=prompt_embed, ensemble_size=ensemble_size,
                                  feature_type="feat_c")
            feat_f = self.forward(img=img, cat=cat, prompt_embed=prompt_embed, ensemble_size=ensemble_size,
                                  feature_type="feat_f")
            feat = self.concat_feature(feat_c=feat_c, feat_f=feat_f, cat=cat)
            return feat
        else:
            raise ValueError(f"Invalid feature type: {feature_type}. Choose 'feat_c', 'feat_f', or 'feat_uni'.")

    def detect_and_describe(self, img: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect keypoints and describe the input image.
        Args:
            img (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs: Additional keyword arguments.
        Returns:
            Tuple: A tuple containing: keypoints (kpts) and descriptors (descs).
        """
        B, _, _H1, _W1 = img.shape
        feat = self.describe(img, **kwargs)
        kpts = self.detect_keypoints(img)
        descs = self.sample_descriptor(kpts=kpts, desc=feat, w=_W1, h=_H1)
        return kpts, descs

    def forward(self,
                img: torch.Tensor,
                cat: str = None,
                prompt_embed: torch.Tensor = None,
                ensemble_size: int = 1,
                feature_type: str = "feat_c") -> torch.Tensor:
        """
        Extract features from the input image using DIFT.
        Args:
            img (torch.Tensor): Input image tensor.
            cat (str, optional): Category of the image. Defaults to None.
            prompt_embed (torch.Tensor, optional): Prompt embedding tensor. Defaults to None.
            ensemble_size (int, optional): Number of ensembles to use. Defaults to 1.
            feature_type (str, optional): Type of feature to extract ('feat_c' or 'feat_f'). Defaults to 'feat_c'.
        Returns:
            torch.Tensor: Extracted features.
        """
        if feature_type == "feat_c":
            # C=1280, 16 times down-sampling
            feat = self.model.forward(
                img=img,
                t=261,
                up_ft_index=1,
                ensemble_size=ensemble_size,
                category=cat,
                prompt_embed=prompt_embed,
            )
        elif feature_type == "feat_f":
            # C=640, 8 times down-sampling
            feat = self.model.forward(
                img=img,
                t=1,
                up_ft_index=2,
                ensemble_size=ensemble_size,
                category=cat,
                prompt_embed=prompt_embed,
            )
        else:
            raise f"Invalid feature type of {feature_type}. Choose 'feat_c' or 'feat_f'."

        return feat
