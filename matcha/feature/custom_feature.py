from typing import Mapping, Any
import torch

from matcha.feature.base_feature import BaseFeature


class CustomFeature(BaseFeature):
    # Update the default configuration as needed
    default_config = {
        "topK": 4096,
        "upsampling": 0,
        "image_size": None,
        "scale_factor": 1,
        "keypoint_method": None,
        "max_length": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, config=None):
        super().__init__(
            config={**self.default_config, **config, } if config else self.default_config,
            name="CustomFeat")
        # Initialize your custom model here
        self.model = None  # Custom model should be defined here

    def describe(self, img: torch.Tensor, **kwargs):
        # Define how to extract features from the image
        pass

    def detect_and_describe(self, img: torch.Tensor, **kwargs):
        # Define how to detect keypoints and describe the image
        pass
