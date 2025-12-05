from .dedode import DeDoDe_GFeature
from .matcha_feature import MatchaFeature
from .matcha_light_feature import MatchaLightFeature
from .dift import DIFTFeature
from .dinov2 import DINOv2Feature

MODELS = {
    "matcha": MatchaFeature,
    "matcha_light": MatchaLightFeature,
    "dedode": DeDoDe_GFeature,
    "dift": DIFTFeature,
    "dinov2": DINOv2Feature,
}
