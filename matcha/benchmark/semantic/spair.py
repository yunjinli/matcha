import torch

from matcha.dataset.semantic.spair import SpairDataset
from matcha.benchmark.semantic.semantic_matching_benchmark import SemanticMatchingBenchmark
from matcha.matcher.base_matcher import BaseMatcher


class SpairBenchmark(SemanticMatchingBenchmark):
    default_config = {
        "image_size": (512, 512),
        "soft_eval": True,
        "semantic_mode": True,
        "norm_desc": True,
    }

    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot: bool = False,
    ):
        config = {**self.default_config, **config}
        super().__init__(
            benchmark_name="spair",
            matcher=matcher,
            config=config,
            device=device,
            plot=plot,
        )

    def init_dataset(self):
        dataset = SpairDataset(
            dataset_path=self.config.dataset_path,
            img_resize=self.config.image_size,
            flip_image=False,
            mode="test",
            category_embed_path=None,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True
        )
