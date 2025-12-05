import torch

from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.geometry.geometric_matching_benchmark import GeometricMatchingBenchmark
from matcha.dataset.geometry.aachen import Aachen1500Dataset


class Aachen1500Benchmark(GeometricMatchingBenchmark):
    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            benchmark_name="aachen1500",
            matcher=matcher,
            config=config,
            device=device,
            plot=plot)

    def init_dataset(self, **kwargs):
        dataset = Aachen1500Dataset(
            dataset_path=self.config.dataset_path,
            img_resize=self.config.image_size,
            scale_factor=self.config.scale_factor,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True
        )

