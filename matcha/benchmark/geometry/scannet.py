import torch

from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.geometry.geometric_matching_benchmark import GeometricMatchingBenchmark
from matcha.dataset.geometry.scannet_eval import Scannet1500Dataset


class Scannet1500Benchmark(GeometricMatchingBenchmark):
    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            benchmark_name="Scannet1500",
            matcher=matcher,
            config=config,
            device=device,
            plot=plot
        )

    def init_dataset(self, **kwargs):
        dataset = Scannet1500Dataset(
            dataset_path=self.config.dataset_path,
            scene_info_path="scannet_test_1500/test.npz",
            img_resize=self.config.image_size,
            scale_factor=self.config.scale_factor,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True
        )
