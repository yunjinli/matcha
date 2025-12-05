import torch

from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.geometry.geometric_matching_benchmark import GeometricMatchingBenchmark
from matcha.dataset.geometry.megadepth_eval import Megadepth1500Dataset


class Megadepth1500Benchmark(GeometricMatchingBenchmark):
    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            benchmark_name="megadepth1500",
            matcher=matcher,
            config=config,
            device=device,
            plot=plot)

    def init_dataset(self, **kwargs):
        all_dataset = None
        scene_info_path_lists = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz"
        ]
        for scene_info_path in scene_info_path_lists:
            dataset = Megadepth1500Dataset(
                dataset_path=self.config.dataset_path,
                scene_info_path=scene_info_path,
                img_resize=self.config.image_size,
                scale_factor=self.config.scale_factor,
            )
            if all_dataset is None:
                all_dataset = dataset
            else:
                all_dataset += dataset

            # print(scene_info_path, dataset.__repr__())
        self.dataloader = torch.utils.data.DataLoader(
            all_dataset, batch_size=1, shuffle=False, pin_memory=True
        )
