import cv2
import numpy as np
import os
import os.path as osp
import torch
from types import SimpleNamespace
from pathlib import Path

from matcha.matcher.base_matcher import BaseMatcher
from .visualization import plot_kpts, plot_matches, resize_img


class Benchmark:
    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            benchmark_name: str,
            plot: bool = False,
    ):
        torch.set_grad_enabled(False)
        self.config = SimpleNamespace(**config)
        print(self.config)
        self.device = device
        self.matcher = matcher
        self.method = matcher.model.name
        self.benchmark_name = benchmark_name
        self.plot = plot
        self.cache_dir = Path(self.config.cache_dir) / self.benchmark_name / self.method
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        if self.plot:
            self.cache_plot_dir = self.cache_dir / "plot"
            self.cache_plot_dir.mkdir(exist_ok=True, parents=True)

        self.init_dataset()

    def init_dataset(self, **kwargs):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def load_cached_dift_feature(self, img: torch.Tensor, img_path: str, ensemble_size=8, cat=None):
        self.dift_cache_dir = getattr(self.config, "dift_cache_dir", None)
        feat_c, feat_f = None, None

        if self.dift_cache_dir is None:
            return feat_c, feat_f

        if not self.dift_cache_dir.exists():
            return feat_c, feat_f

        # Load cached dift features
        img_name = osp.relpath(img_path, self.config.dataset_path).replace("/", "_")
        height, width = img.shape[-2], img.shape[-1]
        feat_file_name = f"{img_name}_H{height}_W{width}_E{ensemble_size}_C{cat}.npy"
        feat_path = self.dift_cache_dir / feat_file_name
        if feat_path.exists():
            out = np.load(feat_path, allow_pickle=True)[()]
            feat_c = out["feat_c"]
            feat_f = out["feat_f"]
            feat_c = torch.from_numpy(feat_c)[None].to(self.device).float()
            feat_f = torch.from_numpy(feat_f)[None].to(self.device).float()
        return feat_c, feat_f

    def write_matching_image(
            self,
            image0: np.ndarray,
            image1: np.ndarray,
            keypoints0: np.ndarray,
            keypoints1: np.ndarray,
            inlier_mask: np.ndarray,
            fig_name: str,
            line_thickness: int = 3,
            plot_outlier: bool = False,
            new_h: int = 512,
            show_text: str = None,
    ):
        fig_path = self.cache_plot_dir / fig_name

        if inlier_mask is None:
            inlier_mask = np.array([True for _ in range(keypoints0.shape[0])])

        img_match = plot_matches(
            img1=image0,
            img2=image1,
            pts1=keypoints0,
            pts2=keypoints1,
            inliers=inlier_mask,
            radius=3,
            line_thickness=line_thickness,
            horizon=True,
            plot_outlier=plot_outlier,
            show_text=show_text,
        )

        if img_match.shape[0] != new_h:
            img_match = resize_img(img=img_match, nh=new_h)
        cv2.imwrite(str(fig_path), img=img_match)
