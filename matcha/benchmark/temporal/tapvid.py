import torch
import torch.nn.functional as F
import os.path as osp
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

from matcha.benchmark.base_benchmark import Benchmark
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.semantic_matching import compute_semantic_matches, compute_semantic_matches_onebyone
from matcha.benchmark.visualization import visualize_matches


class TapvidBenchmark(Benchmark):
    default_config = {
        "image_size": (512, 512),
        "soft_eval": False,
        "semantic_mode": False,
        "norm_desc": False,
        "post_process": True,
        "plot_inlier": False,
    }

    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        config = {**self.default_config, **config}
        super().__init__(
            matcher,
            config=config,
            device=device,
            benchmark_name="tapvid_davis",
            plot=plot,
        )

    def init_dataset(self):
        data_pkl = Path(self.config.dataset_path) / "tapvid_davis.pkl"
        assert data_pkl.exists(), f"Can not find {data_pkl}!"
        self.dataset = np.load(data_pkl, allow_pickle=True)
        self.categories = self.dataset.keys()
        logging.info(f"Total categories={len(self.categories)}")

    @staticmethod
    def post_process(feat_g, feat_s, norm=True):
        if feat_g.shape[2] != feat_s.shape[2] or feat_g.shape[3] != feat_s.shape[3]:
            feat_s = F.interpolate(
                feat_s, size=(feat_g.shape[2], feat_g.shape[3]), mode="bilinear"
            )
            feat_s = F.normalize(feat_s, dim=1)

        stride = feat_s.shape[1] // feat_g.shape[1]
        feat = torch.concat([feat_g, feat_s[:, ::stride, :, :]], dim=1)
        if norm:
            feat = F.normalize(feat, dim=1)
        return feat

    def run(self, test_cats=None, debug=False):
        imsize = self.config.image_size
        thresholds = [0.15, 0.1, 0.05]
        results = {}
        for th in thresholds:
            results[th] = 0
        total = 0

        # Define output dir
        exp_tag = f"imsize{imsize[0]}x{imsize[1]}"
        if self.config.semantic_mode:
            exp_tag += "_sem"
        if self.config.soft_eval:
            exp_tag += "_soft"
        if self.config.post_process:
            exp_tag += "_post"

        save_dir = self.cache_dir / exp_tag
        save_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Experiment output dir: {save_dir}")

        all_cat_results = {}
        for cat in tqdm(self.categories, total=len(self.categories)):
            if test_cats is not None and cat not in test_cats:
                continue

            cat_results = {}
            cat_total = 0
            for th in thresholds:
                cat_results[th] = 0

            # Load data
            points = self.dataset[cat]["points"]
            occluded = self.dataset[cat]["occluded"]
            video = self.dataset[cat]["video"]
            n_frames = video.shape[0]
            print(f"Processing cat of {cat} with {n_frames} frames")

            ref_img = video[0]
            ref_h, ref_w, _ = ref_img.shape
            ref_img = cv2.resize(ref_img, dsize=imsize)
            ref_img_tensor = (
                torch.from_numpy(ref_img.astype(float) / 255.0)
                .permute(2, 0, 1)
                .float()
                .to(self.device)[None]
            )

            # Fake image path
            ref_img_path = osp.join(self.config.dataset_path, f"tapvid_{0:3d}")
            ref_feat_c, ref_feat_f = self.load_cached_dift_feature(
                ref_img_tensor,
                ref_img_path,
            )

            # Load reference feature
            ref_feat = self.matcher.model.describe(
                ref_img_tensor,
                feat_c=ref_feat_c,
                feat_f=ref_feat_f,
                semantic_mode=self.config.semantic_mode,
                normalize=self.config.norm_desc,
            )

            # Post-processing
            if self.config.post_process:
                ref_feat = self.post_process(feat_g=ref_feat, feat_s=ref_feat)

            ds_scale = ref_img_tensor.shape[-1] / ref_feat.shape[-1]

            # Match against the following frames
            for cid in range(1, n_frames):
                curr_img = video[cid]
                curr_h, curr_w, _ = curr_img.shape
                curr_img = cv2.resize(curr_img, dsize=imsize)
                ref_vis = np.logical_not(occluded[:, 0])
                curr_vis = np.logical_not(occluded[:, cid])
                vis_mask = np.logical_and(ref_vis, curr_vis)

                # Pre-defined keypoints
                ref_points = points[:, 0][vis_mask]
                ref_points[:, 0] = ref_points[:, 0] * ref_w
                ref_points[:, 1] = ref_points[:, 1] * ref_h
                if ref_points.shape[0] == 0:
                    continue

                ref_points[:, 0] = ref_points[:, 0] * (imsize[0] / ref_w)
                ref_points[:, 1] = ref_points[:, 1] * (imsize[1] / ref_h)

                curr_points = points[:, cid][vis_mask]
                curr_points[:, 0] *= curr_w
                curr_points[:, 1] *= curr_h
                curr_points[:, 0] = curr_points[:, 0] * (imsize[0] / curr_w)
                curr_points[:, 1] = curr_points[:, 1] * (imsize[1] / curr_h)

                curr_img_tensor = (
                    torch.from_numpy(curr_img.astype(float) / 255.0)
                    .permute(2, 0, 1)
                    .float()
                    .to(self.device)[None]
                )

                # Fake the current image path
                curr_img_path = osp.join(self.config.dataset_path, f"tapvid_{cid:3d}")
                curr_feat_c, curr_feat_f = self.load_cached_dift_feature(
                    curr_img_tensor,
                    curr_img_path,
                )

                # Load current feature
                curr_feat = self.matcher.model.describe(
                    curr_img_tensor,
                    feat_c=curr_feat_c,
                    feat_f=curr_feat_f,
                    semantic_mode=self.config.semantic_mode,
                    normalize=self.config.norm_desc,
                )

                # Post-processing
                if self.config.post_process:
                    curr_feat = self.post_process(feat_g=curr_feat, feat_s=curr_feat)

                # Compute matches
                if self.config.soft_eval:
                    pred_curr_points = compute_semantic_matches(
                        src_ft=ref_feat[0],
                        trg_ft=curr_feat[0],
                        src_kps=torch.from_numpy(ref_points / ds_scale).long(),
                        soft_eval=True,
                    )
                else:
                    pred_curr_points = compute_semantic_matches_onebyone(
                        src_ft=ref_feat[0],
                        trg_ft=curr_feat[0],
                        src_kps=torch.from_numpy(ref_points / ds_scale).long(),
                    )

                pred_curr_points = pred_curr_points.cpu().numpy() * ds_scale

                dist = (pred_curr_points - curr_points) ** 2
                dist = np.sqrt(np.sum(dist, axis=1)) / max(imsize)

                total = total + curr_points.shape[0]
                cat_total = cat_total + curr_points.shape[0]
                for th in thresholds:
                    n_th = np.sum(dist <= th)
                    results[th] = results[th] + n_th
                    cat_results[th] = cat_results[th] + n_th

                if self.plot:
                    cat_save_dir = save_dir / cat
                    cat_save_dir.mkdir(exist_ok=True, parents=True)
                    corr_mask = dist <= 0.1
                    incorr_mask = ~corr_mask
                    if self.config.plot_inlier:
                        img_match = visualize_matches(
                            img0=ref_img[:, :, ::-1],
                            img1=curr_img[:, :, ::-1],
                            pts0=[ref_points[corr_mask]],
                            pts1=[pred_curr_points[corr_mask]],
                            colors=[(0, 255, 0)],
                            lw=5,
                        )
                    else:
                        img_match = visualize_matches(
                            img0=ref_img[:, :, ::-1],
                            img1=curr_img[:, :, ::-1],
                            pts0=[ref_points[corr_mask], ref_points[incorr_mask]],
                            pts1=[
                                pred_curr_points[corr_mask],
                                pred_curr_points[incorr_mask],
                            ],
                            colors=[(0, 255, 0), (0, 0, 255)],
                            lw=5,
                        )
                    cv2.imwrite(cat_save_dir / f"{cid:02d}.png", img_match)

            all_cat_results[cat] = {}
            for th in thresholds:
                all_cat_results[cat][th] = cat_results[th] / cat_total
                print(
                    f"Method: {self.method} {cat} with Accuracy@{th}: {cat_results[th] / cat_total * 100:.2f}%"
                )
            if debug:
                break

        print(f"\n>>>>Summary over {len(all_cat_results)} categories:")
        for th in thresholds:
            print(
                f"Per point - Method: {self.method} with Accuracy@{th}: {results[th] / total * 100:.2f}%"
            )

        for th in thresholds:
            cat_th = [all_cat_results[cat][th] for cat in all_cat_results.keys()]
            cat_th = np.mean(cat_th)
            print(
                f"Per cat - Method: {self.method} with Accuracy@{th}: {cat_th * 100:.2f}%"
            )
