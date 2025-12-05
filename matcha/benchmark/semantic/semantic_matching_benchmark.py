import logging

import torch
import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from matcha.benchmark.base_benchmark import Benchmark
# from matcha.benchmark.base_matcher import BaseMatcher
from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.visualization import visualize_matches
from matcha.utils.semantic_matching import compute_semantic_matches, compute_semantic_matches_onebyone


class SemanticMatchingBenchmark(Benchmark):
    def __init__(
            self,
            benchmark_name: str,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            matcher, config=config, device=device, benchmark_name=benchmark_name, plot=plot
        )

    def run(self, test_cats: list = None, debug: bool = False):
        imsize = self.config.image_size
        error_thresholds = [0.15, 0.1, 0.05, 0.01]
        total_img_pcks = {}
        total_point_pcks = {}
        total_points = 0

        for err_th in error_thresholds:
            total_img_pcks[err_th] = []
            total_point_pcks[err_th] = 0

        cat_img_pcks = {}
        cat_point_pcks = {}
        for err_th in error_thresholds:
            total_img_pcks[err_th] = []
            total_point_pcks[err_th] = 0

        # Define output dir
        exp_tag = f"imsize{imsize[0]}x{imsize[1]}" if imsize is not None else f"imsize-None"
        if self.config.semantic_mode:
            exp_tag += "_sem"
        if self.config.soft_eval:
            exp_tag += "_soft"

        save_dir = self.cache_dir / exp_tag
        save_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Experiment output dir: {save_dir}")

        for bid, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            img_path0 = batch["image_path0"][0]
            img_path1 = batch["image_path1"][0]
            img0 = batch["image0"]
            img1 = batch["image1"]
            kps0 = batch["keypoints0"][0].cpu().numpy()
            kps1 = batch["keypoints1"][0].cpu().numpy()
            visible = kps0[:, 2] * kps1[:, 2] > 0
            kps0 = kps0[visible][..., :2].astype(int)
            kps1 = kps1[visible][..., :2].astype(int)

            threshold = max(imsize)

            cat = batch["cat0"][0]
            if test_cats is not None and cat not in test_cats:
                continue

            # Initialize metric dict
            if cat not in cat_img_pcks.keys():
                cat_img_pcks[cat] = {}
                for err_th in error_thresholds:
                    cat_img_pcks[cat][err_th] = []

            if cat not in cat_point_pcks.keys():
                cat_point_pcks[cat] = {}
                cat_point_pcks[cat]["total"] = 0
                for err_th in error_thresholds:
                    cat_point_pcks[cat][err_th] = 0

            feat_c0, feat_f0 = self.load_cached_dift_feature(
                img0,
                img_path0,
            )
            feat_c1, feat_f1 = self.load_cached_dift_feature(
                img1,
                img_path1,
            )

            # Model inference
            with torch.no_grad():
                feat0 = self.matcher.model.describe(
                    img0,
                    feat_c=feat_c0,
                    feat_f=feat_f0,
                    cat=cat,
                    semantic_mode=self.config.semantic_mode,
                    normalize=self.config.norm_desc,

                )
                feat1 = self.matcher.model.describe(
                    img1,
                    feat_c=feat_c1,
                    feat_f=feat_f1,
                    cat=cat,
                    semantic_mode=self.config.semantic_mode,
                    normalize=self.config.norm_desc,
                )
            src_ft = feat0[0]
            trg_ft = feat1[0]
            h, w = trg_ft.shape[-2], trg_ft.shape[-1]
            scale_x = w / imsize[0]
            scale_y = h / imsize[1]

            total = 0
            correct = {}
            for err_th in error_thresholds:
                correct[err_th] = 0

            gt_matched_pts0 = []
            gt_matched_pts1 = []
            pred_matched_pts0 = []
            pred_matched_pts1 = []
            corr_pred_matched_pts0 = []
            corr_pred_matched_pts1 = []

            scaled_src_kps = deepcopy(kps0).astype(float)
            scaled_src_kps[:, 0] *= scale_x
            scaled_src_kps[:, 1] *= scale_y
            scaled_src_kps = torch.from_numpy(scaled_src_kps).to(src_ft.device)

            if self.config.soft_eval:
                pred_trg_kps = compute_semantic_matches(
                    src_ft=src_ft,
                    trg_ft=trg_ft,
                    src_kps=scaled_src_kps.long(),
                    soft_eval=True,
                )
            else:
                pred_trg_kps = compute_semantic_matches_onebyone(
                    src_ft=src_ft,
                    trg_ft=trg_ft,
                    src_kps=scaled_src_kps.long(),
                )
            pred_trg_kps = pred_trg_kps.cpu().numpy().astype(float)

            # scale to original size
            pred_trg_kps[:, 0] /= scale_x
            pred_trg_kps[:, 1] /= scale_y

            # Compute metrics
            dist = (kps1 - pred_trg_kps) ** 2  # (N, 2)
            dist = np.sqrt(np.sum(dist, axis=1))  # (N,)
            inlier_mask = (dist / threshold) <= 0.1

            for idx in range(kps0.shape[0]):
                total += 1
                total_points += 1
                cat_point_pcks[cat]["total"] += 1

                gt_matched_pts0.append(kps0[idx])
                gt_matched_pts1.append(kps1[idx])
                pred_matched_pts0.append(kps0[idx])
                pred_matched_pts1.append(pred_trg_kps[idx])

                for err_th in error_thresholds:
                    if (dist[idx] / threshold) <= err_th:
                        correct[err_th] += 1

                        total_point_pcks[err_th] += 1
                        cat_point_pcks[cat][err_th] += 1

                        if err_th == 0.1:
                            corr_pred_matched_pts0.append(kps0[idx])
                            corr_pred_matched_pts1.append(pred_trg_kps[idx])

            if self.plot:
                src_img = cv2.imread(img_path0)
                trg_img = cv2.imread(img_path1)
                if imsize is not None:
                    src_img = cv2.resize(src_img, dsize=imsize)
                    trg_img = cv2.resize(trg_img, dsize=imsize)
                gt_matched_pts0 = np.array(gt_matched_pts0)
                gt_matched_pts1 = np.array(gt_matched_pts1)
                pred_matched_pts0 = np.array(pred_matched_pts0)
                pred_matched_pts1 = np.array(pred_matched_pts1)
                corr_pred_matched_pts0 = np.array(corr_pred_matched_pts0)
                corr_pred_matched_pts1 = np.array(corr_pred_matched_pts1)

                img_match = visualize_matches(
                    img0=src_img,
                    img1=trg_img,
                    pts0=[
                        pred_matched_pts0[inlier_mask],
                        pred_matched_pts0[np.logical_not(inlier_mask)],
                    ],
                    pts1=[
                        pred_matched_pts1[inlier_mask],
                        pred_matched_pts1[np.logical_not(inlier_mask)],
                    ],
                    # colors=[(255, 0, 0), (0, 0, 255), (0, 255, 0)],
                    colors=[(0, 255, 0), (0, 0, 255)],
                    lw=5,
                )

                # cv2.imshow("img", img_match)
                # key = cv2.waitKey()
                # if key == ord("q"):
                #     exit(0)
                name_0 = img_path0.split("/")[-1]  # .replace("/", "-")
                name_1 = img_path1.split("/")[-1]  # .replace("/", "-")
                fig_name = f"{cat}_{name_0}_{name_1}.png"
                fig_path = save_dir / "plots" / fig_name
                cv2.imwrite(str(fig_path), img=img_match)

            for err_th in error_thresholds:
                total_img_pcks[err_th].append(correct[err_th] / total)  # per image
                cat_img_pcks[cat][err_th].append(correct[err_th] / total)

            if debug and bid > 5:
                break

        results = {}
        for cat in cat_point_pcks.keys():
            results[cat] = {}
            for err_th in error_thresholds:
                point_pck = (
                                    cat_point_pcks[cat][err_th] / cat_point_pcks[cat]["total"]
                            ) * 100
                image_pck = np.mean(cat_img_pcks[cat][err_th]) * 100  #
                results[cat][f"point_pck@{err_th}"] = point_pck
                results[cat][f"image_pck@{err_th}"] = image_pck
                print(f"{cat} per point @ {err_th}: {point_pck:.3f}%")
                print(f"{cat} per image @ {err_th}: {image_pck:.3f}%")

        for err_th in error_thresholds:
            point_pck = (total_point_pcks[err_th] / total_points) * 100
            image_pck = np.mean(total_img_pcks[err_th]) * 100

            results[f"total_point_pck@{err_th}"] = point_pck
            results[f"total_image_pck@{err_th}"] = image_pck
            print(f"Total per point @ {err_th}: {point_pck:.3f}%")
            print(f"Total per image @ {err_th}: {image_pck:.3f}%")

        cache_path = save_dir / "results.npy"
        np.save(cache_path, results)
        print(f"Save results to {cache_path}")
        return results
