import logging

import torch
import cv2
import numpy as np
from tqdm import tqdm

from matcha.benchmark.base_benchmark import Benchmark
# from matcha.benchmark.base_matcher import BaseMatcher
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy

from matcha.utils.metrics import PoseMetrics
from matcha.utils.geometry import (
    estimate_pose_poselib_general,
    compute_geo_inlier,
    rescale_keypoints,
)


class GeometricMatchingBenchmark(Benchmark, PoseMetrics):
    def __init__(
            self,
            benchmark_name: str,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            matcher=matcher,
            benchmark_name=benchmark_name,
            config=config,
            device=device,
            plot=plot
        )

    def init_dataset(self, **kwargs):
        raise NotImplementedError

    def make_camera(self, camera_model: str, width: int, height: int, params: np.ndarray):
        camera = {
            "model": camera_model,
            "width": width,
            "height": height,
            "params": params,
        }
        return camera

    def run(self, test_cats: list = None, debug: bool = False):
        tot_e_t, tot_e_R, tot_e_pose = [], [], []
        thresholds = [5, 10, 20]

        total_inlier_ratios = []
        total_matches = []
        total_keypoints = []

        # Define output dir
        exp_tag = f"{self.matcher.model.img_process_tag}_{self.matcher.model.detector_tag}ransac{self.config.ransac_threshold}_itr{self.config.shuffle_iter}"
        logging.info(f"Experiment tag: {exp_tag}.")

        for bid, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            img_path0 = batch["image_path0"][0]
            img_path1 = batch["image_path1"][0]
            img0 = batch["image0"]
            img1 = batch["image1"]

            org_K0 = batch["original_K0"][0].cpu().numpy()
            org_K1 = batch["original_K1"][0].cpu().numpy()
            org_width0 = batch["original_width0"][0].cpu().numpy()
            org_height0 = batch["original_height0"][0].cpu().numpy()
            org_width1 = batch["original_width1"][0].cpu().numpy()
            org_height1 = batch["original_height1"][0].cpu().numpy()
            # print(
            #     f"{self.benchmark_name}: {bid} - {img0.shape}, {img1.shape}, {org_width0}-{org_height0}, {org_width1}-{org_height1}")

            camera_model0 = batch["camera_model0"][0]
            camera_model1 = batch["camera_model1"][0]
            org_params0 = batch["original_params0"][0].cpu().numpy()
            org_params1 = batch["original_params1"][0].cpu().numpy()
            camera0 = {"model": camera_model0, "width": org_width0, "height": org_height0, "params": org_params0}
            camera1 = {"model": camera_model1, "width": org_width1, "height": org_height1, "params": org_params1}

            R = batch["R"][0].cpu().numpy()
            t = batch["t"][0].cpu().numpy()

            feat_c0, feat_f0 = self.load_cached_dift_feature(img0, img_path0)
            feat_c1, feat_f1 = self.load_cached_dift_feature(img1, img_path1)

            # Model inference
            data0 = {
                "image": img0,
                "feat_c": feat_c0,
                "feat_f": feat_f0,
            }
            data1 = {
                "image": img1,
                "feat_c": feat_c1,
                "feat_f": feat_f1,
            }
            output = self.matcher(data0=data0, data1=data1)
            # Parse outputs in numpy
            org_matches = to_numpy(output.get("matches", None))
            keypoints0 = to_numpy(output["keypoints0"])
            keypoints1 = to_numpy(output["keypoints1"])
            total_keypoints.append(len(keypoints0))
            total_keypoints.append(len(keypoints1))

            # Init matches & errs
            matches = np.empty((0, 4))
            inlier_ratio = 0
            e_poses = e_ts = e_Rs = [90] * self.config.shuffle_iter

            if org_matches is not None:
                # Resize keypoints back to target image resolution
                keypoints0 = rescale_keypoints(
                    kpts=keypoints0,
                    w=img0.shape[-1],
                    h=img1.shape[-2],
                    wt=org_width0,
                    ht=org_height0,
                )
                keypoints1 = rescale_keypoints(
                    kpts=keypoints1,
                    w=img0.shape[-1],
                    h=img1.shape[-2],
                    wt=org_width1,
                    ht=org_height1,
                )
                matched_ids0 = np.where(org_matches >= 0)[0]
                matched_ids1 = org_matches[org_matches >= 0]
                matches = np.hstack(
                    [matched_ids0.reshape(-1, 1), matched_ids1.reshape(-1, 1)]
                )
                kpts0 = keypoints0[matches[:, 0]]
                kpts1 = keypoints1[matches[:, 1]]

                # Measure inliers
                inlier_mask = compute_geo_inlier(
                    kpts0=kpts0, kpts1=kpts1, K0=org_K0, K1=org_K1, R=R, t=t
                )
                inlier_ratio = np.sum(inlier_mask) / len(kpts0)

                # Compute pose errors
                e_poses = e_ts = e_Rs = []
                plotted = False
                for _ in range(self.config.shuffle_iter):
                    # Proposed by RoMA to have robust pose evaluation
                    # As some ransac solvers are sensitive to the order of the keypoints
                    shuffling = np.random.permutation(
                        np.arange(len(kpts0))
                    )
                    kpts0 = kpts0[shuffling]
                    kpts1 = kpts1[shuffling]

                    try:
                        R_est, t_est, mask = estimate_pose_poselib_general(
                            kpts0,
                            kpts1,
                            camera0,
                            camera1,
                            threshold=self.config.ransac_threshold,
                        )
                        e_t, e_R = self.compute_relative_pose_error(
                            R_est, t_est.flatten(), R, t
                        )
                        e_pose = max(e_t, e_R)

                        if debug:
                            print(
                                f"K:{len(keypoints0)}/{len(keypoints1)}, M: {len(matches)}/{inlier_ratio:.3f}, err_R/t: {e_R:.2f}/{e_t:.2f}"
                            )

                        if self.plot and not plotted:
                            org_img0 = cv2.imread(img_path0)
                            org_img1 = cv2.imread(img_path1)

                            fig_name = f"{img_path0.split('/')[-1]}_{img_path1.split('/')[-1]}_M{len(matches)}_R{e_R:.2f}t{e_t:.2f}.png"
                            if len(org_img0.shape) == 2:
                                org_img0 = cv2.cvtColor(org_img0, cv2.COLOR_GRAY2BGR)
                                org_img1 = cv2.cvtColor(org_img1, cv2.COLOR_GRAY2BGR)
                            else:
                                org_img0 = cv2.cvtColor(org_img0, cv2.COLOR_RGB2BGR)
                                org_img1 = cv2.cvtColor(org_img1, cv2.COLOR_RGB2BGR)

                            self.write_matching_image(
                                image0=org_img0,
                                image1=org_img1,
                                keypoints0=kpts0,
                                keypoints1=kpts1,
                                inlier_mask=mask,
                                new_h=512,
                                plot_outlier=True,
                                show_text=f"R={e_R:.2f} t={e_t:.2f} Inliers={len(inlier_mask)}/{np.sum(mask)}",
                                line_thickness=4,
                                fig_name=fig_name,
                            )
                            plotted = True
                    except Exception as e:
                        print(repr(e))
                        e_pose = e_t = e_R = 90

                    e_ts.append(e_t)
                    e_Rs.append(e_R)
                    e_poses.append(e_pose)

            tot_e_t += e_ts
            tot_e_R += e_Rs
            tot_e_pose += e_poses
            total_inlier_ratios.append(inlier_ratio)
            total_matches.append(len(matches))

            if bid % 100 == 0:
                auc_tmp = self.pose_auc(np.array(tot_e_pose), thresholds)
                results = {
                    "auc_5": auc_tmp[0],
                    "auc_10": auc_tmp[1],
                    "auc_20": auc_tmp[2],
                    "n_matches": np.mean(total_matches),
                    "n_keypoints": np.mean(total_keypoints),
                    "inlier_ratio": np.mean(total_inlier_ratios),
                }  #
                print(
                    f">>>Results of {self.method} on {self.benchmark_name} with {len(total_matches)} samples."
                )
                for k in results.keys():
                    print(f"{k}: {results[k]:.4f}")

            if debug and bid > 5:
                break

        # Summarize evaluation metrics
        tot_e_pose = np.array(tot_e_pose)
        auc = self.pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_15 = (tot_e_pose < 15).mean()
        acc_20 = (tot_e_pose < 20).mean()
        map_5 = acc_5
        map_10 = np.mean([acc_5, acc_10])
        map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])

        results = {
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": map_5,
            "map_10": map_10,
            "map_20": map_20,
            "n_matches": np.mean(total_matches),
            "n_keypoints": np.mean(total_keypoints),
            "inlier_ratio": np.mean(total_inlier_ratios),
        }

        self.print_summary(results=results)
        self.save_results(results=results)
        return results

    def print_summary(self, results: dict):
        print(f"Results of {self.method} on {self.benchmark_name}")
        for k in results.keys():
            print(f"Pose AUC@{k}: {results[k]:.4f}")

    def save_results(self, results: dict):
        cache_path = self.cache_dir / "results.npy"
        np.save(cache_path, np.array(results, dtype=object))
        print(f"Save results to {cache_path}")
