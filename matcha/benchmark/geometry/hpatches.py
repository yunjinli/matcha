import logging
import os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm

from matcha.benchmark.base_benchmark import Benchmark
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy
from matcha.utils.geometry import rescale_keypoints


class HPatchesMatching(Benchmark):
    def __init__(
            self,
            matcher: BaseMatcher,
            config: dict,
            device: torch.device,
            plot=False,
    ):
        super().__init__(
            matcher,
            config=config,
            device=device,
            benchmark_name="hpatches_matching",
            plot=plot,
        )

    def init_dataset(self):
        self.n_i = 52
        self.n_v = 56

        # Following D2Net to ignore scenes  with image resolution larger than 1600x1200
        self.ignored_seqs = [
            "i_contruction",
            "i_crownnight",
            "i_dc i_pencils",
            "i_whitebuilding",
            "v_artisans",
            "v_astronautis",
            "v_talent",
        ]

    def run(self, debug=False):
        dataset_path = self.config.dataset_path
        seq_names = sorted(os.listdir(dataset_path))
        n_feats = []
        n_matches = []
        seq_type = []
        lim = [1, 15]
        rng = np.arange(lim[0], lim[1] + 1)
        i_err = {thr: 0 for thr in rng}
        v_err = {thr: 0 for thr in rng}

        for _, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
            if seq_name in self.ignored_seqs:
                continue
            img_path0 = osp.join(dataset_path, seq_name, f"{1}.ppm")
            img_out0 = self.matcher.model.load_image(img_path0)
            img0 = img_out0["image"]
            img_tensor0 = img_out0["image_tensor"]
            feat_c0, feat_f0 = self.load_cached_dift_feature(
                img_tensor0,
                img_path0,
            )

            for im_idx1 in range(2, 7):
                img_path1 = osp.join(dataset_path, seq_name, f"{im_idx1}.ppm")
                img_out1 = self.matcher.model.load_image(img_path1)
                img1 = img_out1["image"]
                img_tensor1 = img_out1["image_tensor"]
                feat_c1, feat_f1 = self.load_cached_dift_feature(
                    img_tensor1,
                    img_path1,
                )

                # Prepare model input
                data0 = {
                    "image": img_tensor0.to(self.device)[None],
                    "feat_c": feat_c0,
                    "feat_f": feat_f0,
                }
                data1 = {
                    "image": img_tensor1.to(self.device)[None],
                    "feat_c": feat_c1,
                    "feat_f": feat_f1,
                }

                # Model inference
                output = self.matcher(data0=data0, data1=data1)

                # Parse outputs in numpy
                org_matches = to_numpy(output.get("matches", None))
                keypoints0 = to_numpy(output["keypoints0"])
                keypoints1 = to_numpy(output["keypoints1"])
                n_feats.append(len(keypoints0))
                n_feats.append(len(keypoints1))

                # Init matches & err dist
                matches = np.empty((0, 4))
                dist = np.array([float("inf")])
                if org_matches is not None:
                    # Resize keypoints back to target image resolution
                    keypoints0 = rescale_keypoints(
                        kpts=keypoints0,
                        w=img_tensor0.shape[-1],
                        h=img_tensor0.shape[-2],
                        wt=img0.shape[1],
                        ht=img0.shape[0],
                    )
                    keypoints1 = rescale_keypoints(
                        kpts=keypoints1,
                        w=img_tensor1.shape[-1],
                        h=img_tensor1.shape[-2],
                        wt=img1.shape[1],
                        ht=img1.shape[0],
                    )
                    matched_ids0 = np.where(org_matches >= 0)[0]
                    matched_ids1 = org_matches[org_matches >= 0]
                    matches = np.hstack(
                        [matched_ids0.reshape(-1, 1), matched_ids1.reshape(-1, 1)]
                    )

                    # Compute metrics
                    homography = np.loadtxt(
                        os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx1))
                    )
                    pos_0 = keypoints0[matches[:, 0], :2]
                    pos_0_h = np.concatenate(
                        [pos_0, np.ones([matches.shape[0], 1])], axis=1
                    )
                    pos_1_proj_h = np.transpose(
                        np.dot(homography, np.transpose(pos_0_h))
                    )
                    pos_1_proj = pos_1_proj_h[:, :2] / pos_1_proj_h[:, 2:]
                    pos_1 = keypoints1[matches[:, 1], :2]
                    dist = np.sqrt(np.sum((pos_1 - pos_1_proj) ** 2, axis=1))

                # Cache statis
                n_matches.append(len(matches))
                seq_type.append(seq_name[0])
                for thr in rng:
                    if seq_name[0] == "i":
                        i_err[thr] += np.mean(dist <= thr)
                    else:
                        v_err[thr] += np.mean(dist <= thr)

                print(
                    f"{seq_name}-{im_idx1} #kpts: {len(keypoints0)}/{len(keypoints1)} #m: {len(matches)} Acc@3: {np.mean(dist <= 3):.4f}"
                )

                if self.plot:
                    inlier_mask = dist <= 5
                    fig_name = f"{seq_name}_{im_idx1}_K0{len(keypoints0)}_K1{len(keypoints1)}_M{len(matches)}_A{np.sum(inlier_mask)}.png"
                    self.write_matching_image(
                        image0=img0,
                        image1=img1,
                        keypoints0=keypoints0,
                        keypoints1=keypoints1,
                        inlier_mask=inlier_mask,
                        fig_name=fig_name,
                    )

            if debug:
                break

        seq_type = np.array(seq_type)
        n_feats = np.array(n_feats)
        n_matches = np.array(n_matches)

        logging.info(f"Results of {self.method} on HPatch Geometric Matching:")
        self.summary_matching(
            i_err=i_err, v_err=v_err, stats=[seq_type, n_feats, n_matches]
        )
        results = (
            i_err,
            v_err,
            [np.array(seq_type), np.array(n_feats), np.array(n_matches)],
        )
        if not debug:
            self.save_results(results)
        return results

    def summary_matching(self, i_err, v_err, stats):
        np.set_printoptions(precision=3)
        seq_type, n_feats, n_matches = stats
        n_i = 52
        n_v = 56
        summary = "#Features: mean={:.0f} min={:d} max={:d}\n".format(
            np.mean(n_feats), np.min(n_feats), np.max(n_feats)
        )
        summary += "#(Old)Matches: a={:.0f}, i={:.0f}, v={:.0f}\n".format(
            np.sum(n_matches) / ((n_i + n_v) * 5),
            np.sum(n_matches[seq_type == "i"]) / (n_i * 5),
            np.sum(n_matches[seq_type == "v"]) / (n_v * 5),
        )
        summary += "#Matches: a={:.0f}, i={:.0f}, v={:.0f}\n".format(
            np.mean(n_matches),
            np.mean(n_matches[seq_type == "i"]),
            np.mean(n_matches[seq_type == "v"]),
        )
        thres = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        thres = np.array(thres)
        ierr = np.array([i_err[th] / (n_i * 5) for th in thres])
        verr = np.array([v_err[th] / (n_v * 5) for th in thres])
        aerr = np.array([(i_err[th] + v_err[th]) / ((n_i + n_v) * 5) for th in thres])
        summary += "MMA@{} px:\na={}\ni={}\nv={}\n".format(thres, aerr, ierr, verr)
        logging.info(summary)
        return summary

    def save_results(self, results):
        cache_path = self.cache_dir / "results.npy"
        np.save(cache_path, np.array(results, dtype=object))
        logging.info(f"Save results to {cache_path}")
