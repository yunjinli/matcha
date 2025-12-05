from copy import deepcopy

import torch
import torch.nn as nn
from matcha.feature.base_feature import BaseFeature
from matcha.utils.semantic_matching import (
    compute_semantic_matches,
    compute_semantic_matches_onebyone
)


class BaseMatcher(nn.Module):
    def __init__(self, model: BaseFeature, device: torch.device):
        super().__init__()
        self.model = model.eval().to(device)
        self.device = device

    def forward(self, data0: dict, data1: dict, with_keypoint_detection: bool = True):
        # Forward method to handle both keypoint detection and query point matching
        if with_keypoint_detection:
            return self.forward_by_detection(data0=data0, data1=data1)
        else:
            return self.forward_with_query_points(data0=data0, data1=data1)

    def forward_by_detection(self, data0: dict, data1: dict):
        img0 = data0["image"]
        img1 = data1["image"]

        kpts0, desc0 = self.model.detect_and_describe(img=img0, kwargs=data0)
        kpts1, desc1 = self.model.detect_and_describe(img=img1, kwargs=data1)

        matches = self.nearest_neighbor_matching(x0=desc0, x1=desc1)
        return {
            "keypoints0": kpts0[0],
            "keypoints1": kpts1[0],
            "matches": matches[0],
        }

    def forward_with_query_points(self, data0: dict, data1: dict, soft_eval=True):
        """
        Forward method to handle query point matching without keypoint detection.
        Args:
            data0 (dict): Data for the first image, containing "image" and "keypoints".
            data1 (dict): Data for the second image, containing "image".
            soft_eval (bool): Whether to use soft evaluation for matching.
        Returns:
            dict: A dictionary containing keypoints and matches.
        """
        img0 = data0["image"]
        img1 = data1["image"]

        desc0 = self.model.describe(img=img0, kwargs=data0)
        desc1 = self.model.describe(img=img1, kwargs=data1)

        src_ft, trg_ft = desc0[0], desc1[0]

        _, _, im_h, im_w = img0.shape
        ft_c, ft_h, ft_w = src_ft.shape

        scale_x = ft_w / im_w
        scale_y = ft_h / im_h

        src_kps = data0["keypoints"][0]
        scaled_src_kps = deepcopy(src_kps)
        scaled_src_kps[:, 0] *= scale_x
        scaled_src_kps[:, 1] *= scale_y

        with torch.no_grad():
            if soft_eval:
                trg_kps = compute_semantic_matches(
                    src_ft=src_ft,
                    trg_ft=trg_ft,
                    src_kps=scaled_src_kps.long(),
                    soft_eval=True,
                )
            else:
                trg_kps = compute_semantic_matches_onebyone(
                    src_ft=src_ft,
                    trg_ft=trg_ft,
                    src_kps=scaled_src_kps.long(),
                )

        # recover the scale
        trg_kps[:, 0] /= scale_x
        trg_kps[:, 1] /= scale_y

        matches = torch.arange(trg_kps.shape[0], device=src_kps.device)

        return {
            "keypoints0": src_kps,
            "keypoints1": trg_kps,
            "matches": matches,
        }

    @staticmethod
    def nearest_neighbor_matching(x0: torch.Tensor, x1: torch.Tensor, with_mutual_check=True):
        def find_nn(sim, ratio_thresh, distance_thresh):
            sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
            dist_nn = 2 * (1 - sim_nn)
            mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
            if ratio_thresh:
                mask = mask & (dist_nn[..., 0] <= (ratio_thresh ** 2) * dist_nn[..., 1])
            if distance_thresh:
                mask = mask & (dist_nn[..., 0] <= distance_thresh ** 2)
            matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
            scores = torch.where(mask, (sim_nn[..., 0] + 1) / 2, sim_nn.new_tensor(0))
            return matches, scores

        def mutual_check(m0: torch.Tensor, m1: torch.Tensor):
            inds0 = torch.arange(m0.shape[-1], device=m0.device)
            loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
            ok = (m0 > -1) & (inds0 == loop)
            m0_new = torch.where(ok, m0, m0.new_tensor(-1))
            return m0_new

        sim = torch.einsum("bdn,bdm->bnm", x0, x1)
        matches0, _ = find_nn(sim, ratio_thresh=None, distance_thresh=None)

        if with_mutual_check:
            matches1, _ = find_nn(
                sim.transpose(1, 2), ratio_thresh=None, distance_thresh=None
            )
            matches0 = mutual_check(matches0, matches1)
        return matches0
