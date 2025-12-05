import torch
import os
import os.path as osp
import numpy as np
from glob import glob
import json
from tqdm import tqdm
from matcha.dataset.semantic.semantic import SemanticDS


class AP10K(SemanticDS):
    def __init__(
            self,
            dataset_path: str,
            mode="trn",
            img_resize=(800, 608),
            subsample=None,
            bbox_threshold=True,
            max_keypoint_size=30,
            flip_image=True,
            category_embed_path=None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            mode=mode,
            img_resize=img_resize,
            subsample=subsample,
            bbox_threshold=bbox_threshold,
            max_keypoint_size=max_keypoint_size,
            flip_image=flip_image,
            category_embed_path=category_embed_path,
        )

        image_paths, keypoints, cats, used_keypoints, thresholds = self.load_data()
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.cats = cats
        self.used_keypoints = used_keypoints
        self.thresholds = thresholds

    def __len__(self):
        return len(self.image_paths) // 2

    def __getitem__(self, idx):
        img_path0 = self.image_paths[idx * 2]
        img_path1 = self.image_paths[idx * 2 + 1]
        cat0 = self.cats[idx * 2]
        cat1 = self.cats[idx * 2 + 1]
        kpts0 = self.keypoints[idx * 2]
        kpts1 = self.keypoints[idx * 2 + 1]

        org_threshold0 = self.thresholds[idx * 2]
        org_threshold1 = self.thresholds[idx * 2 + 1]

        # pad keypoints to same size
        kpts0 = self.pad_keypoints(keypoints=kpts0, max_size=self.max_keypoint_size)
        kpts1 = self.pad_keypoints(keypoints=kpts1, max_size=self.max_keypoint_size)

        img0_tensor, scale0, img0_flip_tensor, org_size0 = self.load_image(
            path=img_path0, resize=self.img_resize
        )
        img1_tensor, scale1, img1_flip_tensor, org_size1 = self.load_image(
            path=img_path1, resize=self.img_resize
        )  # [scale_w = org_w/new_w, scale_h]

        kpts0_tensor = torch.from_numpy(kpts0.copy())
        kpts0_tensor[:, 0] /= scale0[0]
        kpts0_tensor[:, 1] /= scale0[1]
        kpts1_tensor = torch.from_numpy(kpts1.copy())
        kpts1_tensor[:, 0] /= scale1[0]
        kpts1_tensor[:, 1] /= scale1[1]

        threshold0 = torch.tensor(np.array([org_threshold0])) / torch.max(scale0)
        threshold1 = torch.tensor(np.array([org_threshold1])) / torch.max(scale1)

        data = {
            "image0": img0_tensor.float(),
            "image1": img1_tensor.float(),
            "cat0": cat0,
            "cat1": cat1,
            "keypoints0": kpts0_tensor.float(),
            "keypoints1": kpts1_tensor.float(),
            "thresholds0": threshold0.float(),
            "thresholds1": threshold1.float(),
            "dataset_name": "ap10k",
            "dataset_label": "S",  # Semantic dataset for balanced sampling
            "image_path0": img_path0,
            "image_path1": img_path1,
        }

        if self.category_embedding is not None:
            prompt_embed0 = self.category_embedding[cat0]
            prompt_embed1 = self.category_embedding[cat1]
            data["prompt_embed0"] = torch.from_numpy(prompt_embed0).float()
            data["prompt_embed1"] = torch.from_numpy(prompt_embed1).float()

        if self.mode == "test":
            data["original_keypoints0"] = torch.from_numpy(kpts0)
            data["original_keypoints1"] = torch.from_numpy(kpts1)
            data["original_size0"] = torch.from_numpy(org_size0)
            data["original_size1"] = torch.from_numpy(org_size1)
            data["original_threshold0"] = torch.tensor(org_threshold0)
            data["original_threshold1"] = torch.tensor(org_threshold1)

        if img0_flip_tensor is not None:
            data["image0_flip"] = img0_flip_tensor
            data["image1_flip"] = img1_flip_tensor

        return data

    def load_data(self):
        subfolders = os.listdir(os.path.join(self.dataset_path, "ImageAnnotation"))
        categories = sorted(
            [
                item
                for subfolder in subfolders
                for item in os.listdir(
                os.path.join(self.dataset_path, "ImageAnnotation", subfolder)
            )
            ]
        )

        files, kps, cats, used_points_set, all_thresholds = ([] for _ in range(5))
        for cat_idx, cat in tqdm(
                enumerate(categories), total=len(categories), desc="Processing AP10k"
        ):
            if cat in [
                "argali sheep",
                "black bear",
                "king cheetah",
            ]:  # these three categories is not present in training set of ap10k
                continue

            single_files, single_kps, thresholds, used_points = self.load_category(
                path=self.dataset_path,
                category=cat,
                split=self.mode,
                subsample=self.subsample,
            )

            files.extend(single_files)
            # single_kps = F_pad(single_kps, (0, 0, 0, 30 - single_kps.shape[1], 0, 0), value=0)
            kps.append(single_kps)
            used_points_set.extend([used_points] * (len(single_files) // 2))
            # cats.extend([cat_idx] * (len(single_files) // 2))
            cats.extend([cat] * len(single_files))
            if self.bbox_threshold:
                all_thresholds.extend(thresholds)

        kps = np.concatenate(kps, axis=0)
        # print("files: ", len(files), kps.shape, len(cats), len(all_thresholds))
        return files, kps, cats, used_points_set, all_thresholds

    def load_category(self, path, category, split, subsample=None):
        np.random.seed(42)
        pairs = sorted(glob(f"{path}/PairAnnotation/{split}/*:{category}.json"))
        if subsample is not None and subsample > 0:
            pairs = [
                pairs[ix] for ix in np.random.choice(len(pairs), subsample)
            ]  # overlap may exist
        files = []
        kps = []
        thresholds = []
        for pair in pairs:
            with open(pair) as f:
                data = json.load(f)
            source_json_path = osp.join(self.dataset_path, data["src_json_path"])
            target_json_path = osp.join(self.dataset_path, data["trg_json_path"])
            src_img_path = source_json_path.replace("json", "jpg").replace(
                "ImageAnnotation", "JPEGImages"
            )
            trg_img_path = target_json_path.replace("json", "jpg").replace(
                "ImageAnnotation", "JPEGImages"
            )

            with open(source_json_path) as f:
                src_file = json.load(f)
            with open(target_json_path) as f:
                trg_file = json.load(f)

            source_bbox = np.asarray(src_file["bbox"])  # l t w h
            target_bbox = np.asarray(trg_file["bbox"])

            source_size = np.array([src_file["width"], src_file["height"]])  # (W, H)
            target_size = np.array([trg_file["width"], trg_file["height"]])  # (W, H)

            # print(source_raw_kps.shape)
            source_kps = np.array(src_file["keypoints"], dtype=float).reshape(-1, 3)
            source_kps[:, -1] /= 2
            # source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)

            # target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
            target_kps = np.array(trg_file["keypoints"], dtype=float).reshape(-1, 3)
            target_kps[:, -1] /= 2
            # target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
            # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
            # they are computed as well:
            # thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
            if ("test" in split) or ("val" in split):
                thresholds.append(max(target_bbox[3], target_bbox[2]))
                thresholds.append(max(target_bbox[3], target_bbox[2]))
            elif "trn" in split:
                thresholds.append(max(source_bbox[3], source_bbox[2]))
                thresholds.append(max(target_bbox[3], target_bbox[2]))

            kps.append(source_kps)
            kps.append(target_kps)
            files.append(src_img_path)
            files.append(trg_img_path)

        # kps = torch.stack(kps)
        # used_kps, = torch.where(kps[:, :, 2].any(dim=0))
        kps = np.array(kps)
        (used_kps,) = np.where(kps[:, :, 2].any(axis=0))
        kps = kps[:, used_kps, :]
        # print(f'Final number of used key points: {kps.size(1)}')
        return files, kps, thresholds, used_kps
