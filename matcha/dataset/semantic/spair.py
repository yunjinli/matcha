import torch
import torch.utils as utils
from torch.nn.functional import pad as F_pad
import os
import os.path as osp
import numpy as np
from glob import glob
import json
from tqdm import tqdm
from .semantic import SemanticDS


class SpairDataset(SemanticDS):
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
            "dataset_name": "spair",
            "dataset_label": "S",
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
        categories = sorted(os.listdir(os.path.join(self.dataset_path, "ImageAnnotation")))
        files, kps, cats, used_points_set, all_thresholds = ([] for _ in range(5))
        for cat_idx, cat in tqdm(
                enumerate(categories), total=len(categories), desc="processing Spair"
        ):
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
        pairs = sorted(glob(f"{path}/PairAnnotation/{self.mode}/*:{category}.json"))
        if subsample is not None and subsample > 0:
            pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
        files = []
        thresholds = []
        kps = []
        category_anno = list(glob(f"{path}/ImageAnnotation/{category}/*.json"))[0]
        with open(category_anno) as f:
            num_kps = len(json.load(f)["kps"])
        for pair in pairs:
            source_kps = np.zeros((num_kps, 3), dtype=float)
            target_kps = np.zeros((num_kps, 3), dtype=float)
            with open(pair) as f:
                data = json.load(f)
            assert category == data["category"]
            source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
            target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
            source_json_name = source_fn.replace(
                "JPEGImages", "ImageAnnotation"
            ).replace("jpg", "json")
            target_json_name = target_fn.replace(
                "JPEGImages", "ImageAnnotation"
            ).replace("jpg", "json")
            source_bbox = np.asarray(data["src_bndbox"])  # (x1, y1, x2, y2)
            target_bbox = np.asarray(data["trg_bndbox"])
            with open(source_json_name) as f:
                file = json.load(f)
                kpts_src = file["kps"]
            with open(target_json_name) as f:
                file = json.load(f)
                kpts_trg = file["kps"]

            source_size = data["src_imsize"][:2]  # (W, H)
            target_size = data["trg_imsize"][:2]  # (W, H)

            for i in range(30):
                point = kpts_src[str(i)]
                if point is None:
                    source_kps[i, :3] = 0
                else:
                    # source_kps[i, :2] = torch.Tensor(point).float()  # set x and y
                    source_kps[i, :2] = point  # set x and y
                    source_kps[i, 2] = 1
            # source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)

            for i in range(30):
                point = kpts_trg[str(i)]
                if point is None:
                    target_kps[i, :3] = 0
                else:
                    # target_kps[i, :2] = torch.Tensor(point).float()
                    target_kps[i, :2] = point
                    target_kps[i, 2] = 1
            # target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
            # target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
            # target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
            if split == "test" or split == "val":
                # thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale)
                thresholds.append(
                    max(
                        target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]
                    )
                )
                thresholds.append(
                    max(
                        target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]
                    )
                )
            elif split == "trn":
                # thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]) * src_scale)
                # thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale)

                thresholds.append(
                    max(
                        source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]
                    )
                )
                thresholds.append(
                    max(
                        target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]
                    )
                )

            kps.append(source_kps)
            kps.append(target_kps)
            files.append(source_fn)
            files.append(target_fn)

        # kps = torch.stack(kps)
        kps = np.array(kps)
        (used_kps,) = np.where(kps[:, :, 2].any(axis=0))

        return files, kps, thresholds, used_kps
