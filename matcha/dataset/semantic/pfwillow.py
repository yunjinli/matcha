import torch
import os
import os.path as osp
import numpy as np
import pandas as pd

from tqdm import tqdm
from .semantic import SemanticDS
from .utils import read_mat, process_kps_pascal


class PFWillowDataset(SemanticDS):
    def __init__(
            self,
            dataset_path: str,
            mode="test",
            img_resize=(800, 608),
            subsample=None,
            bbox_threshold=True,
            max_keypoint_size=30,
            flip_image=True,  # by default should be false
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

        image_paths, keypoints, cats, used_keypoints, _ = self.load_data()
        self.image_paths = image_paths
        self.keypoints = keypoints
        self.cats = cats
        self.used_keypoints = used_keypoints

    def __len__(self):
        return len(self.image_paths) // 2

    def __getitem__(self, idx):
        img_path0 = self.image_paths[idx * 2]
        img_path1 = self.image_paths[idx * 2 + 1]
        cat0 = self.cats[idx * 2]
        cat1 = self.cats[idx * 2 + 1]
        kpts0 = self.keypoints[idx * 2]
        kpts1 = self.keypoints[idx * 2 + 1]

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

        threshold0 = torch.tensor(
            np.array([np.max([img0_tensor.shape[-2], img0_tensor.shape[-1]])])
        )
        threshold1 = torch.tensor(
            np.array([np.max([img1_tensor.shape[-2], img1_tensor.shape[-1]])])
        )
        data = {
            "image0": img0_tensor.float(),
            "image1": img1_tensor.float(),
            "cat0": cat0,
            "cat1": cat1,
            "keypoints0": kpts0_tensor.float(),
            "keypoints1": kpts1_tensor.float(),
            "thresholds0": threshold0.float(),
            "thresholds1": threshold1.float(),
            "dataset_name": "willow",
            "dataset_label": "S",
            "image_path0": img_path0,
            "image_path1": img_path1,
        }

        if self.category_embedding is not None:
            prompt_embed0 = self.category_embedding[cat0]
            prompt_embed1 = self.category_embedding[cat1]
            data = {
                "prompt_embed0": torch.from_numpy(prompt_embed0).float(),
                "prompt_embed1": torch.from_numpy(prompt_embed1).float(),
            }

        if self.mode == "test":
            data["original_keypoints0"] = torch.from_numpy(kpts0)
            data["original_keypoints1"] = torch.from_numpy(kpts1)
            data["original_size0"] = torch.from_numpy(org_size0)
            data["original_size1"] = torch.from_numpy(org_size1)

        if img0_flip_tensor is not None:
            data["image0_flip"] = img0_flip_tensor
            data["image1_flip"] = img1_flip_tensor

        return data

    def load_data(self):
        data = pd.read_csv(f"{self.dataset_path}/test_pairs.csv")
        cls = [
            "car(G)",
            "car(M)",
            "car(S)",
            "duck(S)",
            "motorbike(G)",
            "motorbike(M)",
            "motorbike(S)",
            "winebottle(M)",
            "winebottle(wC)",
            "winebottle(woC)",
        ]

        src_imnames = np.array(data.iloc[:, 0])
        trg_imnames = np.array(data.iloc[:, 1])
        src_kps = data.iloc[:, 2:22].values
        trg_kps = data.iloc[:, 22:].values
        # print('src: ', src_imnames.shape, trg_imnames.shape, src_kps.shape, trg_kps.shape)

        files = []
        cats = []
        kps = []
        for i in range(src_imnames.shape[0]):
            src_name = src_imnames[i]  # PF-dataset/car(G)/Cars_006a.png
            trg_name = trg_imnames[i]
            files.append(osp.join(self.dataset_path, src_name))
            files.append(osp.join(self.dataset_path, trg_name))

            cat = src_name.split("/")[1].split("(")[0]
            cats.append(cat)
            cats.append(cat)

            src_xy = np.hstack(
                [src_kps[i, :10].reshape(-1, 1), src_kps[i, 10:].reshape(-1, 1)]
            )
            trg_xy = np.hstack(
                [trg_kps[i, :10].reshape(-1, 1), trg_kps[i, 10:].reshape(-1, 1)]
            )
            src_z = np.zeros(shape=(10, 1))
            src_z[: src_xy.shape[0]] = 1
            trg_z = np.zeros(shape=(10, 1))
            trg_z[: trg_xy.shape[0]] = 1

            # print('src_xy: ', src_xy.shape)
            src_kp_i = np.hstack([src_xy, src_z])
            trg_kp_i = np.hstack([trg_xy, trg_z])

            kps.append(src_kp_i)
            kps.append(trg_kp_i)

            # print('img: ', i, src_imnames[i], trg_imnames[i], cat)
            # print('kp: ', i, src_kps[i].shape, trg_kps[i].shape, src_kps[i])
            # exit(0)

        return files, kps, cats, None, None

    def load_data_(self):
        categories = [
            "car(G)",
            "car(M)",
            "car(S)",
            "duck(S)",
            "motorbike(G)",
            "motorbike(M)",
            "motorbike(S)",
            "winebottle(M)",
            "winebottle(wC)",
            "winebottle(woC)",
        ]
        files, kps, cats, used_points_set, all_thresholds = ([] for _ in range(5))
        for cat_idx, cat in tqdm(
                enumerate(categories), total=len(categories), desc="Processing PF_Willow"
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

            # if cat_idx >= 1:
            #     break

        kps = np.concatenate(kps, axis=0)
        # print("files: ", len(files), kps.shape, len(cats))
        return files, kps, cats, used_points_set, None

    def load_category(self, path, category, split, subsample=None):
        def get_points(point_coords_list, idx):
            X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
            Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
            Xpad = -np.ones(20)
            Xpad[: len(X)] = X
            Ypad = -np.ones(20)
            Ypad[: len(X)] = Y
            Zmask = np.zeros(20)
            Zmask[: len(X)] = 1
            point_coords = np.concatenate(
                (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1, 20)), axis=0
            )
            # make arrays float tensor for subsequent processing
            point_coords = torch.Tensor(point_coords.astype(np.float32))
            return point_coords

        np.random.seed(42)
        files = []
        kps = []
        test_data = pd.read_csv(f"{path}/{split}_pairs.csv")
        cls = [
            "car(G)",
            "car(M)",
            "car(S)",
            "duck(S)",
            "motorbike(G)",
            "motorbike(M)",
            "motorbike(S)",
            "winebottle(M)",
            "winebottle(wC)",
            "winebottle(woC)",
        ]
        cls_ids = test_data.iloc[:, 2].values.astype("int") - 1
        cat_id = cls.index(category)
        subset_id = np.where(cls_ids == cat_id)[0]
        # logger.info(f'Number of Pairs for {category} = {len(subset_id)}')
        subset_pairs = test_data.iloc[subset_id, :]
        src_img_names = np.array(subset_pairs.iloc[:, 0])
        trg_img_names = np.array(subset_pairs.iloc[:, 1])
        print(src_img_names.shape, trg_img_names.shape)
        if not split.startswith("train"):
            point_A_coords = subset_pairs.iloc[:, 3:5]
            point_B_coords = subset_pairs.iloc[:, 5:]
        # print(point_A_coords.shape, point_B_coords.shape)
        for i in range(len(src_img_names)):
            src_fn = f"{path}/../{src_img_names[i]}"
            trg_fn = f"{path}/../{trg_img_names[i]}"
            # src_size = Image.open(src_fn).size
            # trg_size = Image.open(trg_fn).size

            if not split.startswith("train"):
                source_kps = get_points(point_A_coords, i).transpose(1, 0)
                target_kps = get_points(point_B_coords, i).transpose(1, 0)
            else:
                src_anns = (
                        os.path.join(
                            path, "Annotations", category, os.path.basename(src_fn)
                        )[:-4]
                        + ".mat"
                )
                trg_anns = (
                        os.path.join(
                            path, "Annotations", category, os.path.basename(trg_fn)
                        )[:-4]
                        + ".mat"
                )
                source_kps = process_kps_pascal(read_mat(src_anns, "kps"))
                target_kps = process_kps_pascal(read_mat(trg_anns, "kps"))

            # print(src_size)
            # source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
            # target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
            kps.append(source_kps)
            kps.append(target_kps)
            files.append(src_fn)
            files.append(trg_fn)

        # kps = torch.stack(kps)
        # used_kps, = torch.where(kps[:, :, 2].any(dim=0))
        kps = np.array(kps)
        (used_kps,) = np.where(kps[:, :, 2].any(axis=0))

        # kps = kps[:, used_kps, :]
        # logger.info(f'Final number of used key points: {kps.size(1)}')
        return files, kps, None, used_kps
