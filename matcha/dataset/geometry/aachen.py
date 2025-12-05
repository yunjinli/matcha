import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import logging

from matcha.utils.image import read_image_as_tensor
from matcha.utils.geometry import qvec2rotmat, compute_relative_pose


class Aachen1500Dataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            img_resize: tuple[int] = None,
            scale_factor: int = 1,
            **kwargs,
    ):
        """
        :param dataset_path: path to aachen dataset
        :param img_resize: target image size (int, int) or None
        :param scale_factor: target image with dividable factor
        :param kwargs:
        """
        super().__init__()
        self.dataset_name = "Aachen1500"

        self.dataset_path = dataset_path
        self.img_resize = img_resize
        self.scale_factor = scale_factor

        # Load pairs for evaluation
        pair_path = osp.join(dataset_path, "aachen_test_1500_pairs.txt")
        self.pairs = []
        with open(pair_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                self.pairs.append((l[0], l[1]))

        logging.info(f"Load {len(self.pairs)} pairs from {self.dataset_path} of {self.dataset_name}.")

        # Load intrinsics
        intrinsic_path = osp.join(self.dataset_path, "database_intrinsics.txt")
        self.intrinsics = {}
        with open(intrinsic_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                self.intrinsics[l[0]] = {
                    "model": l[1],
                    "width": int(l[2]),
                    "height": int(l[3]),
                    "params": np.array([float(v) for v in l[4:]], dtype=float),
                }

        # Load camera poses
        camera_pose_path = osp.join(self.dataset_path, "db_poses.txt")
        self.camera_pose = {}
        with open(camera_pose_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split()
                qvec = np.array([float(v) for v in l[1:5]], dtype=float)
                R = qvec2rotmat(qvec=qvec).reshape(3, 3)
                tvec = np.array([float(v) for v in l[5:]], dtype=float).reshape(3, 1)
                self.camera_pose[l[0]] = np.hstack([R, tvec])  # [3, 4]

    def __len__(self):
        return len(self.pairs)

    def __repr__(self):
        desc = "Aachen1500Dataset ( \n"
        desc += f" dataset_path: {self.dataset_path}, \n"
        desc += f" resize={self.img_resize}, \n"
        desc += f" scale_factor={self.scale_factor}, \n"
        desc += ")"
        return desc

    def _get_camera_info(self, name):
        camera = self.intrinsics[name]
        f, cx, cy, _ = camera["params"]
        K = np.array(
            [
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ],
            dtype=float,
        )
        return camera, K

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]

        # Compute GT relative pose
        T0 = self.camera_pose[name0]
        T1 = self.camera_pose[name1]
        R0, t0 = T0[:3, :3], T0[:3, 3]
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R, t = compute_relative_pose(R0, t0, R1, t1)
        T0_to_1 = np.concatenate((R, t[:, None]), axis=-1)

        # Load camera intrinsics
        camera0, K0 = self._get_camera_info(name0)
        camera1, K1 = self._get_camera_info(name1)

        # Load image pair
        img_path0 = osp.join(self.dataset_path, "images_upright", name0)
        img_path1 = osp.join(self.dataset_path, "images_upright", name1)

        img0, scale0, org_size0 = read_image_as_tensor(
            path=img_path0,
            resize=self.img_resize,
            scale_factor=self.scale_factor)
        img1, scale1, org_size1 = read_image_as_tensor(
            path=img_path1,
            resize=self.img_resize,
            scale_factor=self.scale_factor)
        # same scale and scale resolution
        scaled_K0 = torch.from_numpy(K0.copy())
        scaled_K0[0, 0] /= scale0[0]
        scaled_K0[1, 1] /= scale0[1]
        scaled_K0[0, 2] /= scale0[0]
        scaled_K0[1, 2] /= scale0[1]

        scaled_K1 = torch.from_numpy(K1.copy())
        scaled_K1[0, 0] /= scale1[0]
        scaled_K1[1, 1] /= scale1[1]
        scaled_K1[0, 2] /= scale1[0]
        scaled_K1[1, 2] /= scale1[1]

        data = {
            "image0": img0,
            "image1": img1,

            "K0": scaled_K0,
            "K1": scaled_K1,

            "T": torch.from_numpy(T0_to_1),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),

            "camera_model0": camera0["model"],
            "camera_model1": camera1["model"],
            "original_K0": torch.from_numpy(K0),
            "original_K1": torch.from_numpy(K1),
            "original_width0": camera0["width"],
            "original_height0": camera0["height"],
            "original_width1": camera1["width"],
            "original_height1": camera1["height"],
            "original_params0": torch.from_numpy(camera0["params"]),
            "original_params1": torch.from_numpy(camera1["params"]),

            "dataset_name": self.dataset_name,
            "image_path0": img_path0,
            "image_path1": img_path1,
        }

        return data

    def _read_image(self, path, resize=None):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        w, h = image.shape[1], image.shape[0]
        scale = torch.tensor([1.0, 1.0], dtype=torch.float)
        if len(resize) == 2:
            image = cv2.resize(image, resize)
            scale = torch.tensor([w / resize[0], h / resize[1]], dtype=torch.float)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        return image, scale, np.array([w, h])
