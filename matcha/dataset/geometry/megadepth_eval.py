import os
import os.path as osp

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import logging

from matcha.utils.image import read_image_as_tensor
from matcha.utils.geometry import compute_relative_pose


class Megadepth1500Dataset(Dataset):
    """
    Megadepth evaluation dataset for one scene
    """

    def __init__(self,
                 dataset_path: str,
                 scene_info_path: str,
                 img_resize: tuple[int] = None,
                 scale_factor: int = 1,
                 **kwargs,
                 ):
        """
        :param dataset_path: root to the dataset
        :param scene_info_path: path of one specific scene
        :param img_resize: image size
        :param scale_factor: make image size dividable by scale_factor
        """
        super().__init__()
        self.dataset_name = "Megadepth1500"

        self.dataset_path = dataset_path
        self.img_resize = img_resize
        self.scale_factor = scale_factor

        scene = np.load(osp.join(self.dataset_path, scene_info_path), allow_pickle=True)
        self.pairs = scene["pair_infos"]
        self.intrinsics = scene["intrinsics"]
        self.poses = scene["poses"]
        self.im_paths = scene["image_paths"]
        self.scene_name = scene_info_path.split('/')[-1].split('_')[0]

    def __repr__(self):
        desc = "Megadepth1500Dataset ( \n"
        desc += f" scene: {self.scene_name}, \n"
        desc += f" dataset_path: {self.dataset_path}, \n"
        desc += f" resize={self.img_resize}, \n"
        desc += f" scale_factor={self.scale_factor}, \n"
        desc += ")"
        return desc

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx][0]
        K0 = self.intrinsics[idx0].copy()
        T0 = self.poses[idx0].copy()
        R0, t0 = T0[:3, :3], T0[:3, 3]
        K1 = self.intrinsics[idx1].copy()
        T1 = self.poses[idx1].copy()
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R, t = compute_relative_pose(R0, t0, R1, t1)
        T0_to_1 = np.concatenate((R, t[:, None]), axis=-1)
        img_path0 = osp.join(self.dataset_path, self.im_paths[idx0])
        img_path1 = osp.join(self.dataset_path, self.im_paths[idx1])

        img0, scale0, org_size0 = read_image_as_tensor(
            path=img_path0, resize=self.img_resize,
            scale_factor=self.scale_factor
        )
        img1, scale1, org_size1 = read_image_as_tensor(
            path=img_path1, resize=self.img_resize,
            scale_factor=self.scale_factor
        )
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

            "camera_model0": "PINHOLE",
            "camera_model1": "PINHOLE",
            "original_K0": torch.from_numpy(K0),
            "original_K1": torch.from_numpy(K1),
            "original_width0": org_size0[0],
            "original_height0": org_size0[1],
            "original_width1": org_size1[0],
            "original_height1": org_size1[1],
            "original_params0": torch.from_numpy(np.array(K0[[0, 1, 0, 1], [0, 1, 2, 2]])),
            "original_params1": torch.from_numpy(np.array(K1[[0, 1, 0, 1], [0, 1, 2, 2]])),

            "dataset_name": self.dataset_name,
            "image_path0": img_path0,
            "image_path1": img_path1,
            "scene_name": self.scene_name,
        }

        return data
