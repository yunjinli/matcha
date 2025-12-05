import torch
from torch.utils.data import Dataset
import os.path as osp
import numpy as np

from matcha.utils.image import read_image_as_tensor


class Scannet1500Dataset(Dataset):
    """
    Scannet dataset for evaluation
    """

    def __init__(self,
                 dataset_path: str,
                 scene_info_path: str,
                 img_resize: tuple[int] = None,
                 scale_factor: int = 1,
                 **kwargs,
                 ):
        """
        :param dataset_path: path to the scannet dataset
        :param scene_info_path: path to the scene_info dataset
        :param img_resize: target image size (int, int)
        :param scale_factor: target image with dividable factor
        """
        super().__init__()
        self.dataset_name = "Scannet1500"
        self.dataset_path = dataset_path
        self.img_resize = img_resize
        self.scale_factor = scale_factor

        tmp = np.load(osp.join(dataset_path, scene_info_path))
        pairs, rel_pose = tmp["name"], tmp["rel_pose"]
        self.pairs = pairs
        self.rel_pose = rel_pose

    def __repr__(self):
        desc = "Scannet1500Dataset ( \n"
        desc += f" dataset_path: {self.dataset_path}, \n"
        desc += f" resize={self.img_resize}, \n"
        desc += f" scale_factor={self.scale_factor}, \n"
        desc += ")"
        return desc

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        scene = self.pairs[idx]
        scene_name = f"scene0{scene[0]}_00"

        img_path0 = osp.join(
            self.dataset_path,
            "test",
            scene_name,
            "color",
            f"{scene[2]}.jpg",
        )

        img_path1 = osp.join(
            self.dataset_path,
            "test",
            scene_name,
            "color",
            f"{scene[3]}.jpg",
        )

        img0, scale0, org_size0 = read_image_as_tensor(path=img_path0, resize=self.img_resize)
        img1, scale1, org_size1 = read_image_as_tensor(path=img_path1, resize=self.img_resize)
        T_gt = self.rel_pose[idx].reshape(3, 4)
        R, t = T_gt[:3, :3], T_gt[:3, 3]
        K = np.stack(
            [
                np.array([float(i) for i in r.split()])
                for r in open(osp.join(
                self.dataset_path, "test", scene_name, "intrinsic",
                "intrinsic_color.txt"), "r", ).read().split("\n") if r
            ]
        )

        # same scale and scale resolution
        scaled_K = torch.from_numpy(K.copy())
        scaled_K[0, 0] /= scale0[0]
        scaled_K[1, 1] /= scale0[1]
        scaled_K[0, 2] /= scale0[0]
        scaled_K[1, 2] /= scale0[1]

        data = {
            "image0": img0,
            "image1": img1,
            "K0": scaled_K,
            "K1": scaled_K,

            "T": torch.from_numpy(T_gt),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),

            "camera_model0": "PINHOLE",
            "camera_model1": "PINHOLE",
            "original_K0": torch.from_numpy(K),
            "original_K1": torch.from_numpy(K),
            "original_width0": org_size0[0],
            "original_height0": org_size0[1],
            "original_width1": org_size1[0],
            "original_height1": org_size1[1],
            "original_params0": torch.from_numpy(np.array(K[[0, 1, 0, 1], [0, 1, 2, 2]])),
            "original_params1": torch.from_numpy(np.array(K[[0, 1, 0, 1], [0, 1, 2, 2]])),

            "dataset_name": self.dataset_name,
            "image_path0": img_path0,
            "image_path1": img_path1,
            "scene_name": scene_name,
        }

        return data
