from os import path as osp
from unicodedata import name

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2
from numpy.linalg import inv


class ScannetDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            npz_path: str,
            intrinsic_path,
            mode='train',
            min_overlap_score=0.4,
            img_resize=(800, 608),
            pose_dir=None,
            subsample=None,
            seed=777,
            **kwargs):
        super().__init__()
        self.dataset_name = "scannet"
        self.dataset_path = dataset_path
        self.pose_dir = pose_dir if pose_dir is not None else dataset_path
        self.mode = mode
        self.im_resize = img_resize

        # prepare data_names, intrinsics and extrinsics (T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]

        if subsample is not None:
            np.random.seed(seed)
            if len(self.data_names) > subsample:
                data_names = [
                    self.data_names[ix] for ix in np.random.choice(len(self.data_names), subsample)
                ]  # overlap may exist
                self.data_names = data_names

        self.intrinsics = dict(np.load(intrinsic_path))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.dataset_path, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.dataset_path, scene_name, 'color', f'{stem_name_1}.jpg')

        image0, scale0 = self._read_image(img_name0, resize=self.im_resize)
        image1, scale1 = self._read_image(img_name1, resize=self.im_resize)

        # read the depth map which is stored as (480, 640)
        if self.mode in ['train', 'val']:
            depth0 = self._read_depth(osp.join(self.dataset_path, scene_name, 'depth', f'{stem_name_0}.png'))
            depth1 = self._read_depth(osp.join(self.dataset_path, scene_name, 'depth', f'{stem_name_1}.png'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics
        K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'image1': image1,
            'depth1': depth1,
            "scale0": scale0,  # [scale_w, scale_h]
            "scale1": scale1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': self.dataset_name,
            "dataset_label": "G",  # Geometric dataset for balanced sampling

            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))
        }

        return data

    def _read_abs_pose(self, scene_name, name):
        path = osp.join(self.pose_dir,
                        scene_name,
                        'pose', f'{name}.txt')
        cam2world = np.loadtxt(path, delimiter=" ")
        world2cam = inv(cam2world)
        return world2cam

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)

        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def _read_depth(self, path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        depth = depth / 1000
        depth = torch.from_numpy(depth).float()  # (h, w)
        return depth

    def _read_image(self, path, resize):
        image = cv2.imread(str(path), 1)
        image = cv2.resize(image, (640, 480))  # align with depth

        # resize image
        w, h = image.shape[1], image.shape[0]
        scale = torch.tensor([1.0, 1.0], dtype=torch.float)

        if len(resize) == 2:
            image = cv2.resize(image, resize)
            scale = torch.tensor([w / resize[0], h / resize[1]], dtype=torch.float)
        # (h, w) -> (1, h, w) and normalized
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        return image, scale
