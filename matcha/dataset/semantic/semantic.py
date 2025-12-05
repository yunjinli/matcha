import logging

import cv2
import numpy as np
import torch
import torch.utils as utils


class SemanticDS(utils.data.Dataset):
    def __init__(
            self,
            dataset_path: str,
            mode="train",
            img_resize=(800, 608),
            subsample=None,
            bbox_threshold=True,
            max_keypoint_size=30,
            flip_image=False,
            category_embed_path=None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.img_resize = img_resize
        self.subsample = subsample
        self.bbox_threshold = bbox_threshold
        self.max_keypoint_size = max_keypoint_size
        self.flip_img = flip_image
        if category_embed_path is not None:
            self.category_embedding = dict(np.load(category_embed_path))
            logging.info(f"Loaded category embedding from {category_embed_path}")
        else:
            self.category_embedding = None

    def load_image(self, path, resize=None):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        w, h = image.shape[1], image.shape[0]
        scale = torch.tensor([1.0, 1.0], dtype=torch.float)
        if resize is not None and len(resize) == 2:
            image = cv2.resize(image, resize)
            scale = torch.tensor([w / resize[0], h / resize[1]], dtype=torch.float)

        if self.flip_img:
            image_flip = self.flip_left_right(image)
            image_flip = torch.from_numpy(image_flip).float().permute(2, 0, 1) / 255
        else:
            image_flip = None
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        return image, scale, image_flip, np.array([w, h])

    def load_data(self):
        pass

    def pad_keypoints(self, keypoints: np.ndarray, max_size: int = 30):
        """ """
        if keypoints.shape[0] == max_size:
            return keypoints
        else:
            new_keypoints = np.zeros(
                shape=(max_size, keypoints.shape[1]), dtype=keypoints.dtype
            )
            new_keypoints[: keypoints.shape[0]] = keypoints
            return new_keypoints

    def flip_left_right(self, img: np.ndarray):
        return cv2.flip(img, 1)
