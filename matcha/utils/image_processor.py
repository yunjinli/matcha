import torch
import torch.nn.functional as F
import cv2


class ImageProcessor:
    def __init__(
            self,
            image_size: tuple[int] = None,
            scale_factor: int = 1,
            gray_scale: bool = False,
            max_length: int = None
    ):
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.gray_scale = gray_scale
        self.max_length = max_length

    def to_tensor(self, image):
        img_tensor = torch.from_numpy(image.astype(float) / 255.0).float()
        if self.gray_scale:
            return img_tensor.unsqueeze(0)
        else:
            return img_tensor.permute(2, 0, 1)  # [H, W, 3] - > [3, H, W]

    def load(self, img_path):
        if self.gray_scale:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR->RGB

        org_h = img.shape[0]
        org_w = img.shape[1]

        if self.image_size is not None:
            new_w = self.image_size[0]
            new_h = self.image_size[1]
        elif self.max_length is not None:
            if org_w > org_h:
                new_w = self.max_length
                new_h = int((new_w / org_w) * org_h)
            else:
                new_h = self.max_length
                new_w = int((new_h / org_h) * org_w)
            if self.scale_factor > 0:
                if new_w % self.scale_factor > 0:
                    new_w = (new_w // self.scale_factor + 1) * self.scale_factor
                if new_h % self.scale_factor > 0:
                    new_h = (new_h // self.scale_factor + 1) * self.scale_factor
        else:
            if self.scale_factor > 0:
                if org_w % self.scale_factor == 0:
                    new_w = org_w
                else:
                    new_w = (org_w // self.scale_factor + 1) * self.scale_factor

                if org_h % self.scale_factor == 0:
                    new_h = org_h
                else:
                    new_h = (org_h // self.scale_factor + 1) * self.scale_factor
            else:
                new_w = org_w
                new_h = org_h

        img_tensor = self.to_tensor(image=img)
        if new_w != org_w or new_h != org_h:
            img_tensor = F.interpolate(
                img_tensor[None], size=(new_h, new_w), mode="bilinear"
            )[0]

        return {
            "image": img,
            "image_tensor": img_tensor,
        }
