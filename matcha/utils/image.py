import torch
import torch.nn.functional as F
import numpy as np
import cv2

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]
    ), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[: inp.shape[0], : inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[: inp.shape[0], : inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, : inp.shape[1], : inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, : inp.shape[1], : inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def fix_path_from_d2net(path):
    if not path:
        return None

    path = path.replace("Undistorted_SfM/", "")
    path = path.replace("images", "dense0/imgs")
    path = path.replace("phoenix/S6/zl548/MegaDepth_v1/", "")

    return path


def read_image_as_tensor(
        path: str,
        resize: tuple = None,
        scale_factor: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    org_w, org_h = image.shape[1], image.shape[0]

    if resize is not None:
        new_w, new_h = resize[0], resize[1]

    elif scale_factor > 0:
        if org_w % scale_factor == 0:
            new_w = org_w
        else:
            new_w = (org_w // scale_factor + 1) * scale_factor

        if org_h % scale_factor == 0:
            new_h = org_h
        else:
            new_h = (org_h // scale_factor + 1) * scale_factor
    else:
        new_w = org_w
        new_h = org_h

    img_tensor = torch.from_numpy(image.astype(float) / 255.0).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    if new_w != org_w or new_h != org_h:
        img_tensor = F.interpolate(
            img_tensor[None], size=(new_h, new_w), mode="bilinear"
        )[0]
    scale = torch.tensor([org_w / new_w, org_h / new_h], dtype=torch.float)
    # print("img_tensor: ", img_tensor.shape, image.shape)

    return img_tensor, scale, torch.tensor([org_w, org_h], dtype=torch.long)