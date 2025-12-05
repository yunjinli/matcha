import scipy.io as sio
import numpy as np


def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


def process_kps_pascal(kps):
    # Step 1: Reshape the array to (20, 2) by adding nan values
    num_pad_rows = 20 - kps.shape[0]
    if num_pad_rows > 0:
        pad_values = np.full((num_pad_rows, 2), np.nan)
        kps = np.vstack((kps, pad_values))

    # Step 2: Reshape the array to (20, 3)
    # Add an extra column: set to 1 if the row does not contain nan, 0 otherwise
    last_col = np.isnan(kps).any(axis=1)
    last_col = np.where(last_col, 0, 1)
    kps = np.column_stack((kps, last_col))

    # Step 3: Replace rows with nan values to all 0's
    mask = np.isnan(kps).any(axis=1)
    kps[mask] = 0

    return kps
    # return torch.tensor(kps).float()


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale
