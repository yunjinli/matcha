import cv2
import torch
import numpy as np
from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.benchmark.visualization import plot_image_matches
from matcha.utils.device import to_numpy


def run_match_pairs(
        img_path0: str,
        img_path1: str,
        src_kpts: np.ndarray,
        matcher: BaseMatcher,
        save_fig: str | None = None,
) -> None:
    """
    Run matching pairs on two images and visualize the matches.

    Args:
        img_path0 (str): Path to the first image.
        img_path1 (str): Path to the second image.
        src_kpts (np.ndarray): Source keypoints (Nx2).
        matcher (BaseMatcher): Matcher object.
        save_fig (str, optional): Path to save the image with matches. Defaults to None.
    """
    # Prepare model input
    img0_out = matcher.model.load_image(img_path0)
    img0_tensor = img0_out["image_tensor"]
    nh, nw = img0_tensor.shape[1:]
    img0 = img0_out["image"]
    h, w = img0.shape[:2]
    if nh != h or nw != w:
        src_kpts[:, 0] *= (nw / w)
        src_kpts[:, 1] *= (nh / h)
    img1_out = matcher.model.load_image(img_path1)

    device = matcher.device

    data0 = {
        "image": img0_out["image_tensor"].to(device)[None],
        "keypoints": torch.from_numpy(src_kpts).float().to(device)[None],
    }

    data1 = {
        "image": img1_out["image_tensor"].to(device)[None],
    }

    # Inference
    output = matcher(data0=data0, data1=data1, with_keypoint_detection=False)
    src_kps = to_numpy(output["keypoints0"])
    trg_kps = to_numpy(output["keypoints1"])
    org_matches = to_numpy(output.get("matches"))
    matched_ids0 = np.where(org_matches >= 0)[0]
    matched_ids1 = org_matches[org_matches >= 0]
    matches = np.concatenate(
        [src_kps[matched_ids0], trg_kps[matched_ids1]], axis=-1
    )

    plot_image_matches(
        img0_out["image_tensor"],
        img1_out["image_tensor"],
        matches,
        save_fig=save_fig,
    )


def run_demo(
        img_path0: str,
        img_path1: str,
        src_kpts: np.ndarray,
        img_size: tuple[int, int] = (512, 512),
        pretrained_path: str = "weights/matcha_pretrained.pth",
        save_path: str | None = None,
) -> None:
    """
    Run semantic matching demo using MatchaFeature and BaseMatcher.
    Args:
        img_path0 (str): Path to the first image.
        img_path1 (str): Path to the second image.
        src_kpts (np.ndarray): Source keypoints (N, 2) in the format (x, y).
        img_size (tuple[int, int]): Image size for model input.
        pretrained_path (str): Path to the pretrained model weights.
        save_path (str | None): Path to save the output image.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    config = {"keypoint_method": None, "image_size": img_size}
    model = MatchaFeature(config=config)
    model.load_state_dict(torch.load(pretrained_path), strict=False)

    matcher = BaseMatcher(model, device)

    # Match pair
    run_match_pairs(img_path0, img_path1, src_kpts=src_kpts, matcher=matcher, save_fig=save_path)


if __name__ == "__main__":
    src_kpts = np.array([[21, 307],
                         [511, 246],
                         [121, 243],
                         [325, 308],
                         [81, 344]], dtype=float)
    run_demo(
        img_path0="assets/examples/pascal_areoplane_2011_001407.png",
        img_path1="assets/examples/pascal_areoplane_2010_004184.png",
        src_kpts=src_kpts,
        save_path="outputs/semantic_demo/pascal_areoplane.png",
    )
