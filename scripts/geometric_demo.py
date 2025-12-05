import torch
import numpy as np

from matcha.feature.matcha_feature import MatchaFeature
from matcha.matcher.base_matcher import BaseMatcher
from matcha.utils.device import to_numpy
from matcha.benchmark.visualization import plot_image_matches
from matcha.utils.geometry import estimate_pose_poselib_general


def run_match_pairs(
        img_path0: str,
        img_path1: str,
        matcher: BaseMatcher,
        geo_verify: bool = False,
        save_fig: str | None = None,
) -> None:
    """
    Run matching pairs on two images and visualize the matches.

    Args:
        img_path0 (str): Path to the first image.
        img_path1 (str): Path to the second image.
        matcher (BaseMatcher): Matcher instance.
        geo_verify (bool, optional): Enable geometric verification with a fake camera model. Defaults to False.
        save_fig (str, optional): Path to save the visualization figure. Defaults to None.
    Returns:
        None: Visualize the matches and save the image if specified.
    """

    # Prepare model input
    img0_out = matcher.model.load_image(img_path0)
    img1_out = matcher.model.load_image(img_path1)
    device = matcher.device
    data0 = {
        "image": img0_out["image_tensor"].to(device)[None],

    }
    data1 = {
        "image": img1_out["image_tensor"].to(device)[None],
    }

    # Inference
    output = matcher(data0=data0, data1=data1)

    # Extract keypoint matches
    org_matches = to_numpy(output.get("matches"))
    keypoints0 = to_numpy(output["keypoints0"])
    keypoints1 = to_numpy(output["keypoints1"])
    matched_ids0 = np.where(org_matches >= 0)[0]
    matched_ids1 = org_matches[org_matches >= 0]
    matches = np.concatenate(
        [keypoints0[matched_ids0], keypoints1[matched_ids1]], axis=-1
    )

    # Geometric verification with fake camera model
    if geo_verify:
        kpts0 = matches[:, :2]
        kpts1 = matches[:, 2:]
        W, H = matcher.config.image_size
        camera = {
            "model": "SIMPLE_PINHOLE",
            "width": W,
            "height": H,
            "params": [1, H / 2, W / 2],
        }
        _, _, mask = estimate_pose_poselib_general(
            kpts0,
            kpts1,
            camera,
            camera,
            threshold=5.0,
        )
        if mask.sum() > 10:
            matches = matches[mask]
    print("Number of matches:", len(matches))

    # Plot images
    skip = max(len(matches) // 50, 1)
    plot_image_matches(
        img0_out["image_tensor"],
        img1_out["image_tensor"],
        matches[::skip],
        save_fig=save_fig,
    )


def run_demo(
        img_path0: str,
        img_path1: str,
        img_size: tuple[int, int] = (512, 512),
        geo_verify: bool = False,
        pretrained_path: str = "weights/matcha_pretrained.pth",
        save_path: str | None = None,
) -> None:
    """
    Run semantic matching demo using MatchaFeature and BaseMatcher.

    Args:
        img_path0 (str): Path to the first image.
        img_path1 (str): Path to the second image.
        img_size (tuple[int, int], optional): Image size. Defaults to (512, 512).
        geo_verify (bool, optional): Enable geometric verification with a fake camera model. Defaults to False.
        pretrained_path (str, optional): Path to the pretrained model weights. Defaults to "weights/matcha_pretrained.pth".
        save_path (str, optional): Path to save the visualization figure. Defaults to None.
    Returns:
        None: Visualize the matches and save the image if specified.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = MatchaFeature(config={"keypoint_method": "disk", "image_size": img_size})
    model.load_state_dict(torch.load(pretrained_path), strict=False)

    matcher = BaseMatcher(model, device)

    # Match pair
    run_match_pairs(img_path0, img_path1, matcher, geo_verify, save_fig=save_path)


if __name__ == "__main__":
    run_demo(
        img_path0="assets/examples/sacre_coeur_A.png",
        img_path1="assets/examples/sacre_coeur_B.png",
        save_path="outputs/geometric_demo/sacre_coeur_matches.png",
    )
