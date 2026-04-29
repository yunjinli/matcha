import torch
import numpy as np

from matcha.third_party.geoaware_sc.flow_matching import get_flow


def compute_semantic_matches(
        src_ft: torch.Tensor,
        trg_ft: torch.Tensor,
        src_kps: torch.Tensor,
        soft_eval: bool = False,
        soft_eval_window: int = 7,
) -> torch.Tensor:
    """
    Compute semantic matches between two feature maps using a similarity matrix.
    Args:
        src_ft (torch.Tensor): Feature map of the source image.
        trg_ft (torch.Tensor): Feature map of the target image.
        src_kps (torch.Tensor): Keypoints in the source image.
        soft_eval (bool): If True, use a flow-based nearest neighbor matching.
        soft_eval_window (int): Window size for flow-based nearest neighbor matching.
    Returns:
        torch.Tensor: Semantic matches between the source and target images.
    """
    C, H, W = src_ft.shape
    # Calculate similarity matrix
    # sim_1_to_2 = torch.matmul(img1_desc, img2_desc.permute(0, 2, 1))[
    #     0
    # ]  # [3600, 3600]
    with torch.no_grad():
        sim_1to2 = torch.matmul(src_ft.reshape(C, H * W).T, trg_ft.reshape(C, -1))  # [H*W, H*W]
    if soft_eval:
        flow = get_flow(sim_1to2, soft_eval_window, num_patches=H)[0]  # (H, W, 2)
        flow_idxed = flow[src_kps[:, 1].long(), src_kps[:, 0].long()]
        nn_y, nn_x = (
            flow_idxed[:, 1].clamp(0, H - 1),
            flow_idxed[:, 0].clamp(0, W - 1),
        )
    else:
        # Find nearest neighbors if soft evaluation is not enabled
        sim_1to2_idxed = sim_1to2.reshape(H, W, -1)[
            src_kps[:, 1], src_kps[:, 0]
        ]  # [N, H, W]
        _, nn_1_to_2 = torch.max(sim_1to2_idxed, dim=-1)
        nn_y, nn_x = nn_1_to_2 // W, nn_1_to_2 % W

    # Stack the transformed keypoints
    kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)

    return kps_1_to_2


def compute_semantic_matches_onebyone(
        src_ft: torch.Tensor,
        trg_ft: torch.Tensor,
        src_kps: torch.Tensor,
):
    num_channel, h, w = trg_ft.shape

    pred_trg_kps = []
    for idx in range(src_kps.shape[0]):
        src_point = src_kps[idx]
        src_vec = src_ft[:, src_point[1], src_point[0]].view(-1, 1)  # C, 1
        trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1)  # HW, C
        cos_map = (trg_vec @ src_vec).view(h, w).cpu().numpy()  # H, W

        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
        pred_trg_kps.append([max_yx[1], max_yx[0]])

    return torch.from_numpy(np.array(pred_trg_kps)).to(src_ft.device)
