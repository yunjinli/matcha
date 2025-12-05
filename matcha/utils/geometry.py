import numpy as np
import poselib


def rescale_keypoints(kpts: np.ndarray, w: int, h: int, wt: int, ht: int):
    """
    Rescale keypoints from original image size to target image size.
    Args:
        kpts (np.ndarray): Keypoints in the format (N, 2) where N is the number of keypoints.
        w (int): Original image width.
        h (int): Original image height.
        wt (int): Target image width.
        ht (int): Target image height.
    Returns:
        np.ndarray: Rescaled keypoints in the same format (N, 2).
    """
    scaled_kpts = kpts * np.array([(wt / w), (ht / h)])
    return scaled_kpts


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def normalize_intrinsic(x, K):
    return (x - K[:2, 2]) / np.diag(K)[:2]


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])
    M = np.stack(
        [
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ],
        axis=1,
    )
    return M


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def compute_E_from_Rt(R, t):
    # relative_extrinsic = np.matmul(np.linalg.inv(ex2), ex1)
    # dR, dt = relative_extrinsic[:3, :3], relative_extrinsic[:3, 3]
    # dt /= np.sqrt(np.sum(dt ** 2))

    norm_t = t / np.sqrt(np.sum(t ** 2))
    e = np.reshape(
        np.matmul(
            np.reshape(
                np_skew_symmetric(norm_t.astype("float64").reshape(1, 3)), (3, 3)
            ),
            np.reshape(R.astype("float64"), (3, 3)),
        ),
        (3, 3),
    )
    e = e / np.linalg.norm(e)
    return e


def compute_epi_inlier(x1, x2, E, inlier_th, return_error=False):
    num_pts1, num_pts2 = x1.shape[0], x2.shape[0]
    x1_h = np.concatenate([x1, np.ones([num_pts1, 1])], -1)
    x2_h = np.concatenate([x2, np.ones([num_pts2, 1])], -1)
    ep_line1 = x1_h @ E.T
    ep_line2 = x2_h @ E
    norm_factor = (
                          1 / np.sqrt((ep_line1[:, :2] ** 2).sum(1))
                          + 1 / np.sqrt((ep_line2[:, :2] ** 2).sum(1))
                  ) / 2
    dis = abs((ep_line1 * x2_h).sum(-1)) * norm_factor
    inlier_mask = dis < inlier_th

    if return_error:
        return inlier_mask, dis
    else:
        return inlier_mask


def compute_geo_inlier(kpts0, kpts1, K0, K1, R, t, inlier_threshold=0.001):
    norm_pts0 = normalize_intrinsic(x=kpts0[:, :2], K=K0)
    norm_pts1 = normalize_intrinsic(x=kpts1[:, :2], K=K1)
    norm_t = t / np.sqrt(np.sum(t ** 2))
    norm_E = np.reshape(
        np.matmul(
            np.reshape(
                np_skew_symmetric(norm_t.astype("float64").reshape(1, 3)), (3, 3)
            ),
            np.reshape(R.astype("float64"), (3, 3)),
        ),
        (3, 3),
    )

    epi_mask = compute_epi_inlier(
        x1=norm_pts0, x2=norm_pts1, E=norm_E, inlier_th=inlier_threshold
    )
    return epi_mask


def estimate_pose_poselib_general(
        kpts0,
        kpts1,
        camera0,
        camera1,
        threshold=1,
):
    if len(kpts0) < 5:
        return None

    relpose, res = poselib.estimate_relative_pose(
        kpts0,
        kpts1,
        camera0,
        camera1,
        ransac_opt={
            "max_reproj_error": threshold,
            "max_epipolar_error": 1.0,
            "min_inliers": 8,
            "max_iterations": 10_000,
        },
    )
    Rt_est = relpose.Rt
    R_est, t_est = Rt_est[:3, :3], Rt_est[:3, 3:]
    mask = np.array(res["inliers"]).astype(bool)
    return R_est, t_est, mask
