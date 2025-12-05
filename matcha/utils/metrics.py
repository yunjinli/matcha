import numpy as np


class PoseMetrics:
    @staticmethod
    def angle_error_vec(v1, v2):
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

    @staticmethod
    def angle_error_mat(R1, R2):
        cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
        cos = np.clip(cos, -1.0, 1.0)
        return np.rad2deg(np.abs(np.arccos(cos)))

    @staticmethod
    def compute_relative_pose_error(R_gt, t_gt, R, t, t_from_essmat: bool = False):
        error_t = PoseMetrics.angle_error_vec(t.squeeze(), t_gt)
        if t_from_essmat:
            # For translations extracted from essential matrices
            error_t = np.minimum(error_t, 180 - error_t)
        error_R = PoseMetrics.angle_error_mat(R, R_gt)
        return error_t, error_R

    @staticmethod
    def pose_auc(errors, thresholds):
        sort_idx = np.argsort(errors)
        errors = np.array(errors)[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.trapezoid(r, x=e) / t)
        return aucs
