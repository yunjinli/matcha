# Code from: https://github.com/Junyi42/GeoAware-SC/blob/master/utils/utils_correspondence.py

import torch
import numpy as np


def softmax_with_temperature(x, beta, d=1):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M  # subtract maximum value for stability
    exp_x = torch.exp(x / beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum


def soft_argmax(corr, beta=0.02):
    r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    # input shape : (B, H_t * W_t, H_s , W_s) e.g., (B, 32*32, 32, 32)
    b, htwt, h, w = corr.size()
    ht, wt = int(np.sqrt(htwt)), int(np.sqrt(htwt))
    x_normal = np.linspace(-1, 1, w)
    x_normal = torch.tensor(x_normal, device=corr.device).float()
    y_normal = np.linspace(-1, 1, h)
    y_normal = torch.tensor(y_normal, device=corr.device).float()

    corr = softmax_with_temperature(corr, beta=beta, d=1)  # (B, H_t * W_t, H_s , W_s)
    corr = corr.view(-1, ht, wt, h, w)  # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
    x_normal = x_normal.expand(b, w)
    x_normal = x_normal.view(b, w, 1, 1)
    grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

    grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
    y_normal = y_normal.expand(b, h)
    y_normal = y_normal.view(b, h, 1, 1)
    grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
    return grid_x, grid_y


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (
            (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
    )  # unormalise
    mapping[:, 1, :, :] = (
            (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
    )  # unormalise

    # xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    # yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    # grid = torch.cat((xx,yy),1).float()

    # if mapping.is_cuda:
    #     grid = grid.cuda()
    # mapping = mapping - grid
    flow = mapping
    return flow


def apply_gaussian_kernel(corr, sigma=5):
    b, hw, h, w = corr.size()  # b, h_t*w_t, h_s, w_s

    idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
    idx_y = (idx // w).view(b, 1, 1, h, w).float()
    idx_x = (idx % w).view(b, 1, 1, h, w).float()
    x = np.linspace(0, 59, 60)
    x = torch.tensor(x, dtype=torch.float, requires_grad=False).to(corr.device)
    y = np.linspace(0, 59, 60)
    y = torch.tensor(y, dtype=torch.float, requires_grad=False).to(corr.device)
    x = x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)
    y = y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

    gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
    gauss_kernel = gauss_kernel.view(b, hw, h, w)

    return gauss_kernel * corr


def get_flow(corr, flow_window=0, num_patches=64):
    # corr: (H_s * W_s, H_t * W_t)
    hsws, htwt = corr.size()
    hs, ws = int(np.sqrt(hsws)), int(np.sqrt(hsws))
    ht, wt = int(np.sqrt(htwt)), int(np.sqrt(htwt))

    if flow_window > 0:  # zero out the corr_map outside the window
        # get the argmax
        max_index_flatten = torch.argmax(corr, dim=-1)
        max_index_x = max_index_flatten % num_patches  # (H_s * W_s, )
        max_index_y = max_index_flatten // num_patches  # (H_s * W_s, )
        corr = corr.view(-1, num_patches, num_patches)

        # Prepare offsets
        offset_range = torch.arange(-flow_window, flow_window + 1, device=corr.device)
        offset_x, offset_y = torch.meshgrid(offset_range, offset_range, indexing="ij")
        offset_x, offset_y = offset_x.flatten(), offset_y.flatten()

        # Compute window mask without loops
        window_positions_x = (max_index_x[:, None] + offset_x[None, :]).clamp(
            0, num_patches - 1
        )
        window_positions_y = (max_index_y[:, None] + offset_y[None, :]).clamp(
            0, num_patches - 1
        )

        # Create indices for gathering values
        batch_indices = torch.arange(corr.shape[0], device=corr.device)[:, None]

        # Using advanced indexing to create the window mask
        window_mask = torch.zeros_like(corr)
        window_mask[batch_indices, window_positions_y, window_positions_x] = 1

        # Apply window mask
        corr = corr * window_mask
    elif flow_window < 0:  # kernel soft_argmax
        corr = corr.permute(1, 0).view(1, num_patches ** 2, num_patches, num_patches)
        corr = apply_gaussian_kernel(corr, sigma=-flow_window)
        corr = corr.view(num_patches ** 2, num_patches ** 2).permute(1, 0)
    x = corr.view(-1, ht, wt, hsws)
    grid_x, grid_y = soft_argmax(x.permute(0, 3, 1, 2))
    x = torch.cat((grid_x, grid_y), dim=1)
    x = unnormalise_and_convert_mapping_to_flow(x)  # (B, 2, H, W)
    # x = self.output_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return x.permute(0, 2, 3, 1)  # (B, H, W, 2)
