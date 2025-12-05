import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matcha.train.losses.semantic_loss_utils import permute_indices, flip_keypoints


class SemanticLoss:
    def __init__(
            self,
            logit_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07)),
            dense_obj=False,
            self_contrast_weight=0,
            adapt_flip_weight=0,
            augment_double_flip_weight=1.0,
            augment_self_flip_weight=0.25,
            loss_type="clip",
    ):
        self.logit_scale = logit_scale
        self.dense_obj = dense_obj
        self.flip_augmentation = (
                adapt_flip_weight > 0
                or augment_double_flip_weight > 0
                or augment_self_flip_weight > 0
        )
        self.adapt_flip_weight = adapt_flip_weight
        self.augment_double_flip_weight = augment_double_flip_weight
        self.augment_self_flip_weight = augment_self_flip_weight
        self.self_contrast_weight = self_contrast_weight
        self.use_clip_loss = loss_type.find("clip") >= 0
        self.use_triplet_loss = loss_type.find("tri") >= 0

    def __call__(self, data: dict, corr_model=None):
        kps0 = data["keypoints0"]
        desc0 = data["feat0"]
        desc0_flip = data["feat0_flip"]

        kps1 = data["keypoints1"]
        desc1 = data["feat1"]
        desc1_flip = data["feat1_flip"]
        threshold = data["threshold"]
        # print("threshold: ", threshold)

        raw_permute_list = data["permute_list"]
        vis = (kps0[:, 2] * kps1[:, 2]).bool()
        if torch.sum(vis) == 0:
            return None

        y0, x0 = kps0[vis, 1].long(), kps0[vis, 0].long()
        y1, x1 = kps1[vis, 1].long(), kps1[vis, 0].long()

        desc_patch0 = desc0[0, :, y0, x0].T  # (c, n)->(n,c)
        desc_patch1 = desc1[0, :, y1, x1].T  # (c, n)->(n,c)

        # print('desc_patch: ', desc_patch0.shape, desc_patch1.shape)

        loss = torch.zeros([], device=desc0.device)
        if self.use_clip_loss:
            loss = loss + self.cal_clip_loss(
                image_features=desc_patch0,
                texture_features=desc_patch1,
                logit_scale=self.logit_scale.exp(),
            )
        if self.use_triplet_loss:
            loss = loss + self.cal_triplet_loss(
                desc0=desc0, desc1=desc1, kps0=kps0, kps1=kps1
            )
        if corr_model is not None:
            epe_loss = self.cal_corr_map_loss(
                corr_model=corr_model,
                desc0=desc0[0],
                desc1=desc1[0],
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                threshold=threshold,
            )

            loss = loss + epe_loss

        # compute losses with flip augmentation
        if (
                self.flip_augmentation > 0
                and desc0_flip is not None
                and desc1_flip is not None
        ):
            loss = [loss]
            loss_weight = [1]
            if raw_permute_list is not None:
                permute_list = permute_indices(raw_permute_list)
                # print('permute_list: ', permute_list)
                kps0_flip = flip_keypoints(
                    keypoints=kps0, img_size=desc0.shape[-1], permute_list=permute_list
                )
                kps1_flip = flip_keypoints(
                    keypoints=kps1, img_size=desc1.shape[-1], permute_list=permute_list
                )
                kps0 = kps0[: len(permute_list), :]
                kps1 = kps1[: len(permute_list), :]
            else:
                kps0_flip = kps0
                kps1_flip = kps1
                desc0_flip = desc0
                desc1_flip = desc1

            # print('flip: ', kps0.shape, kps1.shape)

            # adapt_flip
            if self.adapt_flip_weight > 0:
                vis_flip = kps0_flip[:, 2] * kps1[:, 2] > 0
                if vis_flip.sum() > 0:
                    loss_flip = self.cal_patch_loss(
                        kps0=kps0_flip[vis_flip],
                        kps1=kps1[vis_flip],
                        desc0=desc0_flip,
                        desc1=desc1,
                    )
                    loss.append(loss_flip)
                    loss_weight.append(self.adapt_flip_weight)

            # print('after adapt loss')

            if self.augment_double_flip_weight > 0:
                vis_double_flip = kps0_flip[:, 2] * kps1_flip[:, 2] > 0  #
                if vis_double_flip.sum() > 0:
                    loss_double_flip = self.cal_patch_loss(
                        kps0=kps0_flip[vis_double_flip],
                        kps1=kps1_flip[vis_double_flip],
                        desc0=desc0_flip,
                        desc1=desc1_flip,
                    )
                    loss.append(loss_double_flip)
                    loss_weight.append(self.augment_double_flip_weight)

            if self.augment_self_flip_weight > 0:
                vis_self_flip = kps0_flip[:, 2] * kps0[:, 2] > 0
                if vis_self_flip.sum() > 0:
                    loss_self_flip = self.cal_patch_loss(
                        kps0=kps0_flip[vis_self_flip],
                        kps1=kps0[vis_self_flip],
                        desc0=desc0_flip,
                        desc1=desc0,
                    )
                    loss.append(loss_self_flip)
                    loss_weight.append(self.augment_self_flip_weight)

            # print('after self flip loss')

            # Aggregate losses
            loss = sum([l * w for l, w in zip(loss, loss_weight)]) / sum(loss_weight)

        if self.self_contrast_weight > 0:
            contrast_loss0 = self.cal_self_contrastive_loss(
                feat_map=desc0, instance_mask=None
            )
            contrast_loss1 = self.cal_self_contrastive_loss(
                feat_map=desc1, instance_mask=None
            )

            loss = (
                    loss
                    + (contrast_loss0 + contrast_loss1) * self.self_contrast_weight * 0.5
            )

        return loss

    def get_logits(self, image_features, text_features, logit_scale):
        # Compute base logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def cal_clip_loss(self, image_features, texture_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features=image_features,
            text_features=texture_features,
            logit_scale=logit_scale,
        )
        labels = torch.arange(
            logits_per_image.shape[0], device=device, dtype=torch.long
        )
        total_loss = (
                             F.cross_entropy(logits_per_image, labels)
                             + F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return total_loss

    def cal_self_contrastive_loss(self, feat_map, instance_mask=None):
        device = feat_map.device
        B, C, H, W = feat_map.size()
        if instance_mask is not None:
            # interpolate the mask to the size of the feature map
            instance_mask = (
                    F.interpolate(
                        instance_mask.cuda().unsqueeze(1).float(),
                        size=(H, W),
                        mode="bilinear",
                    )
                    > 0.5
            )
            # mask out the feature map
            feat_map = feat_map * instance_mask
            # make where all zeros to be 1
            feat_map = feat_map + (~instance_mask)
            # Define neighborhood for local loss (8-neighborhood)
        offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        local_loss = 0.0
        for i, j in offsets:
            # Shift feature map
            shifted_map = torch.roll(feat_map, shifts=(i, j), dims=(2, 3))
            # Compute the dot product
            dot_product = (feat_map * shifted_map).sum(
                dim=1
            )  # Sum along channel dimension
            # Only consider valid region (to avoid wrapping around difference)
            if i > 0:
                dot_product[:, :i, :] = 0
            if j > 0:
                dot_product[:, :, :j] = 0
            if i < 0:
                dot_product[:, i:, :] = 0
            if j < 0:
                dot_product[:, :, j:] = 0
            local_loss -= (
                dot_product.mean()
            )  # negative because we want to maximize the dot product for neighbors

        # For global loss, random sample non-neighbor pixels
        num_samples = H * W  # you can adjust this number based on your requirement
        idx_i = torch.randint(0, H, (num_samples,), device=device)
        idx_j = torch.randint(0, W, (num_samples,), device=device)
        idx_k = torch.randint(0, H, (num_samples,), device=device)
        idx_l = torch.randint(0, W, (num_samples,), device=device)

        # Ensure they are not neighbors
        mask = ((idx_k - idx_i).abs() > 1) | ((idx_l - idx_j).abs() > 1)
        if instance_mask is not None:
            mask = (
                    mask
                    & instance_mask[0, 0, idx_i, idx_j]
                    & instance_mask[0, 0, idx_k, idx_l]
            )
        idx_i, idx_j, idx_k, idx_l = idx_i[mask], idx_j[mask], idx_k[mask], idx_l[mask]
        global_loss = 0.0
        for i, j, k, l in zip(idx_i, idx_j, idx_k, idx_l):
            dot_product = (feat_map[:, :, i, j] * feat_map[:, :, k, l]).sum(dim=1)
            global_loss += (
                dot_product.mean()
            )  # positive because we want to minimize the dot product for non-neighbors
        # Combine local and global losses
        lambda_factor = 0.1  # this can be adjusted based on cross-validation
        loss = local_loss + lambda_factor * global_loss
        return loss

    def cal_patch_loss(self, kps0, kps1, desc0, desc1):
        vis = (kps0[:, 2] * kps1[:, 2]).bool()
        y0, x0 = kps0[vis, 1].long(), kps0[vis, 0].long()
        y1, x1 = kps1[vis, 1].long(), kps1[vis, 0].long()

        desc_patch0 = desc0[0, :, y0, x0].T
        desc_patch1 = desc1[0, :, y1, x1].T

        loss = 0
        if self.use_clip_loss:
            loss = loss + self.cal_clip_loss(
                image_features=desc_patch0,
                texture_features=desc_patch1,
                logit_scale=self.logit_scale,
            )
        if self.use_triplet_loss:
            loss = loss + self.cal_triplet_loss(
                desc0=desc0, desc1=desc1, kps0=kps0, kps1=kps1
            )
        return loss

    def cal_corr_map_loss(
            self,
            corr_model,
            desc0,
            desc1,
            x0,
            y0,
            x1,
            y1,
            gaussian_augment=0.1,
            threshold=None,
    ):
        gt_flow = torch.stack(
            [x1 - x0, y1 - y0],
            dim=-1,
        ).to(x1.device)
        if gaussian_augment > 0:
            std = gaussian_augment * threshold / 2
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow = gt_flow + noise

        c, h, w = desc0.shape
        corr_map = torch.matmul(
            desc0.reshape(c, -1).T, desc1.reshape(c, -1)
        )  # [N x D] x [D x N] -> [N N]
        # print("corr_map1: ", corr_map.shape)
        corr_map = corr_map.reshape(h, w, h, w)
        corr_map = corr_model(corr_map[None])
        # print("corr_map2: ", corr_map.shape)

        predict_flow = corr_map[0, y0, x0, :]  # [n x 2]
        epe_loss = torch.norm(predict_flow - gt_flow, dim=-1).mean()
        return epe_loss

        # flow_idx = img1_patch_idx
        # flow_idx2 = img2_patch_idx
        # gt_flow = torch.stack([torch.tensor(img2_x_patch) - torch.tensor(img1_x_patch),
        #                        torch.tensor(img2_y_patch) - torch.tensor(img1_y_patch)], dim=-1).to(device)
        # if args.GAUSSIAN_AUGMENT > 0:
        #     std = args.GAUSSIAN_AUGMENT * img2_threshold / 2  # 2 sigma within the threshold
        #     noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
        #     gt_flow = gt_flow + noise
        # EPE_loss = get_corr_map_loss(img1_desc, img2_desc, corr_map_net, flow_idx, gt_flow, num_patches,
        #                              img2_patch_idx=flow_idx2)
        # loss += EPE_loss

    def cal_triplet_loss1(self, desc0, desc1, kps0, kps1, margin=1.0, radius=3):
        def find_nearest_neighbor_ids(feat0, feat1):
            with torch.no_grad():
                dist = feat0.detach() @ feat1.detach().T  # (n, c) @ (c, m) -> (n, m)
                _, ids = torch.topk(dist, dim=1, largest=True, k=1)
            return ids.reshape(-1)

        _, c, h, w = desc0.shape
        device = desc0.device

        all_ys, all_xs = torch.meshgrid(
            [torch.arange(h, device=device), torch.arange(w, device=device)],
            indexing="ij",
        )
        all_ys = all_ys.reshape(-1)
        all_xs = all_xs.reshape(-1)

        vis = (kps0[:, 2] * kps1[:, 2]).bool()
        y0, x0 = kps0[vis, 1].long(), kps0[vis, 0].long()
        y1, x1 = kps1[vis, 1].long(), kps1[vis, 0].long()

        desc_patch0 = desc0[0, :, y0, x0].T  # (n, c)
        desc_patch1 = desc1[0, :, y1, x1].T  # (n, c)
        pos_dist = 2 - 2 * (desc_patch0 * desc_patch1).sum(dim=1)  # (n, 1)

        ids1 = find_nearest_neighbor_ids(feat0=desc_patch0, feat1=desc1.view(c, -1).T)
        nn_y1 = ids1 // w  # (n, 1)
        nn_x1 = ids1 % w  # (n, 1)
        # print('nn_y1: ', nn_y1, nn_x1, h, w, nn_y1.shape, nn_x1.shape)

        nn_desc_path1 = desc1[0, :, nn_y1, nn_x1].T
        neg_dist01 = 2 - 2 * (desc_patch0 * nn_desc_path1).sum(1)
        dist_xy1 = ((y1 - nn_y1) ** 2 + (x1 - nn_x1) ** 2) ** 0.5  # (n, 1)
        mask1 = (dist_xy1 >= radius).float()
        # neg_dist01 = neg_dist01 * mask1
        # print('post_dist: ', pos_dist.shape, neg_dist01.shape, mask1.shape)
        loss01 = pos_dist - neg_dist01 * mask1 + margin
        loss01 = torch.clamp_min(loss01, min=0).mean()

        ids0 = find_nearest_neighbor_ids(feat0=desc_patch1, feat1=desc0.view(c, -1).T)
        nn_y0 = ids0 // w  # (n, 1)
        nn_x0 = ids0 % w  # (n, 1)

        nn_desc_path0 = desc0[0, :, nn_y0, nn_x0].T
        neg_dist10 = 2 - 2 * (desc_patch1 * nn_desc_path0).sum(1)
        dist_xy0 = ((y0 - nn_y0) ** 2 + (x0 - nn_x0) ** 2) ** 0.5  # (n, 1)
        mask0 = (dist_xy0 >= radius).float()
        loss10 = pos_dist - neg_dist10 * mask0 + margin
        loss10 = torch.clamp_min(loss10, min=0).mean()

        # print('mask1: ', desc_patch0.shape, torch.sum(mask0), torch.sum(mask1))
        # print('distx1: ', dist_xy1)
        # print('distx0: ', dist_xy0)

        return (loss01 + loss10) / 2

    def cal_triplet_loss(self, desc0, desc1, kps0, kps1, margin=1.0, radius=1.5):
        def find_nearest_neighbor_ids(feat0, feat1, yx0, yx1):
            with torch.no_grad():
                dist = 2 - 2 * (
                        feat0.detach() @ feat1.detach().T
                )  # (n, c) @ (c, m) -> (n, m)

                geo_dist = yx0.unsqueeze(-1) - yx1.unsqueeze(
                    0
                )  # (n 2 1) - (1 2 m) -> (n, 2 m)
                geo_dist = (geo_dist ** 2).sum(1) ** 0.5
                mask = (geo_dist <= radius).float()
                dist = dist + mask * 100
                _, ids = torch.topk(dist, dim=1, largest=False, k=1)
            return ids.reshape(-1)

        _, c, h, w = desc0.shape
        device = desc0.device

        with torch.no_grad():
            all_ys, all_xs = torch.meshgrid(
                [torch.arange(h, device=device), torch.arange(w, device=device)],
                indexing="ij",
            )
            all_ys = all_ys.reshape(-1)
            all_xs = all_xs.reshape(-1)
            all_yxs = torch.cat([all_ys.reshape(-1, 1), all_xs.reshape(-1, 1)], dim=1)

        vis = (kps0[:, 2] * kps1[:, 2]).bool()
        y0, x0 = kps0[vis, 1].long(), kps0[vis, 0].long()
        y1, x1 = kps1[vis, 1].long(), kps1[vis, 0].long()
        desc_patch0 = desc0[0, :, y0, x0].T  # (n, c)
        desc_patch1 = desc1[0, :, y1, x1].T  # (n, c)
        pos_dist = 2 - 2 * (desc_patch0 * desc_patch1).sum(dim=1)  # (n, 1)

        yx0 = torch.cat([y0.reshape(-1, 1), x0.reshape(-1, 1)], dim=1)  # [n, 2]
        ids1 = find_nearest_neighbor_ids(
            feat0=desc_patch0, feat1=desc1.view(c, -1).T, yx0=yx0, yx1=all_yxs.T
        )
        nn_y1 = ids1 // w  # (n, 1)
        nn_x1 = ids1 % w  # (n, 1)
        nn_desc_path1 = desc1[0, :, nn_y1, nn_x1].T
        neg_dist01 = 2 - 2 * (desc_patch0 * nn_desc_path1).sum(1)

        yx1 = torch.cat([y1.reshape(-1, 1), x1.reshape(-1, 1)], dim=1)  # [n, 2]
        ids0 = find_nearest_neighbor_ids(
            feat0=desc_patch1, feat1=desc0.view(c, -1).T, yx0=yx1, yx1=all_yxs.T
        )
        nn_y0 = ids0 // w  # (n, 1)
        nn_x0 = ids0 % w  # (n, 1)

        nn_desc_path0 = desc0[0, :, nn_y0, nn_x0].T
        neg_dist10 = 2 - 2 * (desc_patch1 * nn_desc_path0).sum(1)
        # diff = pos_dist - torch.min(neg_dist01, neg_dist10)

        loss01 = (pos_dist - neg_dist01 + margin).clamp_min(0)
        loss10 = (pos_dist - neg_dist10 + margin).clamp_min(0)

        loss = (loss01 + loss10) / 2
        loss = loss.mean()

        return loss

    def cal_triplet_loss3(self, desc0, desc1, kps0, kps1, margin=1.0, radius=5):
        def find_nearest_neighbor_ids(feat0, feat1, yx0, yx1):
            with torch.no_grad():
                dist = 2 - 2 * (
                        feat0.detach() @ feat1.detach().T
                )  # (n, c) @ (c, m) -> (n, m)

                geo_dist = yx0.unsqueeze(-1) - yx1.unsqueeze(
                    0
                )  # (n 2 1) - (1 2 m) -> (n, 2 m)
                geo_dist = (geo_dist ** 2).sum(1) ** 0.5
                mask = (geo_dist <= radius).float()
                dist = dist + mask * 10
                _, ids = torch.topk(dist, dim=1, largest=False, k=1)
            return ids.reshape(-1)

        _, c, h, w = desc0.shape
        device = desc0.device

        with torch.no_grad():
            all_ys, all_xs = torch.meshgrid(
                [torch.arange(h, device=device), torch.arange(w, device=device)],
                indexing="ij",
            )
            all_ys = all_ys.reshape(-1)
            all_xs = all_xs.reshape(-1)
            all_yxs = torch.cat([all_ys.reshape(-1, 1), all_xs.reshape(-1, 1)], dim=1)

        vis = (kps0[:, 2] * kps1[:, 2]).bool()
        y0, x0 = kps0[vis, 1].long(), kps0[vis, 0].long()
        y1, x1 = kps1[vis, 1].long(), kps1[vis, 0].long()
        desc_patch0 = desc0[0, :, y0, x0].T  # (n, c)
        desc_patch1 = desc1[0, :, y1, x1].T  # (n, c)
        pos_dist = 2 - 2 * (desc_patch0 * desc_patch1).sum(dim=1)  # (n, 1)

        yx0 = torch.cat([y0.reshape(-1, 1), x0.reshape(-1, 1)], dim=1)  # [n, 2]
        ids1 = find_nearest_neighbor_ids(
            feat0=desc_patch0, feat1=desc1.view(c, -1).T, yx0=yx0, yx1=all_yxs.T
        )
        nn_y1 = ids1 // w  # (n, 1)
        nn_x1 = ids1 % w  # (n, 1)
        nn_desc_path1 = desc1[0, :, nn_y1, nn_x1].T
        neg_dist01 = 2 - 2 * (desc_patch0 * nn_desc_path1).sum(1)

        yx1 = torch.cat([y1.reshape(-1, 1), x1.reshape(-1, 1)], dim=1)  # [n, 2]
        ids0 = find_nearest_neighbor_ids(
            feat0=desc_patch1, feat1=desc0.view(c, -1).T, yx0=yx1, yx1=all_yxs.T
        )
        nn_y0 = ids0 // w  # (n, 1)
        nn_x0 = ids0 % w  # (n, 1)

        nn_desc_path0 = desc0[0, :, nn_y0, nn_x0].T
        neg_dist10 = 2 - 2 * (desc_patch1 * nn_desc_path0).sum(1)
        diff = pos_dist - torch.min(neg_dist01, neg_dist10)

        # print("dist: ", pos_dist.mean(), neg_dist01.mean(), neg_dist10.mean())

        loss = (diff + margin).mean()

        return loss
