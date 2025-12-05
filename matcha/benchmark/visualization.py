import numpy as np
import cv2
from copy import deepcopy
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt


# code from https://github.com/feixue94/pram/blob/dev/recognition/vis_seg.py


def resize_img(img, nh=-1, nw=-1, rmax=-1, mode=cv2.INTER_NEAREST):
    assert nh > 0 or nw > 0 or rmax > 0
    if nh > 0:
        return cv2.resize(
            img, dsize=(int(img.shape[1] / img.shape[0] * nh), nh), interpolation=mode
        )
    if nw > 0:
        return cv2.resize(
            img, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)), interpolation=mode
        )
    if rmax > 0:
        oh, ow = img.shape[0], img.shape[1]
        if oh > ow:
            return cv2.resize(
                img,
                dsize=(int(img.shape[1] / img.shape[0] * rmax), rmax),
                interpolation=mode,
            )
        else:
            return cv2.resize(
                img,
                dsize=(rmax, int(img.shape[0] / img.shape[1] * rmax)),
                interpolation=mode,
            )

    return cv2.resize(img, dsize=(nw, nh), interpolation=mode)


def plot_matches(
        img1: np.ndarray,
        img2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        inliers: np.ndarray = None,
        radius: int = 3,
        line_thickness: int = 2,
        horizon: bool = True,
        plot_outlier: bool = False,
        show_text: bool = None,
):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    if inliers is None:
        inliers = np.arange(len(pts1))

    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # r = 3
    if horizon:
        img_out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype="uint8")
        # Place the first image to the left
        img_out[:rows1, :cols1, :] = img1
        # Place the next image to the right
        img_out[:rows2, cols1:, :] = img2  # np.dstack([img2, img2, img2])
        for idx in range(inliers.shape[0]):
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)
            pt1 = pts1[idx]
            pt2 = pts2[idx]

            nr = radius
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), nr, color, 2)

            img_out = cv2.circle(
                img_out, (int(pt2[0]) + cols1, int(pt2[1])), nr, color, 2
            )

            img_out = cv2.line(
                img_out,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]) + cols1, int(pt2[1])),
                color,
                line_thickness,
            )
    else:
        img_out = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype="uint8")
        # Place the first image to the left
        img_out[:rows1, :cols1, :] = img1
        # Place the next image to the right of it
        img_out[rows1:, :cols2, :] = img2  # np.dstack([img2, img2, img2])

        for idx in range(inliers.shape[0]):
            # print("idx: ", inliers[idx])
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)

            nr = radius

            pt1 = pts1[idx]
            pt2 = pts2[idx]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), nr, color, 2)

            img_out = cv2.circle(
                img_out, (int(pt2[0]), int(pt2[1]) + rows1), nr, color, 2
            )

            img_out = cv2.line(
                img_out,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1]) + rows1),
                color,
                line_thickness,
            )

    if show_text is not None:
        img_out = cv2.putText(
            img_out, show_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
        )

    return img_out


def plot_kpts(
        img,
        kpts,
        r=3,
        color=(0, 0, 255),
        nh=-1,
        nw=-1,
        shape="o",
        show_text=None,
        thickness=5,
):
    img_out = deepcopy(img)
    for i in range(kpts.shape[0]):
        pt = kpts[i]
        if shape == "o":
            img_out = cv2.circle(
                img_out,
                center=(int(pt[0]), int(pt[1])),
                radius=r,
                color=color,
                thickness=thickness,
            )
        elif shape == "+":
            img_out = cv2.line(
                img_out,
                pt1=(int(pt[0] - r), int(pt[1])),
                pt2=(int(pt[0] + r), int(pt[1])),
                color=color,
                thickness=thickness,
            )
            img_out = cv2.line(
                img_out,
                pt1=(int(pt[0]), int(pt[1] - r)),
                pt2=(int(pt[0]), int(pt[1] + r)),
                color=color,
                thickness=thickness,
            )
    if show_text is not None:
        img_out = cv2.putText(
            img_out, show_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
        )
    if nh == -1 and nw == -1:
        return img_out
    if nh > 0:
        return cv2.resize(img_out, dsize=(int(img.shape[1] / img.shape[0] * nh), nh))
    if nw > 0:
        return cv2.resize(img_out, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)))


def visualize_matches(img0, img1, pts0: list, pts1: list, colors=None, lw=1):
    rows0, cols0 = img0.shape[:2]
    rows1, cols1 = img1.shape[:2]
    img_out = np.zeros((max([rows0, rows1]), cols0 + cols1, 3), dtype="uint8")
    # Place the first image to the left
    img_out[:rows0, :cols0] = img0
    # Place the next image to the right of it
    img_out[:rows1, cols0:] = img1  # np.dstack([img2, img2, img2])

    for l in range(len(pts0)):
        mpts0 = pts0[l]
        mpts1 = pts1[l]
        color = colors[l]

        nr = 5
        for i in range(mpts0.shape[0]):
            pt0 = mpts0[i]
            pt1 = mpts1[i]
            img_out = cv2.circle(img_out, (int(pt0[0]), int(pt0[1])), nr, color, 2)
            img_out = cv2.circle(
                img_out, (int(pt1[0]) + cols0, int(pt1[1])), nr, color, 2
            )

            img_out = cv2.line(
                img_out,
                (int(pt0[0]), int(pt0[1])),
                (int(pt1[0]) + cols0, int(pt1[1])),
                color,
                lw,
            )

    return img_out


def undo_normalize_scale(im, imagenet=False):
    if imagenet:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        im = im * std + mean
    im *= 255.0
    return im.astype(np.uint8)


def torch_rgb_to_PIL(im, unnormalize=True):
    im = im.squeeze()
    if im.shape[0] == 3:
        im = im.permute(1, 2, 0)
    im = im.cpu().data.numpy()
    if unnormalize:
        im = undo_normalize_scale(im)
    else:
        im = im.astype(np.uint8)
    return Image.fromarray(im)


def read_im_PIL(im):
    if isinstance(im, Image.Image):
        I = im
    elif isinstance(im, np.ndarray):
        I = Image.fromarray(im.squeeze())
    elif isinstance(im, str):
        I = Image.open(im).convert("RGB")
    elif isinstance(im, torch.Tensor):
        I = torch_rgb_to_PIL(im)
    return I


def plot_image_matches(
        im1: str,
        im2: str,
        matches,
        inliers=None,
        Npts=None,
        lines=True,
        radius=2,
        dpi=150,
        title=None,
        save_fig=None,
        colors=None,
        ret_fig=False,
):
    # Read images
    I1 = read_im_PIL(im1)
    I2 = read_im_PIL(im2)

    # Resize
    w1, h1 = I1.size
    w2, h2 = I2.size

    if h1 <= h2:
        scale1 = 1
        scale2 = h1 / h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2 / h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        if Npts is not None:
            Npts = Npts if Npts < match_num else match_num
        else:
            Npts = matches.shape[0]
        inliers = range(Npts)  # Everthing as an inlier
    else:
        if Npts is not None and Npts < len(inliers):
            inliers = inliers[:Npts]
    # print('Plotting inliers: ', len(inliers))

    x1 = scale1 * matches[inliers, 0]
    y1 = scale1 * matches[inliers, 1]
    x2 = scale2 * matches[inliers, 2] + w1
    y2 = scale2 * matches[inliers, 3]
    c = np.random.rand(len(inliers), 3)

    if colors is not None:
        c = colors

    # Plot images and matches
    fig = plt.figure(figsize=(20, 10))
    axis = plt.gca()
    cmap = "gray" if len(catI.shape) == 2 else None
    axis.imshow(catI, cmap=cmap)
    axis.axis("off")
    if title:
        axis.set_title(title, fontsize=26)

    for i, inid in enumerate(inliers):
        # Plot
        axis.add_artist(plt.Circle((x1[i], y1[i]), radius=radius, color=c[i, :]))
        axis.add_artist(plt.Circle((x2[i], y2[i]), radius=radius, color=c[i, :]))
        if lines:
            axis.plot(
                [x1[i], x2[i]],
                [y1[i], y2[i]],
                c=c[i, :],
                linestyle="-",
                linewidth=radius,
            )

    if ret_fig:
        fig.tight_layout()
        return fig

    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {save_fig}")
    plt.show()
