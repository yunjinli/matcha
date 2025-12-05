import logging
import os
import os.path as osp
import glob
import tqdm
import torch
from matcha.dataset.geometry.megadepth import MegadepthDataset
from matcha.dataset.geometry.scannet import ScannetDataset
from matcha.dataset.semantic.ap10k import AP10K
from matcha.dataset.semantic.pfpascal import PFPascalDataset
from matcha.dataset.semantic.spair import SpairDataset


def get_dataset(
        dataset_name: str,
        dataset_path: str,
        image_size: tuple[int, int],
        debug: bool = False) -> list:
    if dataset_name == "megadepth":
        train_base_path = osp.join(dataset_path, "train-data", "megadepth_indices")
        train_data_source = osp.join(dataset_path, "train-data/phoenix/S6/zl548/MegaDepth_v1")
        train_info_root = osp.join(train_base_path, "scene_info_0.1_0.7")
        info_paths = glob.glob(train_info_root + "/*.npz")
        print(train_info_root)
        if debug:
            info_paths = glob.glob(train_info_root + "/*.npz")[:50]

        data_set = [
            MegadepthDataset(
                dataset_path=train_data_source,
                npz_path=path,
                img_resize=image_size,
                subsample=None,
            )
            for path in tqdm.tqdm(info_paths, desc="[MegaDepth] Loading metadata")
        ]
    elif dataset_name == "scannet":
        train_data_source = osp.join(dataset_path, "train")
        intrinsic_path = osp.join(dataset_path, "intrinsics.npz")
        train_info_root = osp.join(dataset_path, "train_info")
        info_paths = glob.glob(train_info_root + "/*.npz")
        if debug:
            info_paths = glob.glob(train_info_root + "/*.npz")[:50]

        data_set = [
            ScannetDataset(
                dataset_path=train_data_source,
                intrinsic_path=intrinsic_path,
                npz_path=path,
                img_resize=image_size,
                subsample=None,
            )
            for path in tqdm.tqdm(info_paths, desc="[Scannet] Loading metadata")
        ]
    elif dataset_name == "ap10k":
        data_set = [
            AP10K(
                root_dir=dataset_path,
                img_resize=image_size,
                flip_image=True,
                subsample=1000 if not debug else 50,
                # subsample=50,
            )
        ]
    elif dataset_name == "pascal":
        data_set = [
            PFPascalDataset(dataset_path=dataset_path, img_resize=image_size, flip_image=True)
        ]
    elif dataset_name == "spair":
        data_set = [
            SpairDataset(
                dataset_path=dataset_path,
                img_resize=image_size,
                flip_image=True,
                subsample=None if not debug else 50,
            )
        ]
    else:
        logging.info(f"Dataset {dataset_name} does not exit.")
        data_set = []

    logging.info(f"Loaded {len(data_set)} sequences from {dataset_name} dataset")

    return data_set


def get_combined_dataset(args, dataset_names: list):
    print('dataset_names: ', dataset_names)
    train_set = []
    for idx, name in enumerate(dataset_names):
        if name == "M":
            megadepth_set = get_dataset(
                dataset_name="megadepth",
                dataset_path=args.megadepth_path,
                image_size=args.training_res,
                debug=args.debug,
            )
            mega_set = torch.utils.data.ConcatDataset(megadepth_set)
            train_set.append(mega_set)
            logging.info(f"Loaded {len(mega_set)} samples from megadepth dataset")
        elif name == "S":
            scannet_set = get_dataset(
                dataset_name="scannet",
                dataset_path=args.scannet_path,
                image_size=args.training_res,
                debug=args.debug,
            )

            scannet_set = torch.utils.data.ConcatDataset(scannet_set)
            train_set.append(scannet_set)
            logging.info(f"Loaded {len(scannet_set)} samples from scannet dataset")
        elif name == "A":
            ap10k_set = get_dataset(
                dataset_name="ap10k",
                dataset_path=args.ap10k_path,
                image_size=(512, 512),
                debug=args.debug,
            )

            ap10k_set = torch.utils.data.ConcatDataset(ap10k_set)
            train_set.append(ap10k_set)
            logging.info(f"Loaded {len(ap10k_set)} samples from ap10k dataset")

        elif name == "P":
            pascal_set = get_dataset(
                dataset_name="pascal",
                dataset_path=args.pfpascal_path,
                image_size=(512, 512),
                debug=args.debug,
            )

            pascal_set = torch.utils.data.ConcatDataset(pascal_set)
            train_set.append(pascal_set)
            logging.info(f"Loaded {len(pascal_set)} samples from pascal dataset")
        elif name == "I":
            spair_set = get_dataset(
                dataset_name="spair",
                dataset_path=args.spair_path,
                image_size=(512, 512),
                debug=args.debug,
            )

            spair_set = torch.utils.data.ConcatDataset(spair_set)
            train_set.append(spair_set)
            logging.info(f"Loaded {len(spair_set)} samples from spair dataset")

    return train_set


def get_loader(
        datasets: list,
        batch_sizes: list,
        workers: list,
        with_dist: bool = False) -> list:
    data_loaders = []
    for dset, bs in zip(datasets, batch_sizes):
        if with_dist:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dset, drop_last=True
            )
            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=bs,
                num_workers=workers,
                # pin_memory=False,
                sampler=sampler,
            )
        else:
            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=bs,
                num_workers=workers,
                pin_memory=False,
                shuffle=True,
            )

        data_loaders.append(loader)

    return data_loaders
