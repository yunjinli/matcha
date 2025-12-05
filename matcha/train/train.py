import logging
import os
import sys
import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

from matcha.feature import AttentionFusionNet
from third_party.dift.dift_sd import SDFeaturizer4Eval
from matcha.utils.device import torch_set_gpu
from matcha.train.dataset import get_combined_dataset, get_loader
from matcha.train.trainer import GeoSemTrainer


def get_model(args):
    if args.network == "attenfusionnet":
        model = AttentionFusionNet(
            out_dim_f=args.geo_dim,
            out_dim_c=args.sem_dim,
            ensemble_size=args.ensemble_size,
            hidden_dim=args.hidden_dim,
            dec_depth=args.dec_depth,
            dec_num_heads=args.dec_num_heads,
            ds_scale=args.ds_scale,
            use_corr=True,
        )
    else:
        logging.error("Please specify the network architecture.")
        raise NotImplementedError

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="MATCHA training script.")

    parser.add_argument("--megadepth_path", type=str, default="megadepth",
                        help="Path to the MegaDepth dataset root directory.", )
    parser.add_argument(
        "--scannet_path", type=str, default="scannet", help="Path to the scannet dataset root directory.")
    parser.add_argument("--co3dv2_path", type=str, default="co3dv2",
                        help="Path to the co3dv2 dataset root directory.")
    parser.add_argument("--coco_path", type=str, default="coco_20k",
                        help="Path to the synthetic dataset root directory.")
    parser.add_argument("--ap10k_path", type=str, default="data/ap-10k",
                        help="Path to the ap10k dataset root directory.")
    parser.add_argument("--pfpascal_path", type=str, default="data/PF-dataset-PASCAL",
                        help="Path to the pascal dataset root directory.")
    parser.add_argument("--spair_path", type=str, default="data/SPair-71k",
                        help="Path to the spair dataset root directory.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the checkpoints.")
    parser.add_argument("--network", type=str, default="matcha", help="Network architecture.", )
    parser.add_argument("--ensemble_size", type=int, default=8, help="ensemble size", )
    parser.add_argument("--hidden_dim", type=int, default=512, help="hidden dim for decoder", )

    parser.add_argument("--dec_depth", type=int, default=8, help="depth of decoder", )
    parser.add_argument("--dec_num_heads", type=int, default=8, help="decoder number of heads")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training. Default is 8.")
    parser.add_argument("--batch_size_dataset", type=int, nargs="+", default=None,
                        help="Batch size for different datasets.", )
    parser.add_argument("--n_steps", type=int, default=320000 * 2, help="Number of training steps.", )
    parser.add_argument("--n_epochs", type=int, default=250, help="Number of epochs.")
    parser.add_argument("--n_iteration_per_epoch", type=int, default=1000,
                        help="maximum number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--training_res", type=lambda s: tuple(map(int, s.split(","))), default=(512, 512),
        help="Training resolution as width,height. Default is (800, 608).",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=500,
                        help="Save checkpoints every N steps. Default is 500.")
    parser.add_argument("--workers", type=int, default=4, help="workers per gpu")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="-1 for CPU")
    parser.add_argument("--ds_scale", type=int, default=8, help="downsample scale of descriptor")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="MS",
                        help="dataset used for training (M:Megadepth, S: Scannet, C: Coco, A:AP10k, P:Pascal, I:Spair)")
    parser.add_argument("--port", type=str, default="12350")
    parser.add_argument("--n_matches_per_pair", type=int, default=1000)
    parser.add_argument("--sem_loss_type", type=str, default="clip+obj")
    parser.add_argument("--experiment_name", type=str, default=None, help="save dir for resuming")
    parser.add_argument("--debug", action="store_true", default=False, )
    parser.add_argument("--geo_dim", type=int, default=256, help="output dim (geometric descriptor)")
    parser.add_argument("--sem_dim", type=int, default=768, help="output dim (semantic descriptor)", )
    parser.add_argument("--weight_sem_loss", type=float, default=0.1, help="weight of semantic loss", )
    parser.add_argument("--use_float16", action="store_true", default=False, )

    args = parser.parse_args()

    return args


def setup(port, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=7200000),
    )


def train_ddp(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    logging.info(f"In train_ddp with rank of {rank}")
    dift = SDFeaturizer4Eval(device=device, use_float16=args.use_float16).eval()
    model = get_model(args=args)

    model.to(device)

    # important! otherwise when batch_norm used, error appears
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    setup(port=args.port, rank=rank, world_size=world_size)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=args.local_rank,
        # broadcast_buffers=False,
        # find_unused_parameters=True, # attention! very slow
    )

    n_dataset = len(list(args.dataset))
    train_set = get_combined_dataset(args=args, dataset_names=list(args.dataset))

    if args.batch_size_dataset is not None:
        batch_sizes = args.batch_size_dataset
    else:
        batch_sizes = [max(args.batch_size // world_size, 4) for _ in range(n_dataset)]

    train_loader = get_loader(
        train_set,
        batch_sizes=batch_sizes,
        workers=args.workers,
        with_dist=True,
    )

    args.local_rank = rank
    # update batch_size for each rank
    args.n_steps = args.n_steps // world_size
    trainer = GeoSemTrainer(
        model=model, dift=dift, train_loader=train_loader, eval_loader=None, args=args)

    resume_checkpoint_path = trainer.check_resume_status()

    trainer.resume_from_file(args=args, ckpt_path=resume_checkpoint_path)


try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None

if __name__ == "__main__":
    args = parse_arguments()
    torch_set_gpu(gpus=args.gpu)
    if args.local_rank == 0:
        print(args)

    if AutoResume is not None:
        # Safest is to initialize before you start a training loop
        AutoResume.init(system_stats=True)
        auto_resume_details = AutoResume.get_resume_details()
        logging.info("Auto_resume_details: ", auto_resume_details)
        if auto_resume_details:
            exp_name = auto_resume_details.get("RESUME_EXPERIMENT_NAME", None)
            args.experiment_name = exp_name
            logging.info("Found a requested auto-resume: ", exp_name)
        else:
            logging.info("No auto-resume details from a previous job detected")
    else:
        auto_resume_details = None

    if args.experiment_name is None or args.experiment_name == "None":
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        exp_name = f"{now}_{args.network}_{args.dataset}_S{args.ds_scale}_W{args.training_res[0]}_H{args.training_res[1]}_HD{args.hidden_dim}_ND{args.dec_depth}_NH{args.dec_num_heads}_B{args.batch_size}_E{args.ensemble_size}_GD{args.geo_dim}_SD{args.sem_dim}"
        if args.use_float16:
            exp_name += '_f16'

        args.experiment_name = exp_name

    if len(args.gpu) == 1:
        dift = SDFeaturizer4Eval(use_float16=args.use_float16).eval()
        model = get_model(args=args)
        model = model.cuda()

        train_set = get_combined_dataset(
            args=args, dataset_names=list(args.dataset)
        )  # follow the order
        if args.batch_size_dataset is not None:
            batch_sizes = args.batch_size_dataset
        else:
            batch_sizes = [
                args.batch_size // len(train_set) for _ in range(len(train_set))
            ]

        train_loader = get_loader(
            train_set,
            batch_sizes=batch_sizes,
            workers=args.workers,
            with_dist=False,
        )

        trainer = GeoSemTrainer(
            model=model.cuda(), dift=dift.cuda(), train_loader=train_loader, args=args
        )
        resume_checkpoint_path = trainer.check_resume_status()
        trainer.resume_from_file(args=args, ckpt_path=resume_checkpoint_path)
    else:
        # do not initialize model with frozen parameters before mp.spawn
        mp.spawn(
            train_ddp,
            nprocs=len(args.gpu),
            args=(len(args.gpu), args),
            join=True,
        )
