import sys
import logging
import os
import os.path as osp
import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import glob
import tqdm

from matcha.train.utils import (
    make_batch,
    get_corresponding_pts,
)

from matcha.train.losses.geometric_loss import (
    keypoint_loss,
    dual_softmax_loss,
    coordinate_classification_loss,
    alike_distill_loss
)
from matcha.train.losses.geometric_loss_utils import spvs_coarse
from matcha.train.losses.semantic_loss import SemanticLoss
from matcha.train.losses.semantic_loss_utils import AP10K_FLIP, SPAIR_FLIP_TRN
from matcha.utils.device import save_args

from third_party.dift.dift_sd import extract_dift_feature
from third_party.xfeat.augmentation import AugmentationPipe
from third_party.alike.alike_wrapper import configs
from third_party.alike.alike import ALike

try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
except ModuleNotFoundError:
    AutoResume = None


class GeoTrainer(object):
    def __init__(self, model: torch.nn.Module, args, dift=None, train_loader=None, eval_loader=None, **kwargs):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.network = args.network
        self.dift = dift

        # Move dift to each gpu separately
        if self.dift is not None:
            self.dift = self.dift.to(self.device)

        # Move dift to each gpu separately
        if self.dift is not None:
            self.dift = self.dift.to(self.device)

        # Alike model for geometric loss
        self.alike = ALike(
            **configs["alike-t"],
            device=self.device,
            top_k=4096,
            scores_th=0.1,
            n_limit=8000,
        ).eval()

        self.train_loader = train_loader
        self.batch_size = args.batch_size
        self.local_rank = args.local_rank
        self.n_epochs = args.n_epochs
        self.n_matches_per_pair = args.n_matches_per_pair
        self.shuffle_samples = True
        self.with_dist = len(args.gpu) > 1
        self.args = args

        # Make log path
        self.experiment_name = args.experiment_name
        self.save_path = osp.join(args.save_path, self.experiment_name)
        os.makedirs(self.save_path, exist_ok=True)
        if args.local_rank == 0:
            os.makedirs(osp.join(self.save_path, "logdir"), exist_ok=True)
            save_args(args=args, save_path=Path(self.save_path, "args.txt"))

            self.writer_path = osp.join(self.save_path, f"logdir/{self.experiment_name}")
            os.makedirs(self.writer_path, exist_ok=True)
            self.writer = SummaryWriter(self.writer_path)

            self.log_file = open(osp.join(self.save_path, "log.txt"), "a+")

        self.max_samples = (self.batch_size // len(args.gpu)) // len(list(args.dataset))

        self.init_coco_dataset(args=args)

        self.save_ckpt_every = args.save_ckpt_every
        self.start_epoch = 0
        self.epoch = 0
        self.iteration = 0

        # Optimizer
        self.steps = args.n_steps
        self.optimizer = optim.AdamW(
            params=filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=1e-3,
        )

        self.latest_ckpt_path = None
        # self.latest_opt_path = None

    def train_epoch(self):
        self.model.train()

        data_iters = [iter(v) for v in self.train_loader]

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=0, initial=len(self.train_loader))

        for bidx in range(self.args.n_iteration_per_epoch):
            if 0 < self.n_matches_per_pair < bidx:
                break

            def compute_loss():
                geo_data_list = []
                for d_i, d_iter in enumerate(data_iters):
                    try:
                        d = next(d_iter)
                    except StopIteration:
                        print("End of Dataset.")
                        data_iters[d_i] = iter(self.train_loader[d_i])
                        d = next(data_iters[d_i])
                    if d is not None:
                        dataset_label = d["dataset_label"]
                        for k in d.keys():
                            if isinstance(d[k], torch.Tensor):
                                d[k] = d[k].to(self.device)
                        if dataset_label[0] == "G":
                            geo_data_list.append(d)

                geo_loss = self.process_geometric_data(
                    data_list=geo_data_list, max_samples=self.max_samples
                )

                return geo_loss

            geo_loss = compute_loss()
            if self.args.debug:
                print(f"Rank: {self.local_rank}, bid:{bidx}, Geo loss: {geo_loss}")

            losses = []
            if geo_loss is not None:
                losses.append(geo_loss.unsqueeze(0).clamp_max(10))
            else:
                geo_loss = torch.zeros([], device=self.device)

            if len(losses) > 0:
                loss = torch.cat(losses, -1).mean()
            else:
                loss = torch.zeros([], device=self.device)

            if self.with_dist:
                dist.barrier()
                loss_list = [
                    torch.zeros_like(loss)
                    for _ in range(torch.distributed.get_world_size())
                ]
                dist.all_gather(tensor_list=loss_list, tensor=loss)
                # print("after gather: ", loss_list)
                skip = False
                for v in loss_list:
                    if v <= 0.0001:
                        skip = True
                        break
                if skip:
                    print("Skip of zero (invalid) loss.")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
            else:
                if loss <= 0.001:
                    print("Skip of zero (invalid) loss.")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )  # used for geometric loss
            self.optimizer.step()
            # self.scheduler.step()

            text = "E/I:{:d}/{:d} Loss:{:.3f} Geo loss:{:.3f}".format(
                self.epoch,
                self.iteration,
                loss.item(),
                geo_loss.item()
            )

            if self.local_rank == 0:
                self.iteration = self.iteration + 1
                pbar.set_description(text)
                pbar.update(1)
                # Log metrics
                if self.iteration % 50 == 0:
                    print(text)
                    self.log_file.write(text + "\n")
                    self.log_file.flush()

                    self.writer.add_scalar("Loss", loss.item(), self.iteration)
                    self.writer.add_scalar("Geo_loss", geo_loss.item(), self.iteration)

                if not self.check_autoresume():
                    return False

        return True

    def train(self):
        if self.args.n_iteration_per_epoch > 0:
            self.iteration = self.args.n_iteration_per_epoch * self.start_epoch
        else:
            self.iteration = len(self.train_loader) * self.start_epoch

        for epoch in range(self.start_epoch, self.n_epochs):
            self.epoch = epoch
            self.adjust_learning_rate(epoch=epoch)

            if self.with_dist:
                # print(f"Rank {self.local_rank} - Start resetting sampler")
                if isinstance(self.train_loader, list):
                    for l in self.train_loader:
                        l.sampler.set_epoch(epoch=epoch)
                else:
                    self.train_loader.sampler.set_epoch(epoch=epoch)
                logging.info(f"Rank {self.local_rank} - End resetting sampler")

            # Train one epoch
            status = self.train_epoch()
            if not status:
                return 0

            if self.local_rank == 0:
                print(f"Saving epoch {epoch}")
                torch.save(
                    self.model.module.state_dict()
                    if self.with_dist
                    else self.model.state_dict(),
                    osp.join(self.save_path, f"{self.network}_{epoch}.pth"),
                )
                # if self.epoch % 10 == 0:
                #     self.evaluate_semantic(
                #         model=self.model if not self.with_dist else self.model.module
                #     )
                #     self.evaluate_geometric(
                #         model=self.model if not self.with_dist else self.model.module
                #     )
            torch.cuda.empty_cache()
            if self.with_dist:
                dist.barrier()
        print("Training finished")

    def adjust_learning_rate(self, epoch) -> None:
        # Adjust learning rate
        if 100 <= epoch < 150:
            for p in self.optimizer.param_groups:
                p['lr'] = 0.00005
        elif 150 <= epoch < 200:
            for p in self.optimizer.param_groups:
                p['lr'] = 0.00002
        elif epoch >= 200:
            for p in self.optimizer.param_groups:
                p['lr'] = 0.00001

    def process_geometric_data(
            self, data_list: list, max_num_sem_keypoints=8, max_samples=-1
    ) -> torch.Tensor | None:
        img0, img1 = [], []
        positives_c = []
        dataset_names = []

        for d in data_list:
            positives_md_coarse = spvs_coarse(d, self.args.ds_scale)
            for pi, p in enumerate(positives_md_coarse):
                if len(p) < 32:
                    continue
                positives_c = positives_c + [positives_md_coarse[pi]]

                img0.append(d["image0"][pi: pi + 1])
                img1.append(d["image1"][pi: pi + 1])
                dataset_names.extend([d["dataset_name"][pi]])

        if self.augmentor is not None:
            syn_p1s, syn_p2s, H1, H2 = make_batch(self.augmentor, 0.1)
            h_coarse, w_coarse = (
                syn_p1s[0].shape[-2] // self.args.ds_scale,
                syn_p1s[0].shape[-1] // self.args.ds_scale,
            )
            _, syn_positives_s_coarse = get_corresponding_pts(
                syn_p1s, syn_p2s, H1, H2, self.augmentor, h_coarse, w_coarse
            )

            for pi, p in enumerate(syn_positives_s_coarse):
                if len(p) < 32:
                    continue
                positives_c = positives_c + [syn_positives_s_coarse[pi]]

                img0.append(syn_p1s[pi: pi + 1])
                img1.append(syn_p2s[pi: pi + 1])
                dataset_names.extend(["coco"])

        if len(img0) == 0:
            print("Geo data is valid - return None!")
            return None

        img0 = torch.cat(img0, dim=0)
        img1 = torch.cat(img1, dim=0)

        if self.shuffle_samples:
            shuffle_ids = torch.randperm(img0.shape[0])
            if max_samples > 0:
                shuffle_ids = shuffle_ids[:max_samples]
            img0 = img0[shuffle_ids]
            img1 = img1[shuffle_ids]
            positives_c_shuffled = [positives_c[v] for v in shuffle_ids]
            positives_c = positives_c_shuffled

            dataset_names_shuffled = [dataset_names[v] for v in shuffle_ids]
            dataset_names = dataset_names_shuffled

        # Extract dift features
        if self.dift is not None:
            feat_c0, feat_f0 = extract_dift_feature(
                dift=self.dift,
                x=img0,
                ensemble_size=self.args.ensemble_size,
                require_feat_c=True,
                require_feat_f=True,
                use_float16=self.args.use_float16,

            )
            feat_c1, feat_f1 = extract_dift_feature(
                dift=self.dift,
                x=img1,
                ensemble_size=self.args.ensemble_size,
                require_feat_c=True,
                require_feat_f=True,
                use_float16=self.args.use_float16,
            )
        else:
            feat_c0, feat_f0 = None, None
            feat_c1, feat_f1 = None, None

        # Extract geometric feature maps
        _, feat0, hmap0, kpts0 = self.model(img0, feat_c0, feat_f0)
        _, feat1, hmap1, kpts1 = self.model(img1, feat_c1, feat_f1)

        # Compute geometric losses
        geo_loss = self.compute_geometric_loss(
            img0=img0,
            img1=img1,
            feat0=feat0,
            feat1=feat1,
            kpts0=kpts0,
            kpts1=kpts1,
            hmap0=hmap0,
            hmap1=hmap1,
            dataset_names=dataset_names,
            positives_c=positives_c,
        )

        return geo_loss

    def init_coco_dataset(self, args):
        if args.dataset.find("C"):
            self.augmentor = AugmentationPipe(
                img_dir=args.coco_path,
                device=self.device,
                load_dataset=True,
                batch_size=max(
                    [
                        args.batch_size
                        // (len(args.gpu) * len(list(args.dataset))),
                        2,
                    ]
                ),
                out_resolution=args.training_res,
                warp_resolution=args.training_res,
                sides_crop=0.1,
                max_num_imgs=3_000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4_000,
            )
        else:
            self.augmentor = None

    def compute_geometric_loss(
            self,
            img0: torch.Tensor,
            img1: torch.Tensor,
            feat0: torch.Tensor,
            feat1: torch.Tensor,
            hmap0: torch.Tensor,
            hmap1: torch.Tensor,
            kpts0: torch.Tensor,
            kpts1: torch.Tensor,
            positives_c: list,
            dataset_names,
    ) -> torch.Tensor | None:
        loss_items = []

        batch_size = feat0.shape[0]
        for b in range(batch_size):
            # Get positive correspondences
            pts0, pts1 = positives_c[b][:, :2], positives_c[b][:, 2:]

            # Grab features at corresponding idxs
            m0 = feat0[b, :, pts0[:, 1].long(), pts0[:, 0].long()].permute(1, 0)
            m1 = feat1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)

            # grab heatmaps at corresponding idxs
            h0 = hmap0[
                b, 0, pts0[:, 1].long(), pts0[:, 0].long()
            ]  # heatmap has the same size with feats
            h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]

            coords0 = None
            if self.args.ds_scale > 1:
                if self.with_dist and self.model.module.fine_matcher is not None:
                    coords0 = self.model.module.fine_matcher(
                        torch.cat([m0, m1], dim=-1)
                    )
                elif self.model.fine_matcher is not None:
                    coords0 = self.model.fine_matcher(torch.cat([m0, m1], dim=-1))
                    # [1, ds_scale * ds_scale]

            # Compute losses
            loss_ds, conf = dual_softmax_loss(m0, m1)  # [N]
            loss_kp_pos0, acc_pos0 = alike_distill_loss(
                kpts0[b], img0[b], alike_model=self.alike
            )
            loss_kp_pos1, acc_pos1 = alike_distill_loss(
                kpts1[b], img1[b], alike_model=self.alike
            )
            loss_kp_pos = (loss_kp_pos0 + loss_kp_pos1) * 2.0
            acc_pos = (acc_pos0 + acc_pos1) / 2

            loss_kp = keypoint_loss(h0, conf) + keypoint_loss(
                h1, conf
            )  # works for any ds_scale

            loss_items.append(loss_ds.unsqueeze(0))
            if coords0 is not None:
                loss_coords, acc_coords = coordinate_classification_loss(
                    coords0,
                    pts0,
                    pts1,
                    conf,
                    ds_scale=self.args.ds_scale,
                )
                loss_items.append(loss_coords.unsqueeze(0))
            else:
                loss_coords = torch.zeros_like(loss_ds)
                acc_coords = torch.zeros_like(loss_ds)

            loss_items.append(loss_kp.unsqueeze(0))
            loss_items.append(loss_kp_pos.unsqueeze(0))

        if len(loss_items) == 0:
            return None
        else:
            loss = torch.cat(loss_items, -1).mean()
            return loss

    def check_autoresume(self) -> bool:
        if AutoResume is not None and AutoResume.termination_requested():
            print("Termination request found!", flush=True)
            progress = "Progress %d%% (epoch %d of %d)" % (
                (self.epoch * 100 / self.n_epochs),
                self.epoch,
                self.n_epochs,
            )
            AutoResume.request_resume(
                user_dict={
                    "RESUME_CKPT_PATH": self.latest_ckpt_path,
                    "RESUME_EXPERIMENT_NAME": self.experiment_name,
                },
                message=progress,
            )

            sleep_time = 20
            logging.info(f"Training terminated. Taking a quick nap of {sleep_time} sec")
            time.sleep(sleep_time)
            logging.info("Now quitting as requested")

            return False
        else:
            return True

    def check_resume_status(self):
        if self.save_path is None:
            return None

        prev_ckpts = glob.glob(f"{self.save_path}/{self.network}_*.pth")
        start_epoch = 0
        for v in prev_ckpts:
            ep = v.split("/")[-1].split(".")[0].split("_")[-1]
            ep = int(ep)
            if ep >= start_epoch:
                start_epoch = ep
                self.latest_ckpt_path = v

        return self.latest_ckpt_path

    def resume_from_file(self, args, ckpt_path: str):
        if ckpt_path is None:
            self.start_epoch = 0
        else:
            if len(args.gpu) > 1:
                self.model.module.load_state_dict(torch.load(ckpt_path), strict=True)
            else:
                self.model.load_state_dict(torch.load(ckpt_path), strict=True)
            self.start_epoch = int(
                ckpt_path.split("/")[-1].split(".")[0].split("_")[-1]
            )
            self.start_epoch = self.start_epoch + 1
            logging.info(
                f"Rank {args.local_rank} - Resume from epoch {self.start_epoch} with ckp {ckpt_path}"
            )

        self.train()


class GeoSemTrainer(GeoTrainer):
    def __init__(self, model, args, dift=None, train_loader=None, eval_loader=None):
        super().__init__(model=model, args=args, dift=dift, train_loader=train_loader, eval_loader=eval_loader)

        self.sem_loss_func = SemanticLoss(loss_type=args.sem_loss_type)

    def train_epoch(self):
        self.model.train()

        data_iters = [iter(v) for v in self.train_loader]

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=0, initial=len(self.train_loader))

        for bidx in range(self.args.n_iteration_per_epoch):
            if 0 < self.n_matches_per_pair < bidx:
                break

            def compute_loss():
                sem_data_list = []
                geo_data_list = []
                for d_i, d_iter in enumerate(data_iters):
                    try:
                        d = next(d_iter)
                    except StopIteration:
                        print("End of Dataset.")
                        data_iters[d_i] = iter(self.train_loader[d_i])
                        d = next(data_iters[d_i])
                    if d is not None:
                        dataset_label = d["dataset_label"]
                        for k in d.keys():
                            if isinstance(d[k], torch.Tensor):
                                d[k] = d[k].to(self.device)
                        if dataset_label[0] == "S":
                            sem_data_list.append(d)
                        elif dataset_label[0] == "G":
                            geo_data_list.append(d)

                sem_loss_s = self.process_semantic_data(
                    data_list=sem_data_list, max_samples=self.max_samples
                )
                geo_loss_g = self.process_geometric_data(
                    data_list=geo_data_list, max_samples=self.max_samples
                )

                return sem_loss_s, geo_loss_g

            sem_loss, geo_loss = compute_loss()
            if self.args.debug:
                print(f"Rank: {self.local_rank}, bid:{bidx}, Sem loss: {sem_loss},Geo loss: {geo_loss}")

            losses = []
            if sem_loss is not None and geo_loss is not None:
                losses.append(sem_loss.unsqueeze(0).clamp_max(10) * self.args.weight_sem_loss)
                losses.append(geo_loss.unsqueeze(0).clamp_max(10))
            else:
                sem_loss = torch.zeros([], device=self.device)
                geo_loss = torch.zeros([], device=self.device)

            if len(losses) > 0:
                loss = torch.cat(losses, -1).mean()
            else:
                loss = torch.zeros([], device=self.device)

            if self.with_dist:
                dist.barrier()
                loss_list = [
                    torch.zeros_like(loss)
                    for _ in range(torch.distributed.get_world_size())
                ]
                dist.all_gather(tensor_list=loss_list, tensor=loss)
                # print("after gather: ", loss_list)
                skip = False
                for v in loss_list:
                    if v <= 0.0001:
                        skip = True
                        break
                if skip:
                    print("Skip of zero (invalid) loss.")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
            else:
                if loss <= 0.001:
                    print("Skip of zero (invalid) loss.")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )  # used for geometric loss
            self.optimizer.step()

            text = "E/I:{:d}/{:d} Loss:{:.3f} Sem loss:{:.3f} Geo loss:{:.3f}".format(
                self.epoch,
                self.iteration,
                loss.item(),
                sem_loss.item(),
                geo_loss.item()
            )

            if self.local_rank == 0:
                self.iteration = self.iteration + 1
                pbar.set_description(text)
                pbar.update(1)
                # Log metrics
                if self.iteration % 50 == 0:
                    print(text)
                    self.log_file.write(text + "\n")
                    self.log_file.flush()

                    self.writer.add_scalar("Loss", loss.item(), self.iteration)
                    self.writer.add_scalar("Sem_loss", sem_loss.item(), self.iteration)
                    self.writer.add_scalar("Geo_loss", geo_loss.item(), self.iteration)

                if not self.check_autoresume():
                    return False

        return True

    def process_semantic_data(self, data_list: list, max_samples: int = -1):
        if len(data_list) == 0:
            return None, None

        img0, img1 = [], []
        prompt0, prompt1 = [], []
        threshold = []
        kps0, kps1 = [], []
        cats = []
        dataset_names = []
        for d in data_list:
            img0.append(d["image0"])
            img1.append(d["image1"])
            if "prompt_embed0" in d.keys() and "prompt_embed1" in d.keys():
                prompt0.append(d["prompt_embed0"])
                prompt1.append(d["prompt_embed1"])
            threshold.append(d["thresholds1"])
            kps0.append(d["keypoints0"])
            kps1.append(d["keypoints1"])
            cats.extend(d["cat0"])
            dataset_names.extend(d["dataset_name"])

        img0 = torch.cat(img0, dim=0)
        img1 = torch.cat(img1, dim=0)
        prompt0 = torch.cat(prompt0, dim=0) if len(prompt0) > 0 else None
        prompt1 = torch.cat(prompt1, dim=0) if len(prompt1) > 0 else None
        kps0 = torch.cat(kps0, dim=0)
        kps1 = torch.cat(kps1, dim=0)
        threshold = torch.cat(threshold, dim=0)

        # Samples from multiple dataloaders
        if self.shuffle_samples:
            shuffle_ids = torch.randperm(img0.shape[0])
            if max_samples > 0:
                shuffle_ids = shuffle_ids[:max_samples]
            img0 = img0[shuffle_ids]
            img1 = img1[shuffle_ids]
            prompt0 = prompt0[shuffle_ids] if prompt0 is not None else None
            prompt1 = prompt1[shuffle_ids] if prompt1 is not None else None
            kps0 = kps0[shuffle_ids]
            kps1 = kps1[shuffle_ids]
            threshold = threshold[shuffle_ids]
            shuffled_cats = [cats[v] for v in shuffle_ids]
            cats = shuffled_cats
            shuffled_dataset_names = [dataset_names[v] for v in shuffle_ids]
            dataset_names = shuffled_dataset_names

        # Extract dift features
        with torch.no_grad():
            feat_c0, feat_f0 = extract_dift_feature(
                dift=self.dift,
                x=img0,
                prompt_embed=prompt0,
                ensemble_size=self.args.ensemble_size,
                use_float16=self.args.use_float16,
                category=cats,
            )
            feat_c1, feat_f1 = extract_dift_feature(
                dift=self.dift,
                x=img1,
                prompt_embed=prompt1,
                ensemble_size=self.args.ensemble_size,
                use_float16=self.args.use_float16,
                category=cats,
            )

        # Extract semantic features
        sem_feat0, _, _, _ = self.model(img0, feat_c=feat_c0, feat_f=feat_f0)
        sem_feat1, _, _, _ = self.model(img1, feat_c=feat_c1, feat_f=feat_f1)

        sem_loss = self.compute_semantic_loss(
            img0=img0,
            img1=img1,
            feat0=sem_feat0,
            feat1=sem_feat1,
            feat0_flip=None,
            feat1_flip=None,
            kps0=kps0,
            kps1=kps1,
            threshold=threshold,
            dataset_names=dataset_names,
            cats=cats,
        )
        return sem_loss

    def compute_semantic_loss(
            self,
            img0: torch.Tensor,
            img1: torch.Tensor,
            feat0: torch.Tensor,
            feat1: torch.Tensor,
            feat0_flip: torch.Tensor | None,
            feat1_flip: torch.Tensor | None,
            kps0: torch.Tensor,
            kps1: torch.Tensor,
            threshold: torch.Tensor,
            dataset_names: list[str],
            cats: list[str],
    ) -> torch.Tensor | None:
        scale_h = feat0.shape[-2] / img0.shape[-2]
        scale_w = feat0.shape[-1] / img0.shape[-1]
        kps0[..., 0] *= scale_w
        kps0[..., 1] *= scale_h
        kps1[..., 0] *= scale_w
        kps1[..., 1] *= scale_h
        threshold *= scale_h

        batch_size = feat0.shape[0]
        batch_loss = []
        for b in range(batch_size):
            raw_permute_list = None
            if dataset_names[b] == "ap10k":
                raw_permute_list = AP10K_FLIP
            elif dataset_names[b] == "spair":
                raw_permute_list = SPAIR_FLIP_TRN[str(cats[b])]

            if self.args.debug:
                print(
                    "compute sem_loss: ",
                    self.local_rank,
                    dataset_names[b],
                    kps0[b].shape,
                    torch.sum((kps0[b] * kps1[b])[:, 2] > 0),
                )

            loss = self.sem_loss_func(
                data={
                    "feat0": feat0[b: b + 1],
                    "feat0_flip": feat0_flip[b: b + 1]
                    if feat0_flip is not None
                    else None,
                    "feat1": feat1[b: b + 1],
                    "feat1_flip": feat1_flip[b: b + 1]
                    if feat1_flip is not None
                    else None,
                    "keypoints0": kps0[b],
                    "keypoints1": kps1[b],
                    "permute_list": raw_permute_list,
                    "threshold": threshold[b],
                },
                corr_model=self.model.corr_model
                if not self.with_dist
                else self.model.module.corr_model,
            )

            if loss is not None:
                batch_loss.append(loss)

        if len(batch_loss) > 0:
            loss = sum(batch_loss) / len(batch_loss)
        else:
            loss = None

        return loss
