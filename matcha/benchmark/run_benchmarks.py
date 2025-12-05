import logging

import torch
import argparse
import os

from matcha.benchmark import BENCHMARKS
from matcha.feature import MODELS
from matcha.matcher.base_matcher import BaseMatcher


def init_args():
    # TODO: migrate args to hydra

    parser = argparse.ArgumentParser("Matching Anything Evaluation on Benchmarks")
    parser.add_argument(
        "--method", type=str, choices=list(MODELS.keys()), default="matcha"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(BENCHMARKS.keys()),
        required=True,
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--ransac_threshold", type=float, default=2.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--topK", type=int, default=4096)
    parser.add_argument("--scale_factor", type=int, default=32)
    parser.add_argument("--weight_path", type=str, default="weights/matcha_pretrained.pth")
    parser.add_argument("--keypoint_method", type=str, default=None)
    parser.add_argument("--shuffle_iter", type=int, default=5)
    parser.add_argument("--soft_eval", action="store_true")
    parser.add_argument("--semantic_mode", action="store_true")
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_cats", nargs="+", default=None)
    args = parser.parse_args()
    print(args)
    return args


def init_matcher(args, device):
    model = MODELS[args.method]().eval().to(device)

    # Load pretrained weight
    if args.weight_path and os.path.exists(args.weight_path):
        state = model.load_state_dict(torch.load(args.weight_path), strict=False)
        logging.info(f"Initialize model with {args.weight_path}: {state}")
    else:
        raise ValueError(f"No weight path available from {args.weight_path}.")

    # Wrap as a matching model
    matcher = BaseMatcher(
        model,
        device
    )
    return matcher


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = init_matcher(args, device)
    benchmark = BENCHMARKS[args.benchmark](matcher=matcher, config=vars(args), device=device)
    benchmark.run(debug=args.debug)


if __name__ == "__main__":
    args = init_args()
    run_evaluation(args)
