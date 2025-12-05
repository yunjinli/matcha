import os
import torch
import numpy as np
import json

def save_args(args, save_path):
    with open(save_path, "a+") as f:
        json.dump(args.__dict__, f, indent=2)


def torch_set_gpu(gpus):
    if isinstance(gpus, int):
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpus])
        # print(os.environ['CUDA_VISIBLE_DEVICES'])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ["HOSTNAME"],
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print("Launching on GPUs " + os.environ["CUDA_VISIBLE_DEVICES"])

    else:
        print("Launching on CPU")

    return cuda


def to_numpy(data: torch.Tensor | np.ndarray | None):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data
