import torch
import numpy as np

from ptflops import get_model_complexity_info

COMMON_RESOLUTIONS = [
    # C, H, W
    (3, 224, 224),
    (3, 256, 256),
    (3, 512, 512),
    (3, 480, 640),
    (3, 768, 1024),
]


def measure_model_size(model):
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {num_params:.2f}M")


def measure_flops(model, input_size):
    print(f"\n>>> Measure FLOPs with input size: {input_size}")
    macs, _ = get_model_complexity_info(
        model,
        input_size,
        as_strings=True,
        print_per_layer_stat=False,
    )
    print(f"FLOPs (MACs): {macs}")


def measure_flops2(model, input_size):
    from calflops import calculate_flops

    print(f"\n>>> Measure FLOPs with input size: {input_size}")
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, *input_size),
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    print(f"FLOPs:{flops} MACs:{macs} \n")


def measure_runtime_and_memory(model, input_size, device, iter=10):
    torch.cuda.empty_cache()  # Clear any cache to get accurate measurement
    torch.cuda.reset_peak_memory_stats()

    for _ in range(10):
        # Warm-up the GPU (important to avoid measuring setup overhead)
        _ = model(torch.rand(1, 3, 224, 224).to(device))

    print(f"\n>>> Measure runtime with input size: {input_size}")
    input_tensor = torch.rand(1, *input_size).to(device)
    timers = []
    for i in range(iter):
        timer = model.runtime_benchmark(input_tensor)
        timers.append(timer)

    total = 0
    for k, _ in timer.items():
        vmean = np.mean([tt[k] for tt in timers])
        total += vmean
        print(f"{k}={vmean * 1000:.1f}ms")
    print(f"Total = {total * 1000:.1f}ms")

    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"Peak memory usage: {peak_memory:.2f} GB")


def benchmark_model(model):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    for imsize in COMMON_RESOLUTIONS:
        measure_runtime_and_memory(model, imsize, device)
        measure_flops(model, imsize)
        measure_model_size(model)
        # Leads to inconsistent results as the measulre_flops()
        # measure_flops2(model, imsize)


def get_number_params(model, trainable=False):
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()

    nb_params = sum([np.prod(p.size()) for p in model_parameters])

    print("Number of parameters: {:d}".format(nb_params))
    return nb_params


if __name__ == "__main__":
    from matcha.feature import Matcha

    # Initialize
    model = Matcha()
    # model = MatchaLight()

    state = model.load_state_dict(
        torch.load("weights/matcha_models/matcha/attfnetjp_210.pth"), strict=False
    )
    print(state)

    benchmark_model(model)
