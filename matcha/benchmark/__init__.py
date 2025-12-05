from matcha.benchmark.geometry.hpatches import HPatchesMatching
from matcha.benchmark.semantic.pfwillow import PFWillowBenchmark
from matcha.benchmark.semantic.pfpascal import PFPascalBenchmark
from matcha.benchmark.semantic.spair import SpairBenchmark
from matcha.benchmark.temporal.tapvid import TapvidBenchmark
from matcha.benchmark.geometry.aachen import Aachen1500Benchmark
from matcha.benchmark.geometry.megadepth import Megadepth1500Benchmark
from matcha.benchmark.geometry.scannet import Scannet1500Benchmark

BENCHMARKS = {
    "pfwillow": PFWillowBenchmark,
    "pfpascal": PFPascalBenchmark,
    "spair": SpairBenchmark,
    "tapvid": TapvidBenchmark,
    "hpatches_matching": HPatchesMatching,
    "aachen1500": Aachen1500Benchmark,
    "megadepth1500": Megadepth1500Benchmark,
    "scannet1500": Scannet1500Benchmark,
}
