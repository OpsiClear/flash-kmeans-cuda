"""Side-by-side perf comparison vs upstream flash_kmeans.

Reuses the tile-grid pattern from
``third_party/flash-kmeans/examples/benchmark_backends.py`` and reports
ms/iter and TFLOPS for each backend. If the upstream package cannot import
Triton, it may benchmark the upstream torch fallback instead.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import torch

# Match the Windows-friendly Triton cache redirect from the upstream example.
_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)


from flash_kmeans_cuda import batch_kmeans_Euclid as cuda_kmeans

try:
    from flash_kmeans import batch_kmeans_Euclid as triton_kmeans
except Exception as e:  # pragma: no cover
    triton_kmeans = None
    print(f"WARN: upstream flash_kmeans unavailable: {e}")


@dataclass
class Result:
    name: str
    ms_per_iter: float
    tflops: float


def _tflops(B, N, K, D, n_iters, total_ms):
    # 2 * B * N * K * D MACs per iter * n_iters.
    flops = 2.0 * B * N * K * D * n_iters
    return flops / (total_ms * 1e-3) / 1e12


def _bench(name, fn: Callable, x, K, max_iters, init, warmup=2, rounds=5) -> Result:
    for _ in range(warmup):
        fn(x, K, max_iters=max_iters, tol=0.0, init_centroids=init.clone())
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        _, _, n_iters = fn(x, K, max_iters=max_iters, tol=0.0, init_centroids=init.clone())
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end) / rounds
    ms_per_iter = total_ms / n_iters
    B, N, D = x.shape
    return Result(name=name, ms_per_iter=ms_per_iter, tflops=_tflops(B, N, K, D, n_iters, total_ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-points", type=int, default=32768)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-clusters", type=int, default=256)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    x = torch.randn(args.batch_size, args.num_points, args.dim, device="cuda", dtype=dtype)
    init_idx = torch.randint(0, args.num_points, (args.batch_size, args.num_clusters), device="cuda")
    init = torch.gather(x, 1, init_idx.unsqueeze(-1).expand(-1, -1, args.dim)).contiguous()

    print(f"Shape: B={args.batch_size}, N={args.num_points}, K={args.num_clusters}, "
          f"D={args.dim}, dtype={args.dtype}, iters={args.max_iters}")

    results = []
    results.append(_bench("flash_kmeans_cuda", cuda_kmeans, x, args.num_clusters, args.max_iters, init))
    if triton_kmeans is not None:
        results.append(_bench("triton (heuristic)", triton_kmeans, x, args.num_clusters, args.max_iters, init))

    print(f"{'backend':<25} {'ms/iter':>10} {'TFLOPS':>10}")
    print("-" * 47)
    for r in results:
        print(f"{r.name:<25} {r.ms_per_iter:>10.3f} {r.tflops:>10.2f}")


if __name__ == "__main__":
    main()
