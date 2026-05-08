"""Benchmark dot-product assignment against the chunked PyTorch fallback."""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from flash_kmeans_cuda import similarity_assign
from flash_kmeans_cuda.torch_fallback import similarity_assign_torch_chunked


SHAPES: dict[str, tuple[int, int, int, int]] = {
    "small_d3": (1, 32768, 8192, 3),
    "small_d8": (1, 32768, 8192, 8),
    "small_d16": (1, 32768, 8192, 16),
    "mega": (1, 32768, 8192, 128),
}


def _bench(fn: Callable[[], torch.Tensor], *, warmup: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    if value in SHAPES:
        return SHAPES[value]
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "shape must be one of "
            f"{', '.join(SHAPES)} or an explicit B,N,K,D tuple"
        )
    try:
        b, n, k, d = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    return b, n, k, d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        default=SHAPES["mega"],
        help="shape name or B,N,K,D tuple",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--chunk-size-n", type=int, default=32768)
    parser.add_argument("--chunk-size-k", type=int, default=1024)
    parser.add_argument("--check-accuracy", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B, N, K, D = args.shape
    torch.manual_seed(0)
    x = torch.randn((B, N, D), device="cuda", dtype=dtype)
    centroids = torch.randn((B, K, D), device="cuda", dtype=dtype)

    def ours() -> torch.Tensor:
        return similarity_assign(x, centroids)

    def torch_chunked() -> torch.Tensor:
        return similarity_assign_torch_chunked(
            x,
            centroids,
            chunk_size_N=args.chunk_size_n,
            chunk_size_K=args.chunk_size_k,
        )

    ours_ms = _bench(ours, warmup=args.warmup, rounds=args.rounds)
    torch_ms = _bench(torch_chunked, warmup=args.warmup, rounds=args.rounds)
    tflops = (2.0 * B * N * K * D) / (ours_ms * 1e9)

    print(f"shape: B={B} N={N} K={K} D={D} dtype={args.dtype}")
    print(f"cuda similarity_assign: {ours_ms:.4f} ms ({tflops:.2f} TFLOPS)")
    print(f"torch chunked:          {torch_ms:.4f} ms")
    print(f"speedup:                {torch_ms / ours_ms:.2f}x")

    if args.check_accuracy:
        out = ours()
        ref = torch_chunked()
        disagreement = (out.long() != ref.long()).float().mean().item()
        print(f"disagreement vs chunked torch: {100.0 * disagreement:.3f}%")


if __name__ == "__main__":
    main()
