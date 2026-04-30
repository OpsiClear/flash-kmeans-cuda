"""Quick A/B bench harness for assign-kernel tile variants.

Times the C++-bound euclid_assign directly (no iter loop), so we can
measure the kernel itself, not the K-means convergence cost. The variant
is selected via a hidden helper that we call directly.
"""

from __future__ import annotations

import argparse
import sys
import torch
from flash_kmeans_cuda import _C


def _bench_one(fn, x, c, x_sq, c_sq, warmup=5, rounds=30):
    for _ in range(warmup):
        fn(x, c, x_sq, c_sq, None)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rounds):
        fn(x, c, x_sq, c_sq, None)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=131072)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=30)
    args = ap.parse_args()

    x = torch.randn(1, args.N, args.D, device="cuda", dtype=torch.float16)
    c = x[:, :args.K].contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (c.float() ** 2).sum(-1).contiguous()

    print(f"shape: B=1 N={args.N} K={args.K} D={args.D} fp16, rounds={args.rounds}")
    flops = 2 * args.N * args.K * args.D
    ms = _bench_one(_C.euclid_assign, x, c, x_sq, c_sq, rounds=args.rounds)
    print(f"  euclid_assign:    {ms:.4f} ms,  {flops/(ms*1e-3)/1e12:.2f} TFLOPS")


if __name__ == "__main__":
    main()
