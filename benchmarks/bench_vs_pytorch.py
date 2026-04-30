"""Bench flash_kmeans_cuda assign-step vs PyTorch reference assignment.

Outputs a single line of the form:
  speedup: <our_ms> <torch_ms> <speedup_x>
so the auto-tune loop can grep it cleanly.

Larger dataset by default to expose perf differences; pass --shape big or
--shape small to switch.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import torch

# Force a local Triton/MSVC cache (Windows-friendly). Keeps env predictable.
_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP


def torch_assign(x, centroids, x_sq, c_sq):
    """Reference PyTorch assignment using fp16 matmul + arithmetic.

    Mirrors what a user would naturally write in PyTorch for an Euclidean
    nearest-centroid query. Uses fp16 throughout (matches our kernel's
    compute dtype). torch's matmul will use tensor cores via cublasLt.
    """
    cross = torch.einsum("bnd,bkd->bnk", x, centroids)  # fp16 matmul
    dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross.float()
    return dist.argmin(dim=-1).to(torch.int32)


def _bench(fn, *args, warmup=3, rounds=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rounds):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["tiny", "small", "med", "big", "huge"], default="big")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--check-accuracy", action="store_true",
                    help="verify our cluster_ids agree with torch reference within tolerance")
    args = ap.parse_args()

    shapes = {
        # B, N, K, D
        "tiny": (1, 1024, 32, 64),
        "small": (1, 8192, 128, 64),
        "med": (1, 32768, 256, 128),
        "big": (1, 131072, 2048, 128),     # the K=2048 hot case
        "huge": (1, 262144, 4096, 128),    # very large, well above L2
    }
    B, N, K, D = shapes[args.shape]

    torch.manual_seed(0)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()

    # Import after CUDA init
    from flash_kmeans_cuda import euclid_assign

    # Accuracy check (optional)
    if args.check_accuracy:
        ours = euclid_assign(x, centroids, x_sq, c_sq=c_sq)
        ref = torch_assign(x, centroids, x_sq, c_sq)
        disagreements = (ours.long() != ref.long()).sum().item()
        total = ours.numel()
        frac = disagreements / total
        print(f"accuracy: {disagreements}/{total} ({frac:.3%}) cluster_ids disagree")
        # Allow 2% — fp16 mma scheduling vs fp32-broadcast diff at tied points.
        if frac > 0.02:
            raise SystemExit(f"ACCURACY FAIL: {frac:.2%} > 2.0% threshold")

    # Take median of 5 outer runs to dampen GPU thermal/launch noise.
    import statistics
    ours_runs = [_bench(euclid_assign, x, centroids, x_sq, c_sq, rounds=args.rounds)
                 for _ in range(5)]
    torch_runs = [_bench(torch_assign, x, centroids, x_sq, c_sq, rounds=args.rounds)
                  for _ in range(5)]
    ours_ms = statistics.median(ours_runs)
    torch_ms = statistics.median(torch_runs)

    speedup = torch_ms / ours_ms
    flops = 2.0 * B * N * K * D
    our_tflops = flops / (ours_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    print(f"shape: B={B} N={N} K={K} D={D} dtype=fp16")
    print(f"  ours runs:  {[f'{m:.3f}' for m in ours_runs]}")
    print(f"  ours:   {ours_ms:7.4f} ms ({our_tflops:6.2f} TFLOPS)  median of 5")
    print(f"  torch:  {torch_ms:7.4f} ms ({torch_tflops:6.2f} TFLOPS)")
    print(f"speedup: {ours_ms:.4f} {torch_ms:.4f} {speedup:.3f}")


if __name__ == "__main__":
    main()
