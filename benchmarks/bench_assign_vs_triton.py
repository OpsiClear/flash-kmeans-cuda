"""Bench flash_kmeans_cuda assign-only vs Triton assign-only on a large dataset.

Triton's `euclid_assign_triton` is the hot-loop kernel from the upstream.
We compare it head-to-head with our `_C.euclid_assign`. This is the most
direct apples-to-apples kernel comparison.

Outputs:
  speedup: <our_ms> <triton_ms> <speedup_x>
"""

from __future__ import annotations

import argparse
import os
import tempfile
import statistics
import torch

# Force a local Triton cache (Windows-friendly).
_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)


def _bench(fn, *args, warmup=5, rounds=30):
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
    ap.add_argument("--shape", choices=["med", "big", "huge", "mega"], default="big")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--check-accuracy", action="store_true")
    args = ap.parse_args()

    shapes = {
        "med":  (1, 32768,  256, 128),
        "big":  (1, 131072, 2048, 128),
        "huge": (1, 262144, 4096, 128),
        # mega = "larger dataset" the user asked for
        "mega": (1, 524288, 8192, 128),
    }
    B, N, K, D = shapes[args.shape]

    torch.manual_seed(0)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = x[:, :K].contiguous()
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()

    from flash_kmeans_cuda import _C
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton

    # Wrappers to give both functions the same signature for _bench.
    def ours(x, c, xsq, csq):
        return _C.euclid_assign(x, c, xsq, csq, None)

    def triton(x, c, xsq, csq):
        return euclid_assign_triton(x, c, xsq)

    # Accuracy check
    if args.check_accuracy:
        ours_ids = ours(x, centroids, x_sq, c_sq)
        triton_ids = triton(x, centroids, x_sq, c_sq)
        disagreements = (ours_ids.long() != triton_ids.long()).sum().item()
        total = ours_ids.numel()
        frac = disagreements / total
        print(f"accuracy: {disagreements}/{total} ({frac:.3%}) cluster_ids disagree (vs Triton)")
        if frac > 0.05:
            raise SystemExit(f"ACCURACY FAIL: {frac:.2%} > 5% threshold")

    # Median of 5 outer × rounds inner
    ours_runs = [_bench(ours, x, centroids, x_sq, c_sq, rounds=args.rounds)
                 for _ in range(5)]
    triton_runs = [_bench(triton, x, centroids, x_sq, c_sq, rounds=args.rounds)
                   for _ in range(5)]
    ours_ms = statistics.median(ours_runs)
    triton_ms = statistics.median(triton_runs)

    speedup = triton_ms / ours_ms
    flops = 2.0 * B * N * K * D
    our_tflops = flops / (ours_ms * 1e-3) / 1e12
    triton_tflops = flops / (triton_ms * 1e-3) / 1e12

    print(f"shape: B={B} N={N} K={K} D={D} dtype=fp16")
    print(f"  ours runs:    {[f'{m:.4f}' for m in ours_runs]}")
    print(f"  triton runs:  {[f'{m:.4f}' for m in triton_runs]}")
    print(f"  ours:    {ours_ms:8.4f} ms ({our_tflops:6.2f} TFLOPS)  median of 5")
    print(f"  triton:  {triton_ms:8.4f} ms ({triton_tflops:6.2f} TFLOPS)  median of 5")
    print(f"speedup: {ours_ms:.4f} {triton_ms:.4f} {speedup:.3f}")


if __name__ == "__main__":
    main()
