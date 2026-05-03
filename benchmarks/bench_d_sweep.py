from __future__ import annotations

import argparse
import os
import statistics
import tempfile

import torch


_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))


def _bench(fn, *args, warmup: int, rounds: int) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--d", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    from flash_kmeans_cuda import _C

    print("D\tms\ttflops\tdisagreement")
    for D in args.d:
        torch.manual_seed(args.seed)
        x = torch.randn(1, args.n, D, device="cuda", dtype=torch.float16)
        centroids = torch.randn(1, args.k, D, device="cuda", dtype=torch.float16)
        x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
        c_sq = (centroids.float() ** 2).sum(dim=-1).contiguous()

        def ours(x, c, xs, cs):
            return _C.euclid_assign(x, c, xs, cs, None)

        disagreement = float("nan")
        if args.check:
            ids = ours(x, centroids, x_sq, c_sq)
            cross = torch.einsum("bnd,bkd->bnk", x.float(), centroids.float())
            ref = (x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross).argmin(-1)
            disagreement = (ids.long() != ref.long()).float().mean().item()

        runs = [
            _bench(
                ours, x, centroids, x_sq, c_sq,
                warmup=args.warmup, rounds=args.rounds,
            )
            for _ in range(args.outer)
        ]
        ms = statistics.median(runs)
        flops = 2.0 * args.n * args.k * D
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"{D}\t{ms:.4f}\t{tflops:.2f}\t{disagreement:.6f}")


if __name__ == "__main__":
    main()
