from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass

import torch

_LOCAL_TMP = os.path.abspath(".tmp")
os.makedirs(_LOCAL_TMP, exist_ok=True)
os.environ.setdefault("TMP", _LOCAL_TMP)
os.environ.setdefault("TEMP", _LOCAL_TMP)
os.environ.setdefault("TMPDIR", _LOCAL_TMP)
tempfile.tempdir = _LOCAL_TMP
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath(".triton_cache"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from flash_kmeans_cuda import batch_kmeans_Euclid as cuda_kmeans
from flash_kmeans_cuda import euclid_assign
from flash_kmeans import batch_kmeans_Euclid as triton_kmeans


SHAPES = {
    "med": (1, 32768, 256, 128),
    "big": (1, 131072, 2048, 128),
    "huge": (1, 262144, 4096, 128),
    "mega": (1, 524288, 8192, 128),
}


@dataclass
class Run:
    ids: torch.Tensor
    cents: torch.Tensor
    n_iters: int
    ms: float


def _make_input(B: int, N: int, K: int, D: int, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, N, D, generator=g, device="cuda", dtype=dtype)
    idx = torch.randint(0, N, (B, K), generator=g, device=x.device)
    init = torch.gather(
        x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D)
    ).contiguous()
    return x, init


def _run(fn, x, K: int, iters: int, init: torch.Tensor) -> Run:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    ids, cents, n_iters = fn(
        x, K, max_iters=iters, tol=0.0, init_centroids=init.clone(), verbose=False
    )
    end.record()
    torch.cuda.synchronize()
    return Run(ids=ids, cents=cents, n_iters=n_iters, ms=start.elapsed_time(end))


def _assign_with_fixed_evaluator(x: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
    x_sq = (x.float() ** 2).sum(dim=-1).contiguous()
    c_sq = (cents.float() ** 2).sum(dim=-1).contiguous()
    return euclid_assign(x, cents.contiguous(), x_sq, c_sq=c_sq)


def _assigned_inertia(
    x: torch.Tensor, ids: torch.Tensor, cents: torch.Tensor, chunk_n: int
) -> float:
    B, N, D = x.shape
    total = torch.zeros((), device=x.device, dtype=torch.float64)
    cents_f = cents.float()
    for start in range(0, N, chunk_n):
        end = min(start + chunk_n, N)
        idx = ids[:, start:end].long()
        chosen = torch.gather(
            cents_f, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D)
        )
        diff = x[:, start:end].float() - chosen
        total += diff.square().sum(dtype=torch.float64)
    return float(total.item())


def _empty_clusters(ids: torch.Tensor, K: int) -> int:
    counts = torch.bincount(ids.reshape(-1).long(), minlength=K)
    return int((counts == 0).sum().item())


def _sample_exact_inertia(
    x: torch.Tensor,
    cents: torch.Tensor,
    sample_points: int,
    seed: int,
    chunk_n: int,
) -> float:
    B, N, D = x.shape
    if sample_points <= 0:
        return float("nan")
    sample_points = min(sample_points, N)
    g = torch.Generator(device=x.device).manual_seed(seed)
    sample_idx = torch.randperm(N, generator=g, device=x.device)[:sample_points]
    xs = x[:, sample_idx, :].float()
    cents_f = cents.float()
    c_sq = cents_f.square().sum(dim=-1)
    total = torch.zeros((), device=x.device, dtype=torch.float64)
    for start in range(0, sample_points, chunk_n):
        end = min(start + chunk_n, sample_points)
        chunk = xs[:, start:end, :]
        x_sq = chunk.square().sum(dim=-1)
        cross = torch.matmul(chunk, cents_f.transpose(1, 2))
        dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * cross
        vals = dist.min(dim=-1).values.clamp_min_(0.0)
        total += vals.sum(dtype=torch.float64)
    return float(total.item())


def _fmt_delta(a: float, b: float) -> str:
    return f"{(a - b) / b:+.4%}" if b != 0 else "nan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["big", "huge", "mega"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--chunk-n", type=int, default=65536)
    ap.add_argument("--sample-points", type=int, default=4096)
    ap.add_argument("--sample-chunk-n", type=int, default=512)
    args = ap.parse_args()

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    print(
        "shape\titers\tours_ms\ttriton_ms\tspeedup\t"
        "returned_label_diff\tfinal_label_diff\t"
        "ours_final_inertia_per_point\ttriton_final_inertia_per_point\tfinal_inertia_delta\t"
        "ours_sample_exact_per_point\ttriton_sample_exact_per_point\tsample_exact_delta\t"
        "centroid_rel_frob\tcentroid_mean_l2\tcentroid_max_l2\t"
        "ours_empty\ttriton_empty"
    )

    for shape in args.shapes:
        B, N, K, D = SHAPES[shape]
        x, init = _make_input(B, N, K, D, dtype, args.seed)

        for _ in range(args.warmup):
            cuda_kmeans(x, K, max_iters=args.iters, tol=0.0, init_centroids=init.clone())
            triton_kmeans(x, K, max_iters=args.iters, tol=0.0, init_centroids=init.clone())
        torch.cuda.synchronize()

        ours = _run(cuda_kmeans, x, K, args.iters, init)
        tri = _run(triton_kmeans, x, K, args.iters, init)

        assert ours.n_iters == tri.n_iters == args.iters

        returned_diff = (ours.ids.long() != tri.ids.long()).float().mean().item()

        ours_final_ids = _assign_with_fixed_evaluator(x, ours.cents)
        tri_final_ids = _assign_with_fixed_evaluator(x, tri.cents)
        final_diff = (ours_final_ids.long() != tri_final_ids.long()).float().mean().item()

        ours_inertia = _assigned_inertia(x, ours_final_ids, ours.cents, args.chunk_n)
        tri_inertia = _assigned_inertia(x, tri_final_ids, tri.cents, args.chunk_n)

        ours_sample = _sample_exact_inertia(
            x, ours.cents, args.sample_points, args.seed + 1000, args.sample_chunk_n
        )
        tri_sample = _sample_exact_inertia(
            x, tri.cents, args.sample_points, args.seed + 1000, args.sample_chunk_n
        )

        diff = ours.cents.float() - tri.cents.float()
        centroid_rel_frob = (diff.norm() / (tri.cents.float().norm() + 1e-12)).item()
        centroid_l2 = diff.norm(dim=-1)
        centroid_mean_l2 = centroid_l2.mean().item()
        centroid_max_l2 = centroid_l2.max().item()

        print(
            f"{shape}\t{args.iters}\t"
            f"{ours.ms:.3f}\t{tri.ms:.3f}\t{tri.ms / ours.ms:.3f}x\t"
            f"{returned_diff:.4%}\t{final_diff:.4%}\t"
            f"{ours_inertia / (B * N):.8f}\t{tri_inertia / (B * N):.8f}\t"
            f"{_fmt_delta(ours_inertia, tri_inertia)}\t"
            f"{ours_sample / (B * min(args.sample_points, N)):.8f}\t"
            f"{tri_sample / (B * min(args.sample_points, N)):.8f}\t"
            f"{_fmt_delta(ours_sample, tri_sample)}\t"
            f"{centroid_rel_frob:.6f}\t{centroid_mean_l2:.6f}\t{centroid_max_l2:.6f}\t"
            f"{_empty_clusters(ours_final_ids, K)}\t{_empty_clusters(tri_final_ids, K)}"
        )


if __name__ == "__main__":
    main()
