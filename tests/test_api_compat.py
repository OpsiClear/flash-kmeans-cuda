"""flash-kmeans public API compatibility coverage."""

from __future__ import annotations

import torch

import flash_kmeans_cuda as fkc


def test_public_flash_kmeans_names_exist():
    for name in [
        "FlashKMeans",
        "batch_kmeans_Euclid",
        "batch_kmeans_Cosine",
        "batch_kmeans_Dot",
        "triton_centroid_update_euclid",
        "triton_centroid_update_sorted_euclid",
        "kmeans_largeN",
        "kmeans_largeN_assign",
    ]:
        assert hasattr(fkc, name)
        assert name in fkc.__all__


def test_flash_kmeans_fit_predict_and_predict_2d():
    torch.manual_seed(0)
    N, D, K = 256, 16, 8
    x = torch.randn(N, D, device="cuda", dtype=torch.float16)
    km = fkc.FlashKMeans(d=D, k=K, niter=2, tol=0.0, seed=123, device=torch.device("cuda:0"))

    labels = km.fit_predict(x)
    pred = km.predict(x)

    assert labels.shape == (N,)
    assert pred.shape == (N,)
    assert labels.dtype == torch.int32
    assert pred.dtype == torch.int32
    assert km.centroids_b is not None
    assert km.centroids_b.shape == (1, K, D)
    assert (pred >= 0).all() and (pred < K).all()


def test_flash_kmeans_fit_predict_batched():
    torch.manual_seed(1)
    B, N, D, K = 2, 128, 16, 8
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    km = fkc.FlashKMeans(d=D, k=K, niter=2, tol=0.0, seed=7, device=torch.device("cuda:0"))

    labels = km.fit_predict(x)
    pred = km.predict(x)

    assert labels.shape == (B, N)
    assert pred.shape == (B, N)
    assert km.centroids_b is not None
    assert km.centroids_b.shape == (B, K, D)


def test_cosine_and_dot_batch_api_shapes():
    torch.manual_seed(2)
    B, N, D, K = 1, 192, 16, 12
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)

    cos_ids, cos_centroids, cos_iters = fkc.batch_kmeans_Cosine(x, K, max_iters=2, tol=0.0)
    dot_ids, dot_centroids, dot_iters = fkc.batch_kmeans_Dot(x, K, max_iters=2, tol=0.0)

    assert cos_ids.shape == (B, N)
    assert dot_ids.shape == (B, N)
    assert cos_centroids.shape == (B, K, D)
    assert dot_centroids.shape == (B, K, D)
    assert cos_ids.dtype == torch.int32
    assert dot_ids.dtype == torch.int32
    assert cos_iters == 2
    assert dot_iters == 2
    torch.testing.assert_close(cos_centroids.float().norm(dim=-1), torch.ones((B, K), device="cuda"), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(dot_centroids.float().norm(dim=-1), torch.ones((B, K), device="cuda"), atol=2e-3, rtol=2e-3)


def test_similarity_assign_uses_argmax_dot_semantics():
    torch.manual_seed(22)
    B, N, D, K = 1, 256, 16, 32
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)

    ids = fkc.similarity_assign(x, centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    ids_out = fkc.similarity_assign(x, centroids, out=out)
    ref = torch.bmm(x.float(), centroids.float().transpose(1, 2)).argmax(dim=-1)
    disagreement = (ids.long() != ref.long()).float().mean().item()

    assert ids.shape == (B, N)
    assert ids.dtype == torch.int32
    assert ids_out.data_ptr() == out.data_ptr()
    torch.testing.assert_close(ids_out, ids)
    assert disagreement < 0.02


def test_largeN_and_assign_small_cpu_input():
    torch.manual_seed(3)
    N, D, K = 256, 16, 8
    x = torch.randn(N, D, device="cpu", dtype=torch.float16)

    labels, centroids = fkc.kmeans_largeN(
        x,
        K,
        max_iters=2,
        tol=0.0,
        BLOCK_N=64,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
    )
    pred = fkc.kmeans_largeN_assign(
        x,
        centroids,
        BLOCK_N=64,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
    )

    assert labels.shape == (N,)
    assert pred.shape == (N,)
    assert centroids.shape == (K, D)
    assert labels.device.type == "cuda"
    assert labels.dtype == torch.int32
    assert pred.dtype == torch.int32
    assert (pred >= 0).all() and (pred < K).all()


def test_triton_named_centroid_update_alias():
    torch.manual_seed(4)
    B, N, D, K = 1, 64, 8, 4
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    old = torch.randn(B, K, D, device="cuda", dtype=torch.float16)

    sums = torch.empty((B, K, D), device="cuda", dtype=torch.float32)
    counts = torch.empty((B, K), device="cuda", dtype=torch.int32)
    none_result = fkc.triton_centroid_update_sorted_euclid(
        x, ids, old, centroid_sums=sums, centroid_cnts=counts, calculate_new=False)
    new_centroids = fkc.triton_centroid_update_euclid(x, ids, old)

    assert none_result is None
    assert sums.shape == (B, K, D)
    assert counts.shape == (B, K)
    assert new_centroids.shape == (B, K, D)
    assert new_centroids.dtype == torch.float16
