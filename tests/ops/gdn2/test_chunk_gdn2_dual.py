"""Dual benchmark for chunk_gdn2.

Run:
    python tests/ops/gdn2/test_chunk_gdn2_dual.py
"""

import torch

from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.naive import naive_recurrent_gdn2


def fp64_gt(q, k, v, g, b, w):
    q, k, v, g, b, w = [x.double() for x in (q, k, v, g, b, w)]
    out, _ = naive_recurrent_gdn2(q, k, v, g, b, w)
    return out.double()


def dual_compare(test, gt, bench):
    test_err = torch.sqrt(torch.mean((test.double() - gt) ** 2))
    bench_err = torch.sqrt(torch.mean((bench.double() - gt) ** 2))
    ratio = (test_err / bench_err.clamp_min(1e-12)).item()
    print(
        f"chunk_gdn2: test_rmse={test_err.item():.6e}, "
        f"bench_rmse={bench_err.item():.6e}, ratio={ratio:.3f}"
    )
    assert ratio < 5.0


def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    B, T, H, K, V = 1, 128, 2, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = torch.empty(B, T, H, K, device=device, dtype=dtype).uniform_(-4, -0.1)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)

    gt = fp64_gt(q, k, v, g, b, w)
    bench, _ = naive_recurrent_gdn2(q, k, v, g, b, w)
    test, _ = chunk_gdn2(q, k, v, g, b, w, output_final_state=True)

    dual_compare(test, gt, bench)


if __name__ == '__main__':
    main()
