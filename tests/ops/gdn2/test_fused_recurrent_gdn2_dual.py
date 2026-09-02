"""Dual benchmark for fused_recurrent_gdn2.

Run:
    python tests/ops/gdn2/test_fused_recurrent_gdn2_dual.py

Compares:
    test  : fla fused_recurrent_gdn2
    gt    : fp64 direct recurrent implementation
    bench : low precision eager recurrent reference
"""

import torch
import torch.nn.functional as F

from fla.ops.gdn2 import fused_recurrent_gdn2


def gdn2_reference(q, k, v, g, b, w):
    q, k, v, g, b, w = [x.double() for x in (q, k, v, g, b, w)]
    B, T, H, K = q.shape
    V = v.shape[-1]
    state = torch.zeros(B, H, K, V, dtype=torch.float64, device=q.device)
    out = []
    for t in range(T):
        kt = k[:, t]
        gt = torch.exp(g[:, t])
        erase = (b[:, t] * kt).unsqueeze(-1) * state
        state = state * gt.unsqueeze(-1) + kt.unsqueeze(-1) @ (w[:, t] * v[:, t]).unsqueeze(-2)
        state = state - erase
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state))
    return torch.stack(out, dim=1)


def dual_compare(test, gt, bench, name):
    e_test = torch.sqrt(torch.mean((test.double() - gt) ** 2))
    e_bench = torch.sqrt(torch.mean((bench.double() - gt) ** 2))
    ratio = (e_test / e_bench.clamp_min(1e-12)).item()
    print(f"{name}: test_rmse={e_test.item():.6e}, bench_rmse={e_bench.item():.6e}, ratio={ratio:.3f}")
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

    gt = gdn2_reference(q, k, v, g, b, w)
    bench = gdn2_reference(q, k, v, g, b, w).float()
    test, _ = fused_recurrent_gdn2(q, k, v, g, b, w, output_final_state=True)

    dual_compare(test, gt, bench, 'fused_recurrent_gdn2')


if __name__ == '__main__':
    main()
