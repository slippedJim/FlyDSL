#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlashAttention backward (head_dim 128) tests for gfx1250 against an fp32 reference."""

import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_bwd_gfx1250 import flash_attn_bwd_gfx1250  # noqa: E402

HEAD_DIM = 128
_DT = {"bf16": torch.bfloat16, "f16": torch.float16}


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def reference_attention_bwd(q, k, v, dout, causal, scale):
    """fp32 forward + closed-form backward on BSHD inputs -> (out, lse, dq, dk, dv).

    A fully masked row (causal, seq_q > seq_k) has zero probabilities, so its
    output and its gradient contributions are zero; autograd through
    ``exp(s - lse)`` would turn those rows into NaN instead.
    """
    q, k, v, dout = (t.float() for t in (q, k, v, dout))
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        rows = torch.arange(sq, device=q.device)[:, None]
        cols = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(cols > rows + (sk - sq), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).nan_to_num(0.0)
    out = torch.einsum("bhqk,bkhd->bqhd", p, v)
    dv = torch.einsum("bhqk,bqhd->bkhd", p, dout)
    dp = torch.einsum("bqhd,bkhd->bhqk", dout, v)
    delta = (dout * out).sum(-1).transpose(1, 2)
    ds = p * (dp - delta[..., None])
    dq = torch.einsum("bhqk,bkhd->bqhd", ds, k) * scale
    dk = torch.einsum("bhqk,bqhd->bkhd", ds, q) * scale
    return out, lse, dq, dk, dv


def _make_case(batch, seq_q, seq_k, heads, causal, dtype, scale, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(dtype)

    q, k, v = (
        rand(batch, seq_q, heads, HEAD_DIM),
        rand(batch, seq_k, heads, HEAD_DIM),
        rand(batch, seq_k, heads, HEAD_DIM),
    )
    dout = rand(batch, seq_q, heads, HEAD_DIM)
    out, lse, dq, dk, dv = reference_attention_bwd(q, k, v, dout, causal, scale)
    return (q, k, v, out.to(dtype), dout, lse.contiguous()), (dq, dk, dv)


def _rel_err(x, ref):
    return float((x.float() - ref).norm() / ref.norm().clamp_min(1e-6))


def _run_case(batch, seq_q, seq_k, heads, causal, dtype="bf16", **kw):
    _require_gfx1250()
    scale = 1.0 / math.sqrt(HEAD_DIM)
    (q, k, v, out, dout, lse), refs = _make_case(batch, seq_q, seq_k, heads, causal, _DT[dtype], scale)
    grads = flash_attn_bwd_gfx1250(dout, q, k, v, out, lse, causal=causal, softmax_scale=scale, **kw)
    torch.cuda.synchronize()
    tol = 2e-2 if dtype == "bf16" else 1e-2
    for name, g, ref in zip(("dq", "dk", "dv"), grads, refs):
        assert torch.isfinite(g).all(), f"{name} has non-finite values"
        err = _rel_err(g, ref)
        print(f"  {name}: rel_err={err:.3e} max_abs={float((g.float() - ref).abs().max()):.3e}")
        assert err < tol, f"{name} rel_err {err:.3e} >= {tol}"


# (batch, seq_q, seq_k, heads)
_SHAPES = [
    (1, 128, 128, 1),  # one key block, two query blocks
    (2, 256, 256, 3),
    (1, 300, 300, 2),  # ragged tails on every block size
    (2, 200, 333, 2),  # seq_q < seq_k
    (1, 333, 130, 2),  # seq_q > seq_k: causal leaves leading rows fully masked
    (1, 1, 77, 1),  # a single query row
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch, seq_q, seq_k, heads", _SHAPES)
def test_flash_attn_bwd(batch, seq_q, seq_k, heads, causal):
    _run_case(batch, seq_q, seq_k, heads, causal)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_bwd_f16(causal):
    _run_case(2, 256, 256, 2, causal, dtype="f16")


@pytest.mark.parametrize("block_m, block_n, num_waves", [(32, 32, 4), (128, 128, 2), (64, 128, 4)])
def test_flash_attn_bwd_tiles(block_m, block_n, num_waves):
    _run_case(1, 384, 320, 2, True, block_m=block_m, block_n=block_n, num_waves=num_waves)


@pytest.mark.large_shape
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_bwd_large(causal):
    _run_case(2, 2048, 2048, 8, causal)


def _bench_us(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Manual correctness/perf run for the gfx1250 FlashAttention backward")
    parser.add_argument("-b", type=int, default=2)
    parser.add_argument("-sq", type=int, default=4096)
    parser.add_argument("-sk", type=int, default=None)
    parser.add_argument("-heads", type=int, default=16)
    parser.add_argument("-dtype", default="bf16", choices=sorted(_DT))
    parser.add_argument("-causal", action="store_true")
    parser.add_argument("-block_m", type=int, default=64)
    parser.add_argument("-block_n", type=int, default=64)
    parser.add_argument("-num_waves", type=int, default=8)
    parser.add_argument("-bench", action="store_true")
    args = parser.parse_args()
    sk = args.sq if args.sk is None else args.sk
    kw = dict(block_m=args.block_m, block_n=args.block_n, num_waves=args.num_waves)
    _run_case(args.b, args.sq, sk, args.heads, args.causal, dtype=args.dtype, **kw)
    if args.bench:
        scale = 1.0 / math.sqrt(HEAD_DIM)
        (q, k, v, out, dout, lse), _ = _make_case(args.b, args.sq, sk, args.heads, args.causal, _DT[args.dtype], scale)
        us = _bench_us(lambda: flash_attn_bwd_gfx1250(dout, q, k, v, out, lse, causal=args.causal, **kw))
        # 2.5x the forward's 4 * B * H * Sq * Sk * D, halved for causal.
        flops = 2.5 * 4 * args.b * args.heads * args.sq * sk * HEAD_DIM * (0.5 if args.causal else 1.0)
        print(f"latency {us:.1f} us, {flops / (us * 1e-6) / 1e12:.1f} TFLOPS")


if __name__ == "__main__":
    _main()
