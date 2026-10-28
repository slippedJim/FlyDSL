#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile the gfx1250 FlashAttention backward launchers to gfx1250 ISA without a device."""

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

B, SQ, SK, H = 2, 300, 200, 3
SQ_PAD = 384


@pytest.fixture
def gfx1250_compile_only(monkeypatch):
    monkeypatch.setenv("ARCH", "gfx1250")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def _ptrs(n):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return [flyc.from_c_void_p(fx.Uint8, 0x1000 * (i + 1)) for i in range(n)]


def _compile(launch, *args):
    import flydsl.compiler as flyc

    flyc.compile(launch, *args)
    assert launch._last_compiled is not None


@pytest.mark.parametrize("is_f16", [0, 1])
def test_preprocess_compiles(gfx1250_compile_only, is_f16):
    import flydsl.expr as fx
    from kernels.attention.flash_attn_bwd_gfx1250 import launch_flash_attn_bwd_preprocess_gfx1250 as launch

    _compile(launch, *_ptrs(5), B * SQ * H, SQ, H, SQ_PAD, fx.Stream(None), is_f16)


@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("block, num_waves, is_f16", [(64, 8, 0), (32, 4, 1), (128, 2, 0)])
def test_dkdv_compiles(gfx1250_compile_only, causal, block, num_waves, is_f16):
    import flydsl.expr as fx
    from kernels.attention.flash_attn_bwd_gfx1250 import launch_flash_attn_bwd_dkdv_gfx1250 as launch

    args = (*_ptrs(8), B * H, SQ, SK, H, SQ_PAD, 0.0883883, fx.Stream(None), causal, block, num_waves, is_f16)
    _compile(launch, *args)


@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("block, num_waves, is_f16", [(64, 8, 0), (32, 4, 1), (128, 2, 0)])
def test_dq_compiles(gfx1250_compile_only, causal, block, num_waves, is_f16):
    import flydsl.expr as fx
    from kernels.attention.flash_attn_bwd_gfx1250 import launch_flash_attn_bwd_dq_gfx1250 as launch

    args = (*_ptrs(7), B * H, SQ, SK, H, SQ_PAD, 0.0883883, fx.Stream(None), causal, block, num_waves, is_f16)
    _compile(launch, *args)


@pytest.mark.parametrize("num_waves, block", [(3, 64), (8, 48), (8, 256)])
def test_rejects_bad_tiles(num_waves, block):
    from kernels.attention.flash_attn_bwd_gfx1250 import _check_tiles

    with pytest.raises(ValueError):
        _check_tiles(num_waves, block)
