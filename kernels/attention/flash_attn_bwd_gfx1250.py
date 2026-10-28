# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlashAttention backward (dQ, dK, dV) for gfx1250, head_dim 128: TDM staging + 16x16x32 WMMA.

Tensors follow the forward's conventions: Q/K/V/O/dO and the gradients are
contiguous ``[batch, seq, heads, 128]`` bf16/f16, and LSE is the forward's fp32
``[batch, heads, seq_q]`` natural-log logsumexp of the scaled scores. ``causal``
masks key ``j`` for query ``i`` when ``j > i + seq_k - seq_q`` (bottom-right
aligned, as in ``flash_attn_func``).

Three launches, FlashAttention-2 style without atomics:

* preprocess: ``Delta = rowsum(dO * O)`` and ``LSE * log2(e)``; a fully masked
  row's ``-inf`` LSE becomes ``+inf`` so its probabilities are exactly zero.
* dK/dV: each wave owns 16 keys and the workgroup walks the query blocks,
  keeping ``dV += P^T dO`` and ``dK += dS^T Q`` in registers.
* dQ: each wave owns 16 queries and the workgroup walks the key blocks,
  keeping ``dQ += dS K`` in registers.

``D = wmma(A, B)`` computes ``D[i][j] = sum_k A[i][k] * B[j][k]`` with lane ``l``
holding column ``j = l % 16`` and rows ``i = 8 * (l // 16) + v``. Two such
tiles stacked along ``i`` are exactly one K=32 operand fragment of row ``j``,
so P and dS feed the next WMMA straight from registers. The operand that
reduces over the sequence is read from the same row-major LDS tile with
``ds_load_tr16_b128``.
"""

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr, ReductionOp, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import check_smem_capacity
from kernels.common.kernels_common import LOG2E
from kernels.gemm.gemm_common_gfx1250 import make_lds_copy_ops

__all__ = [
    "flash_attn_bwd_gfx1250",
    "launch_flash_attn_bwd_preprocess_gfx1250",
    "launch_flash_attn_bwd_dkdv_gfx1250",
    "launch_flash_attn_bwd_dq_gfx1250",
]

HEAD_DIM = 128
WAVE = 32
WMMA_K = 32
EB = 2
ROW_B = HEAD_DIM * EB  # bytes per sequence row of one head
LDS_PAD = 16  # keeps consecutive LDS rows 4 dwords apart -> conflict-free b128 row reads
LDS_ROW = ROW_B + LDS_PAD
D_STEPS = HEAD_DIM // WMMA_K  # WMMA K-steps of a reduction over head_dim
D_TILES = HEAD_DIM // 16  # 16-wide output tiles along head_dim
PRE_ROWS = 16  # preprocess: 16 lanes x 8 elements per row
PRE_BLOCK = PRE_ROWS * 16
SEQ_PAD = 128  # Delta / LSE2 rows are padded so every block's TDM read starts 512-byte aligned
SMEM_ARCH = "gfx1250"
# Native v_exp_f32 only; nnan/ninf would let the compiler drop the +-inf LSE handling.
_AFN = fx.arith.FastMathFlags.afn


def _align(x, a):
    return (x + a - 1) // a * a


def _check_tiles(num_waves, seq_block):
    if num_waves not in (2, 4, 8):
        raise ValueError(f"num_waves must be 2, 4 or 8, got {num_waves}")
    if seq_block % WMMA_K or not 32 <= seq_block <= 128:
        raise ValueError(f"the streamed block must be a multiple of 32 in [32, 128], got {seq_block}")


def _wave_lanes():
    tid = fx.Int32(fx.thread_idx.x)
    wave = rocdl.readfirstlane(T.i32, tid // WAVE)
    lane = tid % WAVE
    return wave, lane % 16, lane // 16


def _gv(base, off, shape, stride):
    return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))


def _lv(ptr, shape, stride):
    return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))


def _lds_idx(p):
    return fx.Int64(fx.ptrtoint(p))


def _rows_tdm(gt, rows_left, row_stride, num_warps=1):
    """TDM atom for a [rows, 256 B] head slice; rows past ``rows_left`` load as zero / are not stored."""
    return fx.rocdl.make_tdm_atom(
        gt,
        [rows_left, None],
        strides=[row_stride, None],
        num_warps=num_warps,
        pad_interval=ROW_B,
        pad_amount=LDS_PAD,
    )


def _f32_ptr(arg):
    return fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 4), arg)


class _Frags:
    """LDS -> WMMA operand fragments for one wave (16 bf16 per lane, packed as 8 x i32)."""

    def __init__(self, elem_cls, lane16, kgrp):
        self.elem_cls = elem_cls
        self.load_b128, self.store_b128 = make_lds_copy_ops(128)
        # Row read: lane16 picks the row, each half-wave takes alternate 8-element chunks of K.
        self.row_lane = lane16 * LDS_ROW + kgrp * 16
        # Transposed read: an 8-lane group fetches an 8x8 block that ds_load_tr16_b128 transposes,
        # landing lane16 on the column and the K values on consecutive rows.
        self.tr_lane = (kgrp * 8 + lane16 % 8) * LDS_ROW + (lane16 // 8) * 16
        self.tr_ptr_ty = fx.PointerType.get(elem_cls.ir_type, fx.AddressSpace.Shared, 16)

    def row(self, base, row0, ks):
        """Rows ``row0 + lane16`` of a row-major tile, K-step ``ks`` along head_dim."""
        off = self.row_lane + (row0 * LDS_ROW + ks * WMMA_K * EB)
        v0 = Vec(self.load_b128(base, off))
        v1 = Vec(self.load_b128(base, off + 32))
        return v0.shuffle(v1, list(range(8)))

    def tr(self, base, row0, col0):
        """Columns ``col0 + lane16`` of a row-major tile, K = rows ``row0 .. row0 + 31``."""
        addr = fx.Int32(base) + self.tr_lane + (row0 * LDS_ROW + col0 * EB)
        halves = []
        for h in range_constexpr(2):
            ptr = fx.inttoptr(self.tr_ptr_ty, addr + h * 16 * LDS_ROW)
            raw = rocdl.ds_load_tr16_b128(T.vec(8, self.elem_cls.ir_type), ptr.llvm_ptr)
            halves.append(Vec(raw).bitcast(fx.Int32))
        return halves[0].shuffle(halves[1], list(range(8)))

    def f32x8(self, base, off):
        v0 = Vec(self.load_b128(base, off))
        v1 = Vec(self.load_b128(base, off + 16))
        return v0.shuffle(v1, list(range(8))).bitcast(fx.Float32)

    def pack(self, lo, hi):
        """Two f32 accumulator tiles stacked along K -> one operand fragment."""
        return lo.to(self.elem_cls).shuffle(hi.to(self.elem_cls), list(range(16))).bitcast(fx.Int32)


def _rmem(n, dtype, v):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v)
    return t


def _zeros8():
    return _rmem(8, fx.Float32, fx.full(8, 0.0, fx.Float32))


@flyc.jit
def launch_flash_attn_bwd_preprocess_gfx1250(
    arg_delta: fx.Pointer,
    arg_lse2: fx.Pointer,
    arg_o: fx.Pointer,
    arg_do: fx.Pointer,
    arg_lse: fx.Pointer,
    i32_rows: fx.Int32,
    i32_sq: fx.Int32,
    i32_heads: fx.Int32,
    i32_sq_pad: fx.Int32,
    stream: fx.Stream,
    is_f16: Constexpr[int] = 0,
):
    """Delta[b, h, s] = sum_d O * dO and LSE2 = LSE * log2(e), both fp32 [batch, heads, sq_pad].

    ``i32_rows`` = batch * seq_q * heads (the flattened BSHD rows of O / dO).
    """
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16

    @flyc.kernel(known_block_size=[PRE_BLOCK, 1, 1])
    def flash_attn_bwd_preprocess_gfx1250(
        arg_delta: fx.Pointer,
        arg_lse2: fx.Pointer,
        arg_o: fx.Pointer,
        arg_do: fx.Pointer,
        arg_lse: fx.Pointer,
        i32_rows: fx.Int32,
        i32_sq: fx.Int32,
        i32_heads: fx.Int32,
        i32_sq_pad: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        row0 = fx.Int32(fx.block_idx.x) * PRE_ROWS
        valid = fx.min(i32_rows - row0, PRE_ROWS)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_cls)
        thr = fx.make_tiled_copy_tv(
            atom, fx.make_layout((PRE_ROWS, 16), (16, 1)), fx.make_layout((1, 8), (1, 1))
        ).get_slice(tid)
        elem_ptr = fx.PointerType.get(elem_cls.ir_type, fx.AddressSpace.Global, 16)

        def _load(arg):
            base = fx.recast_iter(fx.Int8, arg) + fx.Int64(row0) * ROW_B
            view = fx.make_view(fx.recast_iter(elem_ptr, base), fx.make_layout((PRE_ROWS, HEAD_DIM), (HEAD_DIM, 1)))
            src = thr.partition_S(fx.rocdl.make_buffer_tensor(view, num_records_bytes=valid * ROW_B))
            frag = fx.make_fragment_like(src)
            fx.copy(atom, src, frag)
            return fx.memref_load_vec(frag).to(fx.Float32)

        dot = (_load(arg_o) * _load(arg_do)).reduce(ReductionOp.ADD)
        for sh in range_constexpr(4):
            dot = dot + fx.Float32(gpu.shuffle_xor(dot, 8 >> sh, WAVE))

        r = row0 + tid // 16
        if tid % 16 == 0:
            if r < i32_rows:
                s = (r // i32_heads) % i32_sq
                bh = (r // (i32_heads * i32_sq)) * i32_heads + r % i32_heads
                lse = fx.Float32(_f32_ptr(arg_lse)[fx.Int64(bh) * fx.Int64(i32_sq) + fx.Int64(s)])
                lse2 = (lse == -math.inf).select(math.inf, lse * LOG2E)
                out = fx.Int64(bh) * fx.Int64(i32_sq_pad) + fx.Int64(s)
                _f32_ptr(arg_delta)[out] = dot
                _f32_ptr(arg_lse2)[out] = fx.Float32(lse2)

    flash_attn_bwd_preprocess_gfx1250(
        arg_delta, arg_lse2, arg_o, arg_do, arg_lse, i32_rows, i32_sq, i32_heads, i32_sq_pad
    ).launch(grid=((i32_rows + (PRE_ROWS - 1)) // PRE_ROWS, 1, 1), block=(PRE_BLOCK, 1, 1), stream=stream)


@flyc.jit
def launch_flash_attn_bwd_dkdv_gfx1250(
    arg_dk: fx.Pointer,
    arg_dv: fx.Pointer,
    arg_q: fx.Pointer,
    arg_k: fx.Pointer,
    arg_v: fx.Pointer,
    arg_do: fx.Pointer,
    arg_lse2: fx.Pointer,
    arg_delta: fx.Pointer,
    i32_bh: fx.Int32,
    i32_sq: fx.Int32,
    i32_sk: fx.Int32,
    i32_heads: fx.Int32,
    i32_sq_pad: fx.Int32,
    f32_scale: fx.Float32,
    stream: fx.Stream,
    causal: Constexpr[int] = 0,
    block_m: Constexpr[int] = 64,
    num_waves: Constexpr[int] = 8,
    is_f16: Constexpr[int] = 0,
):
    """dK, dV for ``num_waves * 16`` keys per workgroup, streaming ``block_m`` queries per step."""
    _check_tiles(num_waves, block_m)
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16
    BLOCK_N = num_waves * 16
    QT = block_m // 16  # 16-query tiles per step
    QS = block_m // WMMA_K  # WMMA K-steps over the step's queries
    KV_B = BLOCK_N * LDS_ROW
    TILE_B = block_m * LDS_ROW
    STAGE_B = _align(2 * TILE_B + 2 * block_m * 4, 1024)
    OFF_STAGE = _align(2 * KV_B, 1024)
    ARENA_B = OFF_STAGE + 2 * STAGE_B
    check_smem_capacity(ARENA_B, SMEM_ARCH)

    @flyc.kernel(known_block_size=[num_waves * WAVE, 1, 1])
    def flash_attn_bwd_dkdv_gfx1250(
        arg_dk: fx.Pointer,
        arg_dv: fx.Pointer,
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_do: fx.Pointer,
        arg_lse2: fx.Pointer,
        arg_delta: fx.Pointer,
        i32_sq: fx.Int32,
        i32_sk: fx.Int32,
        i32_heads: fx.Int32,
        i32_sq_pad: fx.Int32,
        f32_scale: fx.Float32,
    ):
        wave, lane16, kgrp = _wave_lanes()
        bid_n, bid_bh, _ = fx.block_idx
        bh = fx.Int32(bid_bh)
        kv0 = fx.Int32(bid_n) * BLOCK_N
        row_b = fx.Int64(i32_heads) * ROW_B
        batch_off = (fx.Int64(bh // i32_heads) * fx.Int64(i32_heads) * ROW_B) * fx.Int64(i32_sq)
        head_off = fx.Int64(bh % i32_heads) * ROW_B
        q_off = batch_off + head_off
        k_off = (fx.Int64(bh // i32_heads) * fx.Int64(i32_sk) + fx.Int64(kv0)) * row_b + head_off
        stat_off = fx.Int64(bh) * fx.Int64(i32_sq_pad) * 4

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr
        k_lds = fx.add_offset(base_ptr, 0)
        v_lds = fx.add_offset(base_ptr, KV_B)
        frags = _Frags(elem_cls, lane16, kgrp)
        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, WMMA_K, elem_cls, fx.Float32))

        def _stage_ptr(s, off):
            return fx.add_offset(base_ptr, s * STAGE_B + (OFF_STAGE + off))

        gk, gv = fx.recast_iter(fx.Int8, arg_k), fx.recast_iter(fx.Int8, arg_v)
        kv_rows = i32_sk - kv0
        k_gt = _gv(gk, k_off, (BLOCK_N, ROW_B), (ROW_B, 1))
        v_gt = _gv(gv, k_off, (BLOCK_N, ROW_B), (ROW_B, 1))
        kv_jobs = [
            (_rows_tdm(k_gt, kv_rows, row_b), k_gt, k_lds),
            (_rows_tdm(v_gt, kv_rows, row_b), v_gt, v_lds),
        ]

        q_gt = _gv(fx.recast_iter(fx.Int8, arg_q), q_off, (block_m, ROW_B), (ROW_B, 1))
        do_gt = _gv(fx.recast_iter(fx.Int8, arg_do), q_off, (block_m, ROW_B), (ROW_B, 1))
        l_gt = _gv(fx.recast_iter(fx.Int8, arg_lse2), stat_off, (1, block_m * 4), (block_m * 4, 1))
        d_gt = _gv(fx.recast_iter(fx.Int8, arg_delta), stat_off, (1, block_m * 4), (block_m * 4, 1))
        q_jobs = [
            (_rows_tdm(q_gt, i32_sq, row_b), q_gt, 0, True),
            (_rows_tdm(do_gt, i32_sq, row_b), do_gt, TILE_B, True),
            (fx.rocdl.make_tdm_atom(l_gt, [1, i32_sq * 4], num_warps=1), l_gt, 2 * TILE_B, False),
            (fx.rocdl.make_tdm_atom(d_gt, [1, i32_sq * 4], num_warps=1), d_gt, 2 * TILE_B + block_m * 4, False),
        ]

        def issue(qb, s):
            q_start = fx.Int32(qb) * block_m
            for j in range_constexpr(len(q_jobs)):
                atom, gt, off, is_rows = q_jobs[j]
                if wave == (2 + j) % num_waves:
                    if const_expr(is_rows):
                        dst = _lv(_stage_ptr(s, off), (block_m, ROW_B), (LDS_ROW, 1))
                        fx.copy(atom, gt, dst, imm_offset=fx.Int64(q_start) * row_b, extent_0=i32_sq - q_start)
                    else:
                        dst = _lv(_stage_ptr(s, off), (1, block_m * 4), (block_m * 4, 1))
                        fx.copy(atom, gt, dst, imm_offset=fx.Int64(q_start) * 4, extent_1=(i32_sq - q_start) * 4)

        # Queries that see any of this block's keys, and the first one that sees all of them.
        n_qb = (i32_sq + (block_m - 1)) // block_m
        if const_expr(causal):
            offs = i32_sk - i32_sq
            q_lo = fx.min(fx.max(kv0 - offs, 0) // block_m, n_qb)
            q_mid = fx.max(fx.min((fx.max(kv0 + (BLOCK_N - 1) - offs, 0) + (block_m - 1)) // block_m, n_qb), q_lo)
        else:
            q_lo = fx.Int32(0)
            q_mid = fx.Int32(0)

        for j in range_constexpr(len(kv_jobs)):
            atom, gt, dst = kv_jobs[j]
            if wave == j % num_waves:
                fx.copy(atom, gt, _lv(dst, (BLOCK_N, ROW_B), (LDS_ROW, 1)))
        if q_lo < n_qb:
            issue(q_lo, 0)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        my_row = wave * 16
        kw = [_rmem(8, fx.Int32, frags.row(_lds_idx(k_lds), my_row, ks)) for ks in range_constexpr(D_STEPS)]
        vw = [_rmem(8, fx.Int32, frags.row(_lds_idx(v_lds), my_row, ks)) for ks in range_constexpr(D_STEPS)]
        dk = [_zeros8() for _ in range_constexpr(D_TILES)]
        dv = [_zeros8() for _ in range_constexpr(D_TILES)]
        scale_log2 = f32_scale * LOG2E
        # Causal: query q0 + 16 * qt + 8 * kgrp + v sees this lane's key iff 16 * qt + v >= k_thr.
        k_thr = (kv0 + my_row + lane16) - (i32_sk - i32_sq) - 8 * kgrp

        def step(qb, masked):
            s = (fx.Int32(qb) - q_lo) % 2
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            if fx.Int32(qb) + 1 < n_qb:
                issue(fx.Int32(qb) + 1, 1 - s)
            q_base = _lds_idx(_stage_ptr(s, 0))
            do_base = _lds_idx(_stage_ptr(s, TILE_B))
            l_base = _lds_idx(_stage_ptr(s, 2 * TILE_B))
            d_base = _lds_idx(_stage_ptr(s, 2 * TILE_B + block_m * 4))

            sc = [_zeros8() for _ in range_constexpr(QT)]
            dp = [_zeros8() for _ in range_constexpr(QT)]
            for ks in range_constexpr(D_STEPS):
                for qt in range_constexpr(QT):
                    fx.gemm(wmma, sc[qt], _rmem(8, fx.Int32, frags.row(q_base, qt * 16, ks)), kw[ks], sc[qt])
                    fx.gemm(wmma, dp[qt], _rmem(8, fx.Int32, frags.row(do_base, qt * 16, ks)), vw[ks], dp[qt])

            q_thr = k_thr - fx.Int32(qb) * block_m
            p_t, ds_t = [], []
            for qt in range_constexpr(QT):
                stat = (qt * 16) * 4 + kgrp * 32
                p = fx.exp2(sc[qt].load() * scale_log2 - frags.f32x8(l_base, stat), fastmath=_AFN)
                if const_expr(masked):
                    seen = Vec.from_elements([fx.Int32(qt * 16 + v) for v in range(8)], fx.Int32) >= q_thr
                    p = seen.select(p, 0.0)
                p_t.append(p)
                ds_t.append(p * (dp[qt].load() - frags.f32x8(d_base, stat)))
            pf = [_rmem(8, fx.Int32, frags.pack(p_t[2 * i], p_t[2 * i + 1])) for i in range_constexpr(QS)]
            dsf = [_rmem(8, fx.Int32, frags.pack(ds_t[2 * i], ds_t[2 * i + 1])) for i in range_constexpr(QS)]

            for qs in range_constexpr(QS):
                for dt in range_constexpr(D_TILES):
                    fx.gemm(wmma, dv[dt], _rmem(8, fx.Int32, frags.tr(do_base, qs * 32, dt * 16)), pf[qs], dv[dt])
                    fx.gemm(wmma, dk[dt], _rmem(8, fx.Int32, frags.tr(q_base, qs * 32, dt * 16)), dsf[qs], dk[dt])

        for qb in range(q_lo, q_mid):
            step(qb, True)
        for qb in range(q_mid, n_qb):
            step(qb, False)

        # dK/dV tiles reuse the K/V staging rows each wave read in the prologue.
        tdm_ops.tensor_wait(0)
        gpu.barrier()
        for dt in range_constexpr(D_TILES):
            col = (my_row + lane16) * LDS_ROW + (dt * 16 + kgrp * 8) * EB
            fx.ptr_store((dk[dt].load() * f32_scale).to(elem_cls).bitcast(fx.Int8), k_lds + col)
            fx.ptr_store(dv[dt].load().to(elem_cls).bitcast(fx.Int8), v_lds + col)
        gpu.barrier()
        for arg, src in ((arg_dk, k_lds), (arg_dv, v_lds)):
            gt = _gv(fx.recast_iter(fx.Int8, arg), k_off, (BLOCK_N, ROW_B), (ROW_B, 1))
            atom = _rows_tdm(gt, kv_rows, row_b, num_warps=num_waves)
            fx.copy(atom, _lv(src, (BLOCK_N, ROW_B), (LDS_ROW, 1)), gt)
        tdm_ops.tensor_wait(0)

    flash_attn_bwd_dkdv_gfx1250(
        arg_dk,
        arg_dv,
        arg_q,
        arg_k,
        arg_v,
        arg_do,
        arg_lse2,
        arg_delta,
        i32_sq,
        i32_sk,
        i32_heads,
        i32_sq_pad,
        f32_scale,
    ).launch(
        grid=((i32_sk + (BLOCK_N - 1)) // BLOCK_N, i32_bh, 1),
        block=(num_waves * WAVE, 1, 1),
        stream=stream,
    )


@flyc.jit
def launch_flash_attn_bwd_dq_gfx1250(
    arg_dq: fx.Pointer,
    arg_q: fx.Pointer,
    arg_k: fx.Pointer,
    arg_v: fx.Pointer,
    arg_do: fx.Pointer,
    arg_lse2: fx.Pointer,
    arg_delta: fx.Pointer,
    i32_bh: fx.Int32,
    i32_sq: fx.Int32,
    i32_sk: fx.Int32,
    i32_heads: fx.Int32,
    i32_sq_pad: fx.Int32,
    f32_scale: fx.Float32,
    stream: fx.Stream,
    causal: Constexpr[int] = 0,
    block_n: Constexpr[int] = 64,
    num_waves: Constexpr[int] = 8,
    is_f16: Constexpr[int] = 0,
):
    """dQ for ``num_waves * 16`` queries per workgroup, streaming ``block_n`` keys per step."""
    _check_tiles(num_waves, block_n)
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16
    BLOCK_M = num_waves * 16
    KT = block_n // 16  # 16-key tiles per step
    KS = block_n // WMMA_K  # WMMA K-steps over the step's keys
    Q_B = BLOCK_M * LDS_ROW
    TILE_B = block_n * LDS_ROW
    STAGE_B = _align(2 * TILE_B, 1024)
    OFF_STAGE = _align(2 * Q_B, 1024)
    ARENA_B = OFF_STAGE + 2 * STAGE_B
    check_smem_capacity(ARENA_B, SMEM_ARCH)

    @flyc.kernel(known_block_size=[num_waves * WAVE, 1, 1])
    def flash_attn_bwd_dq_gfx1250(
        arg_dq: fx.Pointer,
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_do: fx.Pointer,
        arg_lse2: fx.Pointer,
        arg_delta: fx.Pointer,
        i32_sq: fx.Int32,
        i32_sk: fx.Int32,
        i32_heads: fx.Int32,
        i32_sq_pad: fx.Int32,
        f32_scale: fx.Float32,
    ):
        wave, lane16, kgrp = _wave_lanes()
        bid_m, bid_bh, _ = fx.block_idx
        bh = fx.Int32(bid_bh)
        n_qb = (i32_sq + (BLOCK_M - 1)) // BLOCK_M
        # Causal blocks near the end of the sequence see the most keys: issue them first.
        q0 = (n_qb - 1 - fx.Int32(bid_m)) * BLOCK_M if const_expr(causal) else fx.Int32(bid_m) * BLOCK_M
        row_b = fx.Int64(i32_heads) * ROW_B
        head_off = fx.Int64(bh % i32_heads) * ROW_B
        q_off = (fx.Int64(bh // i32_heads) * fx.Int64(i32_sq) + fx.Int64(q0)) * row_b + head_off
        k_off = fx.Int64(bh // i32_heads) * fx.Int64(i32_sk) * row_b + head_off

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr
        q_lds = fx.add_offset(base_ptr, 0)
        do_lds = fx.add_offset(base_ptr, Q_B)
        frags = _Frags(elem_cls, lane16, kgrp)
        wmma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, WMMA_K, elem_cls, fx.Float32))

        def _stage_ptr(s, off):
            return fx.add_offset(base_ptr, s * STAGE_B + (OFF_STAGE + off))

        q_rows = i32_sq - q0
        q_gt = _gv(fx.recast_iter(fx.Int8, arg_q), q_off, (BLOCK_M, ROW_B), (ROW_B, 1))
        do_gt = _gv(fx.recast_iter(fx.Int8, arg_do), q_off, (BLOCK_M, ROW_B), (ROW_B, 1))
        q_jobs = [
            (_rows_tdm(q_gt, q_rows, row_b), q_gt, q_lds),
            (_rows_tdm(do_gt, q_rows, row_b), do_gt, do_lds),
        ]
        k_gt = _gv(fx.recast_iter(fx.Int8, arg_k), k_off, (block_n, ROW_B), (ROW_B, 1))
        v_gt = _gv(fx.recast_iter(fx.Int8, arg_v), k_off, (block_n, ROW_B), (ROW_B, 1))
        kv_jobs = [
            (_rows_tdm(k_gt, i32_sk, row_b), k_gt, 0),
            (_rows_tdm(v_gt, i32_sk, row_b), v_gt, TILE_B),
        ]

        def issue(kb, s):
            k_start = fx.Int32(kb) * block_n
            for j in range_constexpr(len(kv_jobs)):
                atom, gt, off = kv_jobs[j]
                if wave == (2 + j) % num_waves:
                    dst = _lv(_stage_ptr(s, off), (block_n, ROW_B), (LDS_ROW, 1))
                    fx.copy(atom, gt, dst, imm_offset=fx.Int64(k_start) * row_b, extent_0=i32_sk - k_start)

        # Key blocks this query block sees, and the first one that needs a mask (diagonal or tail).
        n_kb = (i32_sk + (block_n - 1)) // block_n
        if const_expr(causal):
            offs = i32_sk - i32_sq
            kv_hi = fx.min(fx.max(q0 + BLOCK_M + offs, 0) + (block_n - 1), fx.max(i32_sk, 0) + (block_n - 1))
            kv_hi = fx.min(kv_hi // block_n, n_kb)
            kv_mid = fx.min(fx.min(fx.max(q0 + offs + 1, 0) // block_n, i32_sk // block_n), kv_hi)
        else:
            kv_hi = n_kb
            kv_mid = i32_sk // block_n

        for j in range_constexpr(len(q_jobs)):
            atom, gt, dst = q_jobs[j]
            if wave == j % num_waves:
                fx.copy(atom, gt, _lv(dst, (BLOCK_M, ROW_B), (LDS_ROW, 1)))
        if 0 < kv_hi:
            issue(0, 0)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        my_row = wave * 16
        qw = [_rmem(8, fx.Int32, frags.row(_lds_idx(q_lds), my_row, ks)) for ks in range_constexpr(D_STEPS)]
        dow = [_rmem(8, fx.Int32, frags.row(_lds_idx(do_lds), my_row, ks)) for ks in range_constexpr(D_STEPS)]
        q_idx = q0 + my_row + lane16
        stat = fx.Int64(bh) * fx.Int64(i32_sq_pad) + fx.Int64(q_idx)
        lse2 = fx.Float32(_f32_ptr(arg_lse2)[stat])
        delta = fx.Float32(_f32_ptr(arg_delta)[stat])
        dq = [_zeros8() for _ in range_constexpr(D_TILES)]
        scale_log2 = f32_scale * LOG2E
        # Key kv0 + 16 * kt + 8 * kgrp + v is visible iff 16 * kt + v < k_lim + (offset of kv0).
        k_lim = i32_sk - 8 * kgrp
        if const_expr(causal):
            k_lim = fx.min(k_lim, q_idx + (i32_sk - i32_sq) + 1 - 8 * kgrp)

        def step(kb, masked):
            s = fx.Int32(kb) % 2
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            if fx.Int32(kb) + 1 < kv_hi:
                issue(fx.Int32(kb) + 1, 1 - s)
            k_base = _lds_idx(_stage_ptr(s, 0))
            v_base = _lds_idx(_stage_ptr(s, TILE_B))

            sc = [_zeros8() for _ in range_constexpr(KT)]
            dp = [_zeros8() for _ in range_constexpr(KT)]
            for ks in range_constexpr(D_STEPS):
                for kt in range_constexpr(KT):
                    fx.gemm(wmma, sc[kt], _rmem(8, fx.Int32, frags.row(k_base, kt * 16, ks)), qw[ks], sc[kt])
                    fx.gemm(wmma, dp[kt], _rmem(8, fx.Int32, frags.row(v_base, kt * 16, ks)), dow[ks], dp[kt])

            lim = k_lim - fx.Int32(kb) * block_n
            ds_t = []
            for kt in range_constexpr(KT):
                p = fx.exp2(sc[kt].load() * scale_log2 - lse2, fastmath=_AFN)
                if const_expr(masked):
                    seen = Vec.from_elements([fx.Int32(kt * 16 + v) for v in range(8)], fx.Int32) < lim
                    p = seen.select(p, 0.0)
                ds_t.append(p * (dp[kt].load() - delta))
            dsf = [_rmem(8, fx.Int32, frags.pack(ds_t[2 * i], ds_t[2 * i + 1])) for i in range_constexpr(KS)]

            for ks in range_constexpr(KS):
                for dt in range_constexpr(D_TILES):
                    fx.gemm(wmma, dq[dt], _rmem(8, fx.Int32, frags.tr(k_base, ks * 32, dt * 16)), dsf[ks], dq[dt])

        for kb in range(0, kv_mid):
            step(kb, False)
        for kb in range(kv_mid, kv_hi):
            step(kb, True)

        # dQ tiles reuse the Q staging rows each wave read in the prologue.
        tdm_ops.tensor_wait(0)
        gpu.barrier()
        for dt in range_constexpr(D_TILES):
            col = (my_row + lane16) * LDS_ROW + (dt * 16 + kgrp * 8) * EB
            fx.ptr_store((dq[dt].load() * f32_scale).to(elem_cls).bitcast(fx.Int8), q_lds + col)
        gpu.barrier()
        gt = _gv(fx.recast_iter(fx.Int8, arg_dq), q_off, (BLOCK_M, ROW_B), (ROW_B, 1))
        fx.copy(_rows_tdm(gt, q_rows, row_b, num_warps=num_waves), _lv(q_lds, (BLOCK_M, ROW_B), (LDS_ROW, 1)), gt)
        tdm_ops.tensor_wait(0)

    flash_attn_bwd_dq_gfx1250(
        arg_dq, arg_q, arg_k, arg_v, arg_do, arg_lse2, arg_delta, i32_sq, i32_sk, i32_heads, i32_sq_pad, f32_scale
    ).launch(
        grid=((i32_sq + (BLOCK_M - 1)) // BLOCK_M, i32_bh, 1),
        block=(num_waves * WAVE, 1, 1),
        stream=stream,
    )


def flash_attn_bwd_gfx1250(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    *,
    causal=False,
    softmax_scale=None,
    dq=None,
    dk=None,
    dv=None,
    block_m=64,
    block_n=64,
    num_waves=8,
    stream=None,
):
    """FlashAttention backward on gfx1250. Returns ``(dq, dk, dv)``.

    ``q``/``out``/``dout`` are ``[batch, seq_q, heads, 128]``, ``k``/``v`` are
    ``[batch, seq_k, heads, 128]``, all contiguous and of one dtype (bf16 or f16).
    ``softmax_lse`` is the forward's fp32 ``[batch, heads, seq_q]`` natural-log LSE.
    ``block_m`` is the query block the dK/dV kernel streams and ``block_n`` the key
    block the dQ kernel streams; each owns ``num_waves * 16`` rows per workgroup.
    """
    if q.dim() != 4 or q.shape[-1] != HEAD_DIM:
        raise ValueError(f"q must be [batch, seq, heads, {HEAD_DIM}], got {tuple(q.shape)}")
    batch, seq_q, heads, _ = q.shape
    seq_k = k.shape[1]
    if k.shape != (batch, seq_k, heads, HEAD_DIM) or v.shape != k.shape:
        raise ValueError(f"k/v must be [{batch}, seq_k, {heads}, {HEAD_DIM}], got {tuple(k.shape)} / {tuple(v.shape)}")
    if out.shape != q.shape or dout.shape != q.shape:
        raise ValueError("out and dout must match q's shape")
    if softmax_lse.shape != (batch, heads, seq_q) or softmax_lse.dtype != torch.float32:
        raise ValueError(f"softmax_lse must be fp32 [{batch}, {heads}, {seq_q}]")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(t.dtype != q.dtype for t in (k, v, out, dout)):
        raise ValueError("q, k, v, out and dout must share one dtype: bf16 or f16")
    if seq_q < 1 or seq_k < 1:
        raise ValueError("seq_q and seq_k must be positive")
    tensors = (dout, q, k, v, out, softmax_lse)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("flash_attn_bwd_gfx1250 expects contiguous inputs")
    _check_tiles(num_waves, block_m)
    _check_tiles(num_waves, block_n)

    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv
    for name, grad, ref in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
        if grad.shape != ref.shape or grad.dtype != ref.dtype or not grad.is_contiguous():
            raise ValueError(f"{name} must be a contiguous {ref.dtype} tensor of shape {tuple(ref.shape)}")

    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    seq_q_pad = _align(seq_q, SEQ_PAD)
    delta = torch.zeros(batch, heads, seq_q_pad, dtype=torch.float32, device=q.device)
    lse2 = torch.zeros_like(delta)
    is_f16 = int(q.dtype == torch.float16)
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream

    def ptr(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    launch_flash_attn_bwd_preprocess_gfx1250(
        ptr(delta),
        ptr(lse2),
        ptr(out),
        ptr(dout),
        ptr(softmax_lse),
        batch * seq_q * heads,
        seq_q,
        heads,
        seq_q_pad,
        launch_stream,
        is_f16,
    )
    shared = (ptr(q), ptr(k), ptr(v), ptr(dout), ptr(lse2), ptr(delta), batch * heads, seq_q, seq_k, heads, seq_q_pad)
    launch_flash_attn_bwd_dkdv_gfx1250(
        ptr(dk), ptr(dv), *shared, scale, launch_stream, int(causal), block_m, num_waves, is_f16
    )
    launch_flash_attn_bwd_dq_gfx1250(ptr(dq), *shared, scale, launch_stream, int(causal), block_n, num_waves, is_f16)
    return dq, dk, dv
