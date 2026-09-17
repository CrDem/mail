"""
AOT-компиляция Triton-кернела ДО TTIR (без NPU/GPU, только CPU),
плюс отдельная точка вызова backend-компилятора (ttir -> linalg).

Версия с "плоским" циклом: все S1-блоки одного ядра обрабатываются одним
длинным циклом по S2-шагам, чтобы dynamic CV pipeline строил один конвейер
на ядро (пролог/эпилог конвейера один раз, соседние S1-блоки перекрываются
на глубину конвейера, как в Process() NPU-реализации FA).

KERNEL = "orig"  -> _vdv_atn_fwd       (исходный, конвейер на каждый S1-блок) — РАБОЧИЙ, ~0.15 c
KERNEL = "flat"  -> _vdv_atn_fwd_flat  (эксперимент, ~0.223 c — медленнее, см. ниже)
KERNEL = "l1buf" -> _vdv_atn_fwd_l1buf (идея 2: плоский цикл, Q в явном L1-буфере раз на блок;
                                        ssbuf отключается сам из-за scope; CV-разбиение делает bishengir)
KERNEL = "cv"    -> _vdv_atn_fwd_cv    (идея 3: ручной CV-кернел как NPU Process — два scope,
                                        сдвиг стадий C1(t)/V1(t-1)/C2(t-2)/V2(t-3), токены на буферы,
                                        слив 3 итерации на ядро). Компилировать с
                                        disable_auto_inject_block_sync=True, enable_mixed_cv=True,
                                        set_workspace_multibuffer=0 (как ssbuf компилирует свой вывод).

Почему flat медленнее: dynamic CV pipeline не умеет держать Q в L1 весь блок при
плоском цикле. Любая загрузка Q, которая перезагружается внутри цикла и переносится
через итерации (if s2==0 / перезагрузка на последнем шаге), классифицируется как
VECTOR (OpClassifier::patternMatchCUBE не проходит через результат scf.if), и Vector
каждый шаг копирует Q в L1. Безусловная загрузка на Cube = +1 копия 32KB из GM на
каждый S2-шаг: GM-трафик Cube 96KB/шаг вместо 64KB (x1.5, совпадает с 0.223/0.15).
Экономия на прологе/эпилоге конвейера ~3/64 блока, на порядок меньше этой потери.
Main loop прохода — всегда самый внутренний цикл с межъядерными передачами
(MarkMainLoop), поэтому при двухуровневых циклах конвейер строится на внутреннем.
"""

import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget

from triton.backends.ascend.compiler import (
    AscendBackend,
    make_ttir,
    ttir_to_linalg,
)

KERNEL = "orig"


# =============================================================================
# Исходный кернел (без изменений) — для A/B сравнения
# =============================================================================
@triton.jit
def _vdv_atn_fwd_inner_opt(acc, l_i, m_i, q, #
                    K_block_ptr, V_block_ptr, ATTEN_MASK, #
                    stride_am, start_m, qk_scale: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    lo, hi = 0, N_CTX
    q_type : tl.constexpr = q.type
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in tl.range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

        qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p = p.cast(q_type)
        v = tl.load(V_block_ptr)
        pv = tl.dot(p, v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv
    return acc, l_i, m_i


@triton.jit
def _vdv_atn_fwd(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr,
              H: tl.constexpr,
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
    start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
    for block_idx in tl.range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
            offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_kn, stride_kk),
            offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(Q_block_ptr)
        acc, l_i, m_i = _vdv_atn_fwd_inner_opt(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                        stride_am, task_m_idx, sm_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# =============================================================================
# Новый кернел: один плоский цикл на ядро
# =============================================================================
@triton.jit
def _vdv_atn_fwd_flat(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr,
              H: tl.constexpr,
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
    NUM_S2_STEPS = N_CTX // BLOCK_N  # S2-шагов на один S1-блок (64)
    LAST_S2_STEP = NUM_S2_STEPS - 1

    # S1-блоки этого ядра: pid, pid + AICORE_NUM, pid + 2*AICORE_NUM, ... < NUM_BLOCKS
    num_tasks = (NUM_BLOCKS - pid + AICORE_NUM - 1) // AICORE_NUM
    total_steps = num_tasks * NUM_S2_STEPS

    # Состояние S1-блока. Отдельный сброс l_i/acc не нужен: на первом S2-шаге
    # блока m_i = -inf => alpha = exp(-inf - m_ij) = 0 => l_i = l_ij, acc = pv.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for step_idx in tl.range(0, total_steps, 1):
        task = step_idx // NUM_S2_STEPS
        s2_idx = step_idx % NUM_S2_STEPS

        block_idx = pid + task * AICORE_NUM
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        # ---- Q грузится на каждом S2-шаге.
        # Условная загрузка (if s2_idx == 0) с переносом q через итерации
        # классифицируется pipeline-проходом как VECTOR: Vector каждый шаг
        # перекладывает Q в NZ и копирует в L1, а Cube ждёт Vector перед mm1.
        # Безусловный load классифицируется как CUBE (как K).
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
        q = tl.load(Q_block_ptr)

        # ---- mm1
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_kn, stride_kk),
            offsets=(s2_idx * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k))

        # ---- softmax (сброс состояния на первом S2-шаге блока)
        # Без tl.where: where со скалярным условием опускается в extract/insert_slice
        # с динамической формой, и dynamic CV pipeline падает (AllocMultiCache).
        # Прибавляем огромный отрицательный сдвиг только на s2_idx == 0:
        # m_ij = max(qk), alpha = exp(-3e38 - m_ij) = 0 => l_i = l_ij, acc = pv.
        is_block_start = (s2_idx == 0).to(tl.float32)
        m_i = m_i + is_block_start * -3.0e38
        qk = qk * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p = p.cast(q.type)

        # ---- mm2
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
            offsets=(s2_idx * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        v = tl.load(V_block_ptr)
        pv = tl.dot(p, v)

        # ---- update
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv

        # ---- эпилог S1-блока: деление на sum и запись строк в Out
        if s2_idx == LAST_S2_STEP:
            O_block_ptr = tl.make_block_ptr(
                base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
                offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
            tl.store(O_block_ptr, (acc / l_i[:, None]).to(Out.type.element_ty))


# =============================================================================
# Идея 2: плоский цикл, Q блока в явном L1-буфере (запись раз на блок через bind_buffer)
# =============================================================================
@triton.jit
def _vdv_atn_fwd_l1buf(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr,
              H: tl.constexpr,
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
    NUM_S2_STEPS = N_CTX // BLOCK_N
    LAST_S2_STEP = NUM_S2_STEPS - 1
    num_tasks = (NUM_BLOCKS - pid + AICORE_NUM - 1) // AICORE_NUM
    total_steps = num_tasks * NUM_S2_STEPS

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Q блока живёт в явном L1-буфере: пишем в него раз на блок, читаем каждый шаг.
    q_l1 = bl.alloc(tl.float16, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)

    for step_idx in tl.range(0, total_steps, 1):
        task = step_idx // NUM_S2_STEPS
        s2_idx = step_idx % NUM_S2_STEPS
        block_idx = pid + task * AICORE_NUM
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        qvk_offset = (task_hz_idx // H).to(tl.int64) * stride_qz + (task_hz_idx % H).to(tl.int64) * stride_qh

        # Загрузка Q явно на Cube. Любой scope.scope отключает ssbuf (PreCheckBlacklist):
        # с ssbuf этот вариант некорректен (запись в q_l1 уходит на Vector в отдельный буфер).
        with al.scope(core_mode="cube"):
            if s2_idx == 0:
                Q_block_ptr = tl.make_block_ptr(
                    base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
                    offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
                q_new = tl.load(Q_block_ptr)
                bl.to_buffer(tensor=q_new, bind_buffer=q_l1)
        q = bl.to_tensor(q_l1)

        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_kn, stride_kk),
            offsets=(s2_idx * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k))

        is_block_start = (s2_idx == 0).to(tl.float32)
        m_i = m_i + is_block_start * -3.0e38
        qk = qk * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        p = p.cast(tl.float16)

        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
            offsets=(s2_idx * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
        v = tl.load(V_block_ptr)
        pv = tl.dot(p, v)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + pv

        if s2_idx == LAST_S2_STEP:
            O_block_ptr = tl.make_block_ptr(
                base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
                offsets=(task_m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
            tl.store(O_block_ptr, (acc / l_i[:, None]).to(Out.type.element_ty))


# =============================================================================
# Идея 3: ручной CV-кернел (как NPU FlashAttentionScoreKernelTrain::Process)
# Cube:   итерация t -> mm1(t), mm2(t-2);  Vector: итерация t -> vec1(t-1), vec2(t-3)
# Буферы и токены (флаги): P в L1 — 1/4, qk в UB — 2/5, pv в UB — 3/6 (пинг-понг по чётности шага).
# Протокол повторяет рабочий вывод ssbuf для исходного кернела, но один конвейер на ядро.
# =============================================================================
PIPE = al.PIPE


@triton.jit
def _vdv_atn_fwd_cv(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr,
              Z: tl.constexpr,
              H: tl.constexpr,
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
    NUM_S2_STEPS: tl.constexpr = N_CTX // BLOCK_N
    LAST_S2_STEP: tl.constexpr = N_CTX // BLOCK_N - 1
    M1: tl.constexpr = BLOCK_M // 16
    K1: tl.constexpr = BLOCK_N // 16
    num_tasks = (NUM_BLOCKS - pid + AICORE_NUM - 1) // AICORE_NUM
    total_steps = num_tasks * NUM_S2_STEPS
    n_iters = total_steps + 3

    # Межъядерные буферы: по два (пинг-понг по чётности S2-шага), как в DB-политиках NPU.
    qk_ub0 = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    qk_ub1 = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    pv_ub0 = bl.alloc(tl.float32, [BLOCK_M, HEAD_DIM], al.ascend_address_space.UB)
    pv_ub1 = bl.alloc(tl.float32, [BLOCK_M, HEAD_DIM], al.ascend_address_space.UB)
    p_l1_0 = bl.alloc(tl.float16, [K1, M1, 16, 16], al.ascend_address_space.L1)
    p_l1_1 = bl.alloc(tl.float16, [K1, M1, 16, 16], al.ascend_address_space.L1)

    # ============================ CUBE ============================
    # итерация t: mm1(t), mm2(t-2)
    with al.scope(core_mode="cube"):
        # Q блока: два L1-буфера по чётности блока (Q блока k+1 грузится, пока (k,63) ещё в конвейере)
        q_l1_0 = bl.alloc(tl.float16, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)
        q_l1_1 = bl.alloc(tl.float16, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)
        # токены L1-буферов P свободны
        al.sync_block_set("cube", "vector", 1, PIPE.PIPE_M, PIPE.PIPE_MTE3)
        al.sync_block_set("cube", "vector", 4, PIPE.PIPE_M, PIPE.PIPE_MTE3)
        for t in tl.range(0, n_iters, 1):
            # ---------------- mm1(s = t) ----------------
            s = t
            if s < total_steps:
                task = s // NUM_S2_STEPS
                s2 = s % NUM_S2_STEPS
                block_idx = pid + task * AICORE_NUM
                hz = block_idx // NUM_BLOCKS_M
                qvk_offset = (hz // H).to(tl.int64) * stride_qz + (hz % H).to(tl.int64) * stride_qh
                if s2 == 0:
                    Q_ptr = tl.make_block_ptr(
                        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk),
                        offsets=((block_idx % NUM_BLOCKS_M) * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM),
                        order=(1, 0))
                    if task % 2 == 0:
                        q_new0 = tl.load(Q_ptr)
                        bl.to_buffer(tensor=q_new0, bind_buffer=q_l1_0)
                    else:
                        q_new1 = tl.load(Q_ptr)
                        bl.to_buffer(tensor=q_new1, bind_buffer=q_l1_1)
                if task % 2 == 0:
                    q = bl.to_tensor(q_l1_0)
                else:
                    q = bl.to_tensor(q_l1_1)
                K_ptr = tl.make_block_ptr(
                    base=K + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_kn, stride_kk),
                    offsets=(s2 * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
                k = tl.load(K_ptr)
                qk = tl.dot(q, tl.trans(k))
                if s % 2 == 0:
                    al.sync_block_wait("vector", "cube", 2, PIPE.PIPE_V, PIPE.PIPE_FIX)
                    al.fixpipe(qk, qk_ub0)
                    al.sync_block_set("cube", "vector", 2, PIPE.PIPE_FIX, PIPE.PIPE_V)
                else:
                    al.sync_block_wait("vector", "cube", 5, PIPE.PIPE_V, PIPE.PIPE_FIX)
                    al.fixpipe(qk, qk_ub1)
                    al.sync_block_set("cube", "vector", 5, PIPE.PIPE_FIX, PIPE.PIPE_V)
            # ---------------- mm2(s = t - 2) ----------------
            s = t - 2
            if (s >= 0) & (s < total_steps):
                task = s // NUM_S2_STEPS
                s2 = s % NUM_S2_STEPS
                block_idx = pid + task * AICORE_NUM
                hz = block_idx // NUM_BLOCKS_M
                qvk_offset = (hz // H).to(tl.int64) * stride_qz + (hz % H).to(tl.int64) * stride_qh
                if s % 2 == 0:
                    al.sync_block_wait("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE1)
                    pf = bl.to_tensor(p_l1_0)
                else:
                    al.sync_block_wait("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE1)
                    pf = bl.to_tensor(p_l1_1)
                V_ptr = tl.make_block_ptr(
                    base=V + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk),
                    offsets=(s2 * BLOCK_N, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))
                v = tl.load(V_ptr)
                pv = al.dot(pf, v, format_a="fractal", format_b="nd", format_c="nd")
                if s % 2 == 0:
                    al.sync_block_wait("vector", "cube", 3, PIPE.PIPE_V, PIPE.PIPE_FIX)
                    al.fixpipe(pv, pv_ub0)
                    al.sync_block_set("cube", "vector", 3, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    al.sync_block_set("cube", "vector", 1, PIPE.PIPE_M, PIPE.PIPE_MTE3)
                else:
                    al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_V, PIPE.PIPE_FIX)
                    al.fixpipe(pv, pv_ub1)
                    al.sync_block_set("cube", "vector", 6, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    al.sync_block_set("cube", "vector", 4, PIPE.PIPE_M, PIPE.PIPE_MTE3)
        # слив: дождаться возврата токенов UB-буферов
        al.sync_block_wait("vector", "cube", 2, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_wait("vector", "cube", 5, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_wait("vector", "cube", 3, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_wait("vector", "cube", 6, PIPE.PIPE_V, PIPE.PIPE_FIX)

    # =========================== VECTOR ===========================
    # итерация t: vec1(t-1), vec2(t-3)
    with al.scope(core_mode="vector"):
        # токены UB-буферов свободны
        al.sync_block_set("vector", "cube", 2, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_set("vector", "cube", 5, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_set("vector", "cube", 3, PIPE.PIPE_V, PIPE.PIPE_FIX)
        al.sync_block_set("vector", "cube", 6, PIPE.PIPE_V, PIPE.PIPE_FIX)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        zero_row = tl.zeros([BLOCK_M], dtype=tl.float32)
        # сдвиговые регистры: alpha и l_i шага s нужны vec2 через 2 итерации
        a0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        a1 = tl.zeros([BLOCK_M], dtype=tl.float32)
        a2 = tl.zeros([BLOCK_M], dtype=tl.float32)
        li0 = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        li1 = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        li2 = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        for t in tl.range(0, n_iters, 1):
            # ---------------- vec1(s = t - 1) ----------------
            alpha = tl.zeros([BLOCK_M], dtype=tl.float32)
            s = t - 1
            if (s >= 0) & (s < total_steps):
                s2 = s % NUM_S2_STEPS
                if s % 2 == 0:
                    al.sync_block_wait("cube", "vector", 2, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    qk = bl.to_tensor(qk_ub0)
                else:
                    al.sync_block_wait("cube", "vector", 5, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    qk = bl.to_tensor(qk_ub1)
                is_block_start = (s2 == 0).to(tl.float32)
                m_i = m_i + (zero_row + is_block_start) * -3.0e38
                qk = qk * sm_scale
                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                qk = qk - m_ij[:, None]
                p = tl.math.exp(qk).cast(tl.float16)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp(m_i - m_ij)
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                # P -> fractal zN [K1, M1, 16, 16] -> L1 для mm2
                p_nz = tl.reshape(tl.permute(tl.reshape(p, (BLOCK_M, K1, 16)), (1, 0, 2)), (K1, M1, 16, 16))
                p_ub = bl.to_buffer(p_nz, al.ascend_address_space.UB)
                if s % 2 == 0:
                    al.sync_block_wait("cube", "vector", 1, PIPE.PIPE_M, PIPE.PIPE_MTE3)
                    al.copy(p_ub, p_l1_0)
                    al.sync_block_set("vector", "cube", 1, PIPE.PIPE_MTE3, PIPE.PIPE_MTE1)
                    al.sync_block_set("vector", "cube", 2, PIPE.PIPE_V, PIPE.PIPE_FIX)
                else:
                    al.sync_block_wait("cube", "vector", 4, PIPE.PIPE_M, PIPE.PIPE_MTE3)
                    al.copy(p_ub, p_l1_1)
                    al.sync_block_set("vector", "cube", 4, PIPE.PIPE_MTE3, PIPE.PIPE_MTE1)
                    al.sync_block_set("vector", "cube", 5, PIPE.PIPE_V, PIPE.PIPE_FIX)
            a2 = a1
            a1 = a0
            a0 = alpha
            li2 = li1
            li1 = li0
            li0 = l_i
            # ---------------- vec2(s = t - 3) ----------------
            s = t - 3
            if (s >= 0) & (s < total_steps):
                task = s // NUM_S2_STEPS
                s2 = s % NUM_S2_STEPS
                if s % 2 == 0:
                    al.sync_block_wait("cube", "vector", 3, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    pv = bl.to_tensor(pv_ub0)
                else:
                    al.sync_block_wait("cube", "vector", 6, PIPE.PIPE_FIX, PIPE.PIPE_V)
                    pv = bl.to_tensor(pv_ub1)
                acc = acc * a2[:, None] + pv
                if s2 == LAST_S2_STEP:
                    block_idx = pid + task * AICORE_NUM
                    hz = block_idx // NUM_BLOCKS_M
                    qvk_offset = (hz // H).to(tl.int64) * stride_qz + (hz % H).to(tl.int64) * stride_qh
                    O_ptr = tl.make_block_ptr(
                        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_on),
                        offsets=((block_idx % NUM_BLOCKS_M) * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM),
                        order=(1, 0))
                    tl.store(O_ptr, (acc / li2[:, None]).to(Out.type.element_ty))
                if s % 2 == 0:
                    al.sync_block_set("vector", "cube", 3, PIPE.PIPE_V, PIPE.PIPE_FIX)
                else:
                    al.sync_block_set("vector", "cube", 6, PIPE.PIPE_V, PIPE.PIPE_FIX)
        # слив: дождаться возврата токенов L1-буферов P
        al.sync_block_wait("cube", "vector", 1, PIPE.PIPE_M, PIPE.PIPE_MTE3)
        al.sync_block_wait("cube", "vector", 4, PIPE.PIPE_M, PIPE.PIPE_MTE3)


def main():
    signature = {
        "Q": "*fp16",
        "K": "*fp16",
        "V": "*fp16",
        "ATTEN_MASK": "*fp32",
        "M": "*fp32",
        "Out": "*fp16",
        "sm_scale": "constexpr",
        "stride_qz": "constexpr",
        "stride_qh": "constexpr",
        "stride_qm": "constexpr",
        "stride_qk": "constexpr",
        "stride_kz": "constexpr",
        "stride_kh": "constexpr",
        "stride_kn": "constexpr",
        "stride_kk": "constexpr",
        "stride_vz": "constexpr",
        "stride_vh": "constexpr",
        "stride_vn": "constexpr",
        "stride_vk": "constexpr",
        "stride_oz": "constexpr",
        "stride_oh": "constexpr",
        "stride_om": "constexpr",
        "stride_on": "constexpr",
        "stride_am": "constexpr",
        "Z": "constexpr",
        "H": "constexpr",
        "N_CTX": "constexpr",
        "HEAD_DIM": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "STAGE": "constexpr",
        "NUM_BLOCKS_PER_CORE": "constexpr",
        "NUM_BLOCKS": "constexpr",
        "NUM_BLOCKS_M": "constexpr",
        "AICORE_NUM": "constexpr",
    }
    Z = 128
    H = 8
    N_CTX = 8192
    HEAD_DIM = 128
    sm_scale = 0.5
    AICORE_NUM = 28
    BM = 128
    BN = 128

    num_cores = AICORE_NUM
    NUM_BLOCKS_M = triton.cdiv(N_CTX, BM)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H
    num_cores = num_cores if NUM_BLOCKS > num_cores else NUM_BLOCKS
    NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)

    constexprs = {
        "sm_scale": sm_scale,
        "stride_qz": H * N_CTX * HEAD_DIM,
        "stride_qh": N_CTX * HEAD_DIM,
        "stride_qm": HEAD_DIM,
        "stride_qk": 1,
        "stride_kz": H * N_CTX * HEAD_DIM,
        "stride_kh": N_CTX * HEAD_DIM,
        "stride_kn": HEAD_DIM,
        "stride_kk": 1,
        "stride_vz": H * N_CTX * HEAD_DIM,
        "stride_vh": N_CTX * HEAD_DIM,
        "stride_vn": HEAD_DIM,
        "stride_vk": 1,
        "stride_oz": H * N_CTX * HEAD_DIM,
        "stride_oh": N_CTX * HEAD_DIM,
        "stride_om": HEAD_DIM,
        "stride_on": 1,
        "stride_am": N_CTX,
        "Z": Z,
        "H": H,
        "N_CTX": N_CTX,
        "HEAD_DIM": HEAD_DIM,
        "BLOCK_M": BM,
        "BLOCK_N": BN,
        "STAGE": 1,
        "NUM_BLOCKS_PER_CORE": NUM_BLOCKS_PER_CORE,
        "NUM_BLOCKS": NUM_BLOCKS,
        "NUM_BLOCKS_M": NUM_BLOCKS_M,
        "AICORE_NUM": num_cores,
    }

    signature_keys = list(signature.keys())
    target_pointers = ["Q", "K", "V", "ATTEN_MASK", "M", "Out"]
    structured_attrs_map = {}
    for name in target_pointers:
        if name in signature_keys:
            arg_index = signature_keys.index(name)
            structured_attrs_map[(arg_index,)] = [("tt.divisible_by", 16)]

    class Triton36CompilerAttrsMock:
        def __init__(self, mapping_dict):
            self._mapping = mapping_dict
            self.divisible_by_16 = tuple(path[0] for path in mapping_dict.keys())
            self.equal_to_1 = ()

        def get(self, key, default=None):
            return self._mapping.get(key, default if default is not None else [])

    attrs_mock = Triton36CompilerAttrsMock(structured_attrs_map)

    target = GPUTarget(backend="npu", arch="Ascend950PR_958b", warp_size=32)
    backend = AscendBackend(target)
    compile_opts = {"debug": True,
                    "compile_on_910_95": True,
                    "enable_dynamic_cv_pipeline": True,
                    "main_loop_unroll_factor": 1,
                    }
    if KERNEL == "cv":
        # синхронизации заданы вручную: автоинжект bishengir выключаем, как это делает ssbuf-флоу
        compile_opts.update({"enable_dynamic_cv_pipeline": False,
                             "disable_auto_inject_block_sync": True,
                             "enable_mixed_cv": True,
                             "set_workspace_multibuffer": 0})
    elif KERNEL == "l1buf":
        # с ssbuf этот вариант некорректен; scope внутри кернела и так отключает ssbuf (rc=2)
        compile_opts["enable_dynamic_cv_pipeline"] = False
    options = backend.parse_options(compile_opts)
    for key, value in vars(options).items():
        print(f"{key}: {value}")

    kernel_fn = {"orig": _vdv_atn_fwd, "flat": _vdv_atn_fwd_flat,
                 "l1buf": _vdv_atn_fwd_l1buf, "cv": _vdv_atn_fwd_cv}[KERNEL]
    src = triton.compiler.ASTSource(
        fn=kernel_fn,
        signature=signature,
        constexprs=constexprs,
        attrs=attrs_mock,
    )

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)

    metadata = {}
    module = make_ttir(module, metadata, options)
    with open(f"kernel_{KERNEL}.ttir.mlir", "w") as f:
        f.write(str(module))
    print(f"[ok] TTIR сохранён в kernel_{KERNEL}.ttir.mlir")

    metadata.update(options.__dict__)
    metadata.setdefault("hash", "manual-aot-run")
    try:
        linalg_ir = ttir_to_linalg(module, metadata, options, named_ops=True)
        with open(f"kernel_{KERNEL}.ttadapter.mlir", "w") as f:
            f.write(linalg_ir)
        print(f"[ok] Сохранён в kernel_{KERNEL}.ttadapter.mlir")
    except Exception as e:
        print(f"[expected on CPU-only машине без ascend toolchain] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
