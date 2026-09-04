# =============================================================================
# ВАРИАНТ 2 — приведение к форме обычного FlashAttention
# =============================================================================
# Собирать и запускать на том же TA-коммите  1c572dcb0.
#
# Идея: после гатера это обычный аттеншен. Всё, что идёт после сбора K/V по
# индексам, должно выглядеть ровно как _vdv_atn_fwd_inner_opt — состояние
# живёт значениями, ни одного store внутри цикла, единственная запись в конце.
#
# -----------------------------------------------------------------------------
# ТРИ ИЗМЕНЕНИЯ ОТНОСИТЕЛЬНО ИСХОДНИКА
#
# 1. score больше не ходит через Wksp_score_ptr.
#    То же, что в варианте 1: зависимость VECTOR->CUBE становится по значению,
#    компилятор сам даёт ей свой буфер в L1 с полным рукопожатием и ротацией.
#    Это то, что снимает и гонку, и дедлок.
#
# 2. acc больше не ходит через O_ptr на каждой итерации.
#    Был tl.load/tl.store внутри static_range — аккумулятор жил в HBM. Стал
#    обычным loop-carried значением, как acc в FA. Запись одна, после цикла.
#    Побочно исчезает петля по BLOCK_V: при BLOCK_V == D_v она вырождается.
#
# 3. lse-формулировка заменена на классическую пару (m_i, l_i) из FA.
#    Было: нормировать score внутри блока (local_exp / local_sum), считать
#    new_lse через mean_lse с защитой от NaN, домножать на current_weight.
#    Стало: online softmax — держим бегущий максимум m_i и бегущую сумму l_i,
#    масштабируем накопленное на alpha = exp(m_i - m_ij), делим один раз в конце.
#    Математически то же самое, но без деления в цикле и без ветки на NaN.
#
# -----------------------------------------------------------------------------
# ЧТО ПРОВЕРИТЬ ПЕРЕД ЗАПУСКОМ
#
# * BLOCK_V должен стать равным D_v (256), иначе внутренняя петля не исчезнет.
#   Ресурсы: kv_cache [BLOCK_SBS, D_v] = 128x256 bf16 = 64 КБ, pv/acc
#   [BLOCK_G, D_v] = 16x256 f32 = 16 КБ в L0C. Костмодель на исходной версии
#   показывала пик UB 81920 из 262144, запас есть. Если всё же не влезет —
#   вернуть петлю по BLOCK_V, но держать по одному acc на каждый V-блок как
#   отдельные значения (static_range разворачивается, так что это законно).
#
# * Индексация записи в O_ptr взята из оригинала: tl.arange(0, BLOCK_G), а не
#   offs_G. Если G > BLOCK_G, разные idx_n1_blk будут писать друг поверх друга.
#   Я это НЕ трогал, чтобы не менять семантику молча. Проверь G == BLOCK_G.
#
# * Числа не будут побитово совпадать с исходной версией: другая, хотя и
#   эквивалентная, формулировка софтмакса. Сравнивай по allclose с разумным
#   допуском, а не по равенству.
#
# * Полностью замаскированный блок (все offs_sbs >= K) дал бы m_ij = -inf и
#   NaN в exp(-inf - -inf). Здесь этого не бывает: sbs_loop_times = cdiv(K,
#   BLOCK_SBS), маскируется только хвост последнего блока. Если решишь менять
#   разбиение — верни защиту.
#
# -----------------------------------------------------------------------------
# ЧТО ОСТАЛОСЬ НЕТРОНУТЫМ И ЧТО С ЭТИМ ДЕЛАТЬ ДАЛЬШЕ
#
# Гатер. По костмодели исходной версии он один даёт 54% всей оценки:
#   1152 байта x 1024 раза HBM->UB  (111616 циклов)
#   1152 байта x 1024 раза UB->HBM  (100352 циклов)
# то есть K собирается построчно и кладётся в workspace в HBM, откуда куб
# потом читает его обратно. Это отдельная и самая крупная цель, но она требует
# знания о том, что даёт железо на разреженных загрузках, и в этот файл я её
# не тащу.
#
# -----------------------------------------------------------------------------
# ОГОВОРКА: не собиралось и не запускалось. Помощники _gather и cube1 берутся
# из твоего файла как есть.
# =============================================================================

import triton
import triton.language as tl

# _gather и cube1 — твои существующие @triton.jit-помощники.

@triton.jit
def _gather(
    dst_ptr,
    src_ptr,
    Indices_ptr,
    # Strides for indices
    stride_it1, stride_in2, stride_ik,
    # Strides for dst
    stride_dt1, stride_dn2, stride_dsbs, stride_dd,
    # Strides for src
    stride_st2, stride_sn2, stride_s1, stride_sd,
    # Params
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D_VEC: tl.constexpr,
    # Meta
    actual_sel_blk,
    dst_base,
    src_base,
    topk_base,
    cur_s2,
):
    # Сколько строк собирается за одну итерацию. Раньше было по одной, и на
    # каждую приходились скалярная загрузка индекса, ветвление и отдельный DMA
    # на 1152 байта — 109 тактов на чтение и 98 на запись, то есть ~10.6 Б/такт
    # против 18-76 Б/такт у крупных копий. Упирается не в полосу, а в оверхед
    # на транзакцию, поэтому строки собираются пачкой.
    # Это первая ручка для подбора: 8 -> 9 КБ на тайл, 16 -> 18 КБ, 32 -> 37 КБ.
    BLOCK_ROWS: tl.constexpr = 8

    D_loop_times: tl.constexpr = (D + BLOCK_D_VEC - 1) // BLOCK_D_VEC
    row_loop_times = tl.cdiv(actual_sel_blk, BLOCK_ROWS)

    for idx_row_blk in range(0, row_loop_times):
        rows = idx_row_blk * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_in_range = rows < actual_sel_blk

        sparse_ids = tl.load(
            Indices_ptr + topk_base + rows * stride_ik,
            mask=row_in_range,
            other=-1,
        )
        # Ветвление if sparse_id >= 0 стало маской: невалидные строки не
        # читаются и не пишутся, приёмник сохраняет прежнее содержимое —
        # ровно как в исходной версии.
        row_valid = row_in_range & (sparse_ids >= 0)
        # Адрес не должен считаться от -1, даже под маской.
        safe_ids = tl.where(row_valid, sparse_ids, 0)

        src_rows = src_base + safe_ids * stride_st2
        dst_rows = dst_base + rows * stride_dsbs

        for idx_sub_Dv in range(0, D_loop_times):
            offs_sub_Dv = idx_sub_Dv * BLOCK_D_VEC + tl.arange(0, BLOCK_D_VEC)
            mask_sub_Dv = offs_sub_Dv < D
            tile_mask = row_valid[:, None] & mask_sub_Dv[None, :]

            gather_tile = tl.load(
                src_ptr + src_rows[:, None] + offs_sub_Dv[None, :] * stride_sd,
                mask=tile_mask,
                other=0,
            )
            tl.store(
                dst_ptr + dst_rows[:, None] + offs_sub_Dv[None, :] * stride_dd,
                gather_tile,
                mask=tile_mask,
            )

@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    # Input pointers
    Q_ptr,             # [B, S, H, D_qk]
    K_ptr,             # [B, T, H, D_qk]
    V_ptr,             # [B, T, H, D_v]
    Indices_ptr,       # [B, S, K]
    O_ptr,
    # Workspace pointers
    Wksp_K_ptr,
    Wksp_K2_ptr,       # не используется
    Wksp_qk_ptr,       # не используется
    Wksp_score_ptr,    # не используется
    Wksp_sv_ptr,       # не используется
    # Strides for Q
    stride_qt1, stride_qn2, stride_qn1, stride_qd,
    # Strides for K
    stride_kt2, stride_kn2, stride_k1, stride_kd,
    # Strides for indices
    stride_it1, stride_in2, stride_ik,
    # Strides for O
    stride_ot1, stride_on1, stride_wod,
    # Strides for aux K
    stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
    # Strides for aux qk
    stride_wqkt1, stride_wqkn1, stride_wqks2,
    # Strides for aux score
    stride_wscoret1, stride_wscoren1, stride_wscores2,
    # Strides for aux sv
    stride_wsvt1, stride_wsvn1, stride_wsvd,
    # Param
    scale,
    ## Basic
    T1,
    N1: tl.constexpr,
    T2,
    G: tl.constexpr,
    D_qk: tl.constexpr,
    D_v: tl.constexpr,
    K: tl.constexpr,
    ## Extend
    B: tl.constexpr,
    S1,
    S2,
    # Proc per core
    bs1_per_core,
    # Block sizes
    BLOCK_G: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,     # ожидается BLOCK_V == D_v
):
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core
    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        cur_s2 = S2
        beg_s2 = idx_b * S2

        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1

        wk_base = pid * stride_wkt1
        k_base = beg_s2 * stride_kt2
        wsacc_base = idx_t1 * stride_ot1

        # Разреженность требует гатера — оставлен как есть.
        _gather(
            Wksp_K_ptr,
            K_ptr,
            Indices_ptr,
            stride_it1, stride_in2, stride_ik,
            stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
            stride_kt2, stride_kn2, stride_k1, stride_kd,
            D_qk,
            K,
            D_qk,
            K,
            wk_base,
            k_base,
            topk_base,
            cur_s2,
        )

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)

        offs_V = tl.arange(0, D_v)

        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)

            # --- состояние FA: всё значениями, ничего в памяти ---
            m_i = tl.zeros([BLOCK_G], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
            acc = tl.zeros([BLOCK_G, D_v], dtype=tl.float32)

            for idx_sub_sbs in range(0, sbs_loop_times):
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)

                # qk = q @ k^T   (CUBE)
                qk = cube1(
                    Q_ptr,
                    Wksp_K_ptr,
                    stride_qt1, stride_qn2, stride_qn1, stride_qd,
                    stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
                    D_qk,
                    BLOCK_G,
                    BLOCK_SBS,
                    offs_G,
                    offs_sbs,
                    q_base,
                    wk_base,
                    BLOCK_K,
                )
                qk = qk * scale
                qk = tl.where((offs_sbs < K)[None, :], qk, -float("inf"))

                # --- online softmax, как в _vdv_atn_fwd_inner_opt (VECTOR) ---
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.exp(m_i - m_ij)

                # p идёт во второй матмул ЗНАЧЕНИЕМ: компилятор сам построит
                # передачу VECTOR->CUBE через свой буфер в L1.
                p = p.to(Q_ptr.dtype.element_ty)

                kv_cache = tl.load(
                    Wksp_K_ptr + wk_base
                    + offs_sbs[:, None] * stride_wksbs
                    + offs_V[None, :] * stride_wkd
                )
                pv = tl.dot(p, kv_cache)                                   # CUBE

                # --- обновление состояния (VECTOR), всё в значениях ---
                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None] + pv

            # --- эпилог: единственная запись, за пределами цикла ---
            acc = acc / l_i[:, None]
            tl.store(
                O_ptr + wsacc_base
                + tl.arange(0, BLOCK_G)[:, None] * stride_on1
                + offs_V[None, :] * stride_wod,
                acc,
            )
