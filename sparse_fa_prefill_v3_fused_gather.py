# =============================================================================
# ВАРИАНТ B — гатер вфьюжен в петлю по sbs, workspace Wksp_K убран
# =============================================================================
# База — v2 (та, что заработала). Меняется только то, как K и V попадают в
# матмулы. Запускать на том же TA-коммите 1c572dcb0.
#
# -----------------------------------------------------------------------------
# ЧТО ИМЕННО МЕНЯЕТСЯ
#
# Было: _gather один раз на t1 собирает все K строк в Wksp_K_ptr (HBM), а потом
# cube1 и kv_cache читают их оттуда обратно. Каждая строка K пересекает
# глобальную память трижды: читаем K -> пишем Wksp -> читаем Wksp.
#
# Стало: для каждого блока sbs собираются ровно те BLOCK_SBS строк, которые
# нужны прямо сейчас, и идут в tl.dot ЗНАЧЕНИЕМ. Workspace не нужен вовсе.
#
# По старым числам костмодели убирается:
#     100 352 тактов  запись гатера UB->HBM
#      15 456 тактов  перечитывание Wksp_K в L1 для qk
#      14 192 тактов  перечитывание Wksp_K в L1 для kv_cache
# и остаётся неустранимое чтение самих данных K из HBM.
#
# Цена: строки читаются дважды — 576 колонок для qk и 256 для V. Это +44% к
# объёму чтения, но чтение и было единственной неустранимой частью, а ушло
# полтора круга через HBM. Грубая прикидка на старых цифрах: ~241k тактов
# против ~160k.
#
# -----------------------------------------------------------------------------
# ГЛАВНЫЙ РИСК, ЧЕСТНО
#
# Сейчас cube1 читает Wksp_K НАПРЯМУЮ в L1 (в таблице трансферов это
# cube_mte2 hbm:l1) — межъядерной передачи нет вообще, куб грузит свой операнд
# сам. Если после фьюза собранный тайл окажется на VECTOR, добавится передача
# VECTOR->CUBE на 128x576 bf16 = 147 КБ за итерацию, и она может съесть весь
# выигрыш.
#
# Всё решает то, на какое ядро OpClassifier положит собранную загрузку.
# У него есть matchToTensorPattern: производители операндов матмула тянутся на
# CUBE. Поэтому tl.load здесь написан так, чтобы результат шёл в tl.dot
# максимально прямо — без промежуточных векторных операций, которые могли бы
# притянуть его на VECTOR.
#
# Как проверить после сборки:
#     TRITON_ASCEND_CV_COST_VERBOSE=2 ...
# и посмотреть в таблице трансферов строку для чтения K:
#     unit = cube_mte2, path = hbm:l1   -> хорошо, куб грузит сам
#     unit = vec_mte2,  path = hbm:ub   -> плохо, появится V->C передача,
#                                          и тогда этот вариант хуже базового
#
# -----------------------------------------------------------------------------
# ЧТО ЕЩЁ ПРОВЕРИТЬ
#
# * qk считается здесь напрямую (tl.dot(q, tl.trans(k))), как в хорошем FA, а
#   не через cube1. Индексация Q взята по стрейдам из сигнатуры:
#   q_base + offs_G * stride_qn1 + offs_D * stride_qd. Сверь с тем, что делает
#   cube1 — если он режет D по BLOCK_K или делает что-то с раскладкой, здесь
#   это потеряется.
#
# * V берётся из первых D_v колонок тех же строк K — так же, как в оригинале
#   kv_cache читался из Wksp_K по offset_V вдоль stride_wkd.
#
# * Wksp_K_ptr и _gather больше не используются. Параметры оставлены ради ABI;
#   аллокацию workspace можно убрать снаружи.
#
# * Индексация записи в O_ptr по-прежнему через tl.arange(0, BLOCK_G), как в
#   оригинале. При G > BLOCK_G блоки по n1 перезапишут друг друга.
#
# ОГОВОРКА: не собиралось и не запускалось.
# =============================================================================

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    # Input pointers
    Q_ptr,             # [B, S, H, D_qk]
    K_ptr,             # [B, T, H, D_qk]
    V_ptr,             # [B, T, H, D_v]   не используется, V берётся из K_ptr
    Indices_ptr,       # [B, S, K]
    O_ptr,
    # Workspace pointers  (больше не используются)
    Wksp_K_ptr,
    Wksp_K2_ptr,
    Wksp_qk_ptr,
    Wksp_score_ptr,
    Wksp_sv_ptr,
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
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core
    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    offs_D = tl.arange(0, D_qk)      # колонки K для qk
    offs_V = tl.arange(0, D_v)       # колонки K, играющие роль V

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        beg_s2 = idx_b * S2

        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1
        k_base = beg_s2 * stride_kt2
        wsacc_base = idx_t1 * stride_ot1

        # _gather больше не вызывается: сбор строк переехал внутрь петли по sbs.

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)

        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)

            # q остаётся в регистрах на весь цикл, как в FA
            q = tl.load(
                Q_ptr + q_base
                + offs_G[:, None] * stride_qn1
                + offs_D[None, :] * stride_qd
            )

            m_i = tl.zeros([BLOCK_G], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
            acc = tl.zeros([BLOCK_G, D_v], dtype=tl.float32)

            for idx_sub_sbs in range(0, sbs_loop_times):
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)
                in_range = offs_sbs < K

                # --- сбор индексов для текущего блока: BLOCK_SBS штук ---
                sparse_ids = tl.load(
                    Indices_ptr + topk_base + offs_sbs * stride_ik,
                    mask=in_range,
                    other=-1,
                )
                row_valid = in_range & (sparse_ids >= 0)
                safe_ids = tl.where(row_valid, sparse_ids, 0)
                src_rows = k_base + safe_ids * stride_kt2

                # --- K-тайл собирается прямо в операнд матмула ---
                # Никаких векторных операций между load и dot: так у
                # OpClassifier больше шансов положить загрузку на CUBE.
                k_tile = tl.load(
                    K_ptr + src_rows[:, None] + offs_D[None, :] * stride_kd,
                    mask=row_valid[:, None],
                    other=0,
                )
                qk = tl.dot(q, tl.trans(k_tile))                # CUBE
                qk = qk * scale
                qk = tl.where(row_valid[None, :], qk, -float("inf"))

                # --- online softmax, значениями (VECTOR) ---
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                p = tl.exp(qk - m_ij[:, None])
                l_ij = tl.sum(p, 1)
                alpha = tl.exp(m_i - m_ij)
                p = p.to(Q_ptr.dtype.element_ty)

                # --- V-тайл: те же строки, первые D_v колонок ---
                v_tile = tl.load(
                    K_ptr + src_rows[:, None] + offs_V[None, :] * stride_kd,
                    mask=row_valid[:, None],
                    other=0,
                )
                pv = tl.dot(p, v_tile)                          # CUBE

                m_i = m_ij
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None] + pv

            acc = acc / l_i[:, None]
            tl.store(
                O_ptr + wsacc_base
                + tl.arange(0, BLOCK_G)[:, None] * stride_on1
                + offs_V[None, :] * stride_wod,
                acc,
            )
