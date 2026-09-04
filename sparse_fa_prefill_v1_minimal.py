# =============================================================================
# ВАРИАНТ 1 — минимальная правка
# =============================================================================
# Собирать и запускать на TA-коммите  1c572dcb0  ("fix mainloop for indexer").
#
# Там: выбор главной петли по реестру crossCoreDeps + гейт worthwhile/safe,
# и НЕТ нашего memdep-guard (в whitelist только апстримные _hstu_attn_fwd и
# parallel_path_fwd_kernel). Значит четырёхфлаговый кредитный протокол для
# зависимостей через память вообще не строится — зависнуть по той причине,
# по которой зависали f78../dd27.., физически нечем.
#
#   git checkout 1c572dcb0
#
# -----------------------------------------------------------------------------
# ЧТО ИЗМЕНЕНО ОТНОСИТЕЛЬНО ИСХОДНИКА: ровно одно место.
#
# Было:
#     tl.store(Wksp_score_ptr + wdqt_base + ..., score.to(Q_ptr.dtype.element_ty))
#     score_l1 = tl.load(Wksp_score_ptr + wdqt_base + ...)
#
# Стало:
#     score_l1 = score.to(Q_ptr.dtype.element_ty)
#
# Почему это чинит:
#
# Запись и чтение одного и того же адреса подряд — это зависимость VECTOR->CUBE,
# несомая ПАМЯТЬЮ. Буфер принадлежит кернелу, а не компилятору, поэтому
# InterCoreTransferAndSync не может ни продублировать его, ни провернуть по
# итерациям: он ставит только одностороннюю синхронизацию "данные готовы", без
# обратного "буфер свободен". Пока стадии идут в лок-степе этого хватает; после
# софтверного пайплайнинга продюсер итерации i+1 переписывает буфер, который
# потребитель итерации i ещё читает. Плюс сам сигнал "готово" привязан к границе
# compute-блока, а не к самой записи, и после переразметки блоков уезжает ВПЕРЁД
# записи — куб читает scores предыдущей итерации. Это и давало
# allclose=False с равномерной ошибкой ~0.1 при шуме bf16 ~0.004, и это же
# "чинилось" профайлером, который просто замедлял вектор.
#
# Если score идёт значением, зависимость становится VECTOR->CUBE ПО ЗНАЧЕНИЮ.
# Тогда handleVectorToCube выделяет СВОЙ буфер в L1, ставит полное рукопожатие
# в обе стороны и AllocMultiCache может его провернуть. Весь класс проблемы
# исчезает, потому что он живёт только там, где буфер чужой.
#
# Бонусом уходит круг 16x128 bf16 через глобальную память на каждой итерации.
#
# -----------------------------------------------------------------------------
# ЧТО НАМЕРЕННО НЕ ТРОНУТО
#
# 1. Круг acc через O_ptr (tl.load/tl.store внутри static_range). Он не мешает
#    пайплайну: и запись, и чтение на VECTOR, межъядерной зависимости нет.
#    Но он стоит трафика и мешает перекрытию — это вариант 2.
# 2. Гатер K в Wksp_K_ptr. Разреженность его требует, оставлен как есть.
#    По костмодели он даёт 54% всей оценки (1024 копии по 1152 байта
#    HBM->UB->HBM), так что это главная цель для оптимизации, но отдельная.
# 3. Индексация store в O_ptr через tl.arange(0, BLOCK_G), а не offs_G —
#    оставлено как в оригинале. ВНИМАНИЕ: если G > BLOCK_G, разные idx_n1_blk
#    будут писать друг поверх друга. Проверь, что у тебя G == BLOCK_G.
#
# -----------------------------------------------------------------------------
# ОЖИДАЕМОЕ В ЛОГЕ  (TRITON_ASCEND_CV_DEBUG_MAINLOOP=1)
#
#   group N: ... direction=V->C via=transfer   [complete handoff]   <- бывший score
#   verdict : worthwhile (...), safe
#   applicability: at least one id can be pipelined -> proceed
#
# Если увидишь via=memdep у группы, соответствующей score, — значит round-trip
# где-то остался.
#
# -----------------------------------------------------------------------------
# ОГОВОРКА: не собиралось и не запускалось. Помощники _gather и cube1 берутся
# из твоего файла как есть, их сигнатуры я не менял.
# =============================================================================

import triton
import triton.language as tl

# _gather и cube1 — твои существующие @triton.jit-помощники, импортируй/оставь
# их в том же модуле.


@triton.jit(do_not_specialize=["T1", "T2", "S1", "S2"])
def sparse_flash_attention_prefill_kernel(
    # Input pointers
    Q_ptr,             # [B, S, H, D_qk] - queries (combined nope + rope)
    K_ptr,             # [B, T, H, D_qk] - keys (combined nope + rope)
    V_ptr,             # [B, T, H, D_v] - values
    Indices_ptr,       # [B, S, K] - topk indices
    O_ptr,
    # Workspace pointers
    Wksp_K_ptr,
    Wksp_K2_ptr,
    Wksp_qk_ptr,
    Wksp_score_ptr,    # больше не используется, оставлен ради ABI
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
    # Grid: (T1(approx),)
    pid = tl.program_id(0)
    t1_start = pid * bs1_per_core
    t1_end = tl.minimum(T1, (pid + 1) * bs1_per_core)

    for idx_t1 in range(t1_start, t1_end):
        idx_b = idx_t1 // S1
        cur_s2 = S2
        beg_s2 = idx_b * S2

        # Compute base offsets
        topk_base = idx_t1 * stride_it1
        q_base = idx_t1 * stride_qt1

        # Cube WKSP
        wk_base = pid * stride_wkt1
        wdq_base = pid * stride_wqkt1
        wdqt_base = pid * stride_wscoret1
        k_base = beg_s2 * stride_kt2

        wsv_base = pid * stride_wsvt1
        wsacc_base = idx_t1 * stride_ot1

        _gather(
            Wksp_K_ptr,
            K_ptr,
            Indices_ptr,
            # Strides for indices
            stride_it1, stride_in2, stride_ik,
            # Strides for dst
            stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
            # Strides for src
            stride_kt2, stride_kn2, stride_k1, stride_kd,
            # Params
            D_qk,
            K,
            D_qk,
            # Meta
            K,
            wk_base,
            k_base,
            topk_base,
            cur_s2,
        )

        HEAD_LOOP_TIMES: tl.constexpr = triton.cdiv(G, BLOCK_G)     # n1 axis
        sbs_loop_times = tl.cdiv(K, BLOCK_SBS)                      # s2 axis

        for idx_n1_blk in range(0, HEAD_LOOP_TIMES):                # n1 block loop
            offs_G = idx_n1_blk * BLOCK_G + tl.arange(0, BLOCK_G)

            lse = tl.zeros([BLOCK_G], dtype=tl.float32) - float("inf")

            for idx_sub_sbs in range(0, sbs_loop_times):            # s2 loop
                offs_sbs = idx_sub_sbs * BLOCK_SBS + tl.arange(0, BLOCK_SBS)

                ## Calculate qk = q @ k^T
                qk_l1 = cube1(
                    Q_ptr,
                    Wksp_K_ptr,
                    # Strides for Q
                    stride_qt1, stride_qn2, stride_qn1, stride_qd,
                    # Strides for aux K
                    stride_wkt1, stride_wkn2, stride_wksbs, stride_wkd,
                    # Param
                    D_qk,
                    BLOCK_G,
                    BLOCK_SBS,
                    # Meta
                    offs_G,
                    offs_sbs,
                    q_base,
                    wk_base,
                    # Block size
                    BLOCK_K,
                )

                # qk shape: [G, BLOCK_SBS]
                qk = qk_l1
                qk *= scale     # [BLOCK_G, BLOCK_SBS]

                mask = offs_sbs < K
                qk = tl.where(mask[None, :], qk, -float('inf'))
                local_max = tl.max(qk, 1)                                  # [BLOCK_G]
                local_exp = tl.exp(qk - local_max[:, None])                # [BLOCK_G, BLOCK_SBS]
                local_sum = tl.sum(local_exp, 1)                           # [BLOCK_G]
                local_lse = local_max + tl.log(local_sum)                  # [BLOCK_G]
                mean_lse = (lse + local_lse) / 2                           # [BLOCK_G]
                new_lse = mean_lse + tl.log(
                    tl.exp(lse - mean_lse) + tl.exp(local_lse - mean_lse)
                )                                                          # [BLOCK_G]
                new_lse = tl.where(new_lse != new_lse, local_lse, new_lse)  # [BLOCK_G]
                current_weight = tl.exp(local_lse - new_lse)               # [BLOCK_G]
                score = (local_exp / local_sum[:, None])                   # [BLOCK_G, BLOCK_SBS]

                # ============ ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ============
                # Было: круг через глобальную память Wksp_score_ptr —
                #   tl.store(Wksp_score_ptr + wdqt_base + ..., score.to(...))
                #   score_l1 = tl.load(Wksp_score_ptr + wdqt_base + ...)
                # Стало: значение идёт во второй матмул напрямую, и компилятор
                # сам организует передачу VECTOR->CUBE через свой буфер в L1.
                score_l1 = score.to(Q_ptr.dtype.element_ty)
                # ================================================

                blk_v_loop_time: tl.constexpr = triton.cdiv(D_v, BLOCK_V)

                for idx_blk_v in tl.static_range(0, blk_v_loop_time):
                    offset_V = idx_blk_v * BLOCK_V + tl.arange(0, BLOCK_V)
                    kv_cache = tl.load(
                        Wksp_K_ptr + wk_base
                        + offs_sbs[:, None] * stride_wksbs
                        + offset_V[None, :] * stride_wkd
                    )
                    sv = tl.dot(score_l1, kv_cache)

                    acc = tl.load(
                        O_ptr + wsacc_base
                        + tl.arange(0, BLOCK_G)[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod
                    )
                    acc = acc * tl.exp(lse - new_lse)[:, None]

                    acc += sv * current_weight[:, None]

                    tl.store(
                        O_ptr + wsacc_base
                        + tl.arange(0, BLOCK_G)[:, None] * stride_on1
                        + offset_V[None, :] * stride_wod,
                        acc,
                    )

                lse = new_lse
