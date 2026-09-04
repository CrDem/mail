#!/usr/bin/env python3
# =============================================================================
# Изолированное сравнение двух реализаций _gather.
#
#   python test_gather_ab.py
#
# Ничего кроме гатера не запускается: ни аттеншена, ни матмулов. Если версии
# расходятся, расхождение видно здесь и нигде больше.
#
# Данные заполнены узнаваемым узором:  src[r, d] = r * 1000 + d
# Поэтому по любой строке результата сразу читается, из какой строки источника
# она пришла:  r = v[0] // 1000.  Это отличает "взяли не ту строку" от
# "не записали вовсе" и от "записали нули".
#
# Приёмник предзаполнен -1.0, так что нетронутые строки тоже опознаются.
#
# Проверяются три конфигурации:
#   orig        - исходный построчный гатер
#   vec-i32     - векторизованный, адресная арифметика как написана
#   vec-i64     - тот же, но индекс приведён к int64 перед умножением на страйд
# Третья существует чтобы проверить гипотезу о переполнении в указательной
# арифметике: в исходной версии индекс скалярный и часто считается в int64,
# в векторизованной он тензорный и может остаться в int32.
# =============================================================================

import sys

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("torch_npu не найден — скрипт рассчитан на NPU")
    sys.exit(1)

import triton
import triton.language as tl

DEVICE = "npu"

# ---------------------------------------------------------------------------
# Конфигурация. Значения взяты близкими к реальному кернелу.
# ---------------------------------------------------------------------------
T2 = 4096          # строк в источнике
D = 576            # D_qk
K_TOPK = 1024      # сколько строк собираем
BLOCK_D_VEC = 576  # как в вызове: BLOCK_D_VEC = D_qk
BLOCK_ROWS = 8     # ручка векторизованной версии
NEG_FRACTION = 0.1  # доля индексов < 0
DTYPE = torch.float32  # узор точен во float32; попробуй bfloat16 вторым прогоном
SEED = 0


# ---------------------------------------------------------------------------
# Реализация 1: исходная, построчная
# ---------------------------------------------------------------------------
@triton.jit
def _gather_orig(
    dst_ptr,
    src_ptr,
    Indices_ptr,
    stride_it1, stride_in2, stride_ik,
    stride_dt1, stride_dn2, stride_dsbs, stride_dd,
    stride_st2, stride_sn2, stride_s1, stride_sd,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D_VEC: tl.constexpr,
    actual_sel_blk,
    dst_base,
    src_base,
    topk_base,
    cur_s2,
):
    for idx_topk in range(0, actual_sel_blk):
        sparse_id = tl.load(Indices_ptr + topk_base + idx_topk * stride_ik)
        if sparse_id >= 0:
            src_t2 = src_base + sparse_id * stride_st2
            dst_sbs = dst_base + idx_topk * stride_dsbs

            D_loop_times: tl.constexpr = (D + BLOCK_D_VEC - 1) // BLOCK_D_VEC
            for idx_sub_Dv in range(0, D_loop_times):
                offs_sub_Dv = idx_sub_Dv * BLOCK_D_VEC + tl.arange(0, BLOCK_D_VEC)
                mask_sub_Dv = offs_sub_Dv < D

                src_D = offs_sub_Dv * stride_sd
                gather_tensor = tl.load(src_ptr + (src_t2 + src_D), mask=mask_sub_Dv)

                dst_D = offs_sub_Dv * stride_dd
                tl.store(dst_ptr + (dst_sbs + dst_D), gather_tensor, mask=mask_sub_Dv)


# ---------------------------------------------------------------------------
# Реализация 2: векторизованная. USE_I64 переключает ширину индекса.
# ---------------------------------------------------------------------------
@triton.jit
def _gather_vec(
    dst_ptr,
    src_ptr,
    Indices_ptr,
    stride_it1, stride_in2, stride_ik,
    stride_dt1, stride_dn2, stride_dsbs, stride_dd,
    stride_st2, stride_sn2, stride_s1, stride_sd,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D_VEC: tl.constexpr,
    actual_sel_blk,
    dst_base,
    src_base,
    topk_base,
    cur_s2,
    BLOCK_ROWS: tl.constexpr,
    USE_I64: tl.constexpr,
):
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
        row_valid = row_in_range & (sparse_ids >= 0)
        safe_ids = tl.where(row_valid, sparse_ids, 0)
        if USE_I64:
            safe_ids = safe_ids.to(tl.int64)

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


# ---------------------------------------------------------------------------
# Обёртки-кернелы: гатер — device-функция, его надо из чего-то вызвать
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_orig(
    dst_ptr, src_ptr, Indices_ptr,
    stride_it1, stride_in2, stride_ik,
    stride_dt1, stride_dn2, stride_dsbs, stride_dd,
    stride_st2, stride_sn2, stride_s1, stride_sd,
    D: tl.constexpr, K: tl.constexpr, BLOCK_D_VEC: tl.constexpr,
    actual_sel_blk, dst_base, src_base, topk_base, cur_s2,
):
    _gather_orig(
        dst_ptr, src_ptr, Indices_ptr,
        stride_it1, stride_in2, stride_ik,
        stride_dt1, stride_dn2, stride_dsbs, stride_dd,
        stride_st2, stride_sn2, stride_s1, stride_sd,
        D, K, BLOCK_D_VEC,
        actual_sel_blk, dst_base, src_base, topk_base, cur_s2,
    )


@triton.jit
def _kernel_vec(
    dst_ptr, src_ptr, Indices_ptr,
    stride_it1, stride_in2, stride_ik,
    stride_dt1, stride_dn2, stride_dsbs, stride_dd,
    stride_st2, stride_sn2, stride_s1, stride_sd,
    D: tl.constexpr, K: tl.constexpr, BLOCK_D_VEC: tl.constexpr,
    actual_sel_blk, dst_base, src_base, topk_base, cur_s2,
    BLOCK_ROWS: tl.constexpr, USE_I64: tl.constexpr,
):
    _gather_vec(
        dst_ptr, src_ptr, Indices_ptr,
        stride_it1, stride_in2, stride_ik,
        stride_dt1, stride_dn2, stride_dsbs, stride_dd,
        stride_st2, stride_sn2, stride_s1, stride_sd,
        D, K, BLOCK_D_VEC,
        actual_sel_blk, dst_base, src_base, topk_base, cur_s2,
        BLOCK_ROWS, USE_I64,
    )


# ---------------------------------------------------------------------------
# Хост
# ---------------------------------------------------------------------------
def build_inputs():
    g = torch.Generator().manual_seed(SEED)

    # src[r, d] = r * 1000 + d  -- по значению читается номер строки источника
    r = torch.arange(T2, dtype=torch.float32).unsqueeze(1)
    d = torch.arange(D, dtype=torch.float32).unsqueeze(0)
    src = (r * 1000.0 + d).to(DTYPE).to(DEVICE)

    ids = torch.randint(0, T2, (K_TOPK,), generator=g, dtype=torch.int32)
    n_neg = int(K_TOPK * NEG_FRACTION)
    if n_neg:
        neg_pos = torch.randperm(K_TOPK, generator=g)[:n_neg]
        ids[neg_pos] = -1
    ids = ids.to(DEVICE)

    return src, ids


def run_one(which, src, ids, use_i64=False):
    dst = torch.full((K_TOPK, D), -1.0, dtype=DTYPE, device=DEVICE)

    common = dict(
        stride_it1=0, stride_in2=0, stride_ik=1,
        stride_dt1=0, stride_dn2=0, stride_dsbs=D, stride_dd=1,
        stride_st2=D, stride_sn2=0, stride_s1=0, stride_sd=1,
    )

    if which == "orig":
        _kernel_orig[(1,)](
            dst, src, ids,
            common["stride_it1"], common["stride_in2"], common["stride_ik"],
            common["stride_dt1"], common["stride_dn2"], common["stride_dsbs"], common["stride_dd"],
            common["stride_st2"], common["stride_sn2"], common["stride_s1"], common["stride_sd"],
            D, K_TOPK, BLOCK_D_VEC,
            K_TOPK, 0, 0, 0, 0,
        )
    else:
        _kernel_vec[(1,)](
            dst, src, ids,
            common["stride_it1"], common["stride_in2"], common["stride_ik"],
            common["stride_dt1"], common["stride_dn2"], common["stride_dsbs"], common["stride_dd"],
            common["stride_st2"], common["stride_sn2"], common["stride_s1"], common["stride_sd"],
            D, K_TOPK, BLOCK_D_VEC,
            K_TOPK, 0, 0, 0, 0,
            BLOCK_ROWS, use_i64,
        )
    return dst.cpu()


def decode_row(row):
    """Из какой строки источника пришла эта строка. None если не опознаётся."""
    v0 = float(row[0])
    v1 = float(row[1]) if row.numel() > 1 else None
    if v0 == -1.0:
        return "UNTOUCHED"
    if v0 == 0.0 and (v1 is None or v1 == 0.0):
        return "ZEROS"
    r = int(v0) // 1000
    d0 = int(v0) % 1000
    if d0 != 0:
        return f"SHIFTED(d0={d0},r={r})"
    # проверяем, что вся строка из одного r
    expect = torch.arange(row.numel(), dtype=torch.float32) + r * 1000.0
    if torch.equal(row.to(torch.float32), expect):
        return f"src[{r}]"
    return f"MIXED(head=src[{r}])"


def analyse(name, dst, ids_cpu, ref):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print("=" * 70)

    same = torch.eq(dst, ref).all(dim=1)
    n_diff = int((~same).sum())
    print(f"строк всего            : {K_TOPK}")
    print(f"совпало с эталоном     : {int(same.sum())}")
    print(f"разошлось              : {n_diff}")

    if n_diff == 0:
        print("-> идентично исходной реализации")
        return True

    diff_idx = (~same).nonzero().flatten().tolist()

    # --- где именно расходится ---
    neg_rows = set((ids_cpu < 0).nonzero().flatten().tolist())
    diff_set = set(diff_idx)
    print(f"первые расхождения     : {diff_idx[:20]}")
    print(f"минимальный индекс     : {min(diff_idx)}")
    print(f"максимальный индекс    : {max(diff_idx)}")
    print(f"все строки расходятся  : {n_diff == K_TOPK}")
    print(f"строк с id<0 всего     : {len(neg_rows)}")
    print(f"из них разошлось       : {len(diff_set & neg_rows)}")
    print(f"разошлось при id>=0    : {len(diff_set - neg_rows)}")

    # --- позиция внутри блока BLOCK_ROWS ---
    hist = [0] * BLOCK_ROWS
    for i in diff_idx:
        hist[i % BLOCK_ROWS] += 1
    print(f"по позиции в блоке из {BLOCK_ROWS}: {hist}")
    print("   (перекос => ошибка в блочной индексации;"
          " ровно => ошибка общая)")

    # --- хвост ---
    tail_start = (K_TOPK // BLOCK_ROWS) * BLOCK_ROWS
    print(f"только хвост (>= {tail_start}) : "
          f"{min(diff_idx) >= tail_start}")

    # --- что лежит в разошедшихся строках ---
    print("\n  dst_row |    id | ожидалось | получено")
    print("  " + "-" * 58)
    for i in diff_idx[:15]:
        want = decode_row(ref[i])
        got = decode_row(dst[i])
        print(f"  {i:7d} | {int(ids_cpu[i]):5d} | {want:9s} | {got}")
    if n_diff > 15:
        print(f"  ... ещё {n_diff - 15}")

    # --- систематический сдвиг индекса? ---
    shifts = []
    for i in diff_idx[:200]:
        got = decode_row(dst[i])
        if got.startswith("src[") and int(ids_cpu[i]) >= 0:
            got_r = int(got[4:-1])
            shifts.append(got_r - int(ids_cpu[i]))
    if shifts:
        uniq = sorted(set(shifts))
        print(f"\n  сдвиг взятой строки относительно id: {uniq[:10]}"
              f"{' ...' if len(uniq) > 10 else ''}")
        if len(uniq) == 1:
            print(f"  -> постоянный сдвиг {uniq[0]}: ошибка в адресной базе")

    return False


def main():
    print(f"torch  {torch.__version__}")
    print(f"triton {triton.__version__}")
    print(f"\nT2={T2}  D={D}  K_TOPK={K_TOPK}  BLOCK_D_VEC={BLOCK_D_VEC}"
          f"  BLOCK_ROWS={BLOCK_ROWS}")
    print(f"dtype={DTYPE}  index dtype=torch.int32  neg={NEG_FRACTION:.0%}")

    src, ids = build_inputs()
    ids_cpu = ids.cpu()
    print(f"индексы: min={int(ids_cpu.min())} max={int(ids_cpu.max())}"
          f"  уникальных={len(set(ids_cpu.tolist()))}")

    results = {}
    for name, kw in (("orig", {}),
                     ("vec-i32", dict(use_i64=False)),
                     ("vec-i64", dict(use_i64=True))):
        try:
            which = "orig" if name == "orig" else "vec"
            results[name] = run_one(which, src, ids, **kw)
            print(f"\n[ok] {name} отработал")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[FAIL] {name}: {type(exc).__name__}: {exc}")
            results[name] = None

    ref = results.get("orig")
    if ref is None:
        print("\nисходная реализация не отработала — сравнивать не с чем")
        return

    # эталон против самого источника: убеждаемся, что тест корректен
    bad = 0
    for i in range(K_TOPK):
        want = f"src[{int(ids_cpu[i])}]" if int(ids_cpu[i]) >= 0 else "UNTOUCHED"
        if decode_row(ref[i]) != want:
            bad += 1
    print(f"\nсамопроверка эталона: {K_TOPK - bad}/{K_TOPK} строк как ожидается")
    if bad:
        print("  ВНИМАНИЕ: исходный гатер сам не даёт ожидаемого результата,"
              " значит неверна модель страйдов в тесте, а не реализация")

    ok = True
    for name in ("vec-i32", "vec-i64"):
        if results.get(name) is not None:
            ok &= analyse(name, results[name], ids_cpu, ref)

    print(f"\n{'=' * 70}")
    print("ИТОГ:", "векторизованная версия эквивалентна" if ok
          else "есть расхождения, см. выше")
    print("=" * 70)


if __name__ == "__main__":
    main()
