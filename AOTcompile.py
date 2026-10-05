"""
AOT-компиляция Triton-кернела ДО TTIR (без NPU/GPU, только CPU),
плюс отдельная точка вызова backend-компилятора (ttir -> linalg).

Идея: то, что обычно скрыто внутри triton.runtime.jit.JITFunction.run() /
triton.compiler.compiler.compile(), здесь развёрнуто руками
"""

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget

# Модуль вашего Ascend backend-а (тот самый файл, который вы прислали)
from triton.backends.ascend.compiler import (
    AscendBackend,
    make_ttir,        # stages["ttir"]     -- то, до чего хотим дойти
    ttir_to_linalg,   # stages["ttadapter"] -- "backend compiler", вызываем отдельно
)


@triton.jit
def _matmul_activation_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Получаем ID блоков
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Смещения для блоков
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Маски для границ
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Аккумулятор
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Матричное умножение
    for k in range(0, K, BLOCK_K):
        # Загружаем A
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & ((k + offs_k) < K), other=0.0)
        
        # Загружаем B
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & mask_n[None, :], other=0.0)
        
        # Умножаем
        acc = tl.dot(a_tile, b_tile, acc)
    
    # Применяем активацию Swish (x * sigmoid(x))
    result = acc * tl.sigmoid(acc)
    
    # Сохраняем результат
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


def main():
    # -----------------------------------------------------------------------
    # 2. Явная специализация кернела (то, что обычно JIT делает сам по
    #    типам реальных аргументов при вызове add_kernel[grid](x, y, ...))
    #    Здесь мы задаём это руками, без единого реального тензора/девайса.
    # -----------------------------------------------------------------------
    signature = {
        "a_ptr": "*fp32",           # матрица A [M, K]
        "b_ptr": "*fp32",           # матрица B [K, N]
        "c_ptr": "*fp32",           # матрица C [M, N] (результат)
        "M": "constexpr",           # строки A
        "N": "constexpr",           # столбцы B
        "K": "constexpr",           # столбцы A / строки B
        "stride_am": "constexpr",   # шаг по строкам A
        "stride_ak": "constexpr",   # шаг по колонкам A
        "stride_bk": "constexpr",   # шаг по строкам B
        "stride_bn": "constexpr",   # шаг по колонкам B
        "stride_cm": "constexpr",   # шаг по строкам C
        "stride_cn": "constexpr",   # шаг по колонкам C
        "BLOCK_M": "constexpr",     # размер блока по M
        "BLOCK_N": "constexpr",     # размер блока по N
        "BLOCK_K": "constexpr",     # размер блока по K
    }

    constexprs = {
        "M": 1024,          # количество строк A
        "N": 2048,          # количество столбцов B
        "K": 4096,          # размерность внутреннего пространства
        "stride_am": 4096,  # обычно равно K
        "stride_ak": 1,
        "stride_bk": 1,
        "stride_bn": 1,
        "stride_cm": 2048,  # обычно равно N
        "stride_cn": 1,
        "BLOCK_M": 64,      # оптимально подбирать под архитектуру GPU
        "BLOCK_N": 128,     # обычно степень двойки: 64, 128, 256
        "BLOCK_K": 64,      # обычно 32, 64, 128
    }

    # "arch" здесь может быть произвольной строкой на этом этапе — она нужна
    # реальному bishengir-compile'у позже, а не фронтенду/пассам ttir.
    target = GPUTarget(backend="npu", arch="ascend950pr", warp_size=32)

    backend = AscendBackend(target)
    options = backend.parse_options({})  # NPUOptions с дефолтами

    src = triton.compiler.ASTSource(
        fn=_matmul_activation_kernel,          # именно JITFunction, не "голая" python-функция
        signature=signature,
        constants=constexprs,
    )

    # -----------------------------------------------------------------------
    # 3. То, что compile() делает перед стадиями: контекст + диалекты
    # -----------------------------------------------------------------------
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()

    # AST -> "черновой" ttir (ещё без общих оптимизационных пассов)
    module = src.make_ir(options, codegen_fns, module_map, context)

    # -----------------------------------------------------------------------
    # 4. Ровно stages["ttir"] — общий для всех backend-ов проход пассов.
    #    Это и есть финальный TTIR, который в штатном compile() ушёл бы
    #    дальше в backend. Мы просто вызываем его руками и останавливаемся.
    # -----------------------------------------------------------------------
    metadata = {}
    module = make_ttir(module, metadata, options)

    ttir_text = str(module)
    print("=" * 80)
    print("TTIR (то, что дошло бы до backend-компилятора):")
    print("=" * 80)
    print(ttir_text)

    with open("kernel.ttir.mlir", "w") as f:
        f.write(ttir_text)
    print("[ok] TTIR сохранён в kernel.ttir.mlir")

    # -----------------------------------------------------------------------
    # 5. Точка вызова backend-компилятора отдельно ("ttadapter" стадия).
    #    Именно здесь ttir_to_linalg дёргает triton-adapter-opt / готовит
    #    вход для bishengir-compile. На чисто-CPU машине без установленного
    #    ascend-тулчейна (triton-adapter-opt, bishengir-compile и т.п.)
    #    это ожидаемо упадёт -- граница компиляции именно тут, а не раньше.
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("Пробуем вызвать backend-компилятор отдельно (ttadapter stage)...")
    print("=" * 80)

    # ttir_to_linalg читает много полей metadata как metadata["..."] (без .get),
    # поэтому нужно предварительно заполнить дефолты NPUOptions в metadata,
    # как это делает compile() перед прогоном стадий: metadata = {**options.__dict__, ...}
    metadata.update(options.__dict__)
    metadata.setdefault("hash", "manual-aot-run")

    try:
        linalg_ir = ttir_to_linalg(module, metadata, options, named_ops=True)
        print("[ok] Получен linalg IR, backend-toolchain доступен в этой среде.")
        with open("kernel.ttadapter.mlir", "w") as f:
            f.write(linalg_ir)
        print("[ok] Сохранён в kernel.ttadapter.mlir")
    except Exception as e:
        print(f"[expected on CPU-only машине без ascend toolchain] {type(e).__name__}: {e}")
        print("Это нормально — до сюда AOT дошёл без единого реального устройства,")
        print("а дальше нужен уже сам ascend-компилятор (bishengir-compile и т.п.).")


if __name__ == "__main__":
    main()