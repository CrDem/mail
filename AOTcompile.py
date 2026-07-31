"""
AOT-компиляция Triton-кернела ДО TTIR (без NPU/GPU, только CPU),
плюс отдельная точка вызова backend-компилятора (ttir -> linalg).

Идея: то, что обычно скрыто внутри triton.runtime.jit.JITFunction.run() /
triton.compiler.compiler.compile(), здесь развёрнуто руками:

  1. JITFunction (декоратор @triton.jit) сам по себе НИЧЕГО не компилирует.
     Компиляция начинается только когда вызывают compile(ASTSource(...)),
     либо когда JIT-функцию реально вызывают с конкретными аргументами
     (тогда triton сам строит ASTSource на основе типов аргументов).

  2. compile() внутри делает (см. add_stages backend-а):
        ctx = ir.context()
        ir.load_dialects(ctx)
        backend.load_dialects(ctx)
        module = src.make_ir(options, codegen_fns, module_map, ctx)   # AST -> "грязный" ttir
        stages = {}
        backend.add_stages(stages, options)   # {"ttir": make_ttir, "ttadapter": ttir_to_linalg, "npubin": ...}
        module = stages["ttir"](module, metadata)        # <-- обычный, backend-agnostic TTIR
        module = stages["ttadapter"](module, metadata)   # <-- ЭТО и есть "передать в backend-компилятор"
        ...

  Нам нужно остановиться сразу после шага "ttir" и вызвать "ttadapter"
  отдельно, руками, когда сами решим.

ВАЖНО (версионнозависимые места, проверьте под свой triton-ascend):
  - Сигнатуры add_stages/make_ir/get_codegen_implementation могут отличаться
    между версиями triton-core. Здесь используются сигнатуры ИЗ ВАШЕГО
    файла triton/backends/ascend/compiler.py (2 аргумента в add_stages,
    без `options` в get_codegen_implementation, 4 аргумента в make_ir).
  - `NPUOptions.compile_on_910_95` в вашем файле берёт значение из
    `triton.tools.get_ascend_devices.is_compile_on_910_95` — если это
    функция, а не готовый bool, и она пытается спросить реальное
    устройство при импорте модуля — на чисто-CPU машине это может
    бросить исключение ещё на этапе `import`. Если увидите ошибку именно
    на импорте `triton.backends.ascend.compiler` — сначала разберитесь
    с этим импортом (замокать/поставить env-переменную), это не связано
    с нашей логикой AOT.
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


# ---------------------------------------------------------------------------
# 1. Простейший тритон-кернел
# ---------------------------------------------------------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    # -----------------------------------------------------------------------
    # 2. Явная специализация кернела (то, что обычно JIT делает сам по
    #    типам реальных аргументов при вызове add_kernel[grid](x, y, ...))
    #    Здесь мы задаём это руками, без единого реального тензора/девайса.
    # -----------------------------------------------------------------------
    signature = {
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "out_ptr": "*fp32",
        "n_elements": "i32",
        "BLOCK_SIZE": "constexpr",
    }
    constexprs = {"BLOCK_SIZE": 1024}

    # "arch" здесь может быть произвольной строкой на этом этапе — она нужна
    # реальному bishengir-compile'у позже, а не фронтенду/пассам ttir.
    target = GPUTarget(backend="npu", arch="ascend910b", warp_size=32)

    backend = AscendBackend(target)
    options = backend.parse_options({})  # NPUOptions с дефолтами

    src = triton.compiler.ASTSource(
        fn=add_kernel,          # именно JITFunction, не "голая" python-функция
        signature=signature,
        constexprs=constexprs,
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