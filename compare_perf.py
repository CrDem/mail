#!/usr/bin/env python3
"""Сравнение perf-сводок A/B-прогонов mojo_opset hw/950-perf.

Ожидает ~/ta-ab/results/{base,base2,pr}/perf_summary_<TAG>.csv, которые
делает run_ab.sh (они же -- выход perf_statistics_ext.py).

base2 -- повторный прогон той же базы. Он нужен не для красоты: без него
нельзя отличить выигрыш от шума прогона-к-прогону. Кейс считается
регрессией или ускорением только если pr_% выходит за |noise_%|.

Запуск:
    python compare_perf.py                      # ключ и каталог по умолчанию
    python compare_perf.py --key "Op Name"      # если шапка другая
    python compare_perf.py --root /path/results
    python compare_perf.py --threshold 5        # порог значимости, %

Колонки, которые собирает perf_statistics_ext.py:
    Duration(us), aic_mac_ratio, aic_mte2_ratio, aiv_vec_ratio, aiv_mte2_ratio
Сдвиг ratio-колонок между base и pr -- прямая проверка того, перераспределил
ли PR нагрузку между cube и vector, а не просто "стало быстрее".
"""

import argparse
import os
import sys

import pandas as pd

PERF_COLS = [
    "Duration(us)",
    "aic_mac_ratio",
    "aic_mte2_ratio",
    "aiv_vec_ratio",
    "aiv_mte2_ratio",
]
DUR = "Duration(us)"
KEY_CANDIDATES = ["caseid", "case_id", "Name", "Op Name", "op_name", "Kernel Name", "kernel_name"]


def load(root, tag):
    path = os.path.join(root, tag, f"perf_summary_{tag}.csv")
    if not os.path.isfile(path):
        return None, path
    return pd.read_csv(path), path


def pick_key(df, explicit):
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"нет колонки {explicit!r}. Есть: {list(df.columns)}")
        return explicit
    for k in KEY_CANDIDATES:
        if k in df.columns:
            return k
    sys.exit(f"не угадал колонку-ключ, передай --key. Есть: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/ta-ab/results"))
    ap.add_argument("--key", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог значимости в %%; по умолчанию берётся из шума base vs base2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = {}
    for tag in ("base", "base2", "pr"):
        df, path = load(args.root, tag)
        if df is None:
            if tag == "base2":
                print(f"[warn] нет {path} -- шум прогона-к-прогону неизвестен, "
                      f"выводы по perf будут слабыми")
                continue
            sys.exit(f"нет {path}")
        frames[tag] = df

    key = pick_key(frames["base"], args.key)
    use = [c for c in PERF_COLS if c in frames["base"].columns]
    if not use:
        sys.exit(f"ни одной perf-колонки не нашлось. Есть: {list(frames['base'].columns)}")
    print(f"ключ: {key}   колонки: {use}\n")

    merged = None
    for tag, df in frames.items():
        part = df[[key] + use].copy()
        part.columns = [key] + [f"{c}__{tag}" for c in use]
        merged = part if merged is None else merged.merge(part, on=key, how="outer")

    if DUR in use:
        base, pr = f"{DUR}__base", f"{DUR}__pr"
        if "base2" in frames:
            merged["noise_%"] = (merged[f"{DUR}__base2"] / merged[base] - 1) * 100
        merged["pr_%"] = (merged[pr] / merged[base] - 1) * 100
        merged = merged.sort_values("pr_%")

        if args.threshold is not None:
            thr = args.threshold
            src = "задан вручную"
        elif "noise_%" in merged:
            thr = merged["noise_%"].abs().quantile(0.95)
            src = "95-й процентиль |noise_%|"
        else:
            thr = None
            src = None

        if thr is not None:
            merged["verdict"] = merged["pr_%"].apply(
                lambda v: "noise" if pd.isna(v) or abs(v) <= thr
                else ("FASTER" if v < 0 else "SLOWER"))

    # кейсы, которых нет в одном из прогонов -- это тоже результат
    if DUR in use:
        only = merged[merged[f"{DUR}__base"].isna() | merged[f"{DUR}__pr"].isna()]
        if not only.empty:
            print(f"[warn] {len(only)} кейсов есть только в одном прогоне:")
            print(only[[key]].to_string(index=False), "\n")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(merged.to_string(index=False))

    if DUR in use and "verdict" in merged:
        print(f"\nпорог значимости: {thr:.2f}% ({src})")
        print(merged["verdict"].value_counts().to_string())

    out = args.out or os.path.join(args.root, "ab_compare.csv")
    merged.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
