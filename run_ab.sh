#!/bin/bash
# ---------------------------------------------------------------------------
# mojo_opset hw/950-perf  --  A/B двух whl triton-ascend на одной и той же сюите.
#
# Переписано с их CI-скрипта. Отличия от оригинала:
#   * ничего не пишет в чужие каталоги, всё живёт под $AB_ROOT
#   * не сорсит их setenv.bash и не трогает прокси; окружение наше собственное
#   * не доустанавливает transformers/einops (только проверяет и сообщает)
#   * не трогает глобальный gitconfig, токен лежит в отдельном файле
#   * сюита клонируется один раз и переиспользуется -> оба прогона побитово равны
#   * результаты раскладываются по results/$TAG, второй прогон не сносит первый
#   * perf собирается из kernel_details*.csv (как и ждёт perf_statistics_ext.py)
#   * старые kernel_details*.csv убиваются до прогона -> PR не подхватит базу
#   * честный код возврата
#
# Запуск:
#   TAG=base  DEVICE_ID=4 bash run_ab.sh
#   TAG=pr    DEVICE_ID=4 INSTALL_SUITE=0 bash run_ab.sh
#   TAG=base2 DEVICE_ID=4 INSTALL_SUITE=0 bash run_ab.sh   # шум прогона-к-прогону
#
# Если файл приехал с Windows:  sed -i 's/\r$//' run_ab.sh
# ---------------------------------------------------------------------------
set -u -o pipefail

# ============================== конфиг ====================================
AB_ROOT="${AB_ROOT:-$HOME/ta-ab}"
TAG="${TAG:?TAG не задан: TAG=base|pr|base2}"
DEVICE_ID="${DEVICE_ID:?DEVICE_ID не задан, например DEVICE_ID=4 или DEVICE_ID=4,2,3}"

# --- флаги тритона, которые их CI выставляет в setenv.bash ---
# Ставим прямо здесь, никакого внешнего set-env не сорсим.
#
# TRITON_ALWAYS_COMPILE=1 -- кэш triton не переиспользуется вообще, то есть
#   PR-сборка физически не может подхватить кернелы, скомпилированные базой.
#
# TRITON_DISABLE_FFTS -- ВНИМАНИЕ: в свежем TA она deprecated и ИГНОРИРУЕТСЯ,
#   см. third_party/ascend/backend/utils.py:
#     _DEPRECATED_ASCEND_ENV_VARS -> "TRITON_DISABLE_FFTS"
#     "ignored; FFTS policy is derived from the explicit target architecture"
#     force_disable_ffts(arch) -> is_compile_on_910_95(arch)
#     is_ffts_supported(arch)  -> False для 910_95
#   На 950-таргете FFTS выключен и без переменной. Но старый baseline-whl может
#   её ещё читать -- тогда часть разницы в цифрах будет НЕ от нашего PR.
#   Держим выставленной в обоих прогонах и проверяем логи на deprecation warning:
#     grep -rh -i -e deprecat -e TRITON_DISABLE_FFTS results/*/logs/*.log | sort -u
#   Если предупреждение только в одном прогоне -- перепрогнать с пустым значением.
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_DISABLE_FFTS="${TRITON_DISABLE_FFTS:-1}"

# --- сюита ---
SUITE_URL="${SUITE_URL:-https://gitcode.com/TritonAscendTest/mojo_opset}"
SUITE_BRANCH="${SUITE_BRANCH:-hw/950-perf}"
SUITE_DIR="${SUITE_DIR:-$AB_ROOT/suite/mojo_opset}"
GIT_CRED="${GIT_CRED:-$AB_ROOT/.git-credentials}"

TESTDATA_SRC="${TESTDATA_SRC:-/data/PytorchFile/Mojo_opset/dllm}"
TESTDATA_DST="$SUITE_DIR/mojo_opset/tests/specific_shape_perf_acc/test_data"

JOB_DIR="${JOB_DIR:-$AB_ROOT/jobs/mojo950}"
HELPERS="${HELPERS:-$AB_ROOT/jobs/script}"   # template.xml, perf_statistics_ext.py
LIST_FILE="${LIST_FILE:-$JOB_DIR/testcase_hw_950_perf.txt}"

MOJO_BACKEND="${MOJO_BACKEND:-ttx}"
INSTALL_SUITE="${INSTALL_SUITE:-1}"          # pip install -e . --no-deps, нужен 1 раз
# ==========================================================================

RUN_DIR="$AB_ROOT/results/$TAG"
LOG_DIR="$RUN_DIR/logs"
CACHE_DIR="$RUN_DIR/cache"
PERF_DIR="$RUN_DIR/perf"
PROV="$RUN_DIR/provenance.txt"
SUMMARY="$RUN_DIR/summary.txt"

echo "=== TAG=$TAG  DEVICE_ID=$DEVICE_ID  AB_ROOT=$AB_ROOT ==="

[ -f "$LIST_FILE" ]            || { echo "НЕТ списка кейсов: $LIST_FILE"; exit 2; }
[ -f "$HELPERS/template.xml" ] || { echo "НЕТ template.xml в $HELPERS"; exit 2; }
[ -f "$HELPERS/perf_statistics_ext.py" ] || echo "ВНИМАНИЕ: нет perf_statistics_ext.py, perf-сводки не будет"

# ----------------------- preflight по окружению ---------------------------
# Окружение (CANN, bishengir, TA) считаем уже настроенным в контейнере.
# Здесь только убеждаемся в этом, чтобы не обнаружить проблему на 40-м кейсе.
command -v bishengir-compile >/dev/null || { echo "НЕТ bishengir-compile в PATH"; exit 2; }
python -c "import torch, torch_npu, triton" 2>/dev/null \
  || { echo "НЕ импортируются torch / torch_npu / triton -- окружение не готово"; exit 2; }
python -c "import pytest" 2>/dev/null || { echo "НЕТ pytest"; exit 2; }

# Реран-флаги их CI (--reruns/--only-rerun) -- это опции плагина
# pytest-rerunfailures. Без плагина pytest падает на РАЗБОРЕ АРГУМЕНТОВ с rc=4,
# то есть ни один тест не выполняется, а template.xml даёт "всё упало".
# Поэтому флаги добавляем только если плагин есть.
if python -c "import pytest_rerunfailures" 2>/dev/null; then
  RERUN_OPTS=(--reruns 1 --reruns-delay 60
              --only-rerun "Failed to start the device|out of memory")
else
  RERUN_OPTS=()
  echo "ВНИМАНИЕ: нет pytest-rerunfailures -- реранов на флейки (device not started,"
  echo "          out of memory) не будет. Такие падения придётся отличать руками."
fi
python -c "import transformers" 2>/dev/null \
  || echo "ВНИМАНИЕ: нет transformers -- кейсы, которые его импортируют, упадут"

rm -rf "$RUN_DIR"; mkdir -p "$LOG_DIR" "$CACHE_DIR" "$PERF_DIR"

# ---------------- провенанс: что именно тестируем -------------------------
# Узко, без дампа env / pip list: в них лежат прокси-креды и токен gitcode.
{
  date +"%F %T"
  echo "TAG=$TAG DEVICE_ID=$DEVICE_ID MOJO_Backend=$MOJO_BACKEND"
  echo "TRITON_ALWAYS_COMPILE=$TRITON_ALWAYS_COMPILE TRITON_DISABLE_FFTS=$TRITON_DISABLE_FFTS"
  echo "bishengir-compile: $(command -v bishengir-compile)"
  python -c "import triton,torch;print('triton',triton.__version__,triton.__file__);print('torch',torch.__version__)"
  python -c "import torch_npu;print('torch_npu',torch_npu.__version__)" 2>/dev/null
  pip list 2>/dev/null | grep -i -e triton -e torch -e transformers -e einops
} > "$PROV" 2>&1
cat "$PROV"

# ------------------------------- сюита ------------------------------------
if [ ! -d "$SUITE_DIR/.git" ]; then
  echo "--- clone $SUITE_BRANCH (один раз) ---"
  [ -f "$GIT_CRED" ] || { echo "НЕТ $GIT_CRED с токеном"; exit 2; }
  mkdir -p "$(dirname "$SUITE_DIR")"
  GIT_TERMINAL_PROMPT=0 git \
      -c credential.helper="store --file=$GIT_CRED" \
      -c http.sslVerify=false \
      clone -b "$SUITE_BRANCH" "$SUITE_URL" "$SUITE_DIR" || exit 2
else
  echo "--- сюита уже склонирована, переиспользуем ---"
fi
git -C "$SUITE_DIR" log -1 --format='suite commit: %H %s' | tee -a "$PROV"

# --------------------------- тестовые данные ------------------------------
if [ -d "$TESTDATA_SRC" ]; then
  mkdir -p "$TESTDATA_DST"
  [ -d "$TESTDATA_DST/$(basename "$TESTDATA_SRC")" ] || cp -r "$TESTDATA_SRC" "$TESTDATA_DST/"
  echo "test_data: $(ls "$TESTDATA_DST")"
else
  echo "ВНИМАНИЕ: нет $TESTDATA_SRC -- кейсы на этих данных упадут"
fi

# ------------- установка опсета, без подтягивания зависимостей ------------
if [ "$INSTALL_SUITE" = "1" ]; then
  ( cd "$SUITE_DIR" && pip install -e . --no-deps --no-build-isolation ) \
    || echo "ВНИМАНИЕ: pip install -e . не прошёл (PYTHONPATH=./ может спасти)"
fi

python - <<'PY'
import importlib.util as u
mods = ("torch", "torch_npu", "triton", "transformers", "einops", "pytest", "pytest_rerunfailures")
miss = [m for m in mods if u.find_spec(m) is None]
print("ОТСУТСТВУЮТ МОДУЛИ:", miss if miss else "нет, всё на месте")
PY

# TA не должен был перетереться при установке опсета
python -c "import triton;print('triton после установки опсета:',triton.__version__,triton.__file__)" | tee -a "$PROV"

# ------------------------------- прогон -----------------------------------
cd "$SUITE_DIR"
export PYTHONPATH="./:${PYTHONPATH:-}"
rm -rf "$SUITE_DIR/logs" && mkdir -p "$SUITE_DIR/logs"

# КРИТИЧНО: чекаут общий на оба прогона, значит старые профайлерные csv надо
# убить, иначе perf-сводка PR склеится с цифрами базы. У них это не всплывало
# только потому, что каталог реклонится каждый запуск.
stale=$(find "$SUITE_DIR" \( -name 'kernel_details*.csv' -o -name '*operator_perf.csv' \) | wc -l)
echo "--- чищу $stale старых perf-csv ---"
find "$SUITE_DIR" \( -name 'kernel_details*.csv' -o -name '*operator_perf.csv' \) -delete

echo "видно карт: $(ls /dev/davinci? 2>/dev/null | wc -l)"
start_time=$(date +"%F %T"); echo "===== старт: $start_time ====="

fifo="/tmp/ab.$$.fifo"; mkfifo "$fifo"; exec 9<>"$fifo"; rm -f "$fifo"
IFS=',' read -ra ids <<< "$DEVICE_ID"
for id in "${ids[@]}"; do echo "$id" >&9; done

ncase=0
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z $line || $line == \#* ]] && continue
  ncase=$((ncase+1))
  safe="${line%.py}"; safe="${safe//\//_}"; safe="${safe//::/__}"
  test_log="$LOG_DIR/${safe}.log"
  test_xml="$LOG_DIR/${safe}_results.xml"
  read -u 9 npu_id
  {
    cp "$HELPERS/template.xml" "$test_xml"
    TRITON_CACHE_DIR="$CACHE_DIR/$safe" \
    ASCEND_RT_VISIBLE_DEVICES="$npu_id" \
    MOJO_Backend="$MOJO_BACKEND" \
    pytest -sv ${RERUN_OPTS[@]+"${RERUN_OPTS[@]}"} \
        "$line" --junitxml="$test_xml" > "$test_log" 2>&1
    rc=$?
    echo "[done rc=$rc npu=$npu_id] $line"
    echo "$npu_id" >&9
  } &
  echo "[start npu=$npu_id] $line"
done < "$LIST_FILE"
wait; exec 9>&-

end_time=$(date +"%F %T")
echo "===== финиш: $end_time   кейсов запущено: $ncase ====="

# IR оставляем: base vs pr можно диффить
find "$CACHE_DIR" -type f ! -name "*.ttadapter" ! -name "*.ttir" -delete 2>/dev/null

# ---------------------------- статистика ----------------------------------
python - "$LOG_DIR" <<'PY' | tee "$RUN_DIR/stats.txt"
import glob, os, sys, xml.etree.ElementTree as ET
d = sys.argv[1]
t = f = s = e = 0
files = glob.glob(os.path.join(d, '*.xml'))
for x in files:
    try:
        root = ET.parse(x).getroot()
    except Exception as ex:
        print('bad xml', x, ex, file=sys.stderr); continue
    suites = root.findall('testsuite') or ([root] if root.tag == 'testsuite' else [])
    for ts in suites:
        g = lambda k: int(ts.get(k, 0) or 0)
        t += g('tests'); f += g('failures'); s += g('skipped'); e += g('errors')
p = t - f - s - e
den = p + f + e
print(f'xml_files={len(files)}')
print(f'total={t} passed={p} failed={f} skipped={s} errors={e}')
print('rate=%s' % (f'{p*100/den:.1f}%' if den else 'n/a'))
PY

# ------------------------------- perf -------------------------------------
# perf_statistics_ext.py рекурсивно глобает kernel_details.csv и
# kernel_details_*.csv ОТ ТЕКУЩЕГО КАТАЛОГА, caseid берёт из имени файла.
# Поэтому: забираем csv из сюиты с сохранением относительных путей, originals
# удаляем, и запускаем помощник уже в каталоге со собранными csv.
cd "$SUITE_DIR"
found=0
while IFS= read -r f; do
  cp --parents "$f" "$PERF_DIR/" 2>/dev/null && found=$((found+1))
done < <(find . \( -name 'kernel_details*.csv' -o -name '*operator_perf.csv' \))
echo "--- собрано perf-csv: $found ---"
find "$SUITE_DIR" \( -name 'kernel_details*.csv' -o -name '*operator_perf.csv' \) -delete

if [ "$found" -gt 0 ] && [ -f "$HELPERS/perf_statistics_ext.py" ]; then
  cd "$PERF_DIR"
  # помощник написан как pytest-тест (def test_performance_comparison, pytest.fail
  # внутри). Если в нём есть __main__ -- сработает python, иначе добираем pytest.
  # cwd в обоих случаях = $PERF_DIR, так что рекурсивный глоб видит csv.
  python "$HELPERS/perf_statistics_ext.py" > "$RUN_DIR/perf_tool.log" 2>&1 \
    || pytest -q "$HELPERS/perf_statistics_ext.py" >> "$RUN_DIR/perf_tool.log" 2>&1 \
    || echo "ВНИМАНИЕ: perf_statistics_ext.py не отработал, смотри $RUN_DIR/perf_tool.log"
  if [ -f npu_performance_summary.csv ]; then
    cp npu_performance_summary.csv "$RUN_DIR/perf_summary_$TAG.csv"
  else
    echo "ВНИМАНИЕ: npu_performance_summary.csv не создан"
  fi
else
  echo "ВНИМАНИЕ: kernel_details*.csv не найдены -- профайлер не писал, perf-цифр нет."
  echo "          Проверь, что в списке кейсов есть perf-тесты."
fi

# ------------------------------ итоги -------------------------------------
{ echo "=== $TAG  $start_time .. $end_time ==="; cat "$PROV"; cat "$RUN_DIR/stats.txt"; } >> "$SUMMARY"
cp -r "$SUITE_DIR/logs/." "$LOG_DIR/" 2>/dev/null

echo
echo "логи:       $LOG_DIR"
echo "статистика: $RUN_DIR/stats.txt"
echo "perf:       $RUN_DIR/perf_summary_$TAG.csv"
echo "сырые csv:  $PERF_DIR"
echo "IR:         $CACHE_DIR"
echo "провенанс:  $PROV"

bad=$(awk -F'failed=' '/failed=/{split($2,a," ");print a[1]}' "$RUN_DIR/stats.txt")
err=$(awk -F'errors=' '/errors=/{split($2,a," ");print a[1]}' "$RUN_DIR/stats.txt")
[ "${bad:-1}" = "0" ] && [ "${err:-1}" = "0" ] && exit 0 || exit 1
