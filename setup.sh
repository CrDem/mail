#!/bin/bash
# ---------------------------------------------------------------------------
# Разовая раскладка A/B-стенда для mojo_opset hw/950-perf.
#
# Делает три вещи:
#   1. создаёт структуру каталогов под $AB_ROOT
#   2. забирает нужные файлы из их CI-каталога (их самих не трогает, только читает)
#   3. проверяет окружение и говорит, чего не хватает
#
# Запуск (положи setup.sh, run_ab.sh и compare_perf.py в один каталог):
#   bash setup.sh /путь/до/их/run_mojo_*.sh
#   bash setup.sh /путь/до/их/каталога
#
# Токен gitcode по умолчанию вытаскивается из их скрипта. Можно задать свой:
#   GITCODE_USER=xxx GITCODE_TOKEN=yyy bash setup.sh /путь/...
#
# Если файлы приехали с Windows:  sed -i 's/\r$//' setup.sh
# ---------------------------------------------------------------------------
set -u -o pipefail

AB_ROOT="${AB_ROOT:-$HOME/ta-ab}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTDATA_SRC="${TESTDATA_SRC:-/data/PytorchFile/Mojo_opset/dllm}"

problems=0
warn()  { echo "  [!]  $*"; }
bad()   { echo "  [X]  $*"; problems=$((problems+1)); }
ok()    { echo "  [ok] $*"; }

strip_cr() { [ -f "$1" ] && sed -i 's/\r$//' "$1"; }

# --------------------------- аргумент: их каталог -------------------------
SRC="${1:-${SRC:-}}"
[ -n "$SRC" ] || { echo "Укажи путь до их скрипта или каталога:  bash setup.sh /path/to/run_mojo_*.sh"; exit 2; }
[ -e "$SRC" ] || { echo "Не существует: $SRC"; exit 2; }

if [ -d "$SRC" ]; then
  SRC_DIR="$(cd "$SRC" && pwd)"
  SRC_SCRIPT="$(find "$SRC_DIR" -maxdepth 1 -name '*.sh' | head -1)"
else
  SRC_DIR="$(cd "$(dirname "$SRC")" && pwd)"
  SRC_SCRIPT="$SRC_DIR/$(basename "$SRC")"
fi
SRC_HELPERS="$SRC_DIR/../script"

echo "=== их каталог:  $SRC_DIR"
echo "=== их скрипт:   ${SRC_SCRIPT:-не найден}"
echo "=== наш AB_ROOT: $AB_ROOT"
echo

# ----------------------------- 1. структура -------------------------------
echo "--- структура каталогов ---"
mkdir -p "$AB_ROOT"/{upstream,jobs/script,jobs/mojo950,suite,results}
ok "$AB_ROOT/{upstream,jobs/script,jobs/mojo950,suite,results}"
echo

# ------------------------- 2. забираем их файлы ---------------------------
# В upstream/ лежат нетронутые эталоны -- чтобы потом видеть, что мы изменили
# и что изменили они, когда пришлют новую версию джоба.
echo "--- копирую их файлы в upstream/ ---"
for f in "$SRC_SCRIPT" \
         "$SRC_DIR/testcase_hw_950_perf.txt" \
         "$SRC_HELPERS/template.xml" \
         "$SRC_HELPERS/perf_statistics_ext.py" \
         "$SRC_HELPERS/setenv.bash"; do
  [ -n "$f" ] || continue
  if [ -f "$f" ]; then
    cp -a "$f" "$AB_ROOT/upstream/" && ok "$(basename "$f")"
  else
    bad "нет $f"
  fi
done

# accuracy-джоб: в их скрипте он закомментирован, но может лежать рядом.
# Забираем просто для справки -- вдруг он и есть гейт на мерж.
echo "--- ищу отдельный accuracy-джоб ---"
found_acc=0
while IFS= read -r f; do
  cp -a "$f" "$AB_ROOT/upstream/" && ok "$(basename "$f") (для справки)" && found_acc=1
done < <(find "$SRC_DIR" -maxdepth 1 \( -iname '*accuracy*' -o -name 'testcase_a3.txt' \) 2>/dev/null)
[ "$found_acc" = "0" ] && warn "рядом ничего про accuracy -- спросить их, входит ли он в гейт"
echo

echo "--- раскладываю по рабочим местам ---"
for f in template.xml perf_statistics_ext.py; do
  [ -f "$AB_ROOT/upstream/$f" ] && cp "$AB_ROOT/upstream/$f" "$AB_ROOT/jobs/script/" && ok "jobs/script/$f"
done
if [ -f "$AB_ROOT/upstream/testcase_hw_950_perf.txt" ]; then
  cp "$AB_ROOT/upstream/testcase_hw_950_perf.txt" "$AB_ROOT/jobs/mojo950/"
  strip_cr "$AB_ROOT/jobs/mojo950/testcase_hw_950_perf.txt"
  ncase=$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "$AB_ROOT/jobs/mojo950/testcase_hw_950_perf.txt" || true)
  ok "jobs/mojo950/testcase_hw_950_perf.txt -- кейсов: $ncase"
fi

# наши скрипты -- из каталога, где лежит setup.sh
for f in run_ab.sh compare_perf.py; do
  if [ -f "$HERE/$f" ]; then
    cp "$HERE/$f" "$AB_ROOT/jobs/mojo950/"
    strip_cr "$AB_ROOT/jobs/mojo950/$f"
    chmod +x "$AB_ROOT/jobs/mojo950/$f"
    ok "jobs/mojo950/$f"
  else
    bad "нет $HERE/$f -- положи его рядом с setup.sh"
  fi
done
echo

# ------------------------------ 3. токен ----------------------------------
echo "--- токен gitcode ---"
CRED="$AB_ROOT/.git-credentials"
if [ -f "$CRED" ]; then
  ok "$CRED уже есть, не перезаписываю"
elif [ -n "${GITCODE_USER:-}" ] && [ -n "${GITCODE_TOKEN:-}" ]; then
  ( umask 077; printf 'https://%s:%s@gitcode.com\n' "$GITCODE_USER" "$GITCODE_TOKEN" > "$CRED" )
  chmod 600 "$CRED"; ok "записан из GITCODE_USER/GITCODE_TOKEN"
elif [ -n "$SRC_SCRIPT" ] && [ -f "$SRC_SCRIPT" ]; then
  # вытаскиваем из их скрипта, чтобы не гонять секрет через буфер обмена
  cred=$(grep -oE 'https://[^:/@[:space:]]+:[^@[:space:]]+@gitcode\.com' "$SRC_SCRIPT" | head -1)
  if [ -n "$cred" ]; then
    ( umask 077; printf '%s\n' "$cred" > "$CRED" )
    chmod 600 "$CRED"
    u="${cred#https://}"; u="${u%%:*}"
    t="${cred#*:}"; t="${t%@gitcode.com}"
    ok "взят из их скрипта: пользователь ${u:0:4}***, длина токена ${#t}"
    warn "это ИХ общий токен. Стоит попросить отдельный на нас."
  else
    bad "в их скрипте не нашёл креды -- задай GITCODE_USER и GITCODE_TOKEN"
  fi
else
  bad "нет ни $CRED, ни GITCODE_USER/GITCODE_TOKEN, ни их скрипта"
fi
echo

# --------------------------- 4. проверки окружения ------------------------
echo "--- окружение ---"
if command -v bishengir-compile >/dev/null; then
  ok "bishengir-compile: $(command -v bishengir-compile)"
else
  bad "нет bishengir-compile в PATH"
fi

if python -c "import torch, torch_npu, triton" 2>/dev/null; then
  ok "$(python -c "import triton,torch;print('triton',triton.__version__,'| torch',torch.__version__)")"
  echo "       $(python -c "import triton;print(triton.__file__)")"
else
  bad "не импортируются torch / torch_npu / triton"
fi

miss=$(python - <<'PY' 2>/dev/null
import importlib.util as u
mods = ("transformers", "einops", "pytest", "pytest_rerunfailures")
print(",".join(m for m in mods if u.find_spec(m) is None))
PY
)
if [ -z "$miss" ]; then
  ok "transformers / einops / pytest / pytest-rerunfailures на месте"
else
  warn "отсутствуют: $miss"
  warn "их CI ставит transformers==5.3.0 и einops. Если ставить -- то ОДИН раз,"
  warn "до всех прогонов, иначе окружение base и pr будет разным."
fi

ncards=$(ls /dev/davinci? 2>/dev/null | wc -l)
if [ "$ncards" -gt 0 ]; then
  ok "карт видно: $ncards ($(ls /dev/davinci? 2>/dev/null | tr '\n' ' '))"
else
  bad "не вижу /dev/davinci?"
fi

if [ -d "$TESTDATA_SRC" ]; then
  ok "тестовые данные: $TESTDATA_SRC"
else
  bad "нет $TESTDATA_SRC -- часть кейсов упадёт"
fi

if command -v git >/dev/null; then
  ok "git: $(git --version)"
else
  bad "нет git"
fi
echo

# ------------------------------- итог -------------------------------------
echo "=========================================================="
if [ "$problems" -gt 0 ]; then
  echo "НЕ ГОТОВО: проблем $problems (строки [X] выше)"
  echo "=========================================================="
  exit 1
fi

echo "Готово. Дальше, по одному прогону:"
cat <<EOF

  # 1. база
  pip install --force-reinstall --no-deps /путь/до/TA-BASE.whl
  mkdir -p $AB_ROOT/results
  TAG=base DEVICE_ID=4 nohup bash $AB_ROOT/jobs/mojo950/run_ab.sh \\
      > $AB_ROOT/results/base.run.log 2>&1 &

  # смотреть
  tail -f $AB_ROOT/results/base.run.log | grep -e '^\[' -e '^===' -e '^---' -e ВНИМАНИЕ -e НЕТ

  # 2. наш PR (опсет уже установлен -> INSTALL_SUITE=0)
  pip install --force-reinstall --no-deps /путь/до/TA-PR.whl
  TAG=pr DEVICE_ID=4 INSTALL_SUITE=0 nohup bash $AB_ROOT/jobs/mojo950/run_ab.sh \\
      > $AB_ROOT/results/pr.run.log 2>&1 &

  # 3. база ВТОРОЙ раз -- тем же whl, что и base. Без этого прогона
  #    нельзя отличить выигрыш от шума, и цифры нечем защищать.
  pip install --force-reinstall --no-deps /путь/до/TA-BASE.whl
  TAG=base2 DEVICE_ID=4 INSTALL_SUITE=0 nohup bash $AB_ROOT/jobs/mojo950/run_ab.sh \\
      > $AB_ROOT/results/base2.run.log 2>&1 &

  # 4. сравнение
  python $AB_ROOT/jobs/mojo950/compare_perf.py --root $AB_ROOT/results

  # 5. что сломалось именно от PR
  diff <(grep -ho "^FAILED [^ ]*" $AB_ROOT/results/base/logs/*.log | sort -u) \\
       <(grep -ho "^FAILED [^ ]*" $AB_ROOT/results/pr/logs/*.log  | sort -u)

  # 6. ловушка с FFTS: предупреждение должно быть в ОБОИХ прогонах или ни в одном
  grep -rh -i -e deprecat -e TRITON_DISABLE_FFTS \\
       $AB_ROOT/results/base/logs/*.log $AB_ROOT/results/pr/logs/*.log | sort -u

EOF
echo "=========================================================="
