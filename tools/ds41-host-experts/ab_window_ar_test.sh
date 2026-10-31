#!/usr/bin/env bash
# Source-only fixtures for ArN parsing, fresh SHA receipts, startup/KLD guards and arm resets.
# Every checker, container command and server is fake; no model, GPU or service is accessed.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-ar-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
for fn in ar_rows_parse ar_rows_server_gate ar_rows_server_check ar_rows_receipt_ok ar_rows_check ar_rows_gate xt_check kld_finish; do
  eval "$(sed -n "/^$fn() {/,/^}/p" "$W")"
  declare -F "$fn" >/dev/null || { bad "missing $fn"; exit 1; }
done
OUT=$T/out
HERE=$T
mkdir -p "$OUT"
log() { echo "LOG $*" >>"$OUT/window.log"; }
for p in Ar1 Ar64 Ar1024 Ar00064; do
  ar_rows_parse "$p" >/dev/null && ok "parse $p" || bad "parse $p"
done
for p in Ar Ar0 Ar000 Ar-1 Ar1x Ar9999999999; do
  if ar_rows_parse "$p" >/dev/null; then bad "accepted $p"; else ok "refuse $p"; fi
done

line() { echo "radiance: deepseek41: exact AR row slices: at most $1 rows (RADIANCE_EXPERIMENTAL_DS41_AR_ROWS); all_reduce bounds and launches sliced, Engram all_gather independently controlled"; }
line 1024 >"$OUT/on.server.log"
line 16 >"$OUT/clamped.server.log"
echo plain >"$OUT/off.server.log"
expect_log() {
  local want=$1 file=$2 req=$3 cap=$4 rc
  ar_rows_server_gate "$OUT/$file.server.log" "$req" "$cap" >/dev/null; rc=$?
  [[ $rc == "$want" ]] && ok "startup $file requested=$req actual=$cap" || bad "startup $file: $rc != $want"
}
expect_log 0 on 1024 1024
expect_log 0 clamped 1024 16
expect_log 1 on 64 64
expect_log 1 off 1024 1024
expect_log 0 off 0 0
expect_log 1 on 0 0
line 1024 >>"$OUT/on.server.log"
expect_log 1 on 1024 1024
line 1024 >"$OUT/on.server.log"

# KLD follows the same actual-cap gate; all process operations and the corpus comparison are stubbed.
lcp_check() { return 0; }
# Other independently tested instruments also guard KLD in the integration harness.
n128_server_gate() { return 0; }
vp_server_gate() { return 0; }
qsp_gate() { return 0; }
ARGS=()
ag_rows_server_check() { return 0; }
kill() { return 1; }
wait() { return 0; }
python3() { echo compared >>"$T/compares"; return 0; }
ARM_XT= ARM_BT= ARM_LC= ARM_AR_ROWS=1024 ARM_AR_CAP=1024
SERVER_PID=999999 KLD_REF=$T/reference
kld_finish off >/dev/null && bad 'KLD accepted missing cap proof' || ok 'KLD rejects missing cap proof'
SERVER_PID=999999
kld_finish on >/dev/null && ok 'KLD accepts actual cap before comparison' || bad 'KLD rejected matching cap'
[[ -s $T/compares ]] || bad 'KLD did not reach comparison'
SERVER_PID=999999 KLD_REF=
kld_finish on >/dev/null && bad 'Ar became KLD reference' || ok 'Ar cannot become KLD reference'

H=$T/home
mkdir -p "$H/kernels"
LIB=$H/kernels/libr4d.so
echo fixture >"$LIB"
SUM=$(sha256sum "$LIB" | cut -d' ' -f1)
export AR_CALLS=$T/checker.calls
cat >"$T/checker" <<'EOF'
#!/usr/bin/env bash
if [[ -n ${HIP_VISIBLE_DEVICES+x}${ROCR_VISIBLE_DEVICES+x}${CUDA_VISIBLE_DEVICES+x}${R4D_SELFTEST_ONE_CARD+x} ]]; then
  echo 'FAIL inherited card mask'; exit 1
fi
[[ $2 == --case && $3 == ds41_ar_rows_exact ]] || exit 2
echo called >>"$AR_CALLS"
for cap in 1024 64; do for ts in 0 32768; do echo "  ok    ds41_ar_rows_exact cap $cap, ts_min $ts"; done; done
[[ ${MUTATE_AR:-0} == 1 ]] && echo mutation >>"$1"
exit 0
EOF
chmod +x "$T/checker"
SELFTEST_AR_ROWS=$T/checker
declare -A AR_ROWS_PASSED=() AR_ROWS_RECEIPT=()
fake_docker() { echo "${FAKE_STATE:-exited}"; }
DOCKER=(fake_docker) NAME=fake
AR_ROWS_WINDOW_ACTIVE=0
ar_rows_check "$LIB" >/dev/null && bad 'check outside window' || ok 'checker refuses outside window'
[[ ! -e $AR_CALLS ]] || bad 'outside-window checker ran'
AR_ROWS_WINDOW_ACTIVE=1 FAKE_STATE=running
ar_rows_check "$LIB" >/dev/null && bad 'check beside running container' || ok 'checker requires exited container'
FAKE_STATE=exited
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 R4D_SELFTEST_ONE_CARD=1
# A stale file at the expected name must be replaced by this window's fresh checker call.
echo stale >"$OUT/ar-rows-$SUM.log"
ar_rows_check "$LIB" >/dev/null && ok 'fresh checker clears all card masks and replaces stale log' || bad 'fresh checker failed'
ar_rows_check "$LIB" >/dev/null || bad 'same-library cached receipt rejected'
[[ $(wc -l <"$AR_CALLS") == 1 ]] && ok 'one fresh check per full SHA' || bad 'checker repeated'
ar_rows_gate good "$LIB" >/dev/null || bad 'valid receipt gate failed'
F=${AR_ROWS_RECEIPT[$SUM]}
for kind in fail skip missing exit duplicate wrong_sha; do
  cp "$F" "$T/bad.log"
  case $kind in
    fail) echo '  FAIL bad value' >>"$T/bad.log" ;;
    skip) echo '  skip no cards' >>"$T/bad.log" ;;
    missing) sed -i '/cap 64, ts_min 0$/d' "$T/bad.log" ;;
    exit) sed -i 's/ar_rows exit 0/ar_rows exit 1/' "$T/bad.log" ;;
    duplicate) echo "ar_rows sha256 $SUM" >>"$T/bad.log" ;;
    wrong_sha) other=${SUM%?}0; [[ $other == "$SUM" ]] && other=${SUM%?}1
               sed -i "s/$SUM/$other/g" "$T/bad.log" ;;
  esac
  ar_rows_receipt_ok "$T/bad.log" "$SUM" && bad "accepted $kind receipt" || ok "reject $kind receipt"
done
echo changed >>"$LIB"
ar_rows_gate changed "$LIB" >/dev/null && bad 'accepted changed served SHA' || ok 'reject changed served SHA'
export MUTATE_AR=1
ar_rows_check "$LIB" >/dev/null && bad 'accepted library mutation during checker' || ok 'reject pre/post SHA change'
unset MUTATE_AR

# Arm parsing only: no checker or server. Existing no-Ar helpers need no AR receipt variables.
ARMS='base base-Ar64-BT128 base-Ar1024-Kld base-Ar0 base-Arfoo base-Ar64-Ar1024'
out=$(
  set +u
  log() { echo "LOG $*"; }; power_set() { :; }; power_restore() { :; }
  ar_rows_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "CAP ${ARM_AR_CAP}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H PROMPTS=$T/prompts.json
  export RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=777   # each arm must override a window-level leak
  echo '{}' >"$PROMPTS"
  echo '{}' >"$OUT/kld-corpus.jsonl"
  CANCEL=/nonexistent/cancel
  unset RADIANCE_TUNE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ {print NR; exit}' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
)
echo "$out" >"$T/loop.txt"
section() { awk -v name="ARM $1" '$0==name {p=1;next} /^ARM / {p=0} p' "$T/loop.txt"; }
for pair in 'base:0' 'base-Ar64-BT128:64' 'base-Ar1024-Kld:1024'; do
  a=${pair%:*} n=${pair##*:}
  grep -qx "    RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=$n" <<<"$(section "$a")" && ok "$a explicit env=$n" || bad "$a missing explicit env"
done
grep -qx 'CAP 64' <<<"$(section base-Ar64-BT128)" || bad 'wrong BT actual cap'
grep -qx 'CAP 16' <<<"$(section base-Ar1024-Kld)" || bad 'wrong default actual cap'
for a in base-Ar0 base-Arfoo base-Ar64-Ar1024; do
  [[ -z $(section "$a") ]] && ok "skip $a" || bad "launched $a"
done
(( fail )) && exit 1
echo 'ab_window_ar_test: all checks passed'
