#!/usr/bin/env bash
# Source-only fixtures for AgN parsing, fresh SHA receipts, startup/KLD guards and arm resets.
# Every checker, container command and server is fake; no model, GPU or service is accessed.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-ag-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
for fn in serving_cards serving_cards_ok ag_rows_parse ag_rows_server_gate ag_rows_server_check ag_rows_receipt_ok ag_rows_check ag_rows_gate xt_check kld_finish; do
  eval "$(sed -n "/^$fn() {/,/^}/p" "$W")"
  declare -F "$fn" >/dev/null || { bad "missing $fn"; exit 1; }
done
OUT=$T/out
HERE=$T
mkdir -p "$OUT"
log() { echo "LOG $*" >>"$OUT/window.log"; }
for p in Ag1 Ag64 Ag1024 Ag00064; do
  ag_rows_parse "$p" >/dev/null && ok "parse $p" || bad "parse $p"
done
for p in Ag Ag0 Ag000 Ag-1 Ag1x Ag9999999999; do
  if ag_rows_parse "$p" >/dev/null; then bad "accepted $p"; else ok "refuse $p"; fi
done

line() { echo "radiance: deepseek41: exact Engram gather row slices: at most $1 rows (RADIANCE_EXPERIMENTAL_DS41_AG_ROWS); row 6400, output pitch 25600, all_reduce unchanged by AG_ROWS"; }
line 1024 >"$OUT/on.server.log"
line 16 >"$OUT/clamped.server.log"
echo plain >"$OUT/off.server.log"
expect_log() {
  local want=$1 file=$2 req=$3 cap=$4 rc
  ag_rows_server_gate "$OUT/$file.server.log" "$req" "$cap" >/dev/null; rc=$?
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
ar_rows_server_check() { return 0; }
n128_server_gate() { return 0; }
vp_server_gate() { return 0; }
qsp_gate() { return 0; }
ARGS=()
kill() { return 1; }
wait() { return 0; }
python3() { echo compared >>"$T/compares"; return 0; }
ARM_XT= ARM_BT= ARM_LC= ARM_AG_ROWS=1024 ARM_AG_CAP=1024
SERVER_PID=999999 KLD_REF=$T/reference
kld_finish off >/dev/null && bad 'KLD accepted missing cap proof' || ok 'KLD rejects missing cap proof'
SERVER_PID=999999
kld_finish on >/dev/null && ok 'KLD accepts actual cap before comparison' || bad 'KLD rejected matching cap'
[[ -s $T/compares ]] || bad 'KLD did not reach comparison'
SERVER_PID=999999 KLD_REF=
kld_finish on >/dev/null && bad 'Ag became KLD reference' || ok 'Ag cannot become KLD reference'

H=$T/home
mkdir -p "$H/kernels"
LIB=$H/kernels/libr4d.so
echo fixture >"$LIB"
SUM=$(sha256sum "$LIB" | cut -d' ' -f1)
export AG_CALLS=$T/checker.calls
cat >"$T/checker" <<'EOF'
#!/usr/bin/env bash
if [[ -n ${HIP_VISIBLE_DEVICES+x}${CUDA_VISIBLE_DEVICES+x}${R4D_SELFTEST_ONE_CARD+x} ]]; then
  echo 'FAIL inherited one-card mask'; exit 1
fi
# The window's own mask (the serving cards) reaches the checker unchanged.
if [[ ${ROCR_VISIBLE_DEVICES-unset} != "${EXPECT_ROCR-unset}" ]]; then
  echo "FAIL card mask ${ROCR_VISIBLE_DEVICES-unset}, not the window's ${EXPECT_ROCR-unset}"; exit 1
fi
[[ $2 == --case && $3 == ds41_ag_rows_exact ]] || exit 2
echo called >>"$AG_CALLS"
for cap in 1024 64; do for layout in tight pitched; do echo "  ok    ds41_ag_rows_exact cap $cap, $layout"; done; done
[[ ${MUTATE_AG:-0} == 1 ]] && echo mutation >>"$1"
exit 0
EOF
chmod +x "$T/checker"
SELFTEST_AG_ROWS=$T/checker
declare -A AG_ROWS_PASSED=() AG_ROWS_RECEIPT=()
fake_docker() { echo "${FAKE_STATE:-exited}"; }
DOCKER=(fake_docker) NAME=fake
AG_ROWS_WINDOW_ACTIVE=0
ag_rows_check "$LIB" >/dev/null && bad 'check outside window' || ok 'checker refuses outside window'
[[ ! -e $AG_CALLS ]] || bad 'outside-window checker ran'
AG_ROWS_WINDOW_ACTIVE=1 FAKE_STATE=running
ag_rows_check "$LIB" >/dev/null && bad 'check beside running container' || ok 'checker requires exited container'
FAKE_STATE=exited
# The window's four-card mask (the chains' four XT UUIDs) stays; the card-0 checks' one-card settings go.
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 R4D_SELFTEST_ONE_CARD=1
export ROCR_VISIBLE_DEVICES=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1,GPU-23aac62b67afffa9 EXPECT_ROCR=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1,GPU-23aac62b67afffa9
# A stale file at the expected name must be replaced by this window's fresh checker call.
echo stale >"$OUT/ag-rows-$SUM.log"
ag_rows_check "$LIB" >/dev/null && ok 'fresh checker keeps the four-card window mask, drops one-card settings and replaces stale log' || bad 'fresh checker failed'
ag_rows_check "$LIB" >/dev/null || bad 'same-library cached receipt rejected'
[[ $(wc -l <"$AG_CALLS") == 1 ]] && ok 'one fresh check per full SHA' || bad 'checker repeated'
ag_rows_gate good "$LIB" >/dev/null || bad 'valid receipt gate failed'
F=${AG_ROWS_RECEIPT[$SUM]}
for kind in fail skip missing exit duplicate wrong_sha; do
  cp "$F" "$T/bad.log"
  case $kind in
    fail) echo '  FAIL bad value' >>"$T/bad.log" ;;
    skip) echo '  skip no cards' >>"$T/bad.log" ;;
    missing) sed -i '/cap 64, tight$/d' "$T/bad.log" ;;
    exit) sed -i 's/ag_rows exit 0/ag_rows exit 1/' "$T/bad.log" ;;
    duplicate) echo "ag_rows sha256 $SUM" >>"$T/bad.log" ;;
    wrong_sha) other=${SUM%?}0; [[ $other == "$SUM" ]] && other=${SUM%?}1
               sed -i "s/$SUM/$other/g" "$T/bad.log" ;;
  esac
  ag_rows_receipt_ok "$T/bad.log" "$SUM" && bad "accepted $kind receipt" || ok "reject $kind receipt"
done
echo changed >>"$LIB"
ag_rows_gate changed "$LIB" >/dev/null && bad 'accepted changed served SHA' || ok 'reject changed served SHA'
export MUTATE_AG=1
ag_rows_check "$LIB" >/dev/null && bad 'accepted library mutation during checker' || ok 'reject pre/post SHA change'
unset MUTATE_AG
calls_before=$(wc -l <"$AG_CALLS")
echo three >"$H/kernels/three.so"
ROCR_VISIBLE_DEVICES=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1 ag_rows_check "$H/kernels/three.so" >/dev/null &&
  bad 'accepted a mask of three cards' || ok 'refuses a window mask that does not name four cards'
[[ $(wc -l <"$AG_CALLS") == "$calls_before" ]] && ok 'no checker call for a three-card mask' || bad 'checker ran on a three-card mask'
echo nomask >"$H/kernels/nomask.so"
( unset ROCR_VISIBLE_DEVICES EXPECT_ROCR; ag_rows_check "$H/kernels/nomask.so" >/dev/null ) &&
  ok 'without a window mask the checker sees no mask (every visible card, as the servers)' || bad 'no-mask check failed'

# Arm parsing only: no checker or server. No-Ag arms need no AG receipt variables.
ARMS='base base-Ag64-BT128 base-Ag1024-Kld base-Ag0 base-Agfoo base-Ag64-Ag1024'
out=$(
  set +u
  log() { echo "LOG $*"; }; power_set() { :; }; power_restore() { :; }
  ag_rows_gate() { return 0; }
  ar_rows_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "CAP ${ARM_AG_CAP}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H PROMPTS=$T/prompts.json
  export RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=888
  export RADIANCE_EXPERIMENTAL_DS41_AG_ROWS=777   # each arm must override a window-level leak
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
for pair in 'base:0' 'base-Ag64-BT128:64' 'base-Ag1024-Kld:1024'; do
  a=${pair%:*} n=${pair##*:}
  grep -qx "    RADIANCE_EXPERIMENTAL_DS41_AG_ROWS=$n" <<<"$(section "$a")" && ok "$a explicit env=$n" || bad "$a missing explicit env"
done
grep -qx 'CAP 64' <<<"$(section base-Ag64-BT128)" || bad 'wrong BT actual cap'
grep -qx 'CAP 16' <<<"$(section base-Ag1024-Kld)" || bad 'wrong default actual cap'
for a in base-Ag0 base-Agfoo base-Ag64-Ag1024; do
  [[ -z $(section "$a") ]] && ok "skip $a" || bad "launched $a"
done
# A cap line with the wrong full output pitch must not pass even when the cap matches.
line 1024 | sed 's/output pitch 25600/output pitch 6400/' >"$OUT/pitch.server.log"
expect_log 1 pitch 1024 1024
line 1024 | sed 's/row 6400/row 6401/' >"$OUT/alignment.server.log"
expect_log 1 alignment 1024 1024
for a in base base-Ag64-BT128 base-Ag1024-Kld; do
  grep -qx '    RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=0' <<<"$(section "$a")" || bad "$a did not independently reset AR_ROWS"
done
(( fail )) && exit 1
echo 'ab_window_ag_test: all checks passed'
