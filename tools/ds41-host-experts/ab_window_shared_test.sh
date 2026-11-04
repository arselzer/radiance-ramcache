#!/usr/bin/env bash
# Fake-only fixtures: no model, GPU, server or service action is executed.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-shared-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
for fn in serving_cards serving_cards_ok ar_rows_parse ag_rows_parse ar_ag_shared_arm ar_ag_shared_receipt_ok ar_ag_shared_check ar_ag_shared_gate; do
  eval "$(sed -n "/^$fn() {/,/^}/p" "$W")"
  declare -F "$fn" >/dev/null || { bad "missing $fn"; exit 1; }
done
OUT=$T/out
mkdir -p "$OUT"
log() { echo "LOG $*" >>"$OUT/window.log"; }
for a in base-Ar1280-Ag1024-Lc2048 base-Ag1024-Ar1280-BT4096 base-Ar9999-Ag1024-BT1280; do
  ar_ag_shared_arm "$a" && ok "coordinated actual caps: $a" || bad "missed $a"
done
for a in base base-Ar1280 base-Ag1024 base-Ar1280-Ag1024 base-Ar1280-Ag1024-BT1024 \
         base-Ar1024-Ag1024-BT4096 base-Ar1280-Ag1024-Lcfoo base-Ar1280-Ar1280-Ag1024-BT4096; do
  ar_ag_shared_arm "$a" && bad "incorrect coordinated match: $a" || ok "no mixed gate: $a"
done
H=$T/home
mkdir -p "$H/kernels"
LIB=$H/kernels/libr4d.so
echo fixture >"$LIB"
SUM=$(sha256sum "$LIB" | cut -d' ' -f1)
export SHARED_CALLS=$T/checker.calls
cat >"$T/checker" <<'EOF'
#!/usr/bin/env bash
if [[ -n ${HIP_VISIBLE_DEVICES+x}${CUDA_VISIBLE_DEVICES+x}${R4D_SELFTEST_ONE_CARD+x} ]]; then
  echo 'FAIL inherited one-card mask'; exit 1
fi
# The window's own mask (the serving cards) reaches the checker unchanged.
if [[ ${ROCR_VISIBLE_DEVICES-unset} != "${EXPECT_ROCR-unset}" ]]; then
  echo "FAIL card mask ${ROCR_VISIBLE_DEVICES-unset}, not the window's ${EXPECT_ROCR-unset}"; exit 1
fi
[[ $2 == --case && $3 == ds41_ar_ag_shared_exact ]] || exit 2
echo called >>"$SHARED_CALLS"
for order in ar-first ag-first; do for seed in 0 1; do
  echo "  ok    ds41_ar_ag_shared_exact order $order, seed $seed"
done; done
[[ ${MUTATE_SHARED:-0} == 1 ]] && echo mutation >>"$1"
exit 0
EOF
chmod +x "$T/checker"
SELFTEST_AR_AG_SHARED=$T/checker
declare -A AR_AG_SHARED_PASSED=() AR_AG_SHARED_RECEIPT=()
fake_docker() { echo "${FAKE_STATE:-exited}"; }
DOCKER=(fake_docker) NAME=fake
AR_AG_SHARED_WINDOW_ACTIVE=0
ar_ag_shared_check "$LIB" >/dev/null && bad 'check outside window' || ok 'refuses outside window'
[[ ! -e $SHARED_CALLS ]] || bad 'outside-window checker ran'
AR_AG_SHARED_WINDOW_ACTIVE=1 FAKE_STATE=running
ar_ag_shared_check "$LIB" >/dev/null && bad 'check beside running QFN' || ok 'requires stopped QFN'
FAKE_STATE=exited
# The window's four-card mask (the chains' four XT UUIDs) stays; the card-0 checks' one-card settings go.
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 R4D_SELFTEST_ONE_CARD=1
export ROCR_VISIBLE_DEVICES=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1,GPU-23aac62b67afffa9 EXPECT_ROCR=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1,GPU-23aac62b67afffa9
echo stale >"$OUT/ar-ag-shared-$SUM.log"
ar_ag_shared_check "$LIB" >/dev/null && ok 'fresh receipt replaces stale data on the four window cards, one-card settings dropped' || bad 'fresh checker failed'
ar_ag_shared_check "$LIB" >/dev/null || bad 'same-library receipt rejected'
[[ $(wc -l <"$SHARED_CALLS") == 1 ]] || bad 'checker repeated within the window'
ar_ag_shared_gate good "$LIB" >/dev/null || bad 'valid receipt gate failed'
F=${AR_AG_SHARED_RECEIPT[$SUM]}
for kind in fail skip missing exit duplicate wrong_sha; do
  cp "$F" "$T/bad.log"
  case $kind in
    fail) echo '  FAIL bad value' >>"$T/bad.log" ;;
    skip) echo '  skip no cards' >>"$T/bad.log" ;;
    missing) sed -i '/order ag-first, seed 1$/d' "$T/bad.log" ;;
    exit) sed -i 's/ar_ag_shared exit 0/ar_ag_shared exit 1/' "$T/bad.log" ;;
    duplicate) echo "ar_ag_shared sha256 $SUM" >>"$T/bad.log" ;;
    wrong_sha) other=${SUM%?}0; [[ $other == "$SUM" ]] && other=${SUM%?}1
               sed -i "s/$SUM/$other/g" "$T/bad.log" ;;
  esac
  ar_ag_shared_receipt_ok "$T/bad.log" "$SUM" && bad "accepted $kind receipt" || ok "reject $kind receipt"
done
echo changed >>"$LIB"
ar_ag_shared_gate changed "$LIB" >/dev/null && bad 'accepted changed served SHA' || ok 'reject changed served SHA'
export MUTATE_SHARED=1
ar_ag_shared_check "$LIB" >/dev/null && bad 'accepted library mutation' || ok 'reject pre/post library mutation'
unset MUTATE_SHARED
calls_before=$(wc -l <"$SHARED_CALLS")
echo three >"$H/kernels/three.so"
ROCR_VISIBLE_DEVICES=GPU-e49179ac81aef5ad,GPU-e770be134869264b,GPU-be0851a556d0f2c1 ar_ag_shared_check "$H/kernels/three.so" >/dev/null &&
  bad 'accepted a mask of three cards' || ok 'refuses a window mask that does not name four cards'
[[ $(wc -l <"$SHARED_CALLS") == "$calls_before" ]] && ok 'no checker call for a three-card mask' || bad 'checker ran on a three-card mask'
echo nomask >"$H/kernels/nomask.so"
( unset ROCR_VISIBLE_DEVICES EXPECT_ROCR; ar_ag_shared_check "$H/kernels/nomask.so" >/dev/null ) &&
  ok 'without a window mask the checker sees no mask (every visible card, as the servers)' || bad 'no-mask check failed'

# Main arm parsing only. Both independent gates must still run; the mixed gate can only add a refusal.
ARMS='base base-Ar1280-Ag1024-BT2048 base-Ar1280-Ag1024-BT1024 base-Ar1280-BT2048 base-Ar1280-Ag1024-BT4096 base-Ar1280-Ag1024-BT8192'
out=$(
  set +u
  log() { echo "LOG $*"; }; power_set() { :; }; power_restore() { :; }
  ar_rows_gate() { echo "AR_GATE $1"; return 0; }
  ag_rows_gate() { echo "AG_GATE $1"; [[ $1 != *-BT4096 ]]; }
  ar_ag_shared_gate() { echo "SHARED_GATE $1"; [[ $1 != *-BT8192 ]]; }
  arm() { echo "ARM $1"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H PROMPTS=$T/prompts.json
  echo '{}' >"$PROMPTS"
  CANCEL=/nonexistent/cancel
  unset RADIANCE_TUNE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ {print NR; exit}' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
)
for a in base base-Ar1280-Ag1024-BT2048 base-Ar1280-Ag1024-BT1024 base-Ar1280-BT2048; do
  grep -qx "ARM $a" <<<"$out" && ok "allowed $a" || bad "skipped $a"
done
for a in base-Ar1280-Ag1024-BT2048 base-Ar1280-Ag1024-BT4096 base-Ar1280-Ag1024-BT8192; do
  grep -qx "AR_GATE $a" <<<"$out" || bad "missing independent AR gate for $a"
  grep -qx "AG_GATE $a" <<<"$out" || bad "missing independent AG gate for $a"
done
grep -qx 'SHARED_GATE base-Ar1280-Ag1024-BT2048' <<<"$out" || bad 'missing mixed gate'
for a in base-Ar1280-Ag1024-BT4096 base-Ar1280-Ag1024-BT8192; do
  grep -qx "ARM $a" <<<"$out" && bad "bypassed a refusing gate: $a"
done
for a in base base-Ar1280-Ag1024-BT1024 base-Ar1280-BT2048 base-Ar1280-Ag1024-BT4096; do
  grep -qx "SHARED_GATE $a" <<<"$out" && bad "unexpected mixed gate: $a"
done
(( fail )) && exit 1
echo 'ab_window_shared_test: all checks passed'
