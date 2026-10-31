#!/usr/bin/env bash
# Source-level window gate cases with synthetic logs/receipts and a stubbed arm loop.
# No GPU, server or model is opened. Run under the box build lock like every other test.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-n128-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
eval "$(sed -n '/^n128_server_gate() {/,/^}/p' "$W")"
eval "$(sed -n '/^n128_arm_gate() {/,/^}/p' "$W")"
declare -F n128_server_gate >/dev/null && declare -F n128_arm_gate >/dev/null || exit 2

expect() { # status, description, command and arguments
  local want=$1 what=$2 why rc
  shift 2
  why=$("$@"); rc=$?
  if [[ $rc == "$want" ]]; then ok "$what${why:+ ($why)}"
  else bad "$what: exit $rc, expected $want${why:+ ($why)}"; fi
}
marker() { echo "[r4d] ds41 n128 active: normal dispatch rank=$1 world=$2 n=128 M=73 old_thr=256"; }
{ marker 0 4; marker 1 4; marker 2 4; marker 3 4; } >"$T/all.log"
{ marker 0 4; marker 1 4; marker 2 4; } >"$T/three.log"
{ marker 0 4; marker 0 4; marker 0 4; marker 0 4; } >"$T/duplicate.log"
{ marker 0 1; marker 1 1; marker 2 1; marker 3 1; } >"$T/world.log"
echo '[r4d] ds41 n128 active: normal dispatch n=128 M=73 old_thr=256' >"$T/legacy.log"
echo 'no n128 dispatch' >"$T/plain.log"
expect 0 'all active TP4 ranks dispatched' n128_server_gate "$T/all.log" 1 4
expect 1 'one missing rank refuses' n128_server_gate "$T/three.log" 1 4
expect 1 'four copies of rank0 are insufficient' n128_server_gate "$T/duplicate.log" 1 4
expect 1 'wrong world refuses' n128_server_gate "$T/world.log" 1 4
expect 1 'legacy process-wide marker is insufficient' n128_server_gate "$T/legacy.log" 1 4
expect 0 'plain arm has no fast dispatch' n128_server_gate "$T/plain.log" '' 4
expect 1 'plain arm with active dispatch refuses' n128_server_gate "$T/all.log" '' 4

# A fast arm must not become its own whole-model byte reference if the plain arm failed.
(
  eval "$(sed -n '/^kld_finish() {/,/^}/p' "$W")"
  kill() { return 1; }; wait() { return 0; }; log() { :; }; xt_check() { return 0; }
  OUT=$T/kld SERVER_PID=stub KLD_REF= ARM_XT= ARM_LC= ARM_N128=1
  mkdir -p "$OUT"
  cp "$T/all.log" "$OUT/fast.server.log"
  expect 1 'Fn128 cannot establish the KLD reference' kld_finish fast
  [[ -z "$KLD_REF" ]] && ok 'KLD reference remains unset' || bad 'Fn128 became the KLD reference'
  (( fail == 0 ))
) || fail=1

H=$T/home
mkdir -p "$H/kernels"
echo 'checked library' >"$H/kernels/libr4d.so"
echo 'other library' >"$T/other.so"
LIB=$H/kernels/libr4d.so
SUM=$(sha256sum "$LIB" | cut -d' ' -f1)
N128_OK=1 N128_LIB=$LIB N128_SUM=$SUM N128_MAX_M=4096 N128_WHY=test
expect 0 'matching library and covered work bound' n128_arm_gate "$LIB" 4096
expect 1 'bound above oracle refuses' n128_arm_gate "$LIB" 4097
expect 1 'different library SHA refuses' n128_arm_gate "$T/other.so" 16
expect 1 'missing library refuses' n128_arm_gate "$T/missing.so" 16
N128_OK=
expect 1 'missing oracle refuses' n128_arm_gate "$LIB" 16
N128_OK=1 N128_MAX_M=2048
expect 1 'shorter oracle cannot cover taller work' n128_arm_gate "$LIB" 2049

# The parser must put 0/1 in every arm's environment even when the enclosing window says 1.
echo '{}' >"$T/prompts.json"
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe
run_loop() { # oracle-present flag, SHA
  (
    set +u
    log() { echo "LOG: $*"; }
    power_set() { :; }; power_restore() { :; }; vpack_gate() { return 0; }
    arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; }
    HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T MAX_SEQS=1
    PROMPTS=$T/prompts.json OUT=$T/out CANCEL_FILE=/nonexistent/cancel CANCEL=$CANCEL_FILE
    mkdir -p "$OUT"
    unset RADIANCE_TUNE
    export RADIANCE_EXPERIMENTAL_DS41_N128=1
    N128_OK=$1 N128_SUM=$2 N128_MAX_M=4096 N128_WHY='missing test oracle'
    ARMS="$B $B-Fn128 $B-Fn128-BT4096"
    eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
    eval "$(sed -n '/^n128_arm_gate() {/,/^}/p' "$W")"
    HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
    s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
    e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
    eval "$(sed -n "${s},${e}p" "$W")"
  ) 2>&1
}
run_loop 1 "$SUM" >"$T/loop.log"
section() { awk -v a="ARM $1" '$0 == a { p=1; next } /^ARM / { p=0 } p' "$T/loop.log"; }
plain=$(section "$B") active=$(section "$B-Fn128")
[[ "$plain" == *'RADIANCE_EXPERIMENTAL_DS41_N128=0'* ]] && ok 'plain arm explicitly off' || bad 'plain arm is not explicitly off'
[[ "$active" == *'RADIANCE_EXPERIMENTAL_DS41_N128=1'* ]] && ok 'Fn128 explicitly on' || bad 'Fn128 was not launched on'
[[ -z "$(section "$B-Fn128-BT4096")" ]] && grep -q 'work bound 4097 exceeds oracle M4096' "$T/loop.log" &&
  ok 'unpinned max_tok+max_seqs bound enforced' || bad 'unpinned work bound was not refused'
run_loop '' "$SUM" >"$T/absent.log"
if grep -qxF "ARM $B-Fn128" "$T/absent.log"; then bad 'Fn128 launched without oracle'
else ok 'Fn128 skipped without oracle'; fi
run_loop 1 invalid-sha >"$T/sha.log"
if grep -qxF "ARM $B-Fn128" "$T/sha.log"; then bad 'Fn128 launched with wrong oracle SHA'
else ok 'Fn128 skipped with wrong oracle SHA'; fi

(( fail )) && { echo 'ab_window_n128_test: FAILED'; exit 1; }
echo 'ab_window_n128_test: all checks passed'
