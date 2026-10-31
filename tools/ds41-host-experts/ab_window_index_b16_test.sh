#!/usr/bin/env bash
# Source fixture only: actual arm-loop parsing, all actions stubbed. No GPU/server/model.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-index-b16-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
bad() { echo "FAIL $*"; fail=1; }
ok() { echo "ok $*"; }
H=$T/home
mkdir -p "$H/kernels" "$H/architectures"
echo fake >"$H/kernels/libr4d.so"
echo '{}' >"$T/prompts.json"
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe
run_loop() {
  (
    set +u
    log() { echo "LOG $*"; }; power_set() { :; }; power_restore() { :; }; vpack_gate() { return 0; }
    arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; }
    # Only the NEW receipt process is stubbed. Existing arm parser still runs.
    python3() { [[ " $* " != *" receipt "* ]] || [[ "$ORACLE" == ok ]]; }
    ORACLE=$1
    HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T MAX_SEQS=1
    PROMPTS=$T/prompts.json OUT=$T/out CANCEL_FILE=/nonexistent/cancel CANCEL=$CANCEL_FILE
    INDEX_B16_COHORT=$T/cohort INDEX_B16_RECEIPT=$T/receipt INDEX_B16_WINDOW_ID=fresh-nonce
    mkdir -p "$OUT"
    unset RADIANCE_TUNE
    export RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_BF16=1
    ARMS="$B $B-Ib16"
    eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
    HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
    s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
    e=$(awk -v s="$s" 'NR>s && /^done$/ { print NR; exit }' "$W")
    eval "$(sed -n "${s},${e}p" "$W")"
  ) 2>&1
}
run_loop ok >"$T/loop.log"
section() { awk -v a="ARM $1" '$0==a { p=1;next } /^ARM / { p=0 } p' "$T/loop.log"; }
plain=$(section "$B") active=$(section "$B-Ib16")
[[ "$plain" == *'RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_BF16=0'* ]] && ok 'every plain arm explicitly resets0' || bad 'plain arm inherited global enable'
[[ "$active" == *'RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_BF16=1'* ]] && ok 'Ib16 arm explicitly enables1 after receipt' || bad 'Ib16 failed to enable'
run_loop missing >"$T/missing.log"
grep -qxF "ARM $B-Ib16" "$T/missing.log" && bad 'Ib16 ran without receipt' || ok 'missing receipt skips enabled arm'
(
  set +u
  eval "$(sed -n '/^kld_finish() {/,/^}/p' "$W")"
  kill() { return 1; }; wait() { return 0; }; log() { :; }; xt_check() { return 0; }; lcp_check() { return 0; }
  n128_server_gate() { return 0; }; vp_server_gate() { return 0; }; qsp_gate() { return 0; }
  OUT=$T/kld SERVER_PID=fake KLD_REF= ARM_INDEX_B16=1
  mkdir -p "$OUT"; echo ordinary >"$OUT/active.server.log"; ARGS=()
  kld_finish active >/dev/null && exit 1
  [[ -z "$KLD_REF" ]]
) && ok 'Ib16 cannot establish KLD reference' || bad 'Ib16 became its own byte reference'
((fail)) && exit 1
echo 'ab_window_index_b16_test: all checks passed'
