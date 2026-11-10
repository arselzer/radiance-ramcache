#!/usr/bin/env bash
# ab_window_lh_test.sh [ab_window.sh] -- ab_window.sh's Lh part (LhS: the lent slots follow live decode
# routing, RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=S) with no GPU, no server and no model:
#   1. lcp_check on synthetic server logs: an Lh arm is measured only with rank 0's "ds41 p1: lent heat: on
#      (..._P1_LEND_HEAT=S;" line naming its S, and an arm without the part only without such a line;
#   2. the arm loop, every action stubbed (as ab_window_p1_test.sh): P1 at Lc2048 with Qe64 and Lh2 (Lh8; Lh1
#      beside Ql10 and Qe64) is launched with the switch at its S, the lent arena on and the two tunables
#      stated empty; the same arm without Lh, launched right after an Lh arm, the Ql arm and the adopted
#      large-chunk arm state the switch 0 (and the tunables empty); Lh with the lent ring alone, with P1
#      alone, without P1, Lh0, Lh9, Lh2x, Lhx and a second Lh part are skipped.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-lh-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
HEAT=RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT

# ---- 1. the server-log gate
eval "$(sed -n '/^lcp_check() {/,/^}/p' "$W")"
declare -F lcp_check >/dev/null || { echo "FAIL no lcp_check in $W"; exit 1; }
OUT=$T/logs
mkdir -p "$OUT"
log() { echo "LOG: $*" >> "$OUT/window.log"; }
lc_line='I[0] large-chunk prefill: passes of up to 2048 tokens (RADIANCE_EXPERIMENTAL_DS41_LCP=2048): ...'
p1_line="I[0] ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of 320 to 2048 rows on the cards (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=320; ...)"
arena_line="I[0] ds41 p1: lent arena: 42 slots (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=64; 42 fit) in the activation arena's top, ..."
# The engine's line (FileStager::p1_lend_init, ds41p1::kLogLendHeat).
heat_line() { echo "I[0] ds41 p1: lent heat: on (${HEAT}=$1; half-life 100 decode rows, ${HEAT}_HALF_ROWS; margin 150%, ${HEAT}_MARGIN_PCT): each rank's lent slots (rank 0: 42) follow live decode routing, at most $1 handed to another unit a pass; ..."; }
mk() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.server.log"; }
mk arena "$lc_line" "$p1_line" "$arena_line"
mk heat2 "$lc_line" "$p1_line" "$arena_line" "$(heat_line 2)"
mk heat22 "$lc_line" "$p1_line" "$arena_line" "$(heat_line 22)"
mk plain "$lc_line"
expect() {   # want(0/1) what log P1 QE LH
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=$4 ARM_QM=320 ARM_QL= ARM_QE=$5 ARM_LH=$6 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect 0 "Qe64 with Lh2, its line"                                   heat2  1 64 2
expect 1 "Qe64 with Lh2, no lent-heat line"                          arena  1 64 2
expect 1 "Lh4 on a server that took 2"                               heat2  1 64 4
expect 1 "Lh2 on a server whose line says 22"                        heat22 1 64 2
expect 1 "no Lh part on a server whose lent slots follow routing"    heat2  1 64 ""
expect 0 "Qe64 without Lh, no lent-heat line"                        arena  1 64 ""
expect 0 "the plain Lc arm"                                          plain  "" "" ""
# An Lh arm's line goes to the window log.
: > "$OUT/window.log"
ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL= ARM_QE=64 ARM_LH=2 lcp_check heat2 >/dev/null 2>&1
grep -q "ds41 p1: lent heat: on (${HEAT}=2;" "$OUT/window.log" && ok "the window log takes the lent-heat line" ||
  bad "the window log has no lent-heat line"

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
pre=all-t40-f
mid=H-p256-m196608-G-R
post=W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc1024
post2k=${post%-Lc1024}-Lc2048
REF=$pre-M1024-$mid-B1-$post-Cs-Mb1-Lm4096
RING=$pre-M1024-$mid-B6-$post-P1-Ql10-Cs-Mb1-Lm4096
BIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
LH2=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Cs-Mb1-Lm4096
LH8=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh8-Cs-Mb1-Lm4096
LH1BOTH=$pre-M2048-$mid-B6-$post2k-P1-Ql10-Qe64-Lh1-Cs-Mb1-Lm4096
LHRING=$pre-M1024-$mid-B6-$post-P1-Ql10-Lh2-Cs-Mb1-Lm4096
LHP1=$pre-M2048-$mid-B6-$post2k-P1-Lh2-Cs-Mb1-Lm4096
LHNOP1=$pre-M1024-$mid-B1-$post-Lh2-Cs-Mb1-Lm4096
LH0=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh0-Cs-Mb1-Lm4096
LH9=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh9-Cs-Mb1-Lm4096
LH2X=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2x-Cs-Mb1-Lm4096
LHX=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lhx-Cs-Mb1-Lm4096
LHTWICE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lh3-Cs-Mb1-Lm4096
# BIG right after the Lh arms: a switch an arm set must not reach the next one.
ARMS="$REF $LH2 $LH8 $LH1BOTH $BIG $RING $LHRING $LHP1 $LHNOP1 $LH0 $LH9 $LH2X $LHX $LHTWICE"
run_loop() {   # every model action stubbed
 (
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; echo "  ARM_LH: ${ARM_LH}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE
  # The window's own environment tries to turn the switch on and retune it for every arm: each arm states all three.
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=4 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_HALF_ROWS=7
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_MARGIN_PCT=900
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
 ) 2>&1
}
out=$(run_loop)
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
# The last value an arm states for a variable (later entries win in the arm's environment); "unstated" if none.
val() { local l; l=$(grep -E "^    $2=" <<< "$1" | tail -1); [[ -n "$l" ]] && echo "${l#*=}" || echo unstated; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
launched() {   # what arm heat [lend_arena [lend]]
  local sec
  sec=$(section "$2")
  if [[ -z "$sec" ]]; then bad "$1: arm $2 not launched: $(grep -m3 "arm $2" "$T/loop.txt")"; return; fi
  if grep -q "LOG: arm $2: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "$1: $(grep -m1 "LOG: arm $2: " "$T/loop.txt")"; fi
  has "$1" "$sec" ${HEAT} "$3"
  has "$1" "$sec" ${HEAT}_HALF_ROWS ""
  has "$1" "$sec" ${HEAT}_MARGIN_PCT ""
  [[ -n "${4:-}" ]] && has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA "$4"
  [[ -n "${5:-}" ]] && has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND "$5"
  local lh
  lh=$(grep -m1 '^  ARM_LH: ' <<< "$sec" | cut -d' ' -f4)
  local want_lh=$3; [[ $want_lh == 0 ]] && want_lh=""
  if [[ "$lh" == "$want_lh" ]]; then ok "$1: lcp_check will ask for '${want_lh}'"; else bad "$1: ARM_LH is '$lh', want '$want_lh'"; fi
}
launched "P1+Qe64+Lh2" "$LH2" 2 64 0
launched "P1+Qe64+Lh8" "$LH8" 8 64 0
launched "P1+Ql10+Qe64+Lh1" "$LH1BOTH" 1 64 10
launched "P1+Qe64, right after the Lh arms" "$BIG" 0 64 0
launched "P1+Ql10" "$RING" 0 0 10
launched "the adopted large-chunk arm" "$REF" 0 0 0
sec=$(section "$LH2")
has "P1+Qe64+Lh2" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
has "P1+Qe64+Lh2" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
has "the adopted large-chunk arm" "$(section "$REF")" RADIANCE_EXPERIMENTAL_DS41_P1 0
skipped() {   # arm, the reason's words
  if [[ -n "$(section "$1")" ]]; then bad "arm $1 was launched"
  elif grep -q "LOG: arm $1: .*$2.*skipped" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $1: .*$2" "$T/loop.txt" | sed "s|LOG: arm $1: ||")"
  else bad "arm $1 not skipped for '$2': $(grep -m2 "LOG: arm $1: " "$T/loop.txt" | tr '\n' ' ')"; fi
}
skipped "$LHRING" "needs the lent arena"
skipped "$LHP1" "needs the lent arena"
skipped "$LHNOP1" "needs the lent arena"
skipped "$LH0" "Lh takes 1-8"
skipped "$LH9" "Lh takes 1-8"
skipped "$LH2X" "Lh takes 1-8"
skipped "$LHX" "unknown part Lhx"
skipped "$LHTWICE" "duplicate Lh part"

(( fail )) && { echo "ab_window_lh_test: FAILED"; echo "$out" | tail -40; exit 1; }
echo "ab_window_lh_test: all checks passed"
