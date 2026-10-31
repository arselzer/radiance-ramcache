#!/usr/bin/env bash
# ab_window_p1_test.sh [ab_window.sh] -- ab_window.sh's P1 parts (P1, QmN, QlK, QeK, QhMS) with no GPU, no
# server and no model:
#   1. lcp_check on synthetic server logs: a P1 arm is measured only when the card path's startup line
#      states its threshold and chunk, a Ql arm only with its lent-ring line, a Qe arm only with its
#      lent-arena line naming its K, and an arm without a part only without its line;
#   2. the arm loop, every action stubbed (as ab_window_vp_test.sh): the adopted large-chunk arm with P1
#      on top of it (B6, Ql10, M1024, Mr32, Mb1, Cs) is launched with the switches it names and the rest
#      stated off; the adopted arm itself states P1 off; P1 at the default threshold with M16, Ql12 on
#      B6's ring, Qh without Ql and Ql without P1 are skipped; Qm17 with M16 is launched; P1 at Lc2048 with
#      Qe64 and M2048 is launched with the lent arena on, Qh beside Qe alone too; Qe at M1024 beside Lc2048
#      and Qe without P1 are skipped.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-p1-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# ---- 1. the server-log gate
eval "$(sed -n '/^lcp_check() {/,/^}/p' "$W")"
declare -F lcp_check >/dev/null || { echo "FAIL no lcp_check in $W"; exit 1; }
OUT=$T/logs
mkdir -p "$OUT"
log() { echo "LOG: $*" >> "$OUT/window.log"; }
lc_line='I[0] large-chunk prefill: passes of up to 1024 tokens (RADIANCE_EXPERIMENTAL_DS41_LCP=1024): ...'
p1_line() { echo "I[0] ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of $1 to 1024 rows on the cards (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=$1; ...)"; }
lend_line() { echo "I[0] ds41 p1: lent ring: $1 of the ring's 12 slots (10 of them past the file buffers' alias bases; RADIANCE_EXPERIMENTAL_DS41_P1_LEND=$1) hold ..."; }
arena_line() { echo "I[0] ds41 p1: lent arena: $2 slots (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=$1; $2 fit) in the activation arena's top, ..."; }
mk() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.server.log"; }
mk p1def "$lc_line" "$(p1_line 320)"
mk p1lend "$lc_line" "$(p1_line 320)" "$(lend_line 10)"
mk p1q17 "$lc_line" "$(p1_line 17)"
mk plain "$lc_line"
mk p1arena "$lc_line" "$(p1_line 320)" "$(arena_line 64 18)"
mk p1both "$lc_line" "$(p1_line 320)" "$(lend_line 10)" "$(arena_line 64 18)"
expect() {   # want(0/1) what log LC P1 QM QL [QE]
  local want=$1 what=$2 rc
  ARM_LC=$4 ARM_P1=$5 ARM_QM=$6 ARM_QL=$7 ARM_QE=${8:-} lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect 0 "P1 at the default threshold, its line"                      p1def  1024 1 320 ""
expect 1 "P1 at Qm17 on a server that took 320"                       p1def  1024 1 17  ""
expect 0 "P1 at Qm17, its line"                                       p1q17  1024 1 17  ""
expect 0 "P1 with Ql10, both lines"                                   p1lend 1024 1 320 10
expect 1 "P1 with Ql10, no lent-ring line"                            p1def  1024 1 320 10
expect 1 "P1 with Ql8 on a server that lent 10"                       p1lend 1024 1 320 8
expect 1 "P1 without Ql on a server that lends (the knob from the window)" p1lend 1024 1 320 ""
expect 1 "no P1 part on a server that runs the card path"             p1def  1024 "" 320 ""
expect 0 "the plain Lc arm"                                           plain  1024 "" 320 ""
expect 0 "P1 with Qe64, its line"                                     p1arena 1024 1 320 "" 64
expect 1 "P1 with Qe64, no lent-arena line"                           p1def  1024 1 320 "" 64
expect 1 "P1 with Qe32 on a server that took 64"                      p1arena 1024 1 320 "" 32
expect 1 "P1 without Qe on a server that lends the arena"             p1arena 1024 1 320 ""
expect 0 "P1 with Ql10 and Qe64, both lines"                          p1both 1024 1 320 10 64
expect 1 "P1 with Qe64 alone on a server that lends the ring too"     p1both 1024 1 320 "" 64

# Qsp's startup and useful-work gates, independent of ordinary P1 startup admission.
eval "$(sed -n '/^qsp_gate() {/,/^}/p' "$W")"
declare -F qsp_gate >/dev/null || { echo "FAIL no qsp_gate in $W"; exit 1; }
: > "$T/qsp-good.log"
for rank in 0 1 2 3; do
  echo "ds41 p1: rank $rank's wave SwiGLU spans on (test)" >> "$T/qsp-good.log"
  echo "ds41 p1 stats, rank $rank: test; wave SwiGLU spans 12, rows 20 of 200, full-replay fallbacks 0" >> "$T/qsp-good.log"
done
sed '/rank 3/d' "$T/qsp-good.log" > "$T/qsp-missing.log"
sed 's/spans 12/spans 0/' "$T/qsp-good.log" > "$T/qsp-idle.log"
sed 's/rows 20 of 200/rows 200 of 200/' "$T/qsp-good.log" > "$T/qsp-full.log"
: > "$T/qsp-off.log"
qexpect() {
  local want=$1 label=$2 rc; shift 2
  qsp_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
qexpect 0 'Qsp every-rank startup' "$T/qsp-good.log" 1 start
qexpect 0 'Qsp every-rank work reduction' "$T/qsp-good.log" 1 exit
qexpect 1 'Qsp missing rank' "$T/qsp-missing.log" 1 start
qexpect 1 'Qsp enabled without actual waves' "$T/qsp-idle.log" 1 exit
qexpect 1 'Qsp no reduced work' "$T/qsp-full.log" 1 exit
qexpect 1 'Qsp unrequested activation' "$T/qsp-good.log" '' start
qexpect 0 'Qsp explicitly off' "$T/qsp-off.log" '' exit

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
pre=all-t40-f
mid=H-p256-m196608-G-R
post=W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc1024
REF=$pre-M1024-$mid-B1-$post-Cs-Mb1-Lm4096
NEW=$pre-M1024-$mid-B6-$post-P1-Ql10-Cs-Mb1-Lm4096
NEWG=$pre-M1024-$mid-B6-$post-P1-Ql10-Qh3000-Cs-Mb1-Lm4096
SHORTM=$pre-M16-$mid-B6-$post-P1-Cs-Mb1-Lm4096
Q17=$pre-M16-$mid-B6-$post-P1-Qm17-Cs-Mb1-Lm4096
OVERL=$pre-M1024-$mid-B6-$post-P1-Ql12-Cs-Mb1-Lm4096
GNOL=$pre-M1024-$mid-B6-$post-P1-Qh3000-Cs-Mb1-Lm4096
LNOP=$pre-M1024-$mid-B1-$post-Ql10-Cs-Mb1-Lm4096
post2k=${post%-Lc1024}-Lc2048
BIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
BIGG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Qh3000-Cs-Mb1-Lm4096
BIGM=$pre-M1024-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
ENOP=$pre-M1024-$mid-B1-$post-Qe64-Cs-Mb1-Lm4096
ARMS="$REF $NEW $NEWG $SHORTM $Q17 $OVERL $GNOL $LNOP $BIG $BIGG $BIGM $ENOP"
out=$(
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=2048
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) 2>&1
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
# The last value an arm states for a variable (later entries win in the arm's environment).
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
sec=$(section "$REF")
if [[ -z "$sec" ]]; then bad "the adopted arm was not launched: $(grep -m3 "arm $REF" "$T/loop.txt")"
else
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 1024
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
fi
for a in "$NEW" "$NEWG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Ql10
  [[ "$a" == *-Qh3000-* ]] && n=P1+Ql10+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 10
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS 6
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 1024
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS 32
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP 1
  [[ "$sec" == *"--max-num-batched-tokens 1024"* ]] && ok "$n: chunks of 1024" || bad "$n: no --max-num-batched-tokens 1024"
done
has "P1+Ql10+Qh3000" "$(section "$NEWG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
has "P1+Ql10" "$(section "$NEW")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS ""
sec=$(section "$Q17")
[[ -n "$sec" ]] && has "Qm17 with M16" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS 17 || bad "the Qm17 M16 arm was not launched"
for a in "$BIG" "$BIGG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Lc2048+Qe64
  [[ "$a" == *-Qh3000-* ]] && n=P1+Lc2048+Qe64+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 64
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 2048
  [[ "$sec" == *"--max-num-batched-tokens 2048"* ]] && ok "$n: chunks of 2048" || bad "$n: no --max-num-batched-tokens 2048"
done
has "P1+Lc2048+Qe64+Qh3000" "$(section "$BIGG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
for a in "$SHORTM" "$OVERL" "$GNOL" "$LNOP" "$BIGM" "$ENOP"; do
  if [[ -n "$(section "$a")" ]]; then bad "arm $a was launched"
  elif grep -q "LOG: arm $a: .*skipped" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $a: " "$T/loop.txt" | sed "s|LOG: arm $a: ||")"
  else bad "arm $a neither launched nor skipped"; fi
done
for a in "$REF" "$NEW" "$NEWG" "$Q17" "$BIG" "$BIGG"; do
  if grep -q "LOG: arm $a: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "arm $a: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
done

(( fail )) && { echo "ab_window_p1_test: FAILED (loop output in $T/loop.txt, removed on exit)"; exit 1; }
echo "ab_window_p1_test: all checks passed"
