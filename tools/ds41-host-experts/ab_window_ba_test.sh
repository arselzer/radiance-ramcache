#!/usr/bin/env bash
# ab_window_ba_test.sh [ab_window.sh] -- ab_window.sh's Ba / BaN parts (RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT,
# whole prompts waiting together admitted into one prefill-only step) with no GPU, no server and no model:
#   1. the arm loop, every action stubbed (as ab_window_p1_test.sh): the recommended N8 arm is launched with
#      the switch 0, the same arm with Ba with 256 and with Ba128 with 128 (prefill alone on in all three),
#      and with Kld (a KL run) with the switch 0; Ba without Pf, without Lc, at N1, beside Pc, with Kld (the
#      scheduler refuses it under --kld-record, and a Kld arm never reaches arm()'s startup-line check), at
#      a row count outside 32-256, and twice are skipped, each with its reason;
#   2. the startup-line guard arm() applies (its Ba row of the switch table): a Ba arm's pattern matches the
#      scheduler's line for its N only, and an arm without the part flags any such line.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-ba-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# ---- 1. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
R=all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs
S16=all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw
N8=L1024-Pf-N8-C8-Pmr8
BASE=$R-$N8-r1
BA=$R-$N8-Ba-r1
BA128=$R-$N8-Ba128
NOPF=$R-L1024-N8-C8-Pmr8-Ba
NOLC=$S16-$N8-Ba
ONE=$R-L1024-Pf-N1-Ba
PC=$R-$N8-Pc-Ba
KLDOK=$R-$N8-Kld
KLD=$R-$N8-Ba-Kld
LOW=$R-$N8-Ba16
HIGH=$R-$N8-Ba300
TWICE=$R-$N8-Ba-Ba128
ARMS="$BASE $BA $BA128 $KLDOK $NOPF $NOLC $ONE $PC $KLD $LOW $HIGH $TWICE"
(
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
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  echo '{}' > "$OUT/kld-corpus.jsonl"   # a Kld arm's corpus already there (nothing to build)
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) > "$T/loop.txt" 2>&1
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
for spec in "$BASE|0|base" "$BA|256|Ba" "$BA128|128|Ba128"; do
  IFS='|' read -r a want what <<< "$spec"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "$what not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "$what" "$sec" RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT "$want"
  has "$what" "$sec" RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE 1
  has "$what" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
  has "$what" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
done
# A KL run without Ba is launched (the switch 0, its --kld-record), so the Kld skip below is Ba's.
sec=$(section "$KLDOK")
if [[ -z "$sec" ]]; then bad "Kld not launched: $(grep -m3 "arm $KLDOK" "$T/loop.txt")"
else
  has "Kld" "$sec" RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT 0
  if grep -qF -- '--kld-record' <<< "$sec"; then ok "Kld: --kld-record"; else bad "Kld: no --kld-record"; fi
fi
skipped() {   # arm, the reason's words, what
  if [[ -n "$(section "$1")" ]]; then bad "$3 launched"
  elif grep -qF "arm $1: " "$T/loop.txt" && grep "arm $1: " "$T/loop.txt" | grep -qF "$2"; then ok "$3 skipped: $2"
  else bad "$3: no '$2' ($(grep -m2 "arm $1: " "$T/loop.txt" | tr '\n' ' '))"; fi
}
skipped "$NOPF" "Ba needs Pf" "Ba without Pf"
skipped "$NOLC" "Ba needs an Lc part" "Ba without Lc"
skipped "$ONE" "Ba needs NK with K above 1" "Ba at N1"
skipped "$PC" "Ba with a Pc part" "Ba beside Pc"
skipped "$KLD" "Ba with a Kld part" "Ba with Kld"
skipped "$LOW" "a batch holds 32-256 rows" "Ba16"
skipped "$HIGH" "a batch holds 32-256 rows" "Ba300"
skipped "$TWICE" "duplicate Ba part" "Ba twice"

# ---- 2. the startup-line guard: arm()'s Ba row of its switch table, as the heredoc expands it
row=$(grep -E '^Ba\$\{ARM_BA' "$W")
if [[ -z "$row" ]]; then bad "no Ba row in arm()'s switch table"
else
  line() { echo "I batch admission: on (RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=$1) -- whole prompts waiting together share one prefill-only step ..."; }
  guard() {   # ARM_BA, the line's N, want (1 match, 0 none), what
    local ARM_BA=$1 r part on pat n
    eval "r=\"$row\""
    IFS='|' read -r part on pat <<< "$r"
    n=$(line "$2" | grep -cE "$pat")
    if [[ "$n" == "$3" && "$on" == "$1" ]]; then ok "guard: $4"; else bad "guard: $4 (pattern '$pat', on '$on', $n matches)"; fi
  }
  guard 256 256 1 "Ba's pattern matches the line for 256"
  guard 128 128 1 "Ba128's pattern matches the line for 128"
  guard 128 256 0 "Ba128's pattern does not match the line for 256"
  guard "" 256 1 "an arm without Ba flags the line (the switch set for the window)"
fi
(( fail == 0 )) && echo "PASS ab_window_ba_test" || echo "FAIL ab_window_ba_test (loop output: $T/loop.txt)"
exit $fail
