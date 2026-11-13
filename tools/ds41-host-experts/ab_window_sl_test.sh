#!/usr/bin/env bash
# ab_window.sh's SlN part (--max-seq-len N): the server-log gate on synthetic logs and the arm loop
# with a stubbed arm(). No GPU, server or model is opened. Run under the box build lock like every
# other test.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-sl-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
eval "$(sed -n '/^seqlen_server_gate() {/,/^}/p' "$W")"
declare -F seqlen_server_gate >/dev/null || exit 2

expect() { # status, description, command and arguments
  local want=$1 what=$2 why rc
  shift 2
  why=$("$@"); rc=$?
  if [[ $rc == "$want" ]]; then ok "$what${why:+ ($why)}"
  else bad "$what: exit $rc, expected $want${why:+ ($why)}"; fi
}
line() { echo "[info] max seq len: $1 tokens a sequence (--max-seq-len $1); block tables, step staging and rotary tables are sized for $3 tokens a sequence; the KV pool stays --max-model-len $2"; }
line 4096 720896 12418 >"$T/on.log"
line 2048 720896 10370 >"$T/other.log"
line 4096 393216 12418 >"$T/pool.log"
echo '[info] step batch: 1 MiB pinned staging ring + 2 MiB device slab a rank' >"$T/plain.log"
expect 0 'Sl4096 with its startup line' seqlen_server_gate "$T/on.log" 4096 720896
expect 1 'Sl4096 without a line refuses' seqlen_server_gate "$T/plain.log" 4096 720896
expect 1 'a line for another N refuses' seqlen_server_gate "$T/other.log" 4096 720896
expect 1 'a line naming another pool refuses' seqlen_server_gate "$T/pool.log" 4096 720896
expect 0 'no Sl part and no line' seqlen_server_gate "$T/plain.log" '' 720896
expect 1 'no Sl part but a capped server refuses' seqlen_server_gate "$T/on.log" '' 720896
expect 0 'Sl at the pool bound states no cap' seqlen_server_gate "$T/plain.log" 4096 4096
expect 1 'Sl at the pool bound with a cap line refuses' seqlen_server_gate "$T/on.log" 4096 4096

# The parser: SlN passes --max-seq-len N and states the arm's N and context to the gate; a bad,
# duplicate or past-the-pool part skips the arm.
echo '{}' >"$T/prompts.json"
H=$T/home
mkdir -p "$H/kernels"
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-Pp4-Ei-Pm56-Pe-N512-L720896
run_loop() {
  (
    set +u
    log() { echo "LOG: $*"; }
    power_set() { :; }; power_restore() { :; }; vpack_gate() { return 0; }
    arm() { echo "ARM $1 SL=${ARM_SL:-} CTX=${ARM_CTX:-}"; shift; printf '    %s\n' "$@" "${ARGS[@]}"; }
    HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T MAX_SEQS=1
    PROMPTS=$T/prompts.json OUT=$T/out CANCEL_FILE=/nonexistent/cancel CANCEL=$CANCEL_FILE
    mkdir -p "$OUT"
    unset RADIANCE_TUNE
    ARMS="$B $B-Sl4096 $B-Sl63 $B-Sl4096-Sl2048 ${B%-L720896}-L2048-Sl4096 $B-Slx"
    eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
    HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
    s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
    e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
    eval "$(sed -n "${s},${e}p" "$W")"
  ) 2>&1
}
run_loop >"$T/loop.log"
section() { awk -v a="$1" 'index($0, "ARM " a " ") == 1 { p=1; next } /^ARM / { p=0 } p' "$T/loop.log"; }
plain=$(section "$B") on=$(section "$B-Sl4096")
grep -qxF "ARM $B SL= CTX=720896" "$T/loop.log" && ok 'plain arm states no Sl' || bad 'plain arm header'
[[ "$plain" != *'--max-seq-len'* ]] && ok 'plain arm passes no --max-seq-len' || bad 'plain arm passes --max-seq-len'
grep -qxF "ARM $B-Sl4096 SL=4096 CTX=720896" "$T/loop.log" && ok 'Sl4096 states N and the context' || bad 'Sl4096 header'
if grep -A1 -x -- '    --max-seq-len' <<<"$on" | grep -qx '    4096'; then ok 'Sl4096 passes --max-seq-len 4096'
else bad 'Sl4096 does not pass --max-seq-len 4096'; fi
for x in "$B-Sl63" "$B-Sl4096-Sl2048" "${B%-L720896}-L2048-Sl4096"; do
  if grep -q "^ARM $x " "$T/loop.log"; then bad "$x launched"; else ok "$x skipped"; fi
done
grep -q 'Sl4096 is past the arm.s --max-model-len 2048' "$T/loop.log" && ok 'past-the-pool reason logged' ||
  bad 'past-the-pool reason missing'
grep -q 'duplicate Sl part' "$T/loop.log" && ok 'duplicate reason logged' || bad 'duplicate reason missing'
if grep -q "^ARM $B-Slx " "$T/loop.log"; then bad "$B-Slx launched"; else ok 'a part without a number is not Sl'; fi

(( fail )) && { cat "$T/loop.log" | grep -E '^LOG|^ARM' | head -20; echo 'ab_window_sl_test: FAILED'; exit 1; }
echo 'ab_window_sl_test: all checks passed'
