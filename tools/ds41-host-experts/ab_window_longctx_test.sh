#!/usr/bin/env bash
# ab_window_longctx_test.sh [ab_window.sh] -- ab_window.sh's long-context prompt sets (U parts whose receipt
# declares max_context_tokens: Ufill32k, Ulong48k, Ulong100k from make_longctx_receipts.py), with no GPU, no
# server and no model:
#   1. the arm loop, every action stubbed (as ab_window_ba_test.sh), receipts in a scratch EXTRA_PROMPTS: the
#      window CR arms are launched with their receipt, --max-model-len and per-request timeout (300 s plus
#      the wave at 100 tok/s with Lc, 25 without, at least 600); an arm without a U part, or whose receipt
#      declares no max_context_tokens, gets none; a context shorter than the receipt's (no Lm, a smaller
#      Lm, an L part after Lm), two U parts and an unreadable receipt are skipped with their reasons; a
#      window's AB_REQUEST_TIMEOUT_S wins for every arm, and one that is not whole seconds skips them;
#   2. arm()'s ab_requests.py command, run against a stub: AB_REQUEST_TIMEOUT_S is the arm's timeout, and
#      absent without one;
#   3. the end of the window: a prompt set with <name>-khref.json has its Kh arms (only those) compared
#      with it (ab_requests.py --hashes-ref), a set without one is not;
#   4. DEVICE_CHECKS=lsparse (the sparse ops at long index widths) against a fake card harness: card 0, the
#      window's LCP_CHECK_C and LSPARSE_WIDTHS, the switches unset for it; only a PASS line with exit 0
#      sets LSPARSE_OK, and it gates no arm.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-longctx-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
unset AB_REQUEST_TIMEOUT_S AB_CONCURRENCY

# ---- the receipts (only what the harness reads: the request list and max_context_tokens)
X=$T/extra
mkdir -p "$X"
receipt() {   # name, max_context_tokens (empty: none)
  python3 -I -c 'import json, sys
r = {"benchmark_requests": [{"name": sys.argv[1] + "-1", "request": {"prompt": [0, 1], "max_tokens": 4}}]}
if sys.argv[2]:
    r["max_context_tokens"] = int(sys.argv[2])
json.dump(r, open(sys.argv[3], "w"))' "$1" "$2" "$X/$1-receipt.json"
}
receipt fill32k 32832
receipt long48k 48045
receipt long100k 100115
receipt long8k ""
echo '{"benchmark_requests": [' > "$X/broken-receipt.json"
echo '{"schema": "ds41-prefill-hash-reference-v1", "prompts": []}' > "$X/fill32k-khref.json"

# ---- 1. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
R=all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs
C4=all-t40-f-M4096-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc4096-P1-Qe64-Mb1-Pci4096-Rb-Vp-Cs
S16=all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Rb-Vp-Cs
ABS=$R
ANCHOR=$R-Kh-Lm131072-Ulong8k
FILL=$R-Kh-Lm131072-Ufill32k
L48=$R-Kh-Lm131072-Ulong48k
L48C4=$C4-Kh-Lm131072-Ulong48k
L100=$R-Kh-Lm131072-Ulong100k
L100C4=$C4-Kh-Lm131072-Ulong100k
N4S=$R-L1024-Pf-N4-C4-Pmr4-Ba
N4L=$R-L131072-Pf-N4-C4-Pmr4-Ba
M16FILL=$S16-Kh-Lm32832-Ufill32k
N4LONG=$R-L131072-Pf-N4-C4-Pmr4-Ba-Ulong48k
EXACT=$R-Kh-Lm100115-Ulong100k
NOLM=$R-Kh-Ulong100k
SHORT=$R-Kh-Lm65536-Ulong100k
LAST=$R-Lm131072-L1024-Ulong48k
TWO=$R-Kh-Lm131072-Ulong48k-Ulong100k
BROKEN=$R-Lm131072-Ubroken
ARMS_ALL="$ABS $ANCHOR $FILL $L48 $L48C4 $L100 $L100C4 $N4S $N4L $M16FILL $N4LONG $EXACT $NOLM $SHORT $LAST $TWO $BROKEN"
loop() {   # output file, then VAR=value assignments for the loop's shell (a window's environment)
  local out=$1; shift
  (
    set +u
    for v in "$@"; do export "${v?}"; done
    log() { echo "LOG: $*"; }
    power_set() { :; }; power_restore() { :; }
    vpack_gate() { return 0; }
    arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; echo "  PROMPTS: ${ARM_PROMPTS:-}"
            echo "  TIMEOUT: ${ARM_TIMEOUT:-}"; echo "  CONC: ${ARM_CONC:-}"; }
    HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T EXTRA_PROMPTS=$X
    PROMPTS=$T/prompts.json
    unset RADIANCE_TUNE
    SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
    TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
    RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
    BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
    OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
    eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
    s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
    e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
    eval "$(sed -n "${s},${e}p" "$W")"
  ) > "$out" 2>&1
}
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$2"; }
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
field() { grep -oE "^  $2: .*" <<< "$1" | head -1 | cut -d' ' -f4-; }
launched() {   # arm, loop output, what, want timeout, want receipt (empty: the held-out set), want context
  local sec got
  sec=$(section "$1" "$2")
  if [[ -z "$sec" ]]; then bad "$3 not launched: $(grep -m2 "arm $1: " "$2" | tr '\n' ' ')"; return; fi
  got=$(field "$sec" TIMEOUT)
  if [[ "$got" == "$4" ]]; then ok "$3: timeout '${4}'"; else bad "$3: timeout '$got', want '$4'"; fi
  got=$(field "$sec" PROMPTS)
  if [[ "$got" == "${5:+$X/$5-receipt.json}" ]]; then ok "$3: prompts ${5:-held-out}"; else bad "$3: prompts '$got'"; fi
  got=$(grep -oE -- '--max-model-len [0-9]+' <<< "$(field "$sec" ARGS)" | tail -1 | cut -d' ' -f2)
  if [[ "$got" == "$6" ]]; then ok "$3: --max-model-len ${6:-512 by default}"; else bad "$3: --max-model-len '$got', want '$6'"; fi
}
skipped() {   # arm, loop output, the reason's words, what
  if [[ -n "$(section "$1" "$2")" ]]; then bad "$4 launched"
  elif grep "arm $1: " "$2" | grep -qF "$3"; then ok "$4 skipped: $3"
  else bad "$4: no '$3' ($(grep -m2 "arm $1: " "$2" | tr '\n' ' '))"; fi
}
L1=$T/loop.txt
loop "$L1" ARMS="$ARMS_ALL"
launched "$ABS" "$L1" "absorber (no U part)" "" "" ""
launched "$ANCHOR" "$L1" "Ulong8k anchor (no max_context_tokens)" "" long8k 131072
launched "$FILL" "$L1" "Ufill32k C2" 629 fill32k 131072
launched "$L48" "$L1" "Ulong48k C2" 781 long48k 131072
launched "$L48C4" "$L1" "Ulong48k C4" 781 long48k 131072
launched "$L100" "$L1" "Ulong100k C2" 1302 long100k 131072
launched "$L100C4" "$L1" "Ulong100k C4" 1302 long100k 131072
launched "$N4S" "$L1" "N4 Ba at L1024" "" "" 1024
launched "$N4L" "$L1" "N4 Ba at L131072" "" "" 131072
launched "$M16FILL" "$L1" "Ufill32k at 16-token passes (25 tok/s)" 1614 fill32k 32832
launched "$N4LONG" "$L1" "Ulong48k sent 4 at a time" 2224 long48k 131072
launched "$EXACT" "$L1" "Ulong100k at exactly its max_context_tokens" 1302 long100k 100115
sec=$(section "$L48C4" "$L1")
[[ "$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP)" == 4096 ]] && ok "C4 arm: LCP 4096" || bad "C4 arm: LCP '$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP)'"
for a in "$N4S" "$N4L"; do
  sec=$(section "$a" "$L1")
  [[ "$(val "$sec" RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT)" == 256 && "$(field "$sec" CONC)" == 4 ]] &&
    ok "N4 Ba arm: BATCH_ADMIT 256, 4 at a time" || bad "N4 Ba arm $a: BATCH_ADMIT '$(val "$sec" RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT)'"
done
skipped "$NOLM" "$L1" "needs 100115 tokens of context (its receipt's max_context_tokens), more than the arm's --max-model-len 512" "Ulong100k without Lm"
skipped "$SHORT" "$L1" "more than the arm's --max-model-len 65536" "Ulong100k at Lm65536"
skipped "$LAST" "$L1" "more than the arm's --max-model-len 1024" "Ulong48k with an L1024 after Lm131072"
skipped "$TWO" "$L1" "Ulong100k after Ulong48k (one receipt an arm)" "two U parts"
skipped "$BROKEN" "$L1" "the receipt does not read as JSON" "an unreadable receipt"

# A window's AB_REQUEST_TIMEOUT_S: every arm's, long receipt or not; anything but whole seconds skips.
L2=$T/loop-window.txt
loop "$L2" ARMS="$ABS $L100 $M16FILL" AB_REQUEST_TIMEOUT_S=900
launched "$ABS" "$L2" "window AB_REQUEST_TIMEOUT_S=900, absorber" 900 "" ""
launched "$L100" "$L2" "window AB_REQUEST_TIMEOUT_S=900, Ulong100k" 900 long100k 131072
launched "$M16FILL" "$L2" "window AB_REQUEST_TIMEOUT_S=900, Ufill32k M16" 900 fill32k 32832
L3=$T/loop-bad.txt
loop "$L3" ARMS="$ABS $L100" AB_REQUEST_TIMEOUT_S=10m
skipped "$ABS" "$L3" "AB_REQUEST_TIMEOUT_S=10m is not a whole number of seconds" "absorber under AB_REQUEST_TIMEOUT_S=10m"
skipped "$L100" "$L3" "AB_REQUEST_TIMEOUT_S=10m is not a whole number of seconds" "Ulong100k under AB_REQUEST_TIMEOUT_S=10m"

# ---- 2. arm()'s ab_requests.py command against a stub that records its environment
S=$T/stub
mkdir -p "$S"
cat > "$S/ab_requests.py" <<'EOF'
import os, sys
with open(os.environ["STUB_OUT"], "w") as f:
    f.write("timeout=%s conc=%s argv=%s\n" % (os.environ.get("AB_REQUEST_TIMEOUT_S", "<unset>"),
                                              os.environ.get("AB_CONCURRENCY", "<unset>"), " ".join(sys.argv[1:])))
EOF
cmd=$(awk '/^arm\(\) \{/ { p = 1 } p && /^  env AB_CONCURRENCY=/ { print; getline; print; exit }' "$W")
if [[ -z "$cmd" || "$cmd" != *ab_requests.py* ]]; then bad "no ab_requests.py command in arm()"
else
  run_cmd() {   # ARM_TIMEOUT, ARM_CONC, want timeout, want conc, what
    local got
    (
      set +u
      HERE=$S OUT=$T/cmd PORT=1 n=3 name=x ARM_PROMPTS=$X/long100k-receipt.json PROMPTS=$T/prompts.json
      ARM_TIMEOUT=$1 ARM_CONC=$2 STUB_OUT=$T/stub.out
      export STUB_OUT
      mkdir -p "$OUT"
      eval "$cmd"
    )
    got=$(cat "$T/stub.out" 2>/dev/null)
    if [[ "$got" == "timeout=$3 conc=$4 argv=$X/long100k-receipt.json http://127.0.0.1:1 3 $T/cmd/x.json" ]]; then ok "arm(): $5"
    else bad "arm(): $5: '$got'"; fi
    rm -f "$T/stub.out"
  }
  run_cmd 1302 "" 1302 1 "AB_REQUEST_TIMEOUT_S=1302 passed for an arm with a timeout"
  run_cmd "" 4 "<unset>" 4 "no AB_REQUEST_TIMEOUT_S without one"
fi

# ---- 3. the end of the window: --hashes-ref for each prompt set with a reference, its Kh arms only
end=$(awk '/^declare -A BY_PROMPTS=\(\) KH_BY_PROMPTS=\(\)/ { p = 1 } p { print } p && /python3 .*--hashes-ref/ { getline; print; exit }' "$W")
if [[ "$end" != *--hashes-ref* ]]; then bad "no --hashes-ref step at the end of the window"
else
  cat > "$S/ab_requests.py" <<'EOF'
import os, sys
with open(os.environ["STUB_OUT"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
EOF
  (
    set +u
    log() { :; }
    HERE=$S OUT=$T/end EXTRA_PROMPTS=$X STUB_OUT=$T/end.calls
    export STUB_OUT
    mkdir -p "$OUT"
    ARMS="$ABS $FILL $R-Lm131072-Ufill32k $L48 $L48C4"
    eval "$end"
  ) > /dev/null 2>&1
  refs=$(grep -- '^--hashes-ref' "$T/end.calls" 2>/dev/null)
  if [[ "$refs" == "--hashes-ref $X/fill32k-khref.json $T/end $FILL " || "$refs" == "--hashes-ref $X/fill32k-khref.json $T/end $FILL" ]]; then
    ok "end: Ufill32k's Kh arm (only) against fill32k-khref.json; no reference call for Ulong48k"
  else bad "end: --hashes-ref calls '$refs'"; fi
  grep -q -- "^--hashes $T/end $L48 $L48C4" "$T/end.calls" && ok "end: Ulong48k's Kh arms against each other (--hashes)" ||
    bad "end: no --hashes for Ulong48k ($(tr '\n' '|' < "$T/end.calls"))"
fi

# ---- 4. DEVICE_CHECKS=lsparse: the long-context sparse ops' card check, run against a fake harness that
# records how it was called; its verdict is logged and sets LSPARSE_OK, and it gates no arm.
blk=$(awk '/^LSPARSE_OK=""$/ { p = 1 } p { print } p && /^fi$/ { exit }' "$W")
if [[ "$blk" != *'lsparse $((10#$LC))'* ]]; then bad "no lsparse device-check block in ab_window.sh"
else
  cat > "$S/tc" <<'EOF'
#!/usr/bin/env bash
echo "argv=$* hip=${HIP_VISIBLE_DEVICES-unset} lcp=${RADIANCE_EXPERIMENTAL_DS41_LCP-unset} tall=${RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL-unset} p1=${RADIANCE_EXPERIMENTAL_DS41_P1-unset}" >> "$TC_CALLS"
echo "lsparse W=65537 (ratio 2, positions 130816..131071, 256 rows): index_score 0, ..."
echo "lsparse: ${TC_VERDICT:-PASS}: ..."
exit ${TC_EXIT:-0}
EOF
  chmod +x "$S/tc"
  echo lib > "$T/libr4d.so"
  lsp() {   # verdict, exit, then VAR=value; prints LSPARSE_OK and the log lines
    local v=$1 x=$2; shift 2
    (
      set +u
      log() { echo "LOG: $*"; }
      OUT=$T/lsp; mkdir -p "$OUT"
      DEVICE_CHECKS="swiglu lcp lsparse" TALL_CHECK=$S/tc TALL_LIBR4D=$T/libr4d.so LCP_CHECK_C=4096
      export TC_CALLS=$T/tc.calls TC_VERDICT=$v TC_EXIT=$x RADIANCE_EXPERIMENTAL_DS41_LCP=999 RADIANCE_EXPERIMENTAL_DS41_P1=1
      for kv in "$@"; do eval "$kv"; done
      eval "$blk"
      echo "LSPARSE_OK=$LSPARSE_OK"
    ) 2>&1
  }
  rm -f "$T/tc.calls"
  out=$(lsp PASS 0)
  [[ "$out" == *"LSPARSE_OK=1"* && "$out" == *"(it gates no arm)"* ]] && ok "lsparse: a PASS sets LSPARSE_OK" || bad "lsparse PASS: $out"
  [[ "$(cat "$T/tc.calls")" == "argv=$T/libr4d.so lsparse 4096 hip=0 lcp=unset tall=unset p1=unset" ]] &&
    ok "lsparse: card 0, the window's LCP_CHECK_C, LCP / EXACT_TALL / P1 unset for the harness" || bad "lsparse call: $(cat "$T/tc.calls")"
  rm -f "$T/tc.calls"
  out=$(lsp PASS 0 'LSPARSE_WIDTHS="131073:1 524289:1"')
  [[ "$(cat "$T/tc.calls")" == "argv=$T/libr4d.so lsparse 4096 131073:1 524289:1 hip=0 lcp=unset tall=unset p1=unset" ]] &&
    ok "lsparse: LSPARSE_WIDTHS passed as the width list" || bad "lsparse widths call: $(cat "$T/tc.calls")"
  out=$(lsp FAIL 0)
  [[ "$out" == *"LSPARSE_OK="$'\n'* || "$out" == *"LSPARSE_OK=" ]] && [[ "$out" == *"did NOT pass"* ]] &&
    ok "lsparse: a FAIL line leaves LSPARSE_OK empty, logged" || bad "lsparse FAIL: $out"
  out=$(lsp PASS 3)
  [[ "$out" == *"did NOT pass"* && "$out" != *"LSPARSE_OK=1"* ]] && ok "lsparse: a non-zero exit is not a pass" || bad "lsparse exit 3: $out"
  out=$(lsp PASS 0 TALL_CHECK=/nonexistent)
  [[ "$out" == *"not run"* && "$out" != *"LSPARSE_OK=1"* ]] && ok "lsparse: no harness, not run" || bad "lsparse without harness: $out"
  out=$(lsp PASS 0 'DEVICE_CHECKS="swiglu lcp"')
  [[ "$out" != *"long-context sparse"* ]] && ok "lsparse: not in DEVICE_CHECKS, not run" || bad "lsparse ran without the entry: $out"
fi
(( fail == 0 )) && echo "PASS ab_window_longctx_test" || echo "FAIL ab_window_longctx_test (loop output: $T/loop.txt)"
exit $fail
