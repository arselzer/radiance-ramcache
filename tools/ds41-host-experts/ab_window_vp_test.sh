#!/usr/bin/env bash
# ab_window_vp_test.sh [ab_window.sh] -- ab_window.sh's Vp part beside a DSpark bank (Ds), with no GPU,
# no server and no model:
#   1. vp_server_gate on synthetic server logs: a Vp arm with a bank is measured only when every rank
#      logged the engine's bank check ('DSpark bank bound; N trunk weights packed', N > 0), and the
#      plain and no-bank cases read as before;
#   2. the arm loop, every action stubbed (as work/int4-tmp/dryrun.sh does): a Ds + Vp arm is launched
#      with the switch and the bank, not skipped, and a Ds arm without Vp states the switch off.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-vp-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# ---- 1. the server-log gate
eval "$(sed -n '/^vp_server_gate() {/,/^}/p' "$W")"
declare -F vp_server_gate >/dev/null || { echo "FAIL no vp_server_gate in $W"; exit 1; }
on='VRAM packed experts (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED): device.ds41_vram_packed selects moe_gemm_ggml_mxfp4_gemv_vp'
held() { echo "VRAM packed experts: rank $1 holds $2 static residents in the packed-scale form (x gate_up and y down weights)"; }
bankl() { echo "VRAM packed experts: rank $1: DSpark bank bound; $2 trunk weights packed from the main container; the bank's 576 weights stay block_mxfp4 (144 VRAM residents) and its 6 draft moe_gemm_ggml ops keep their row"; }
mk() { local f=$T/$1.log; shift; printf '%s\n' "$@" > "$f"; echo "$f"; }
L_new=$(mk new "$on" "$(held 0 740)" "$(bankl 0 1480)" "$(held 1 741)" "$(bankl 1 1482)" "$(held 2 739)" "$(bankl 2 1478)" "$(held 3 742)" "$(bankl 3 1484)")
L_old=$(mk old "$on" "VRAM packed experts: rank 0: 0 routed weights can be held packed; the experts are not read from one container (a checkpoint, or a DSpark bank)" "$(held 0 0)" "$(held 1 0)" "$(held 2 0)" "$(held 3 0)")
L_three=$(mk three "$on" "$(held 0 740)" "$(bankl 0 1480)" "$(held 1 741)" "$(bankl 1 1482)" "$(held 2 739)" "$(bankl 2 1478)" "$(held 3 742)")
L_zero=$(mk zero "$on" "$(held 0 740)" "$(bankl 0 1480)" "$(held 1 741)" "$(bankl 1 1482)" "$(held 2 739)" "$(bankl 2 1478)" "$(held 3 742)" "$(bankl 3 0)")
L_nobank=$(mk nobank "$on" "$(held 0 740)" "$(held 1 741)" "$(held 2 739)" "$(held 3 742)")
L_plain=$(mk plain "nothing packed here")
expect() {   # want(0/1) what log vp bank
  local want=$1 what=$2 why rc
  why=$(vp_server_gate "$3" "$4" "$5"); rc=$?
  if [[ $rc == "$want" ]]; then ok "$what${why:+ ($why)}"; else bad "$what: exit $rc, want $want${why:+ ($why)}"; fi
}
expect 0 "Vp + bank: every rank packed the trunk and checked the bank"  "$L_new"    1 1
expect 1 "Vp + bank on a radiance that refused the switch with a bank"  "$L_old"    1 1
expect 1 "Vp + bank with one rank's bank check missing"                 "$L_three"  1 1
expect 1 "Vp + bank with a rank that packed no trunk weight"            "$L_zero"   1 1
expect 0 "Vp without a bank: the switch's line and residents held"      "$L_nobank" 1 ""
expect 1 "Vp without a bank, nothing held"                              "$L_plain"  1 ""
expect 0 "no Vp, nothing packed (with or without a bank)"               "$L_plain"  "" 1
expect 1 "no Vp, but the server packed (the switch from the window)"    "$L_nobank" "" ""

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/hot.json"
echo '{}' > "$T/prompts.json"
B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Rb
DS=$B-Ss-Pu-Is-Pi8-Ns7-Fw-Rm7-Ds1-Dkt-Db2-Dv24-Pmr-Df
ARMS="$DS $DS-Vp $DS-Vp-r2"
out=$(
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T DRAFT_BANK=$T/draft.rad
  DRAFT_HOTSET=; export DRAFT_HOTSET_heldout=$T/hot.json DRAFT_HOTSET_heldout_db2=$T/hot.json
  PROMPTS=$T/prompts.json
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=1024
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) 2>&1
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
for a in "$DS-Vp" "$DS-Vp-r2"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  [[ "$sec" == *"RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1"* ]] && ok "$a: the switch on" || bad "$a: no VRAM_PACKED=1"
  [[ "$sec" == *"--draft-bank $T/draft.rad"* ]] && ok "$a: the bank bound" || bad "$a: no --draft-bank"
  [[ "$sec" == *"RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=24"* && "$sec" == *"RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=2"* ]] \
    && ok "$a: the draft hot set and its GEMV path as in the Ds arm" || bad "$a: DRAFT_VRAM / DRAFT_GEMV missing"
done
sec=$(section "$DS")
[[ -n "$sec" && "$sec" == *"RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=0"* ]] && ok "$DS: the switch stated off" || bad "$DS: not launched with VRAM_PACKED=0"
if grep -E "LOG: arm [^ ]*: .*(skipped|refused)" "$T/loop.txt"; then bad "an arm was skipped or refused"; else ok "no arm skipped"; fi

(( fail )) && { echo "ab_window_vp_test: FAILED (loop output: see above)"; exit 1; }
echo "ab_window_vp_test: all checks passed"
