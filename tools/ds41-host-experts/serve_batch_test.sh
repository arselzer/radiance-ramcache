#!/usr/bin/env bash
# serve_batch_test.sh [ab_window.sh] -- serve-batch.sh against the window harness, with no GPU, no server and no model:
#   1. the harness's own arm loop (every action stubbed, as ab_window_ba_test.sh) builds the measured four-card batch
#      arm, and arm()'s own launch words (ab_window.sh launch_words) give the server's environment and flags;
#      serve-batch.sh --dry-run gives the same environment, each variable at its last assignment, and the same
#      flags, each at its last value (serve_batch_match.py compare), apart from the port, --radiance-home and
#      --served-model-name, and with serve_batch_match.py params' arguments all of them;
#   2. --streams 128 gives the 128-stream arm's pool (L147456, N128), --reserve-mib the mAN variable, --shared-head
#      the PchN floor, and no --profile / --mirror leaves those flags out;
#   3. a missing --silu-table or --model, and a reserve under 1024 MiB, are refused.
# Exit 0 when every check passes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HERE/ab_window.sh}
SB=$HERE/serve-batch.sh
T=$(mktemp -d /tmp/serve-batch-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

ARM=all-t40-f-M2048-H-p256-m229376-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs-mA-Qd64-Qa0-Pc-Pch32-Pcs128-L294912-Pf-N256-C256-Ba2048-Bs-Qy60-Xqprof
mkdir -p "$T/home/kernels" "$T/mirror" "$T/profiles"; echo "{}" > "$T/profiles/prof.json"
echo "not a library" > "$T/home/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
LW=$(sed -n '/^launch_words() {/,/^}/p' "$W")
[[ $LW == *'"$BIN" --radiance-home'* ]] || bad "no launch_words in $W"
(
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  eval "$LW"
  arm() { local name=$1; shift; echo "ARM $name"; launch_words "$@"; printf '%s\n' "${LAUNCH[@]}"; }
  HX_THREADS=16 SILU=SILU RHOME=$T/home RADIANCE_HOME_DIR2=$T/home MIRROR=$T/mirror PROMPTS=$T/prompts.json
  BIN=BIN MODEL=MODEL PORT=18089 PROFILE=SEALED EXTRA_PROFILES=$T/profiles
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$T/home/kernels/libr4d.so" | cut -d' ' -f1); LIB=$T/home/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  ARMS=$ARM
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) > "$T/loop.txt" 2>&1
grep -q "^ARM $ARM$" "$T/loop.txt" && ok "the harness launches the arm" ||
  { bad "the harness did not launch the arm: $(grep -m3 -E 'skip|unknown' "$T/loop.txt")"; }
awk -v a="ARM $ARM" '$0 == a { p = 1; next } p' "$T/loop.txt" > "$T/harness.txt"

RADIANCE_BIN=BIN "$SB" --model MODEL --silu-table SILU --profile "$T/profiles/prof.json" --mirror "$T/mirror" --dry-run > "$T/sb.txt" ||
  bad "serve-batch.sh --dry-run failed"

M=$HERE/serve_batch_match.py
if python3 -I "$M" compare "$T/harness.txt" "$T/sb.txt" --port --radiance-home --served-model-name > "$T/cmp.txt"; then
  ok "$(cat "$T/cmp.txt") (the port, --radiance-home and the served name aside)"
else bad "serve-batch.sh against the arm: $(tr '\n' ';' < "$T/cmp.txt")"; fi
if python3 -I "$M" params "$T/harness.txt" > "$T/params.txt" && mapfile -t P < "$T/params.txt" &&
   RADIANCE_BIN=BIN "$SB" "${P[@]}" --dry-run > "$T/sb2.txt" && python3 -I "$M" compare "$T/harness.txt" "$T/sb2.txt" > "$T/cmp2.txt"; then
  ok "with serve_batch_match.py params' arguments: $(cat "$T/cmp2.txt"), nothing aside"
else bad "params: $(tr '\n' ' ' < "$T/params.txt") $(tr '\n' ';' < "$T/cmp2.txt" 2>/dev/null)"; fi

# ---- 2. the parameters
d128=$("$SB" --model M --silu-table S --streams 128 --dry-run)
[[ $d128 == *"--max-model-len 147456 "* && $d128 == *"--max-num-seqs 128 "* ]] && ok "128 streams: L147456, N128" ||
  bad "128 streams: $(grep '^cmd' <<< "$d128")"
dres=$("$SB" --model M --silu-table S --reserve-mib 20480 --shared-head 64 --port 9000 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB=20480' <<< "$dres" && ok "--reserve-mib: the mAN variable" ||
  bad "--reserve-mib"
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=64' <<< "$dres" && ok "--shared-head 64: Pch64's floor" ||
  bad "--shared-head"
[[ $dres == *"--port 9000"* ]] && ok "--port" || bad "--port"
d0=$("$SB" --model M --silu-table S --dry-run)
if grep -q 'RAM_RESERVE_MIB' <<< "$d0" || [[ $d0 == *--expert-profile* || $d0 == *--expert-plane-mirror* ]]; then
  bad "the defaults add a reserve, a profile or a mirror"
else ok "the defaults: no reserve variable, no --expert-profile, no mirror"; fi
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=32' <<< "$d0" && ok "the shared-head floor's default: 32" ||
  bad "the shared-head default"

dh=$("$SB" --model M --silu-table S --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=60' <<< "$dh" && grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT=' <<< "$dh" &&
  ok "hybrid decode by default: Qy60, the share adapting" || bad "the hybrid default"
d0h=$("$SB" --model M --silu-table S --hybrid 0 --max-seq-len 4096 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=0' <<< "$d0h" && [[ $d0h == *"--max-seq-len 4096"* ]] &&
  ok "--hybrid 0: off (an arm without Qy); --max-seq-len 4096: SlN's flag" || bad "--hybrid 0 / --max-seq-len"
[[ $dh != *--max-seq-len* ]] && ok "no --max-seq-len by default" || bad "a --max-seq-len by default"

# ---- 3. refusals
"$SB" --model M --silu-table S --hybrid 401 --dry-run > /dev/null 2>&1 && bad "--hybrid 401 accepted" || ok "--hybrid above 400 refused"
"$SB" --model M --dry-run > /dev/null 2>&1 && bad "no --silu-table accepted" || ok "no --silu-table refused"
"$SB" --silu-table S --dry-run > /dev/null 2>&1 && bad "no --model accepted" || ok "no --model refused"
"$SB" --model M --silu-table S --reserve-mib 512 --dry-run > /dev/null 2>&1 && bad "a 512 MiB reserve accepted" ||
  ok "a reserve under 1024 MiB refused"
"$SB" --model "$T/none.rad" --silu-table S > /dev/null 2>&1 && bad "a missing container accepted" ||
  ok "a missing container refused before any start"

(( fail == 0 )) && echo "serve_batch_test: all passed"
exit $fail
