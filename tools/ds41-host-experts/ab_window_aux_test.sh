#!/usr/bin/env bash
# ab_window_aux_test.sh [ab_window.sh] -- ab_window.sh's aux expert worker parts (Ax / AxN / Axv / AxvN / Ax0 /
# AxfN; RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS, work/aux/SPEC.md §13 and §15.6) with no GPU, no server and no model:
#   1. the arm loop, every action stubbed (as ab_window_ba_test.sh): an arm without an Ax part states the six
#      switches with the worker off and keeps the window's mask; Ax (26000), Ax24000, Axv, Axv20000, Ax0, Ax-Axf
#      and Axv-Axf8 launch with their switches and the extra card appended last to the four-UUID mask, and the
#      window's AUX_MIB, AUX_DEADLINE_US, AUX_BACKOFF and AUX_HEADROOM_MIB reach them; the skips, each with its
#      reason: no mask, a mask of five, no extra card, no K / Vp / BN / all, M8, Tb, Rt, Ds, Dk, two Ax parts, Ax
#      beside Ax0, Ax512, Ax2000000, Axf without an Ax part, Axf0, no AUX_OK, another library, no AXV_OK (or one
#      measured on another library), a client not allowed, too many foreign MiB, too little free VRAM;
#   2. arm()'s startup-line guards (its switch table's Ax / Axv rows and the "on:" line's six values) on lines
#      printed with the engine's own format strings (core/engine_aux.cpp, core/place/aux_expert_worker.cpp);
#   3. aux_find on fabricated KFD topologies: the rank cards skipped, another ISA, no VRAM, no PCI location, two
#      candidates (the root rule, AUX_UUID, AUX_BDF), VRAM then the BDF on independent roots, a mask of three,
#      rank cards of two ISAs; its outputs (the UUID of a unique_id above 2^63, BDF, gpu_id, drm directory);
#   4. aux_clients, aux_clients_gate and aux_clients_after on fabricated KFD process entries (the stopped server's
#      own entry and a gone process's left out, the wait for the server's entry bounded);
#   5. aux_exit_gate and aux_arm_exit (the margin, AXV_OK, results set aside, a client arriving) on the engine's
#      exit lines;
#   6. the DEVICE_CHECKS=aux section with stub checks and a stub docker: it passes, and each way it must not (the
#      shared extra card's clients and free VRAM included: no check runs then);
#   7. the aux_any line before the loop leaves the loop the first `for a in $ARMS; do` (the other tests' anchor);
#   8. kld_finish's aux gates (aux_kld_check): a Kld arm's records are compared only when the worker ran on every
#      rank with its arm's values, no launch failure, no row differing and no client arriving; an Ax arm is never
#      the reference; Ax0 and plain arms as before;
#   9. the loop skips an Ax / Axv part beside Kld when the arm's step is over 16 rows (Lc, BT32), and launches it on
#      the 16-token base;
#  10. arm(): the aux startup lines logged before the switch table can return, the server's pid kept for the exit;
#  12. AxrN (RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS): the loop's switch and skips, aux_want's rows, the exit gate's acted
#      check, the DEVICE_CHECKS=aux section's auxtall step.
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
SRC=${AUX_TEST_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
T=$(mktemp -d /tmp/ab-aux-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

FUNCS=$(sed -n '/^# ---- aux expert worker: functions/,/^# ---- aux expert worker: end of functions/p' "$W")
[[ -n "$FUNCS" ]] || { echo "FAIL no aux expert worker functions block in $W"; exit 1; }

# The engine's format strings: a call's string literals joined (the call found by a phrase inside it).
cfmt() {   # source file, a phrase inside the call, the call's name
  python3 -I - "$@" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
k = src.index(sys.argv[2])
i = src.rindex(sys.argv[3] + "(", 0, k) + len(sys.argv[3]) + 1
lit = re.compile(r'\s*"((?:[^"\\]|\\.)*)"')
out, m = "", lit.match(src, i)
while m:
    out += m.group(1)
    m = lit.match(src, m.end())
sys.stdout.write(out)
PY
}
ONFMT=$(cfmt "$SRC/core/engine_aux.cpp" 'resident on device' RAD_INFO)
RANKFMT=$(cfmt "$SRC/core/place/aux_expert_worker.cpp" 'layer jobs, %llu units' fmt)
[[ "$ONFMT" == *"forced lateness %lld"* ]] || bad "the engine's 'on:' format not found in core/engine_aux.cpp ('$ONFMT')"
[[ "$RANKFMT" == *"launch failures; verify:"* ]] || bad "the engine's rank format not found in core/place/aux_expert_worker.cpp"
on_line() {   # policy deadline backoff headroom cap forced
  printf "I $ONFMT\n" "aux experts: on: " 1513 "25.4 GiB" 4 "rank 0 380, rank 1 378, rank 2 377, rank 3 378" 0.4123 \
    0.9 15360 61.2 1500 13 0 16 "$1" "$2" "$3" "$4" "$5" "$6"
}
rank_line() {   # rank, layer jobs, launch failures, rows compared, rows differing, forced misses[, released text]
  # The 7th: what a controller adds after "... after it failed" (", L while released"); "" as the merged worker prints.
  printf "I $RANKFMT\n" "aux experts: rank " "$1" "$2" $(( $2 * 3 )) 900 2000 16 0 0 0 3 0 "${7:-}" 0 1.5 310.2 1900.0 \
    "$6" "$6" 4 9 2.0 "$3" "$4" "$5" $(( $5 * 2 )) 0
}

# ---- fabricated sysfs trees
mkcpu() {   # sysfs: the CPU node (gfx_target_version 0)
  local d=$1/class/kfd/kfd/topology/nodes/0
  mkdir -p "$d"
  printf 'cpu_cores_count 48\nsimd_count 0\ngfx_target_version 0\nunique_id 0\n' > "$d/properties"
  echo 0 > "$d/gpu_id"
}
mknode() {   # sysfs node gfx_target_version unique_id(hex) bdf("" none) gpu_id vram_bytes
  local d=$1/class/kfd/kfd/topology/nodes/$2 uid
  mkdir -p "$d/mem_banks/0"
  uid=$(python3 -I -c 'import sys; print(int(sys.argv[1], 16))' "$4")
  {
    echo "cpu_cores_count 0"
    echo "simd_count 128"
    echo "gfx_target_version $3"
    echo "unique_id $uid"
    if [[ $5 =~ ^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$ ]]; then
      echo "domain $((16#${BASH_REMATCH[1]}))"
      echo "location_id $(( (16#${BASH_REMATCH[2]} << 8) | (16#${BASH_REMATCH[3]} << 3) | BASH_REMATCH[4] ))"
    fi
  } > "$d/properties"
  echo "$6" > "$d/gpu_id"
  printf 'heap_type 1\nsize_in_bytes %s\nflags 0\n' "$7" > "$d/mem_banks/0/properties"
}
mkpci() {   # sysfs bdf root-complex (pciDDDD:BB): /sys/bus/pci/devices/<bdf> -> devices/<complex>/<port>/<bdf>
  local br=${3#pci}:01.1
  mkdir -p "$1/devices/$3/$br/$2" "$1/bus/pci/devices"
  ln -sfn "../../../devices/$3/$br/$2" "$1/bus/pci/devices/$2"
}
mkdrm() {   # sysfs card bdf vram_total vram_used
  local d=$1/class/drm/card$2/device
  mkdir -p "$d"
  printf 'DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_SLOT_NAME=%s\n' "$3" > "$d/uevent"
  echo "$4" > "$d/mem_info_vram_total"
  echo "$5" > "$d/mem_info_vram_used"
}
mkclient() {   # sysfs proc-root pid comm gpu_id bytes
  mkdir -p "$1/class/kfd/kfd/proc/$3" "$2/$3"
  echo "$6" > "$1/class/kfd/kfd/proc/$3/vram_$5"
  echo "$4" > "$2/$3/comm"
}
ranks() {   # sysfs: the CPU node and the four rank cards (gfx1201 at 03/23/43/63, each on its own root complex)
  local i bdf
  mkcpu "$1"
  for i in 1 2 3 4; do
    bdf=$(printf '0000:%02x:00.0' $(( (i - 1) * 32 + 3 )))
    mknode "$1" "$i" 120001 "0a0000000000000$i" "$bdf" $((10000 + i)) 17163091968
    mkpci "$1" "$bdf" "$(printf 'pci0000:%02x' $(( (i - 1) * 32 )))"
    mkdrm "$1" "$i" "$bdf" 17163091968 1000000
  done
}
MASK=GPU-0A00000000000001,GPU-0a00000000000002,GPU-0a00000000000003,GPU-0a00000000000004
XUUID=GPU-fb00000000000005 XBDF=0000:c3:00.0 XGID=50005
extra() {   # sysfs [vram_used]: the R9700-like extra card at c3 on its own root complex, 32 GiB
  mknode "$1" 5 120001 fb00000000000005 "$XBDF" "$XGID" 34359738368
  mkpci "$1" "$XBDF" pci0000:c0
  mkdrm "$1" 5 "$XBDF" 34359738368 "${2:-5368709120}"
}

# ---- 1. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
OTHER=$(echo "another library" | sha256sum | cut -d' ' -f1)
echo '{}' > "$T/prompts.json"
R=all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs
N8=L1024-Pf-N8-C8-Pmr8
C2=$R-$N8
# The loop's extra card: whisper holding 4200 MiB, 27648 MiB free of 32768.
LS=$T/lsys LP=$T/lproc
extra "$LS"
mkclient "$LS" "$LP" 111 whisper-server "$XGID" 4404019200
mkclient "$LS" "$LP" 222 radiance "$XGID" 0            # holds nothing there: not a client
mkclient "$LS" "$LP" 333 qfn 10001 9999999             # another card's
window() {   # the window's state as the script has it when the loop starts; output directory
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  AX: ${ARM_AX}|${ARM_AXV}|${ARM_AXF}|${ARM_AXVIS}|${ARM_AXSUM}"; echo "  ARGS: ${ARGS[*]}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE AUX_MIB AUX_DEADLINE_US AUX_BACKOFF AUX_HEADROOM_MIB AUX_ALLOW_CLIENTS AUX_FOREIGN_MAX_MIB
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$1; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  echo '{}' > "$OUT/kld-corpus.jsonl"
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  eval "$FUNCS"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  ROCR_VISIBLE_DEVICES=$MASK
  AUX_UUID=$XUUID AUX_BDF=$XBDF AUX_GPUID=$XGID AUX_DRM=$LS/class/drm/card5/device AUX_SYSFS=$LS AUX_PROC=$LP
  AUX_WHY=test AUX_OK=1 AUX_LIB=$LIB AUX_SUM=$SUM AXV_OK=1 AXV_SUM=$SUM
  AUX_CHECK_WHY="no aux device check in this window (test)"
}
loop() {   # the script's arm loop over $ARMS, as the other loop tests take it
  local s e
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
}
section() { awk -v a="ARM $2" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$1"; }
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
launched() {   # loop file, arm, what; then the six switches: experts device policy deadline headroom backoff forced, mask
  local sec
  sec=$(section "$1" "$2")
  if [[ -z "$sec" ]]; then bad "$3 not launched: $(grep "arm $2: " "$1" | head -3 | tr '\n' ' ')"; return; fi
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS "$4"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_DEVICE "$5"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_POLICY "$6"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_DEADLINE_US "$7"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_HEADROOM_MIB "$8"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_BACKOFF "$9"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_FORCE_LATE "${10}"
  local n v missing=""
  for v in EXPERTS DEVICE POLICY DEADLINE_US HEADROOM_MIB BACKOFF FORCE_LATE; do
    grep -q "^    RADIANCE_EXPERIMENTAL_DS41_AUX_$v=" <<< "$sec" || missing+=" $v"
  done
  if [[ -z "$missing" ]]; then ok "$3: states every aux switch"; else bad "$3: does not state$missing"; fi
  n=$(grep -c '^    ROCR_VISIBLE_DEVICES=' <<< "$sec")
  if [[ -z "${11}" ]]; then
    if (( n == 0 )); then ok "$3: the window's mask kept"; else bad "$3: sets ROCR_VISIBLE_DEVICES ($(val "$sec" ROCR_VISIBLE_DEVICES))"; fi
  else
    has "$3" "$sec" ROCR_VISIBLE_DEVICES "${11}"
  fi
}
skipped() {   # loop file, arm, the reason's words, what
  if [[ -n "$(section "$1" "$2")" ]]; then bad "$4 launched"
  elif grep -F "arm $2: " "$1" | grep -qF -- "$3"; then ok "$4 skipped: $3"
  else bad "$4: no '$3' ($(grep -F "arm $2: " "$1" | head -3 | tr '\n' ' '))"; fi
}

L1=$T/loop1.txt
ARMS="$C2 $C2-Ax $C2-Ax24000 $C2-Axv $C2-Axv20000 $C2-Ax0 $C2-Ax-Axf $C2-Axv-Axf8"
NOK=${C2/-K-/-}-Ax NOVP=${C2/-Vp-/-}-Ax NOB=${C2/-B6-/-}-Ax GREEDY=greedy-${C2#all-}-Ax M8=${C2/-M2048-/-M8-}-Ax
ARMS+=" $NOK $NOVP $NOB $GREEDY $M8 $C2-Ax-Tb $C2-Ax-Rt $C2-Ax-Ds $C2-Ax-Dk $C2-Ax-Axv $C2-Ax-Ax0 $C2-Ax512"
ARMS+=" $C2-Ax2000000 $C2-Axf $C2-Ax0-Axf $C2-Ax-Axf0"
( set +u; window "$T/out1"; loop ) > "$L1" 2>&1
VIS=$MASK,$XUUID
launched "$L1" "$C2" "no Ax part" 0 "" "" 2000 1024 64 0 ""
launched "$L1" "$C2-Ax" "Ax" 26000 "$XBDF" "" 2000 1024 64 0 "$VIS"
launched "$L1" "$C2-Ax24000" "Ax24000" 24000 "$XBDF" "" 2000 1024 64 0 "$VIS"
launched "$L1" "$C2-Axv" "Axv" 26000 "$XBDF" verify 2000 1024 64 0 "$VIS"
launched "$L1" "$C2-Axv20000" "Axv20000" 20000 "$XBDF" verify 2000 1024 64 0 "$VIS"
launched "$L1" "$C2-Ax0" "Ax0" 0 "" "" 2000 1024 64 0 "$VIS"
launched "$L1" "$C2-Ax-Axf" "Ax-Axf" 26000 "$XBDF" "" 2000 1024 0 4 "$VIS"
launched "$L1" "$C2-Axv-Axf8" "Axv-Axf8" 26000 "$XBDF" verify 2000 1024 0 8 "$VIS"
sec=$(section "$L1" "$C2-Axv")
if grep -qxF "  AX: 26000|1||$VIS|$SUM" <<< "$sec"; then ok "Axv: arm() sees ARM_AX, ARM_AXV, ARM_AXVIS and ARM_AXSUM"
else bad "Axv: arm()'s state is '$(grep '^  AX:' <<< "$sec")'"; fi
if grep -qx '111 whisper-server 4404019200' "$T/out1/$C2-Ax.aux-clients" 2>/dev/null; then ok "Ax: its clients written at the start"
else bad "Ax: $C2-Ax.aux-clients is '$(cat "$T/out1/$C2-Ax.aux-clients" 2>/dev/null)'"; fi
[[ -e "$T/out1/$C2.aux-clients" ]] && bad "no Ax part: the extra card's clients listed" || ok "no Ax part: no client list"
skipped "$L1" "$NOK" "Ax needs K" "Ax without K"
skipped "$L1" "$NOVP" "Ax needs Vp" "Ax without Vp"
skipped "$L1" "$NOB" "Ax needs a BN part" "Ax without B"
skipped "$L1" "$GREEDY" "Ax needs policy all" "Ax under greedy"
skipped "$L1" "$M8" "Ax needs M16 or more" "Ax with M8"
skipped "$L1" "$C2-Ax-Tb" "Ax with Tb" "Ax beside Tb"
skipped "$L1" "$C2-Ax-Rt" "Ax with Rt" "Ax beside Rt"
skipped "$L1" "$C2-Ax-Ds" "Ax with a Ds / Dk part" "Ax beside Ds"
skipped "$L1" "$C2-Ax-Dk" "Ax with a Ds / Dk part" "Ax beside Dk"
skipped "$L1" "$C2-Ax-Axv" "two Ax parts" "Ax twice"
skipped "$L1" "$C2-Ax-Ax0" "two Ax parts" "Ax beside Ax0"
skipped "$L1" "$C2-Ax512" "the worker's cap is 1024-1048576 MiB" "Ax512"
skipped "$L1" "$C2-Ax2000000" "the worker's cap is 1024-1048576 MiB" "Ax2000000"
skipped "$L1" "$C2-Axf" "Axf4 needs an Ax or Axv part" "Axf alone"
skipped "$L1" "$C2-Ax0-Axf" "Axf4 needs an Ax or Axv part" "Axf beside Ax0"
skipped "$L1" "$C2-Ax-Axf0" "forced lateness takes 1-1000000" "Axf0"

# The window's variables reach the arms (and an arm without an Ax part states them with the worker off).
L2=$T/loop2.txt
ARMS="$C2 $C2-Ax"
( set +u; window "$T/out2"; AUX_MIB=20000 AUX_DEADLINE_US=1500 AUX_BACKOFF=32 AUX_HEADROOM_MIB=2048; loop ) > "$L2" 2>&1
launched "$L2" "$C2" "no Ax part, window variables" 0 "" "" 1500 2048 32 0 ""
launched "$L2" "$C2-Ax" "Ax, window variables" 20000 "$XBDF" "" 1500 2048 32 0 "$VIS"

# The device checks and the Axv order.
L3=$T/loop3.txt
ARMS="$C2-Ax $C2-Axv $C2-Ax0"
( set +u; window "$T/out3"; AUX_OK=""; loop ) > "$L3" 2>&1
skipped "$L3" "$C2-Ax" "Ax refused: no aux device check in this window" "Ax without AUX_OK"
skipped "$L3" "$C2-Axv" "Ax refused: no aux device check in this window" "Axv without AUX_OK"
launched "$L3" "$C2-Ax0" "Ax0 without AUX_OK" 0 "" "" 2000 1024 64 0 "$VIS"
L4=$T/loop4.txt
ARMS="$C2-Ax $C2-Axv"
( set +u; window "$T/out4"; AUX_SUM=$OTHER; loop ) > "$L4" 2>&1
skipped "$L4" "$C2-Ax" "is not the one the aux device checks passed on" "Ax on another library"
skipped "$L4" "$C2-Axv" "is not the one the aux device checks passed on" "Axv on another library"
L5=$T/loop5.txt
( set +u; window "$T/out5"; AXV_OK=""; loop ) > "$L5" 2>&1
skipped "$L5" "$C2-Ax" "no Axv arm of this window compared the worker's rows" "Ax before an Axv arm"
launched "$L5" "$C2-Axv" "Axv before an Axv arm" 26000 "$XBDF" verify 2000 1024 64 0 "$VIS"
L6=$T/loop6.txt
( set +u; window "$T/out6"; AXV_SUM=$OTHER; loop ) > "$L6" 2>&1
skipped "$L6" "$C2-Ax" "no Axv arm of this window compared the worker's rows" "Ax after an Axv arm on another library"

# The mask and the extra card.
L7=$T/loop7.txt
ARMS="$C2 $C2-Ax $C2-Ax0"
( set +u; window "$T/out7"; unset ROCR_VISIBLE_DEVICES; loop ) > "$L7" 2>&1
launched "$L7" "$C2" "no Ax part, no mask" 0 "" "" 2000 1024 64 0 ""
skipped "$L7" "$C2-Ax" "needs the window's ROCR_VISIBLE_DEVICES to name exactly the four rank cards' UUIDs" "Ax without a mask"
skipped "$L7" "$C2-Ax0" "needs the window's ROCR_VISIBLE_DEVICES to name exactly the four rank cards' UUIDs" "Ax0 without a mask"
L8=$T/loop8.txt
( set +u; window "$T/out8"; ROCR_VISIBLE_DEVICES=$MASK,$XUUID; loop ) > "$L8" 2>&1
skipped "$L8" "$C2-Ax" "needs the window's ROCR_VISIBLE_DEVICES to name exactly the four rank cards' UUIDs" "Ax with a mask of five"
L9=$T/loop9.txt
( set +u; window "$T/out9"; AUX_UUID="" AUX_BDF="" AUX_WHY="no candidate beyond the rank cards: none"; loop ) > "$L9" 2>&1
skipped "$L9" "$C2-Ax" "no extra card (no candidate beyond the rank cards" "Ax without an extra card"
skipped "$L9" "$C2-Ax0" "no extra card (no candidate beyond the rank cards" "Ax0 without an extra card"
launched "$L9" "$C2" "no Ax part, no extra card" 0 "" "" 2000 1024 64 0 ""

# The extra card's clients and room.
LS2=$T/lsys2
cp -a "$LS" "$LS2"
mkclient "$LS2" "$LP" 444 python3 "$XGID" 104857600
L10=$T/loop10.txt
ARMS="$C2 $C2-Ax $C2-Ax0"
( set +u; window "$T/out10"; AUX_SYSFS=$LS2; loop ) > "$L10" 2>&1
skipped "$L10" "$C2-Ax" "python3 (pid 444) holds 100 MiB and is not allowed" "Ax beside a client not allowed"
skipped "$L10" "$C2-Ax0" "python3 (pid 444) holds 100 MiB and is not allowed" "Ax0 beside a client not allowed"
launched "$L10" "$C2" "no Ax part beside a client not allowed" 0 "" "" 2000 1024 64 0 ""
L11=$T/loop11.txt
( set +u; window "$T/out11"; AUX_SYSFS=$LS2 AUX_ALLOW_CLIENTS='^(whisper|python3$)'; loop ) > "$L11" 2>&1
launched "$L11" "$C2-Ax" "Ax with python3 allowed" 26000 "$XBDF" "" 2000 1024 64 0 "$VIS"
L12=$T/loop12.txt
( set +u; window "$T/out12"; AUX_FOREIGN_MAX_MIB=4000; loop ) > "$L12" 2>&1
skipped "$L12" "$C2-Ax" "its clients hold 4200 MiB, above AUX_FOREIGN_MAX_MIB=4000" "Ax beside too many foreign MiB"
LS3=$T/lsys3
cp -a "$LS" "$LS3"
echo 8589934592 > "$LS3/class/drm/card5/device/mem_info_vram_used"     # 24576 MiB free
L13=$T/loop13.txt
ARMS="$C2-Ax $C2-Ax20000 $C2-Ax0"
( set +u; window "$T/out13"; AUX_SYSFS=$LS3 AUX_DRM=$LS3/class/drm/card5/device; loop ) > "$L13" 2>&1
skipped "$L13" "$C2-Ax" "it has 24576 MiB free, under the cap 26000 + headroom 1024 + 512 MiB" "Ax26000 in 24576 MiB"
launched "$L13" "$C2-Ax20000" "Ax20000 in 24576 MiB" 20000 "$XBDF" "" 2000 1024 64 0 "$VIS"
launched "$L13" "$C2-Ax0" "Ax0 in 24576 MiB" 0 "" "" 2000 1024 64 0 "$VIS"

# ---- 2. arm()'s startup-line guards, as its heredoc and its "on:" check expand them
row_ax=$(grep -m1 -F 'Ax${ARM_AX:-}|${ARM_AX:-}|' "$W")
row_axv=$(grep -m1 -F 'Axv|${ARM_AXV:-}|' "$W")
want_src=$(grep -cF 'ax_want=$(aux_want)' "$W")
[[ "$row_ax" == Ax* ]] || bad "no Ax row in arm()'s switch table"
[[ "$row_axv" == Axv* ]] || bad "no Axv row in arm()'s switch table"
[[ "$want_src" == 1 ]] || bad "arm() does not take its 'on:' values from aux_want ($want_src)"
SERVE=$(on_line serve 2000 64 1024 26000 0)
VERIFY=$(on_line verify 2000 64 1024 26000 0)
OFFL="I aux experts: off: no candidate beyond the rank cards"
DEVL="I aux experts: device 4 (0000:c3:00.0, AMD Radeon AI PRO R9700): ..."
guard() {   # row, ARM_AX, ARM_AXV, line, want: 1 (a match) or 0, what[, ARM_AXTEN]
  local ARM_AX=$2 ARM_AXV=$3 ARM_AXTEN=${7:-} r part on pat n
  # arm()'s own patterns for the Ax / Axv rows (an AxF arm may start released), as it sets them
  eval "$(sed -n "/^  local ax_on='/,/^  fi$/p" "$W")"
  eval "r=\"$1\""
  IFS='|' read -r part on pat <<< "$r"
  n=$(grep -cE "$pat" <<< "$4")
  if [[ "$n" == "$5" ]]; then ok "guard: $6"; else bad "guard: $6 (part '$part', on '$on', pattern '$pat', $n matches)"; fi
}
guard "$row_ax" 26000 "" "$SERVE" 1 "the Ax row's pattern matches the serve line"
guard "$row_ax" 26000 1 "$VERIFY" 1 "the Ax row's pattern matches the verify line"
guard "$row_ax" 26000 "" "$OFFL" 0 "the Ax row's pattern does not take the off line (an Ax arm without the worker is flagged)"
guard "$row_ax" 26000 "" "$DEVL" 0 "the Ax row's pattern does not take the device line"
guard "$row_axv" 26000 "" "$SERVE" 0 "the Axv row's pattern does not take the serve line"
guard "$row_axv" 26000 1 "$VERIFY" 1 "the Axv row's pattern matches the verify line"
ARM_AXTEN=""; eval "$(sed -n "/^  local ax_on='/,/^  fi$/p" "$W" | sed 's/^  local /  /')"   # arm()'s patterns, unscoped
r=""; ARM_AX=""; eval "r=\"$row_ax\""; IFS='|' read -r part on pat <<< "$r"
if [[ -z "$on" ]] && grep -qE "$pat" <<< "$SERVE"; then ok "guard: an arm without Ax flags the line (the switch set for the window)"
else bad "guard: an arm without Ax does not flag the line (on '$on')"; fi
want() {   # ARM_AX ARM_AXV ARM_AXF ARM_AXDL ARM_AXBO ARM_AXHR, line, want 1 / 0, what
  local ARM_AX=$1 ARM_AXV=$2 ARM_AXF=$3 ARM_AXDL=$4 ARM_AXBO=$5 ARM_AXHR=$6 ax_want n
  ax_want=$( eval "$FUNCS"; aux_want )
  n=$(grep -cF -- "$ax_want" <<< "$7")
  if [[ "$n" == "$8" ]]; then ok "on: $9"; else bad "on: $9 ('$ax_want', $n matches)"; fi
}
want 26000 "" "" 2000 64 1024 "$SERVE" 1 "Ax's six values match its line"
want 26000 1 "" 2000 64 1024 "$VERIFY" 1 "Axv's six values match its line"
want 26000 1 "" 2000 64 1024 "$SERVE" 0 "Axv does not take a serve line"
want 24000 "" "" 2000 64 1024 "$SERVE" 0 "Ax24000 does not take a line with cap 26000"
want 26000 "" 4 2000 0 1024 "$(on_line serve 2000 0 1024 26000 4)" 1 "Ax-Axf's values (forced 4, backoff 0) match"
want 26000 "" 4 2000 0 1024 "$(on_line serve 2000 0 1024 26000 40)" 0 "forced 4 does not take a line with forced 40"
want 26000 "" "" 2000 64 1024 "$(on_line serve 1500 64 1024 26000 0)" 0 "deadline 2000 does not take a line with 1500"
want 26000 "" "" 2000 64 1024 "$(on_line serve 2000 64 2048 26000 0)" 0 "headroom 1024 does not take a line with 2048"

# ---- 3. aux_find on fabricated KFD topologies
find_case() {   # what, sysfs, mask, AUX_UUID wanted, AUX_BDF wanted, want rc, want "uuid|bdf|gpu_id|drm|vram", why words
  local got rc rest
  got=$( set +u; log() { :; }; eval "$FUNCS"
         AUX_SYSFS=$2; ROCR_VISIBLE_DEVICES=$3; AUX_UUID_WANT=$4; AUX_BDF_WANT=$5
         aux_find; echo "$?|$AUX_UUID|$AUX_BDF|$AUX_GPUID|$AUX_DRM|$AUX_VRAM|$AUX_WHY" )
  rc=${got%%|*} rest=${got#*|}
  if [[ "$rc" != "$6" ]]; then bad "aux_find: $1: returned $rc ($rest)"; return; fi
  if [[ -n "$7" && "${rest%|*}" != "$7" ]]; then bad "aux_find: $1: '${rest%|*}', want '$7'"; return; fi
  if [[ "${rest##*|}" != *"$8"* ]]; then bad "aux_find: $1: why '${rest##*|}' lacks '$8'"; return; fi
  ok "aux_find: $1"
}
S1=$T/s1; ranks "$S1"; extra "$S1"
find_case "the one candidate" "$S1" "$MASK" "" "" 0 "$XUUID|$XBDF|$XGID|$S1/class/drm/card5/device|34359738368" \
  "1 candidate(s); chose $XUUID at $XBDF (gpu_id $XGID, 32768 MiB of VRAM, root pci0000:c0, no rank card on it)"
find_case "the mask in upper case" "$S1" "${MASK^^}" "" "" 0 "$XUUID|$XBDF|$XGID|$S1/class/drm/card5/device|34359738368" "1 candidate(s)"
find_case "a mask of three" "$S1" "${MASK%,*}" "" "" 1 "" "names 3 of the KFD nodes' UUIDs, not the four rank cards"
find_case "the ranks' card as AUX_UUID" "$S1" "$MASK" "GPU-0a00000000000001" "" 1 "" "(not AUX_UUID)"
S2=$T/s2; ranks "$S2"; mknode "$S2" 6 110000 0c00000000000006 0000:e3:00.0 60006 17163091968; mkpci "$S2" 0000:e3:00.0 pci0000:e0
find_case "another ISA" "$S2" "$MASK" "" "" 1 "" "no candidate beyond the rank cards: GPU-0c00000000000006 (gfx_target_version 110000);"
S3=$T/s3; ranks "$S3"; mknode "$S3" 6 120001 0c00000000000006 0000:e3:00.0 60006 0; mkpci "$S3" 0000:e3:00.0 pci0000:e0
find_case "no VRAM" "$S3" "$MASK" "" "" 1 "" "GPU-0c00000000000006 (no VRAM);"
S4=$T/s4; ranks "$S4"; mknode "$S4" 6 120001 0c00000000000006 "" 60006 17163091968
find_case "no PCI location" "$S4" "$MASK" "" "" 1 "" "GPU-0c00000000000006 (no PCI location);"
S5=$T/s5; ranks "$S5"; extra "$S5"
mknode "$S5" 7 120001 0d00000000000007 0000:05:00.0 70007 51539607552; mkpci "$S5" 0000:05:00.0 pci0000:00
mkdrm "$S5" 7 0000:05:00.0 51539607552 0
find_case "two: the root rule over more VRAM" "$S5" "$MASK" "" "" 0 "$XUUID|$XBDF|$XGID|$S5/class/drm/card5/device|34359738368" "2 candidate(s)"
find_case "two: AUX_UUID" "$S5" "$MASK" "GPU-0D00000000000007" "" 0 "GPU-0d00000000000007|0000:05:00.0|70007|$S5/class/drm/card7/device|51539607552" \
  "shared with a rank card or unknown"
find_case "two: AUX_BDF short" "$S5" "$MASK" "" "05:00.0" 0 "GPU-0d00000000000007|0000:05:00.0|70007|$S5/class/drm/card7/device|51539607552" "1 candidate(s)"
find_case "two: AUX_BDF" "$S5" "$MASK" "" "0000:C3:00.0" 0 "$XUUID|$XBDF|$XGID|$S5/class/drm/card5/device|34359738368" "1 candidate(s)"
find_case "two: an AUX_UUID naming neither" "$S5" "$MASK" "GPU-ffffffffffffffff" "" 1 "" "(not AUX_UUID);"
S6=$T/s6; ranks "$S6"; extra "$S6"
mknode "$S6" 8 120001 0e00000000000008 0000:a3:00.0 80008 34359738368; mkpci "$S6" 0000:a3:00.0 pci0000:a0
find_case "two independent, equal VRAM: the lower BDF" "$S6" "$MASK" "" "" 0 "GPU-0e00000000000008|0000:a3:00.0|80008||34359738368" "2 candidate(s)"
S7=$T/s7; ranks "$S7"; extra "$S7"
mknode "$S7" 8 120001 0e00000000000008 0000:a3:00.0 80008 17179869184; mkpci "$S7" 0000:a3:00.0 pci0000:a0
find_case "two independent: more VRAM" "$S7" "$MASK" "" "" 0 "$XUUID|$XBDF|$XGID|$S7/class/drm/card5/device|34359738368" "2 candidate(s)"
S8=$T/s8; ranks "$S8"; extra "$S8"; mknode "$S8" 4 120000 0a00000000000004 0000:63:00.0 10004 17163091968
find_case "rank cards of two ISAs" "$S8" "$MASK" "" "" 1 "" "the rank cards' gfx_target_version differ"

# ---- 4. the extra card's clients
CS=$T/csys CP=$T/cproc
extra "$CS"
mkclient "$CS" "$CP" 111 whisper-server "$XGID" 4404019200
mkclient "$CS" "$CP" 222 radiance "$XGID" 0
mkclient "$CS" "$CP" 333 qfn 10001 9999999
clients() {   # what, then a command run with the aux functions over $CS (output, and its status last)
  local what=$1; shift
  ( set +u; log() { echo "LOG: $*"; }; eval "$FUNCS"
    AUX_SYSFS=$CS AUX_PROC=$CP AUX_GPUID=$XGID AUX_DRM=$CS/class/drm/card5/device OUT=$T/cout
    mkdir -p "$OUT"
    eval "$*"; echo "rc=$?" ) 2>&1
}
got=$(clients list aux_clients)
if [[ "$got" == $'111 whisper-server 4404019200\nrc=0' ]]; then ok "aux_clients: whisper only (a zero entry and another card's skipped)"
else bad "aux_clients: '$got'"; fi
gate() {   # what, the command, want rc, words in its output ("" none)
  local got
  got=$(clients "$1" "$2")
  if [[ "$got" != *"rc=$3" ]]; then bad "aux_clients_gate: $1: $(tail -2 <<< "$got" | tr '\n' ' ')"; return; fi
  if [[ -n "$4" && "$got" != *"$4"* ]]; then bad "aux_clients_gate: $1: no '$4' in '$got'"; return; fi
  ok "aux_clients_gate: $1"
}
gate "whisper, Ax26000 in 27648 MiB" 'aux_clients_gate a1 26000' 0 "the extra card's clients at the start: 111 whisper-server 4404019200;"
gate "Ax26200 in 27648 MiB" 'aux_clients_gate a2 26200' 1 "it has 27648 MiB free, under the cap 26200 + headroom 1024 + 512 MiB"
gate "Ax26200 with headroom 512" 'AUX_HEADROOM_MIB=512; aux_clients_gate a3 26200' 0 ""
gate "Ax0 needs no room" 'echo 34359738368 > $AUX_DRM/mem_info_vram_used; aux_clients_gate a4 0; s=$?; echo 5368709120 > $AUX_DRM/mem_info_vram_used; (exit $s)' 0 ""
gate "free VRAM unknown" 'AUX_DRM=; aux_clients_gate a5 26000' 1 "its free VRAM is unknown"
gate "too many foreign MiB" 'AUX_FOREIGN_MAX_MIB=4000; aux_clients_gate a6 26000' 1 "its clients hold 4200 MiB, above AUX_FOREIGN_MAX_MIB=4000"
mkclient "$CS" "$CP" 444 python3 "$XGID" 104857600
gate "a client not allowed" 'aux_clients_gate a7 26000' 1 "python3 (pid 444) holds 100 MiB and is not allowed (AUX_ALLOW_CLIENTS=^whisper)"
gate "the client allowed" "AUX_ALLOW_CLIENTS='^(whisper|python3\$)'; aux_clients_gate a8 26000" 0 ""
rm -rf "$CS/class/kfd/kfd/proc/444" "$CP/444"
got=$(clients "after, none new" 'aux_clients_gate b1 26000 >/dev/null; aux_clients_after b1; s=$?; echo "new=[$AUX_NEW_CLIENTS]"; (exit $s)')
if [[ "$got" == *$'new=[]\nrc=0' ]]; then ok "aux_clients_after: no new client"; else bad "aux_clients_after (none new): '$got'"; fi
got=$(clients "after, one new" "aux_clients_gate b2 26000 >/dev/null; mkdir -p $CS/class/kfd/kfd/proc/555 $CP/555; echo 1048576000 > $CS/class/kfd/kfd/proc/555/vram_$XGID; echo boltzgen > $CP/555/comm; aux_clients_after b2; s=\$?; echo \"new=[\$AUX_NEW_CLIENTS]\"; (exit \$s)")
if [[ "$got" == *$'new=[ boltzgen (pid 555, 1000 MiB)]\nrc=1' ]]; then ok "aux_clients_after: a client arriving is new"
else bad "aux_clients_after (one new): '$got'"; fi
rm -rf "$CS/class/kfd/kfd/proc/555" "$CP/555"
# The stopped server's own KFD entry (its pid given) is not a new client, nor is an entry whose process is gone: the
# driver drops both only after the last reference goes. Both appear after the start list (the server's whole life
# is the arm) and are still in the after-list as read.
during="mkdir -p $CS/class/kfd/kfd/proc/556 $CP/556 $CS/class/kfd/kfd/proc/557; echo 2147483648 > $CS/class/kfd/kfd/proc/556/vram_$XGID; echo radiance > $CP/556/comm; echo 1048576 > $CS/class/kfd/kfd/proc/557/vram_$XGID"
got=$(clients "after, the server's entry and a gone process" "aux_clients_gate b3 26000 >/dev/null; $during; AUX_KFD_WAIT_DS=0; aux_clients_after b3 556; s=\$?; echo \"new=[\$AUX_NEW_CLIENTS] listed=\$(grep -c . \$OUT/b3.aux-clients-after)\"; (exit \$s)")
rm -rf "$CS/class/kfd/kfd/proc/556" "$CP/556" "$CS/class/kfd/kfd/proc/557"
if [[ "$got" == *$'new=[] listed=3\nrc=0' ]]; then ok "aux_clients_after: the stopped server's entry and a gone process's are left out"
else bad "aux_clients_after (server entry, gone process): '$got'"; fi
got=$(clients "after, the server's entry not named" "aux_clients_gate b4 26000 >/dev/null; $during; AUX_KFD_WAIT_DS=0; aux_clients_after b4 ''; s=\$?; echo \"new=[\$AUX_NEW_CLIENTS]\"; (exit \$s)")
rm -rf "$CS/class/kfd/kfd/proc/556" "$CP/556" "$CS/class/kfd/kfd/proc/557"
if [[ "$got" == *$'new=[ radiance (pid 556, 2048 MiB)]\nrc=1' ]]; then ok "aux_clients_after: a live process unnamed is new (the gone one still left out)"
else bad "aux_clients_after (server pid not given): '$got'"; fi
# The server's entry is waited for (bounded) before the list is read.
mkdir -p "$CS/class/kfd/kfd/proc/558" "$CP/558"
echo 1048576 > "$CS/class/kfd/kfd/proc/558/vram_$XGID"; echo radiance > "$CP/558/comm"
got=$(clients "after, waits for the server's entry" "aux_clients_gate b5 26000 >/dev/null; ( sleep 0.3; rm -rf $CS/class/kfd/kfd/proc/558 ) & aux_clients_after b5 558; s=\$?; wait; echo \"listed558=\$(grep -c '^558 ' \$OUT/b5.aux-clients-after)\"; (exit \$s)")
if [[ "$got" == *$'listed558=0\nrc=0' ]]; then ok "aux_clients_after: waits for the stopped server's KFD entry to go"
else bad "aux_clients_after (wait): '$got'"; fi
got=$(clients "after, the wait is bounded" "aux_clients_gate b6 26000 >/dev/null; mkdir -p $CS/class/kfd/kfd/proc/558; echo 1 > $CS/class/kfd/kfd/proc/558/vram_$XGID; AUX_KFD_WAIT_DS=3; t0=\$(date +%s%N); aux_clients_after b6 558; s=\$?; echo \"waited=\$(( (\$(date +%s%N) - t0) / 100000000 ))\"; (exit \$s)")
if [[ "$got" =~ waited=([0-9]+) ]] && (( BASH_REMATCH[1] >= 2 && BASH_REMATCH[1] <= 20 )) && [[ "$got" == *"rc=0" ]]; then
  ok "aux_clients_after: the wait for the server's entry is bounded (AUX_KFD_WAIT_DS)"
else bad "aux_clients_after (bounded wait): '$got'"; fi
rm -rf "$CS/class/kfd/kfd/proc/558" "$CP/558"

# ---- 5. the worker's exit lines
EX=$T/exit; mkdir -p "$EX"
lines() {   # file, then per rank "jobs:failures:compared:differ:forced"
  local f=$1 r=0 spec j fl c d fo; shift
  : > "$f"
  echo "I something else" >> "$f"
  for spec in "$@"; do IFS=: read -r j fl c d fo <<< "$spec"; rank_line "$r" "$j" "$fl" "$c" "$d" "$fo" >> "$f"; r=$((r + 1)); done
}
egate() {   # what, file, cap, verify, forced, want rc, words ("" none)
  local got rc
  got=$( set +u; eval "$FUNCS"; aux_exit_gate "$2" "$3" "$4" "$5"; echo "rc=$?" )
  rc=${got##*rc=}
  if [[ "$rc" != "$6" ]]; then bad "aux_exit_gate: $1: rc $rc ('$got')"; return; fi
  if [[ -n "$7" && "$got" != *"$7"* ]]; then bad "aux_exit_gate: $1: no '$7' in '$got'"; return; fi
  ok "aux_exit_gate: $1"
}
lines "$EX/serve.log" 120:0:0:0:0 118:0:0:0:0 119:0:0:0:0 121:0:0:0:0
egate "serve, every rank ran" "$EX/serve.log" 26000 "" "" 0 ""
egate "no cap: nothing read" "/nonexistent" "" "" "" 0 ""
egate "Ax0: nothing read" "/nonexistent" 0 "" "" 0 ""
lines "$EX/three.log" 120:0:0:0:0 118:0:0:0:0 119:0:0:0:0
egate "three ranks' lines" "$EX/three.log" 26000 "" "" 1 "3 of 4 ranks' 'aux experts: rank' lines"
lines "$EX/idle.log" 120:0:0:0:0 118:0:0:0:0 0:0:0:0:0 121:0:0:0:0
egate "a rank ran no job" "$EX/idle.log" 26000 "" "" 1 "3 of 4 ranks ran layer jobs on the extra card"
lines "$EX/fail2.log" 120:0:0:0:0 118:2:0:0:0 119:0:0:0:0 121:0:0:0:0
egate "a rank with launch failures" "$EX/fail2.log" 26000 "" "" 1 "1 rank(s) had launch failures"
lines "$EX/fail10.log" 120:0:0:0:0 118:0:0:0:0 119:0:0:0:0 121:10:0:0:0
egate "10 launch failures are not 0" "$EX/fail10.log" 26000 "" "" 1 "1 rank(s) had launch failures"
lines "$EX/verify.log" 120:0:500:0:0 118:0:480:0:0 119:0:490:0:0 121:0:510:0:0
egate "verify, none differ" "$EX/verify.log" 26000 1 "" 0 ""
egate "serve accepts a verify log" "$EX/verify.log" 26000 "" "" 0 ""
lines "$EX/differ.log" 120:0:500:1:0 118:0:480:0:0 119:0:490:0:0 121:0:510:0:0
egate "verify, a row differs" "$EX/differ.log" 26000 1 "" 1 "AUX VERIFY FAIL: 3 of 4 ranks compared rows with 0 differing"
lines "$EX/differ10.log" 120:0:500:10:0 118:0:480:0:0 119:0:490:0:0 121:0:510:0:0
egate "verify, 10 rows differ" "$EX/differ10.log" 26000 1 "" 1 "AUX VERIFY FAIL"
lines "$EX/nocmp.log" 120:0:500:0:0 118:0:480:0:0 119:0:0:0:0 121:0:510:0:0
egate "verify, a rank compared none" "$EX/nocmp.log" 26000 1 "" 1 "AUX VERIFY FAIL"
egate "forced, none forced" "$EX/serve.log" 26000 "" 4 1 "forced lateness every 4, but no rank forced a deadline miss"
lines "$EX/forced.log" 120:0:0:0:0 118:0:0:0:3 119:0:0:0:0 121:0:0:0:0
egate "forced, a rank forced misses" "$EX/forced.log" 26000 "" 4 0 ""
arm_exit() {   # name, ARM_AX, ARM_AXV, ARM_AXF, server log source, aux-vram lines ("" none): output and AXV state
  ( set +u; log() { echo "LOG: $*"; }; eval "$FUNCS"
    OUT=$T/aout; mkdir -p "$OUT"
    AUX_SYSFS=$CS AUX_PROC=$CP AUX_GPUID=$XGID AUX_DRM=$CS/class/drm/card5/device
    aux_clients > "$OUT/$1.aux-clients"
    echo '{}' > "$OUT/$1.json"
    cp "$5" "$OUT/$1.server.log"
    [[ -n "$6" ]] && printf '%s\n' "$6" > "$OUT/$1.aux-vram"
    ARM_AXVIS=$MASK,$XUUID ARM_AX=$2 ARM_AXV=$3 ARM_AXF=$4 ARM_AXSUM=$SUM AXV_OK="" AXV_SUM=""
    eval "${7:-:}"   # during the arm (a client arriving)
    aux_arm_exit "$1" "${8:-}"; echo "rc=$?"
    echo "AXV=$AXV_OK|$AXV_SUM"
    ls "$OUT" | grep -E "^$1\.(json|unmeasured\.json)$" ) 2>&1
}
newcl="mkdir -p $CS/class/kfd/kfd/proc/555 $CP/555; echo 1048576000 > $CS/class/kfd/kfd/proc/555/vram_$XGID; echo boltzgen > $CP/555/comm"
got=$(arm_exit c1 26000 "" "" "$EX/serve.log" "" "$newcl")
rm -rf "$CS/class/kfd/kfd/proc/555" "$CP/555"
if [[ "$got" == *"arm c1: the extra card gained clients during the arm: boltzgen (pid 555, 1000 MiB); not measured (results kept as c1.unmeasured.json)"* &&
      "$got" == *"rc=1"* && "$got" == *"c1.unmeasured.json"* ]]; then ok "aux_arm_exit: a client arriving sets the results aside"
else bad "aux_arm_exit (a new client): '$got'"; fi
got=$(arm_exit c2 26000 1 "" "$EX/differ.log" "" "$newcl")
rm -rf "$CS/class/kfd/kfd/proc/555" "$CP/555"
if [[ "$got" == *"gained clients during the arm: boltzgen (pid 555, 1000 MiB); AUX VERIFY FAIL: 3 of 4 ranks"* && "$got" == *"AXV=|"* ]]; then
  ok "aux_arm_exit: a new client and a differing row, both said"
else bad "aux_arm_exit (a new client and a differing row): '$got'"; fi
srv="mkdir -p $CS/class/kfd/kfd/proc/559 $CP/559; echo 2147483648 > $CS/class/kfd/kfd/proc/559/vram_$XGID; echo radiance > $CP/559/comm; AUX_KFD_WAIT_DS=0"
got=$(arm_exit c3 26000 "" "" "$EX/serve.log" "" "$srv" 559)
rm -rf "$CS/class/kfd/kfd/proc/559" "$CP/559"
if [[ "$got" == *"rc=0"* && "$got" == *"c3.json"* && "$got" != *"gained clients"* ]]; then
  ok "aux_arm_exit: the stopped server's own KFD entry is not a new client"
else bad "aux_arm_exit (the server's entry): '$got'"; fi
got=$(arm_exit m1 26000 "" "" "$EX/serve.log" "12:00:00 34359738368 5368709120
12:00:02 34359738368 33822867456
12:00:04 34359738368 6442450944")
if [[ "$got" == *"at least 512 MiB"* && "$got" == *"AUX MARGIN LOW: the extra card's free VRAM went down to 512 MiB"* &&
      "$got" == *"rc=0"* && "$got" == *"m1.json"* ]]; then ok "aux_arm_exit: a low margin is flagged, the arm still measured"
else bad "aux_arm_exit (margin): '$got'"; fi
got=$(arm_exit m2 26000 "" "" "$EX/serve.log" "12:00:00 34359738368 5368709120")
if [[ "$got" == *"at least 27648 MiB"* && "$got" != *"AUX MARGIN LOW"* ]]; then ok "aux_arm_exit: an ample margin is not flagged"
else bad "aux_arm_exit (ample margin): '$got'"; fi
got=$(arm_exit v1 26000 1 "" "$EX/verify.log" "")
if [[ "$got" == *"rc=0"* && "$got" == *"AXV=1|$SUM"* && "$got" == *"Ax arms on this library may run"* ]]; then
  ok "aux_arm_exit: a measured Axv arm sets AXV_OK on its library"
else bad "aux_arm_exit (Axv): '$got'"; fi
got=$(arm_exit v2 26000 1 "" "$EX/differ.log" "")
if [[ "$got" == *"rc=1"* && "$got" == *"AXV=|"* && "$got" == *"v2.unmeasured.json"* && "$got" == *"AUX VERIFY FAIL"* ]]; then
  ok "aux_arm_exit: a differing row sets the Axv arm aside, no AXV_OK"
else bad "aux_arm_exit (Axv differs): '$got'"; fi
got=$(arm_exit f1 26000 "" 4 "$EX/serve.log" "")
if [[ "$got" == *"rc=1"* && "$got" == *"f1.unmeasured.json"* ]]; then ok "aux_arm_exit: AxfN without a forced miss is set aside"
else bad "aux_arm_exit (Axf): '$got'"; fi
got=$(arm_exit z1 0 "" "" "/dev/null" "")
if [[ "$got" == *"rc=0"* && "$got" == *"z1.json"* ]]; then ok "aux_arm_exit: Ax0 reads no worker lines"
else bad "aux_arm_exit (Ax0): '$got'"; fi

# ---- 6. the DEVICE_CHECKS=aux section
s=$(grep -n '^if \[\[ " \${DEVICE_CHECKS:-1} " == \*" aux "\* \]\]; then' "$W" | head -1 | cut -d: -f1)
e=$(awk -v s="${s:-0}" 'NR > s && /^fi$/ { print NR; exit }' "$W")
SECTION=$( [[ -n "$s" && -n "$e" ]] && sed -n "${s},${e}p" "$W")
VPASS=$(sed -n '/^vpack_pass() {/,/^}/p' "$W")
if [[ -z "$SECTION" || -z "$VPASS" ]]; then bad "no DEVICE_CHECKS=aux section or vpack_pass in $W"
else
  B=$T/bin; mkdir -p "$B"
  cat > "$B/docker" <<'EOF'
#!/usr/bin/env bash
echo "${FAKE_QFN:-exited}"
EOF
  cat > "$B/auxcheck" <<'EOF'
#!/usr/bin/env bash
# auxcheck LIB MODE [OUT]: ds41_mxfp4_device_check's aux / swiglu / auxrt modes, stubbed
echo "$2 card ${ROCR_VISIBLE_DEVICES:-}/${HIP_VISIBLE_DEVICES:-} live tail ${RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL:-unset}" >> "$FAKE_RUNS"
case $2 in
  aux)
    o=same; [[ ${ROCR_VISIBLE_DEVICES:-} == "$FAKE_XUUID" && -n ${FAKE_AUX_DIFFER:-} ]] && o=differs
    echo "$o" > "$3"
    [[ -n ${FAKE_AUX_FAIL:-} ]] && { echo "FAIL: aux: a row differs"; exit 1; }
    echo "PASS: aux: compacted gemv_vp and gemv rows equal the full launch's at M1/M4/M8/M16 (48 cases)" ;;
  swiglu) cp "$FAKE_SILU" "$3"; echo "PASS: host swiglu with the captured table equals libr4d's" ;;
  auxtall)   # auxtall OUT MMAX (AxrN)
    o=same; [[ ${ROCR_VISIBLE_DEVICES:-} == "$FAKE_XUUID" && -n ${FAKE_TALL_DIFFER:-} ]] && o=differs
    echo "$o" > "$3"
    echo "auxtall rows $4" >> "$FAKE_RUNS"
    [[ -n ${FAKE_TALL_FAIL:-} ]] && { echo "FAIL: auxtall: a row differs"; exit 1; }
    echo "PASS: auxtall: the P1 twin's compacted rows equal its full launch's and the gemv_vp and gemv chains' in passes of 16 at M17..$4 (60 cases)" ;;
  auxrt) echo "auxrt M1 units 1: median 301.0 us, p90 340.0 us, max 900.0 us"
         [[ -n ${FAKE_TOUCH_LIB:-} ]] && echo changed >> "$1" ;;
  *) exit 2 ;;
esac
exit 0
EOF
  cat > "$B/vpack" <<'EOF'
#!/usr/bin/env bash
echo "vpack $2 $3 ${4:-} card ${ROCR_VISIBLE_DEVICES:-}" >> "$FAKE_RUNS"
echo "PASS: gemv_vp over packed experts is bit-identical (stub)"
EOF
  chmod +x "$B/docker" "$B/auxcheck" "$B/vpack"
  echo "table" > "$T/silu.bin"; echo "another table" > "$T/silu2.bin"
  dcheck() {   # what; then VAR=val ... for the window: prints the section's outcome
    local what=$1; shift
    ( set +u; log() { echo "LOG: $*"; }; eval "$FUNCS"; eval "$VPASS"
      OUT=$T/dc-$what; mkdir -p "$OUT"; cp "$LIB" "$OUT/lib.so"
      DEVICE_CHECKS=aux AUX_CHECK=$B/auxcheck VPACK_CHECK=$B/vpack AUX_LIBR4D=$OUT/lib.so DOCKER=("$B/docker") NAME=qfn
      SILU=$T/silu.bin RHOME=$H AUX_UUID=$XUUID AUX_BDF=$XBDF AUX_WHY=test
      AUX_SYSFS=$CS AUX_PROC=$CP AUX_GPUID=$XGID AUX_DRM=$CS/class/drm/card5/device   # whisper, 27648 MiB free
      export FAKE_RUNS=$OUT/runs FAKE_XUUID=$XUUID FAKE_SILU=$T/silu.bin ROCR_VISIBLE_DEVICES=$MASK
      unset RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL TALL_CHECK
      AUX_OK="" AUX_LIB="" AUX_SUM="" AUX_CHECK_WHY=""
      for kv in "$@"; do
        case $kv in FAKE_*|RADIANCE_*) export "${kv?}" ;; *) eval "${kv%%=*}=\"\${kv#*=}\"" ;; esac
      done
      eval "$SECTION"
      echo "AUX=$AUX_OK|$AUX_SUM|$AUX_LIB"
      echo "TALL=${AUX_TALL_OK:-}|${AUX_TALL_SUM:-}|${AUX_TALL_ROWS:-}|${AUX_TALL_WHY:-}"
      echo "WHY=$AUX_CHECK_WHY"
      cat "$OUT/runs" 2>/dev/null ) 2>&1
  }
  LSUM=$SUM
  got=$(dcheck pass)
  if [[ "$got" == *"AUX=1|$LSUM|$T/dc-pass/lib.so"* && "$got" == *"aux card $MASK/0 live tail 0"* &&
        "$got" == *"aux card $XUUID/0 live tail 0"* && "$got" == *"swiglu card $XUUID/0"* && "$got" == *"vpack check 48  card $XUUID"* &&
        "$got" == *"auxrt card $XUUID/0"* && "$got" != *"vpack check 48 tail"* ]]; then
    ok "aux checks: pass on both cards, without the live tail (its vpack tail check not run)"
  else bad "aux checks (pass): '$got'"; fi
  got=$(dcheck tail RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1)
  if [[ "$got" == *"AUX=1|"* && "$got" == *"aux card $MASK/0 live tail 1"* && "$got" == *"aux card $XUUID/0 live tail 1"* &&
        "$got" == *"vpack check 48 tail card $XUUID"* ]]; then ok "aux checks: the window's live tail reaches every mode, with check 48 tail"
  else bad "aux checks (live tail): '$got'"; fi
  got=$(dcheck differ FAKE_AUX_DIFFER=1)
  if [[ "$got" == *"AUX=||"* && "$got" == *"the extra card's aux outputs differ from card 0's"* ]]; then ok "aux checks: outputs differing between the cards refuse"
  else bad "aux checks (outputs differ): '$got'"; fi
  got=$(dcheck silu FAKE_SILU="$T/silu2.bin")
  if [[ "$got" == *"AUX=||"* && "$got" == *"the extra card's silu table differs from SILU_TABLE"* ]]; then ok "aux checks: another silu table refuses"
  else bad "aux checks (silu): '$got'"; fi
  got=$(dcheck fail FAKE_AUX_FAIL=1)
  if [[ "$got" == *"AUX=||"* && "$got" == *"aux-xt exited non-zero"* && "$got" == *"aux-ext printed FAIL"* ]]; then ok "aux checks: a FAIL refuses"
  else bad "aux checks (FAIL): '$got'"; fi
  got=$(dcheck touch FAKE_TOUCH_LIB=1)
  if [[ "$got" == *"AUX=||"* && "$got" == *"the library changed during the checks"* ]]; then ok "aux checks: a library changed during them refuses"
  else bad "aux checks (library changed): '$got'"; fi
  got=$(dcheck running FAKE_QFN=running)
  if [[ "$got" == *"AUX=||"* && "$got" == *"refused: QFN (qfn) is not stopped"* && "$got" != *"aux card"* ]]; then
    ok "aux checks: refused while QFN is not stopped (no check ran)"
  else bad "aux checks (QFN running): '$got'"; fi
  got=$(dcheck nocard AUX_UUID=)
  if [[ "$got" == *"AUX=||"* && "$got" == *"WHY=no extra card (test)"* && "$got" != *"aux card"* ]]; then ok "aux checks: no extra card, no check"
  else bad "aux checks (no card): '$got'"; fi
  got=$(dcheck nobin AUX_CHECK=/nonexistent/check)
  if [[ "$got" == *"AUX=||"* && "$got" == *"is not an executable"* ]]; then ok "aux checks: no check binary"
  else bad "aux checks (no binary): '$got'"; fi
  # The shared extra card: the checks start only beside the clients the arms allow, with room for their allocations.
  got=$(dcheck pass2)
  if [[ "$got" == *"AUX=1|"* && "$got" == *"aux device checks: the extra card's clients: 111 whisper-server 4404019200;"* ]]; then
    ok "aux checks: whisper alone and room enough: they run, the clients logged first"
  else bad "aux checks (the clients' line): '$got'"; fi
  mkdir -p "$CS/class/kfd/kfd/proc/777" "$CP/777"
  echo 104857600 > "$CS/class/kfd/kfd/proc/777/vram_$XGID"; echo python3 > "$CP/777/comm"
  got=$(dcheck client)
  if [[ "$got" == *"AUX=||"* &&
        "$got" == *"WHY=refused: the extra card is not free for its checks (its clients: 111 whisper-server 4404019200;777 python3 104857600;): python3 (pid 777) holds 100 MiB and is not allowed"* &&
        "$got" != *"aux card"* ]]; then ok "aux checks: a client not allowed refuses them, with the clients listed (no check ran)"
  else bad "aux checks (a client not allowed): '$got'"; fi
  got=$(dcheck allowed "AUX_ALLOW_CLIENTS=^(whisper|python3\$)")
  if [[ "$got" == *"AUX=1|"* && "$got" == *"aux card $XUUID/0"* ]]; then ok "aux checks: the client allowed, they run"
  else bad "aux checks (the client allowed): '$got'"; fi
  rm -rf "$CS/class/kfd/kfd/proc/777" "$CP/777"
  got=$(dcheck lowfree AUX_CHECK_FREE_MIB=30000)
  if [[ "$got" == *"AUX=||"* && "$got" == *"it has 27648 MiB free, under AUX_CHECK_FREE_MIB=30000"* && "$got" != *"aux card"* ]]; then
    ok "aux checks: too little free VRAM on the extra card refuses them (no check ran)"
  else bad "aux checks (too little free VRAM): '$got'"; fi
  got=$(dcheck foreign AUX_FOREIGN_MAX_MIB=4000)
  if [[ "$got" == *"AUX=||"* && "$got" == *"its clients hold 4200 MiB, above AUX_FOREIGN_MAX_MIB=4000"* && "$got" != *"aux card"* ]]; then
    ok "aux checks: too many foreign MiB on the extra card refuse them"
  else bad "aux checks (too many foreign MiB): '$got'"; fi
fi

# ---- 7. the aux_any line before the loop: not the loop the other tests find, and it sees the Ax parts
first=$(grep -n '^for a in \$ARMS; do' "$W" | head -1)
if [[ "$first" == *":for a in \$ARMS; do" ]] && sed -n "$(( ${first%%:*} + 1 ))p" "$W" | grep -q 'CANCEL'; then
  ok "the first 'for a in \$ARMS; do' is the arm loop"
else bad "the first 'for a in \$ARMS; do' is '$first'"; fi
anyl=$(grep -m1 -F 'for aux_a in $ARMS; do' "$W")
for spec in "$C2-Ax|1" "$C2-Axv24000|1" "$C2-Ax0|1" "$C2-Axf|" "$C2|"; do
  IFS='|' read -r a want <<< "$spec"
  got=$( ARMS="$C2 $a"; aux_any=""; eval "$anyl"; echo "$aux_any" )
  if [[ "$got" == "$want" ]]; then ok "aux_any for ${a#$C2-}: '${want}'"; else bad "aux_any for ${a#$C2-}: '$got', want '$want'"; fi
done

# ---- 8. a Kld arm's aux gates (kld_finish -> aux_kld_check), every action stubbed: the records are compared only
# when the worker demonstrably ran (its "on:" values, every rank's jobs, no launch failure, the clients unchanged)
KF=$(sed -n '/^kld_finish() {/,/^}/p' "$W")
[[ -n "$KF" ]] || bad "no kld_finish in $W"
kld() {   # name, ARM_AX, ARM_AXV, ARM_AXF, server log, KLD_REF ("" none), a command during the run ("" none)
  ( set +u; log() { echo "LOG: $*"; }; eval "$FUNCS"; eval "$KF"
    kill() { return 1; }; wait() { return 0; }; python3() { echo "COMPARED $*"; return 0; }
    n128_server_gate() { return 0; }; xt_check() { return 0; }; lcp_check() { return 0; }
    vp_server_gate() { return 0; }; qsp_gate() { return 0; }
    OUT=$T/kout; mkdir -p "$OUT"; HERE=$T; ARGS=()
    AUX_SYSFS=$CS AUX_PROC=$CP AUX_GPUID=$XGID AUX_DRM=$CS/class/drm/card5/device AUX_KFD_WAIT_DS=0
    aux_clients > "$OUT/$1.aux-clients"
    cp "$5" "$OUT/$1.server.log"
    ARM_AX=$2 ARM_AXV=$3 ARM_AXF=$4 ARM_AXDL=2000 ARM_AXHR=1024 ARM_AXBO=64 ARM_AXVIS=""
    [[ -n $4 ]] && ARM_AXBO=0
    [[ -n $2 ]] && ARM_AXVIS=$MASK,$XUUID
    ARM_XT= ARM_LC= ARM_N128= ARM_AR_ROWS= ARM_AG_ROWS= ARM_VP= ARM_QSP=
    KLD_REF=$6 SERVER_PID=4242
    eval "${7:-:}"
    kld_finish "$1"; echo "rc=$?" ) 2>&1
}
KL=$T/kld; mkdir -p "$KL"
{ on_line serve 2000 64 1024 26000 0; for r in 0 1 2 3; do rank_line $r 120 0 0 0 0; done; } > "$KL/served.log"
{ on_line serve 2000 64 1024 26000 0; for r in 0 1 2 3; do rank_line $r 0 0 0 0 0; done; } > "$KL/idle.log"
echo "I aux experts: off: no usable device beyond the ranks: too little free VRAM" > "$KL/off.log"
{ on_line serve 2000 64 1024 24000 0; for r in 0 1 2 3; do rank_line $r 120 0 0 0 0; done; } > "$KL/cap.log"
{ on_line verify 2000 64 1024 26000 0; rank_line 0 120 0 500 1 0; for r in 1 2 3; do rank_line $r 120 0 500 0 0; done; } > "$KL/differ.log"
{ on_line verify 2000 64 1024 26000 0; for r in 0 1 2 3; do rank_line $r 120 0 500 0 0; done; } > "$KL/verify.log"
{ on_line serve 2000 64 1024 26000 0; rank_line 0 120 3 0 0 0; for r in 1 2 3; do rank_line $r 120 0 0 0 0; done; } > "$KL/fail.log"
echo "I device: AQL backend, 5 device(s)" > "$KL/ax0.log"
{ echo "I device: AQL backend, 5 device(s)"; echo "I aux experts: off: the switch read as 0"; } > "$KL/ax0aux.log"
echo "I device: AQL backend, 4 device(s)" > "$KL/plain.log"
kcase() {   # what, want rc, words in the output ("" none), compared (1 / ""), then kld's arguments
  local what=$1 want=$2 words=$3 cmp=$4 got; shift 4
  got=$(kld "$@")
  if [[ "$got" != *"rc=$want" ]]; then bad "kld_finish: $what: '$got'"; return; fi
  if [[ -n "$words" && "$got" != *"$words"* ]]; then bad "kld_finish: $what: no '$words' in '$got'"; return; fi
  if [[ -n "$cmp" && "$got" != *"COMPARED "*"--kld-compare $T/ref $T/kout/kld-$1"* ]] || [[ -z "$cmp" && "$got" == *"COMPARED"* ]]; then
    bad "kld_finish: $what: compared '$cmp' expected ('$got')"; return
  fi
  ok "kld_finish: $what"
}
kcase "Ax, the worker served on every rank: compared" 0 "" 1 k1 26000 "" "" "$KL/served.log" "$T/ref"
kcase "Ax, no rank ran a layer job: not compared" 1 "arm k2: 0 of 4 ranks ran layer jobs on the extra card; Kld not measured" "" \
  k2 26000 "" "" "$KL/idle.log" "$T/ref"
kcase "Ax, the worker off: not compared, the reason said" 1 \
  "arm k3: Ax, but the worker did not start (I aux experts: off: no usable device beyond the ranks: too little free VRAM); Kld not measured" "" \
  k3 26000 "" "" "$KL/off.log" "$T/ref"
kcase "Ax, the 'on:' line with another cap: not compared" 1 "does not state 'passes of up to 16 rows, policy serve, deadline 2000 us, backoff 64, headroom 1024 MiB, cap 26000 MiB, forced lateness 0 '" "" \
  k4 26000 "" "" "$KL/cap.log" "$T/ref"
kcase "Axv, a row differs: not compared" 1 "AUX VERIFY FAIL: 3 of 4 ranks compared rows with 0 differing; Kld not measured" "" \
  k5 26000 1 "" "$KL/differ.log" "$T/ref"
kcase "Axv, none differ: compared" 0 "" 1 k6 26000 1 "" "$KL/verify.log" "$T/ref"
kcase "Ax, launch failures: not compared" 1 "1 rank(s) had launch failures; Kld not measured" "" k7 26000 "" "" "$KL/fail.log" "$T/ref"
kcase "Ax, a client arriving during the run: not compared" 1 \
  "arm k8: the extra card gained clients during the arm: boltzgen (pid 555, 1000 MiB); Kld not measured" "" \
  k8 26000 "" "" "$KL/served.log" "$T/ref" "$newcl"
rm -rf "$CS/class/kfd/kfd/proc/555" "$CP/555"
kcase "Ax, the window's first Kld arm: never the reference" 1 "an Xt, Lc, Fn128, Ar, Ag or Ax arm cannot be the window's kld reference" "" \
  k9 26000 "" "" "$KL/served.log" ""
kcase "Ax0, five devices and no aux line: compared" 0 "" 1 k10 0 "" "" "$KL/ax0.log" "$T/ref"
kcase "Ax0 with an aux line: not compared" 1 "Ax0, but the server did not see five devices or logged an aux line; Kld not measured" "" \
  k11 0 "" "" "$KL/ax0aux.log" "$T/ref"
kcase "no Ax part, the worker on: not compared" 1 "no Ax part, but the worker ran (the switch set for the window?); Kld not measured" "" \
  k12 "" "" "" "$KL/served.log" "$T/ref"
kcase "no Ax part: compared" 0 "" 1 k13 "" "" "" "$KL/plain.log" "$T/ref"
kcase "no Ax part, the first Kld arm: the reference" 0 "the window's kld reference" "" k14 "" "" "" "$KL/plain.log" ""

# ---- 9. the arm loop: an Ax / Axv part with Kld only in passes of at most 16 rows (KL mode is prefill only)
R16=all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Rb-Vp
L14=$T/loop14.txt
ARMS="$R-Ax-Kld $R16-Ax-Kld $R16-Axv-Kld $R16-BT32-Ax-Kld $R-Ax0-Kld $R16-Kld"
( set +u; window "$T/out14"; loop ) > "$L14" 2>&1
skipped "$L14" "$R-Ax-Kld" "Ax with Kld needs passes of at most 16 rows, not Lc2048's" "Ax-Kld beside Lc2048"
skipped "$L14" "$R16-BT32-Ax-Kld" "Ax with Kld needs passes of at most 16 rows, not BT32's" "Ax-Kld beside BT32"
launched "$L14" "$R16-Ax-Kld" "Ax-Kld on the 16-token base" 26000 "$XBDF" "" 2000 1024 64 0 "$VIS"
launched "$L14" "$R16-Axv-Kld" "Axv-Kld on the 16-token base" 26000 "$XBDF" verify 2000 1024 64 0 "$VIS"
launched "$L14" "$R-Ax0-Kld" "Ax0-Kld beside Lc2048 (the worker off)" 0 "" "" 2000 1024 64 0 "$VIS"
launched "$L14" "$R16-Kld" "the 16-token Kld reference" 0 "" "" 2000 1024 64 0 ""
if grep -qF -- "--kld-record $T/out14/kld-$R16-Ax-Kld" "$L14"; then ok "Ax-Kld on the 16-token base: records the corpus"
else bad "Ax-Kld on the 16-token base: no --kld-record"; fi

# ---- 10. arm(): the worker's startup lines (an off line's reason) reach window.log before the switch table can
# return, and the stopped server's pid goes to the exit gates (its KFD entry may outlive the reap)
ARMSRC=$(sed -n '/^arm() {/,/^}/p' "$W")
g=$(grep -nF "grep -E 'aux experts: (off|device|on): '" <<< "$ARMSRC" | head -1 | cut -d: -f1)
r=$(grep -nF 'if [[ -n "$sw_bad" ]]; then' <<< "$ARMSRC" | head -1 | cut -d: -f1)
if [[ -n "$g" && -n "$r" ]] && (( g < r )); then ok "arm(): the aux startup lines are logged before the switch table's return"
else bad "arm(): the aux startup grep (line ${g:-none}) is not before the switch table's return (line ${r:-none})"; fi
seq=$(grep -A2 -F 'local spid=$SERVER_PID' <<< "$ARMSRC" | sed 's/^ *//')
if [[ "$seq" == *$'\nstop_server\naux_arm_exit "$name" "$spid"'* ]]; then
  ok "arm(): the server's pid is kept across stop_server for aux_arm_exit"
else bad "arm(): aux_arm_exit does not get the stopped server's pid"; fi

# ---- 11. card sharing and the online set (work/aux-dyn/SPEC.md §8): AxS / AxSx / AxO / AxON / AxOd / AxT / AxF
# The engine's format strings the guards and gates key on.
SHFMT=$(cfmt "$SRC/core/place/aux_control.cpp" 'clients from %lld MiB' fmt)
OLFMT=$(cfmt "$SRC/core/place/aux_control.cpp" 'swaps a token, at most' fmt)
OSFMT=$(cfmt "$SRC/core/place/aux_control.cpp" 'decisions, %llu swaps' fmt)
SSFMT=$(cfmt "$SRC/core/place/aux_control.cpp" 'releases (%llu foreign' fmt)
RLFMT=$(cfmt "$SRC/core/engine_aux.cpp" 'nothing loaded on device' RAD_INFO)
[[ "$SHFMT" == "%spolicy %s, clients from"* ]] && ok "fmt: the sharing startup line" || bad "fmt: no sharing startup format ('$SHFMT')"
[[ "$OLFMT" == "%s%.2f swaps a token, at most"* ]] && ok "fmt: the online-set startup line" || bad "fmt: no online-set format ('$OLFMT')"
[[ "$OSFMT" == *"read back (%llu differ)"* && "$OSFMT" == *"%s%s" ]] && ok "fmt: the online shutdown line" || bad "fmt: the online shutdown format ('$OSFMT')"
[[ "$SSFMT" == "%s%llu releases ("* ]] && ok "fmt: the sharing shutdown line" || bad "fmt: the sharing shutdown format ('$SSFMT')"
[[ "$RLFMT" == *"forced lateness %lld (0 off)%s" ]] && ok "fmt: the released-at-start line states the six values" || bad "fmt: '$RLFMT'"
SHL=$(printf "I $SHFMT\n" "aux experts: sharing: " "allow (^whisper, foreign at most 6144 MiB)" 1 1024 100 2 60 3000 \
      "/x/a.aux-ctl (holders /x/a.aux-ctl.d, state /x/a.aux-ctl.state)" 50005)
OLL=$(printf "I $OLFMT\n" "aux experts: online set: " 2.00 16 1 512 256 100 16 1497 1513 3000 "every 16th swap" "")
RLL=$(printf "I $RLFMT\n" "aux experts: released at start: " "free 15000 MiB, under cap 26000 + headroom 1024 + 512 MiB" 4 1513 4000 1.2 16 serve 2000 64 1024 26000 0)
on_sh() {   # swaps, read-back differ, the tail (dry / stopped)
  printf "I $OSFMT\n" "aux experts: online: " 300 "$1" 1.2 2.1 3.0 140.0 260.0 0 12 0 "$1" "$2" 2 1497 26.2 0.71 0.98 0.52 "$1" "$3" ""
}
# The switch table's new rows and the Ax / Axv rows of an AxF arm (released at start).
row_axs=$(grep -m1 -F 'AxS|${ARM_AXS:-}|' "$W")
row_axo=$(grep -m1 -F 'AxO|${ARM_AXO:-}|' "$W")
gdyn() {   # row, ARM_AXS, ARM_AXO_STR, line, want, what
  local ARM_AXS=$2 ARM_AXO=$3 ARM_AXO_STR=$3 r part on pat n
  eval "r=\"$1\""
  IFS='|' read -r part on pat <<< "$r"
  n=$(grep -cE "$pat" <<< "$4")
  if [[ "$n" == "$5" ]]; then ok "guard: $6"; else bad "guard: $6 (on '$on', pattern '$pat', $n matches)"; fi
}
gdyn "$row_axs" allow "" "$SHL" 1 "AxS's row takes the allow line"
gdyn "$row_axs" exclusive "" "$SHL" 0 "AxSx's row does not take an allow line"
gdyn "$row_axs" "" "" "$SHL" 1 "an arm without AxS flags a sharing line (its pattern takes any policy)"
gdyn "$row_axo" "" 2.00 "$OLL" 1 "AxO's row takes the online line at 2.00 a token"
gdyn "$row_axo" "" 4.00 "$OLL" 0 "AxO400's row does not take a line at 2.00"
gdyn "$row_axo" "" "" "$OLL" 1 "an arm without AxO flags an online line"
guard "$row_ax" 26000 "" "$RLL" 1 "an AxF arm's Ax row takes the released-at-start line" 12000
guard "$row_ax" 26000 "" "$RLL" 0 "another Ax arm's row does not take it (it must serve from its start)"
want 26000 "" "" 2000 64 1024 "$RLL" 1 "the released-at-start line states the arm's six values"
# The arm loop: launches, every new switch stated, the window's sub-switches, skips.
TEN=$T/aux_tenant_stub
printf '#!/bin/sh\nexit 0\n' > "$TEN"; chmod +x "$TEN"
L15=$T/loop15.txt
ARMS="$C2 $C2-Ax $C2-Axv-AxO $C2-Axv-AxT $C2-Ax-AxO $C2-Ax-AxO400 $C2-Ax-AxOd $C2-Ax-AxS $C2-Ax-AxSx $C2-Ax-AxT6"
ARMS+=" $C2-Ax-AxF $C2-Ax-AxF8000 $C2-AxO $C2-Ax0-AxS $C2-Ax-AxO0 $C2-Ax-AxO2000 $C2-Ax-AxT1 $C2-Ax-AxF500"
ARMS+=" $C2-Ax-AxS-AxSx $C2-Ax-AxO-AxOd"
( set +u; window "$T/out15"; AUX_SWAP_OK=1 AUX_SWAP_SUM=$SUM AXV_OK_O=1 AXV_SUM_O=$SUM AXV_OK_T=1 AXV_SUM_T=$SUM
  AUX_TENANT=$TEN AUX_SHARE_FLOOR_MIB=512 AUX_ONLINE_SPARE=32; loop ) > "$L15" 2>&1
dyn() {   # loop file, arm, what, then SHARE CONTROL(1: the arm's file) ALLOW FOREIGN REACQUIRE FLOOR ONLINE RATE DRY SPARE
  local sec v
  sec=$(section "$1" "$2")
  if [[ -z "$sec" ]]; then bad "$3 not launched: $(grep "arm $2: " "$1" | head -3 | tr '\n' ' ')"; return; fi
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE "$4"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL "$([[ -n $5 ]] && echo "$T/out15/$2.aux-ctl")"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_ALLOW "$6"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FOREIGN_MIB "$7"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_REACQUIRE_S "$8"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FLOOR_MIB "$9"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE "${10}"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_RATE "${11}"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_DRY "${12}"
  has "$3" "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_SPARE "${13}"
  local missing=""
  for v in SHARE SHARE_ALLOW SHARE_FOREIGN_MIB SHARE_MIN_MIB SHARE_FLOOR_MIB SHARE_POLL_MS SHARE_RELEASE_POLLS \
           SHARE_REACQUIRE_S SHARE_FILL_MBPS CONTROL ONLINE ONLINE_RATE ONLINE_MAX ONLINE_EVERY ONLINE_HALFLIFE \
           ONLINE_PRIOR ONLINE_MARGIN ONLINE_SPARE ONLINE_MBPS ONLINE_CHECK ONLINE_DRY; do
    grep -q "^    RADIANCE_EXPERIMENTAL_DS41_AUX_$v=" <<< "$sec" || missing+=" $v"
  done
  if [[ -z "$missing" ]]; then ok "$3: states every new aux switch"; else bad "$3: does not state$missing"; fi
}
dyn "$L15" "$C2" "no Ax part" "" "" "" "" "" "" "" "" "" ""
dyn "$L15" "$C2-Ax" "Ax alone" "" "" "" "" "" "" "" "" "" ""
dyn "$L15" "$C2-Axv-AxO" "Axv-AxO" "" "" "" "" "" "" 1 200 "" 32
dyn "$L15" "$C2-Axv-AxT" "Axv-AxT (implies AxS)" allow 1 "^whisper" 6144 0 512 "" "" "" ""
dyn "$L15" "$C2-Ax-AxO" "Ax-AxO" "" "" "" "" "" "" 1 200 "" 32
dyn "$L15" "$C2-Ax-AxO400" "Ax-AxO400" "" "" "" "" "" "" 1 400 "" 32
dyn "$L15" "$C2-Ax-AxOd" "Ax-AxOd" "" "" "" "" "" "" 1 200 1 32
dyn "$L15" "$C2-Ax-AxS" "Ax-AxS" allow 1 "^whisper" 6144 "" 512 "" "" "" ""
dyn "$L15" "$C2-Ax-AxSx" "Ax-AxSx" exclusive 1 "" "" "" 512 "" "" "" ""
dyn "$L15" "$C2-Ax-AxT6" "Ax-AxT6" allow 1 "^whisper" 6144 0 512 "" "" "" ""
dyn "$L15" "$C2-Ax-AxF" "Ax-AxF (implies AxS, reacquire 5 s)" allow 1 "^whisper" 6144 5 512 "" "" "" ""
dyn "$L15" "$C2-Ax-AxF8000" "Ax-AxF8000" allow 1 "^whisper" 6144 5 512 "" "" "" ""
skipped "$L15" "$C2-AxO" "needs an Ax or Axv part" "AxO alone"
skipped "$L15" "$C2-Ax0-AxS" "needs an Ax or Axv part" "AxS beside Ax0"
skipped "$L15" "$C2-Ax-AxO0" "the swap rate is 1-1600" "AxO0"
skipped "$L15" "$C2-Ax-AxO2000" "the swap rate is 1-1600" "AxO2000"
skipped "$L15" "$C2-Ax-AxT1" "the toggle period is 2-600 s" "AxT1"
skipped "$L15" "$C2-Ax-AxF500" "the tenant takes 1024-30000 MiB" "AxF500"
skipped "$L15" "$C2-Ax-AxS-AxSx" "two AxS parts" "AxS twice"
skipped "$L15" "$C2-Ax-AxO-AxOd" "two AxO parts" "AxO twice"
# arm() sees the parts (the toggler, the tenant, the guards read them).
sec=$(section "$L15" "$C2-Ax-AxT6")
grep -q "^    RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_REACQUIRE_S=0$" <<< "$sec" && ok "AxT: reacquire at once" || bad "AxT: reacquire not 0"
# Gating: the auxswap check, per-feature Axv, the tenant binary, Kld.
L16=$T/loop16.txt
ARMS="$C2-Axv-AxO $C2-Ax-AxO $C2-Ax-AxS"
( set +u; window "$T/out16"; AUX_TENANT=$TEN; loop ) > "$L16" 2>&1
skipped "$L16" "$C2-Axv-AxO" "no auxswap device check passed" "Axv-AxO without AUX_SWAP_OK"
skipped "$L16" "$C2-Ax-AxS" "no auxswap device check passed" "AxS without AUX_SWAP_OK"
L17=$T/loop17.txt
ARMS="$C2-Axv-AxO $C2-Ax-AxO $C2-Ax-AxT $C2-Ax-AxF $C2-Ax-AxO-Kld"
( set +u; window "$T/out17"; AUX_SWAP_OK=1 AUX_SWAP_SUM=$SUM; unset AUX_TENANT; loop ) > "$L17" 2>&1
dyn "$L17" "$C2-Axv-AxO" "Axv-AxO without AXV_OK_O (a verify arm needs none)" "" "" "" "" "" "" 1 200 "" ""
skipped "$L17" "$C2-Ax-AxO" "no Axv arm of this window with AxO" "serve AxO without AXV_OK_O"
skipped "$L17" "$C2-Ax-AxT" "no Axv arm of this window with a sharing part" "serve AxT without AXV_OK_T"
skipped "$L17" "$C2-Ax-AxF" "AxF needs AUX_TENANT" "AxF without the tenant binary"
skipped "$L17" "$C2-Ax-AxO-Kld" "AxS / AxO with Kld" "AxO beside Kld"
L18=$T/loop18.txt
ARMS="$C2-Ax-AxO"
( set +u; window "$T/out18"; AUX_SWAP_OK=1 AUX_SWAP_SUM=$SUM AXV_OK_O=1 AXV_SUM_O=$OTHER; loop ) > "$L18" 2>&1
skipped "$L18" "$C2-Ax-AxO" "no Axv arm of this window with AxO" "AXV_OK_O on another library"
# The exit gates of the dynamic parts (aux_dyn_gate) on lines in the engine's formats.
EX=$T/exitdyn
mkdir -p "$EX"
dyng() {   # log text, ARM_AXS ARM_AXO ARM_AXOD ARM_AXTOG, want rc (0 / 1), what[, words in the reason]
  local got rc
  printf '%s\n' "$1" > "$EX/s.log"
  got=$( ARM_AXS=$2 ARM_AXO=$3 ARM_AXOD=$4 ARM_AXTOG=$5; eval "$FUNCS"; aux_dyn_gate "$EX/s.log" ); rc=$?
  if [[ "$rc" == "$6" && "$got" == *"${8:-}"* ]]; then ok "dyn gate: $7"; else bad "dyn gate: $7 (rc $rc, '$got')"; fi
}
SH_OK=$(printf "I $SSFMT\n" "aux experts: sharing: " 2 0 0 0 0 2 2 0 0 50.1 20.2 0.1 1.0 12.0 30.0 15.0 40.0 0.5 3.1 9000 0)
dyng "$(on_sh 120 0 "")" "" 200 "" "" 0 "AxO: swaps, none differing"
dyng "$(on_sh 0 0 "")" "" 200 "" "" 1 "AxO: no swap" "no swap"
dyng "$(on_sh 120 1 "")" "" 200 "" "" 1 "AxO: a read-back differing" "AUX ONLINE FAIL"
dyng "$(on_sh 120 0 "; stopped: a copy on the swap queue failed")" "" 200 "" "" 1 "AxO: stopped" "stopped"
dyng "$(on_sh 120 0 "")" "" 200 1 "" 1 "AxOd: the line must say dry" "dry"
dyng "$(on_sh 120 0 "; dry (no flip)")" "" 200 1 "" 0 "AxOd: dry"
dyng "$SH_OK"$'\n'"I aux experts: released in 3.0 ms"$'\n'"I aux experts: serving again in 0.4 s"$'\n'"I aux experts: released in 2.0 ms" \
  allow "" "" 4 1 "AxT: two releases, one acquire" "1 acquires"
dyng "$SH_OK"$'\n'"I aux experts: released in 3.0 ms"$'\n'"I aux experts: serving again in 0.4 s"$'\n'"I aux experts: released in 2.0 ms"$'\n'"I aux experts: serving again in 0.5 s" \
  allow "" "" 4 0 "AxT: two of each"
dyng "$SH_OK"$'\n'"W aux experts: dead: the swap queue did not drain; the CPU path computes every row from here on" \
  allow "" "" "" 1 "AxS: a dead line" "went dead"
dyng "I aux experts: rank 0: ..." allow "" "" "" 1 "AxS: no sharing shutdown line" "no 'aux experts: sharing"
# The toggler: its timeline against a 16 s request phase (an acquire in about 1 s), then a short real run.
plan=$( eval "$FUNCS"; aux_toggle_plan 4 16 1 )
read -r tr ta <<< "$plan"
(( tr >= 2 && ta >= 2 )) && ok "toggler: AxT4 over 16 s of requests: $tr releases, $ta acquires" || bad "toggler: AxT4 over 16 s: '$plan'"
plan=$( eval "$FUNCS"; aux_toggle_plan 6 16 1 )
read -r tr ta <<< "$plan"
(( tr >= 2 && ta >= 2 )) && ok "toggler: AxT6 over 16 s: $tr releases, $ta acquires" || bad "toggler: AxT6 over 16 s: '$plan'"
plan=$( eval "$FUNCS"; aux_toggle_plan 4 16 2.5 )
[[ $plan == "4 0" ]] && ok "toggler: AxT4 with acquires of 2.5 s shows no acquire (the period must exceed twice the acquire)" || bad "toggler: '$plan'"
TG=$T/tog
mkdir -p "$TG"
( eval "$FUNCS"; aux_toggler "$TG/ctl" 0.4 "$TG/log" ) & tp=$!
sleep 1.1; kill "$tp" 2>/dev/null; wait "$tp" 2>/dev/null
nw=$(wc -l < "$TG/log" 2>/dev/null || echo 0)
if (( nw >= 4 )) && [[ "$(awk '{ print $2 }' "$TG/log" | head -2 | tr '\n' ' ')" == "off auto " ]] &&
   [[ "$(cat "$TG/ctl")" =~ ^(off|auto)$ ]] && ! compgen -G "$TG/ctl.tmp.*" >/dev/null; then
  ok "toggler: $nw writes, off then auto, each through a temp file and rename"
else bad "toggler: $nw writes ($(tr '\n' ';' < "$TG/log" 2>/dev/null)), control file '$(cat "$TG/ctl" 2>/dev/null)'"; fi
# The tenant's checks (aux_tenant_check) and its exemption from the client after-check.
TN=$T/tenant
mkdir -p "$TN"
tcheck() {   # tenant1 log, tenant2 log, server log, want: "" or words, what
  printf '%s\n' "$1" > "$TN/a.tenant1.log"
  printf '%s\n' "$2" > "$TN/a.tenant2.log"
  printf '%s\n' "$3" > "$TN/a.server.log"
  local got
  got=$( OUT=$TN; log() { :; }; eval "$FUNCS"; AUX_TENANT_FAIL=""; aux_tenant_check a; echo "$AUX_TENANT_FAIL" )
  if [[ -z "$4" && -z "$got" ]] || [[ -n "$4" && "$got" == *"$4"* ]]; then ok "tenant: $5"; else bad "tenant: $5 ('$got')"; fi
}
T1=$'aux_tenant: open 1\naux_tenant: room after 10 ms, holding 12000 MiB\naux_tenant: freed'
T2=$'aux_tenant: open 2\naux_tenant: room after 1200 ms, holding 12000 MiB\naux_tenant: freed\ntenant2 exit 0'
SV=$'I aux experts: released at start: x\nI aux experts: serving again in 0.5 s\nI aux experts: releasing (foreign: aux_tenant (pid 9) 64 MiB)\nI aux experts: serving again in 0.4 s'
tcheck "$T1" "$T2" "$SV" "" "both tenants, the server's lines in order"
tcheck "$T1" "${T2/1200/4000}" "$SV" "over AUX_TENANT_ROOM_MS" "the second tenant's room too late"
tcheck "$T1" "${T2/exit 0/exit 2}" "$SV" "did not exit 0" "the second tenant failed"
tcheck "${T1/freed/}" "$T2" "$SV" "did not free" "the first tenant did not free"
tcheck "$T1" "$T2" $'I aux experts: on: 1513 units\nI aux experts: releasing (foreign: aux_tenant (pid 9) 64 MiB)\nI aux experts: serving again in 0.4 s' \
  "not released at start" "a server that did not start released"
LS11=$T/lsys11 LP11=$T/lproc11
extra "$LS11"
mkclient "$LS11" "$LP11" 111 whisper-server "$XGID" 4404019200
O3=$T/out19; mkdir -p "$O3"
got=$( set +u; OUT=$O3; log() { :; }; eval "$FUNCS"; AUX_SYSFS=$LS11 AUX_PROC=$LP11 AUX_GPUID=$XGID AUX_KFD_WAIT_DS=1
       aux_clients > "$O3/a.aux-clients"; mkclient "$LS11" "$LP11" 444 aux_tenant "$XGID" 67108864
       ARM_AXTEN=12000; aux_clients_after a "" && echo "none" || echo "new:$AUX_NEW_CLIENTS"
       ARM_AXTEN=""; aux_clients_after a "" && echo "none" || echo "new:$AUX_NEW_CLIENTS" )
[[ "$got" == $'none\nnew: aux_tenant (pid 444, 64 MiB)' ]] && ok "clients after: an AxF arm's tenant exempt, another arm's not" ||
  bad "clients after: '$got'"
# aux_yield.sh against a fake server state: no server, released after the holder, dead held, a timeout.
AY=$(dirname "$W")/aux_yield.sh
YD=$T/yield; mkdir -p "$YD"
out=$("$AY" "$YD/c1" 5 -- echo ran 2>&1); [[ "$out" == *"no server"*ran* ]] && ok "aux_yield: no server: runs" || bad "aux_yield: no server: '$out'"
( exec 7>"$YD/c2.lock"; flock -x 7; sleep 8 ) & lp=$!
sleep 0.2
echo "dead held 26112 1 pid 1 1 mode auto (default) a lane faulted" > "$YD/c2.state"
out=$("$AY" "$YD/c2" 5 -- echo ran 2>&1); rc=$?
[[ $rc == 3 && "$out" != *ran* ]] && ok "aux_yield: dead held: refuses (exit 3)" || bad "aux_yield: dead held: rc $rc '$out'"
echo "serving 1 pid 1 1 mode auto (default)" > "$YD/c2.state"
out=$("$AY" "$YD/c2" 1 -- echo ran 2>&1); rc=$?
[[ $rc == 4 && "$out" != *ran* ]] && ok "aux_yield: serving past the timeout: refuses (exit 4)" || bad "aux_yield: timeout: rc $rc '$out'"
( sleep 0.5; t=${EPOCHREALTIME//[!0-9]/}; printf 'released %s pid 1 1 mode off (holder x) manual\n' "$(( 10#$t / 1000 + 1000 ))" > "$YD/c2.state" ) &
out=$("$AY" "$YD/c2" 5 -- sh -c 'ls "$0"/c2.d | grep -c "^job-"' "$YD" 2>&1); rc=$?
[[ $rc == 0 && "$out" == *"given back (released)"*1* ]] && ok "aux_yield: released after its holder: runs, the holder in place" ||
  bad "aux_yield: released: rc $rc '$out'"
[[ -z "$(ls "$YD/c2.d")" ]] && ok "aux_yield: the holder removed after the command" || bad "aux_yield: holder left: $(ls "$YD/c2.d")"
wait "$lp" 2>/dev/null

# ---- 12. AxrN: the worker's taller passes (RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS, ds41_aux.h "taller passes")
#   the loop: Axr / AxrN with Ax or Axv states the switch (every other arm states it empty), its skips (no Ax part,
#   Ax0, no P1, no Lc, Kld, N outside 17-512, two parts, no auxtall pass or on another library, N above the check's,
#   a serve arm without an Axv arm with Axr of N rows or more); aux_want's rows; the exit gate's acted check; the
#   DEVICE_CHECKS=aux section's auxtall step (both cards at the window's tallest N, outputs compared).
L20=$T/loop20.txt
NOP1=${C2/-P1-/-}
ARMS="$C2 $C2-Ax $C2-Axv-Axr $C2-Axv-Axr256 $C2-Ax-Axr $C2-Ax-Axr64 $C2-Axr $C2-Ax0-Axr $NOP1-Ax-Axr $C2-Ax-Axr16"
ARMS+=" $C2-Ax-Axr600 $C2-Ax-Axr-Axr64 $C2-Ax-Axr384"
( set +u; window "$T/out20"; AUX_TALL_OK=1 AUX_TALL_SUM=$SUM AUX_TALL_ROWS=256 AXV_OK_R=1 AXV_SUM_R=$SUM AXV_ROWS_R=128; loop ) > "$L20" 2>&1
for spec in "$C2|" "$C2-Ax|" "$C2-Axv-Axr|128" "$C2-Axv-Axr256|256" "$C2-Ax-Axr|128" "$C2-Ax-Axr64|64"; do
  IFS='|' read -r a want <<< "$spec"
  sec=$(section "$L20" "$a")
  if [[ -z "$sec" ]]; then bad "Axr: ${a#$C2} not launched: $(grep "arm $a: " "$L20" | head -2 | tr '\n' ' ')"; continue; fi
  if grep -qx "    RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS=$want" <<< "$sec"; then ok "Axr: ${a#$C2} states RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS='$want'"
  else bad "Axr: ${a#$C2}: RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS is '$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS)', want '$want'"; fi
done
skipped "$L20" "$C2-Axr" "Axr128 needs an Ax or Axv part" "Axr alone"
skipped "$L20" "$C2-Ax0-Axr" "Axr128 needs an Ax or Axv part" "Axr beside Ax0"
skipped "$L20" "$NOP1-Ax-Axr" "Axr128 needs P1 and an Lc part" "Axr without P1"
skipped "$L20" "$C2-Ax-Axr16" "the worker's tallest pass is 17-512 rows" "Axr16"
skipped "$L20" "$C2-Ax-Axr600" "the worker's tallest pass is 17-512 rows" "Axr600"
skipped "$L20" "$C2-Ax-Axr-Axr64" "two Axr parts" "Axr twice"
skipped "$L20" "$C2-Ax-Axr384" "the auxtall device check passed up to 256 rows only" "Axr384 above the check's 256"
L21=$T/loop21.txt
ARMS="$C2-Ax-Axr256 $C2-Axv-Axr $C2-Ax-Axr"
( set +u; window "$T/out21"; AUX_TALL_OK=1 AUX_TALL_SUM=$SUM AUX_TALL_ROWS=256 AXV_OK_R=1 AXV_SUM_R=$SUM AXV_ROWS_R=128; loop ) > "$L21" 2>&1
skipped "$L21" "$C2-Ax-Axr256" "no Axv arm of this window with an Axr part of at least 256 rows" "Axr256 after an Axv arm of 128 rows"
L22=$T/loop22.txt
( set +u; window "$T/out22"; AXV_OK_R="" AUX_TALL_OK=1 AUX_TALL_SUM=$SUM AUX_TALL_ROWS=256; loop ) > "$L22" 2>&1
skipped "$L22" "$C2-Ax-Axr" "no Axv arm of this window with an Axr part" "Axr serve before an Axv arm with Axr"
[[ -n "$(section "$L22" "$C2-Axv-Axr")" ]] && ok "Axr: an Axv arm with Axr runs before any" || bad "Axr: Axv-Axr not launched in loop22"
L23=$T/loop23.txt
( set +u; window "$T/out23"; AUX_TALL_OK="" AUX_TALL_WHY="the auxtall device check did not pass on x: y;"; loop ) > "$L23" 2>&1
skipped "$L23" "$C2-Axv-Axr" "Axr refused: the auxtall device check did not pass" "Axv-Axr without the auxtall pass"
L24=$T/loop24.txt
( set +u; window "$T/out24"; AUX_TALL_OK=1 AUX_TALL_SUM=$OTHER AUX_TALL_ROWS=256; loop ) > "$L24" 2>&1
skipped "$L24" "$C2-Axv-Axr" "Axr refused: no auxtall device check passed on this library" "Axv-Axr on another library"
# aux_want: the rows lead the "on:" values; the engine's line with N rows (its format, the rows its 16th argument).
on_rows() {   # rows, policy
  printf "I $ONFMT\n" "aux experts: on: " 1513 "25.4 GiB" 4 "rank 0 380, rank 1 378, rank 2 377, rank 3 378" 0.4123 \
    0.9 15360 61.2 1500 13 0 "$1" "$2" 2000 64 1024 26000 0 \
    "; RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS=$1: passes of 17-$1 rows by moe_gemm_ggml_mxfp4_gemv_tall_p1, the deadline D * ceil(rows / 16), lanes of 40.1 MiB each"
}
wantr() {   # ARM_AXR, ARM_AXV, line, want, what
  local ARM_AX=26000 ARM_AXR=$1 ARM_AXV=$2 ARM_AXF="" ARM_AXDL=2000 ARM_AXBO=64 ARM_AXHR=1024 w n
  w=$( eval "$FUNCS"; aux_want )
  n=$(grep -cF -- "$w" <<< "$3")
  if [[ "$n" == "$4" ]]; then ok "on: $5"; else bad "on: $5 ('$w', $n matches)"; fi
}
wantr 128 "" "$(on_rows 128 serve)" 1 "Axr128's line (passes of up to 128 rows)"
wantr 128 1 "$(on_rows 128 verify)" 1 "Axv-Axr128's line"
wantr 128 "" "$(on_rows 256 serve)" 0 "Axr128 does not take a line of 256 rows"
wantr "" "" "$(on_rows 128 serve)" 0 "an Ax arm without Axr does not take a line of 128 rows"
wantr "" "" "$(on_line serve 2000 64 1024 26000 0)" 1 "an Ax arm without Axr takes today's line (16 rows)"
# The exit gate's acted check: the height part on every rank line, a job above 16 rows on some rank.
tall_lines() {   # file, then per rank the jobs above 16 rows ("-" no height part)
  local f=$1 r=0 n l; shift
  : > "$f"
  for n in "$@"; do
    l=$(rank_line "$r" 120 0 0 0 0)
    [[ $n != - ]] && l+="; by height: 1-16: 100 jobs, 300 of 900 rows; 17-32: 0 jobs, 0 of 0 rows; 33-64: 0 jobs, 0 of 0 rows; 65-128: $n jobs, $((n * 40)) of $((n * 90)) rows"
    echo "$l" >> "$f"; r=$((r + 1))
  done
}
egate_r() {   # what, file, ARM_AXR, want rc, words
  local got rc
  got=$( set +u; ARM_AXR=$3; eval "$FUNCS"; aux_exit_gate "$2" 26000 "" ""; echo "rc=$?" )
  rc=${got##*rc=}
  if [[ "$rc" != "$4" ]]; then bad "aux_exit_gate (Axr): $1: rc $rc ('$got')"; return; fi
  if [[ -n "$5" && "$got" != *"$5"* ]]; then bad "aux_exit_gate (Axr): $1: no '$5' in '$got'"; return; fi
  ok "aux_exit_gate (Axr): $1"
}
tall_lines "$EX/tall.log" 40 38 0 41
egate_r "Axr128: tall jobs on three ranks" "$EX/tall.log" 128 0 ""
tall_lines "$EX/tall0.log" 0 0 0 0
egate_r "Axr128: no tall job on any rank" "$EX/tall0.log" 128 1 "no rank ran a job taller than 16 rows (the part did not act)"
tall_lines "$EX/tallby.log" 40 38 - 41
egate_r "Axr128: a rank line without heights" "$EX/tallby.log" 128 1 "3 of 4 ranks' lines count jobs by height"
egate_r "no Axr part, heights in the lines" "$EX/tall.log" "" 1 "no Axr part, but 4 rank line(s) count jobs by height"
lines "$EX/serve16.log" 120:0:0:0:0 118:0:0:0:0 119:0:0:0:0 121:0:0:0:0
egate_r "no Axr part, today's lines" "$EX/serve16.log" "" 0 ""
egate_r "Axr128 on today's lines" "$EX/serve16.log" 128 1 "0 of 4 ranks' lines count jobs by height"
# The DEVICE_CHECKS=aux section's auxtall step.
if declare -f dcheck >/dev/null; then
  got=$(dcheck tall aux_tall_rows=256)
  if [[ "$got" == *"TALL=1|$LSUM|256|"* && "$got" == *"auxtall card $MASK/0 live tail 0"* && "$got" == *"auxtall card $XUUID/0 live tail 0"* &&
        "$got" == *"auxtall rows 256"* ]]; then ok "auxtall check: both cards at the window's tallest N, passed"
  else bad "auxtall check (pass): '$got'"; fi
  got=$(dcheck tall0)
  if [[ "$got" == *"TALL=|||"* && "$got" != *"auxtall card"* ]]; then ok "auxtall check: no Axr arm, not run"
  else bad "auxtall check (no Axr arm): '$got'"; fi
  got=$(dcheck talldiffer aux_tall_rows=128 FAKE_TALL_DIFFER=1)
  if [[ "$got" == *"TALL=|||"*"the extra card's auxtall outputs differ from card 0's"* ]]; then ok "auxtall check: outputs differing between the cards refuse"
  else bad "auxtall check (outputs differ): '$got'"; fi
  got=$(dcheck tallfail aux_tall_rows=128 FAKE_TALL_FAIL=1)
  if [[ "$got" == *"TALL=|||"*"auxtall-xt exited non-zero"* ]]; then ok "auxtall check: a FAIL refuses"
  else bad "auxtall check (FAIL): '$got'"; fi
  got=$(dcheck tallaux aux_tall_rows=128 FAKE_AUX_FAIL=1)
  if [[ "$got" == *"AUX=||"* && "$got" == *"TALL=|||"* && "$got" != *"auxtall card"* ]]; then ok "auxtall check: not run when aux did not pass"
  else bad "auxtall check (aux failed): '$got'"; fi
else bad "no dcheck (section 6) for the auxtall step"; fi

(( fail == 0 )) && echo "PASS ab_window_aux_test" || echo "FAIL ab_window_aux_test (loop outputs: $T)"
(( fail == 0 )) || trap - EXIT
exit $fail
