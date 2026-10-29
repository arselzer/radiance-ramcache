#!/usr/bin/env bash
# The engine A/B for host experts: one radiance binary and one set of plugins serving
# DeepSeek-V4.1 at TP4 on the loop's eight held-out prompts (greedy, 64 tokens each), once per arm:
#
#   base     no host experts
#   verify   the pool computes its share and the card still computes everything; every CPU row
#            is compared with the card's (the shutdown report counts differing rows; want 0)
#   greedy   the pool takes the cached experts it is expected to finish before the copy link
#   all      the pool takes every cached expert
#
# Run under both research locks, compiler first, as the research controllers take them:
#   flock -w 3600 research-compiler.lock flock -w 600 research-heavy-runtime.lock ab_window.sh OUT
# Waits for 30 s of QFN idleness, stops the QFN container, runs the arms, and always restarts the
# same container and checks it answers before returning.
#
# Before the arms (DEVICE_CHECKS=1, the default), the device harness checks the served libr4d.so on
# card 0: swiglu over every (gate, up) pair with mismatches classified, and the host chain.
#
# Arms are named POLICY[-tN][-gG][-sS][-nP][-mM][-MR][-PsK|-PoK|-PpK][-f][-x] (see the loop below), e.g. "base greedy-t20-g65 all-t24".
# An arm with an unknown part is skipped, not run without it; F8 (the FP8-stored dense weights) and
# Rb (the routers read as bf16) are also checked in the server log, and so is a predictor part's
# router predictor. An Rb arm runs only after SELFTEST_F32BF16's card checks passed in the window,
# and a Vp arm only on a libr4d.so VPACK_CHECK passed (below).
#
# Environment: RADIANCE_BIN (the binary), RADIANCE_HOME_DIR (kernels/ and architectures/ for it),
# SILU_TABLE (the card's silu table), ARMS (default "base verify greedy"), HX_THREADS (16),
# VERIFY_REQUESTS (3), DEVICE_CHECKS (1), SELFTEST_F32BF16 (an r4d_selftest: gemm_nt_f32bf16's
# checks and timings on card 0 before the arms; unset, none, and no Rb arm runs).
set -u
OUT=${1:?output directory}
BIN=${RADIANCE_BIN:?path to the radiance binary}
RHOME=${RADIANCE_HOME_DIR:?radiance home with kernels/ and architectures/}
SILU=${SILU_TABLE:?the card silu table}
ARMS=${ARMS-"base verify greedy"}
HX_THREADS=${HX_THREADS:-16}
VERIFY_REQUESTS=${VERIFY_REQUESTS:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/as/radiance-setup/research/deepseek-v41-1017-20261024
MODEL=$R/native-models/ds41-native-mxfp4-v3-20261027T121600Z/model.rad
MIRROR=${MIRROR:-}   # the checkpoint's model-*.safetensors for R arms; no default
DRAFT_BANK=${DRAFT_BANK:-}   # a DSpark draft bank (draft.rad) bound to $MODEL, for Ds arms; no default
PROFILE=$R/ds41-dev-window-profile-calibration-20261028T034300Z/sealed-decode-profile.json
PROMPTS=${PROMPTS:-$R/ds41-pack-tail-dev-window-tail1-heldout-20261028T085400Z/receipt.json}
PORT=18089
DOCKER=(/usr/bin/sudo -n /usr/bin/docker)
NAME=radiance-flashnext
URL=http://127.0.0.1:8078
mkdir -p "$OUT"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/window.log"; }

idle_for() {   # seconds of continuous idleness required, give up after $2 seconds
  local need=$1 limit=$2 quiet=0 waited=0
  while (( waited < limit )); do
    local s; s=$(curl -s -m 3 "$URL/stats" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["running"]+d["waiting"])' 2>/dev/null || echo busy)
    if [[ "$s" == 0 ]]; then quiet=$((quiet + 2)); else quiet=0; fi
    (( quiet >= need )) && return 0
    sleep 2; waited=$((waited + 2))
  done
  return 1
}

SERVER_PID=
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -INT "$SERVER_PID" 2>/dev/null
  local t=0
  while kill -0 "$SERVER_PID" 2>/dev/null && (( t < 180 )); do sleep 1; t=$((t + 1)); done
  if kill -0 "$SERVER_PID" 2>/dev/null; then log "server $SERVER_PID still up after ${t}s; SIGKILL"; kill -KILL "$SERVER_PID"; fi
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=
}

# Power settings a Q arm changes for its own run: g the performance CPU governor, n no NVMe power
# state with an exit latency (pm_qos latency tolerance 0), c the C2 idle state off. Every file's
# value is read here, before any arm, and written back after each Q arm and on any exit.
POWER_FILES=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor
             /sys/class/nvme/nvme[0-9]*/power/pm_qos_latency_tolerance_us
             /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state2/disable)
POWER_SAVED=$(for f in "${POWER_FILES[@]}"; do [[ -r $f ]] && echo "$f $(cat "$f")"; done)
POWER_SET=
power_write() {   # lines "file value" on stdin, written as root
  /usr/bin/sudo -n /bin/bash -c 'while read -r f v; do echo "$v" > "$f" || echo "power: cannot write $f" >&2; done'
}
power_state() {
  echo "governor $(sort /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | uniq -c | xargs)," \
       "nvme latency tolerance $(cat /sys/class/nvme/nvme[0-9]*/power/pm_qos_latency_tolerance_us | xargs) us," \
       "C2 disabled $(sort /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state2/disable | uniq -c | xargs)"
}
power_set() {   # the letters of an arm's Q part
  local f
  for f in "${POWER_FILES[@]}"; do
    case $f in
      */scaling_governor)              [[ $1 == *g* ]] && echo "$f performance" ;;
      */pm_qos_latency_tolerance_us)   [[ $1 == *n* ]] && echo "$f 0" ;;
      */state2/disable)                [[ $1 == *c* ]] && echo "$f 1" ;;
    esac
  done | power_write 2>>"$OUT/window.log"
  POWER_SET=1
  log "power Q$1: $(power_state)"
}
power_restore() {
  [[ -n "$POWER_SET" ]] || return 0
  echo "$POWER_SAVED" | power_write 2>>"$OUT/window.log"
  POWER_SET=
  log "power restored: $(power_state)"
}

restore() {
  # Runs once, ignores further signals, and retries the start until the container really runs:
  # a start issued while a stop is still in flight is a no-op, and the stop then wins.
  trap '' INT TERM HUP
  [[ -n "${RESTORED:-}" ]] && return 0
  RESTORED=1
  stop_server
  power_restore
  log "restoring $NAME"
  local t=0 state
  while :; do
    state=$("${DOCKER[@]}" inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)
    [[ "$state" == running ]] && curl -s -m 3 "$URL/stats" >/dev/null 2>&1 && break
    [[ "$state" != running ]] && "${DOCKER[@]}" start "$NAME" >>"$OUT/window.log" 2>&1
    sleep 5; t=$((t + 5))
    if (( t > 1200 )); then log "QFN did not come back within 20 min (state $state)"; return 1; fi
  done
  local r; r=$(curl -s -m 120 "$URL/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"qwen-vllm","prompt":"The capital of France is","max_tokens":4,"temperature":0}')
  log "QFN up after ${t}s; completion: $(echo "$r" | head -c 300)"
  rm -f /home/as/ds41-claude-build/work/qfn_stopped_by
  "${DOCKER[@]}" inspect "$NAME" --format '{{.Id}} {{.State.Status}} {{.State.StartedAt}} {{.Config.Image}}' >>"$OUT/window.log" 2>&1
}

arm() {   # name, then env assignments for the server; CACHE_MIB the RAM cache (default 81920)
  local name=$1; shift
  local n=${ARM_N:-${AB_REQUESTS:-8}}; [[ "$name" == verify* ]] && n=$VERIFY_REQUESTS
  log "arm $name: $*"
  env "$@" RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS=1 RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1 \
      RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2=1 \
    "$BIN" --radiance-home "${ARM_HOME:-$RHOME}" --kernels libr4d,libref --model "$MODEL" \
      --served-model-name ds41-native-smoke --tp 4 --tp-wire exact --placement expert_tiered \
      --weights-disk-tier --expert-ram-cache-total-mib ${CACHE_MIB:-81920} --host-pool-mib ${HOST_POOL_MIB:-4096} \
      --host-headroom-mib 8192 --gpu-headroom-mib 256 --embedding-placement vram \
      --vision-tower-placement vram --ngram-placement disk --max-model-len 512 \
      --max-num-batched-tokens 16 --max-num-seqs ${MAX_SEQS:-1} --kv-cache-dtype bf16 \
      --num-speculative-tokens 0 --spec-ngram 0 --no-prefix-cache --deterministic \
      --host 127.0.0.1 --port $PORT --expert-profile "$PROFILE" ${EXTRA_ARGS:-} "${ARGS[@]}" >"$OUT/$name.server.log" 2>&1 &
  SERVER_PID=$!
  # Memory guard: stops this server (never anything else) if the host runs short.
  ( while kill -0 "$SERVER_PID" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
      if (( avail < ${MEMGUARD_MIB:-3072} )); then
        echo "[$(date -u +%FT%TZ)] arm $name: MemAvailable ${avail} MiB below the guard; stopping the server" >> "$OUT/window.log"
        kill -INT "$SERVER_PID"; break
      fi
      echo "$(date -u +%T) $avail" >> "$OUT/$name.mem"
      sleep 5
    done ) &
  local t=0
  until curl -s -m 3 "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "arm $name: server exited during startup"; SERVER_PID=; return 1; fi
    sleep 2; t=$((t + 2))
    if (( t > 600 )); then log "arm $name: no server after 600 s"; stop_server; return 1; fi
  done
  log "arm $name: serving after ${t}s"
  # RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8 (the F8 parts): the plugin names the weights it reads from
  # E4M3 codes on each rank, and an arm must have exactly what its name says. An F8 arm whose
  # plugin did not take the knob, or a plain arm that took it from the window's environment, is
  # not measured. A tune cache would give the bf16 GEMMs another split than gemm_nt_bf16f8's, which
  # always takes the split law, so an F8 arm with one is not measured either.
  local f8_lines; f8_lines=$(grep -cE '[1-9][0-9]* dense weights read by gemm_nt_bf16f8' "$OUT/$name.server.log")
  if [[ -n "${ARM_F8:-}" && "$f8_lines" == 0 ]]; then
    log "arm $name: F8, but the server log has no 'dense weights read by gemm_nt_bf16f8' line (a plugin without the knob?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_F8:-}" && "$f8_lines" != 0 ]]; then
    log "arm $name: no F8 part, but the server read dense weights from E4M3 codes (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  if [[ -n "${ARM_F8:-}" ]] && grep -q 'tune cache: [0-9]* row' "$OUT/$name.server.log"; then
    log "arm $name: F8 with a tune cache loaded ($(grep -m1 'tune cache:' "$OUT/$name.server.log")); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_F8:-}" ]] && grep -E 'dense weights read by gemm_nt_bf16f8' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 (the Rb parts): rank 0's plugin names the routers it
  # reads as bf16, and an arm must have exactly what its name says. An Rb arm whose plugin did not
  # take the knob, or a plain arm that took it from the window's environment, is not measured.
  local rb_lines; rb_lines=$(grep -cE 'ROUTER_BF16: [1-9][0-9]* routers' "$OUT/$name.server.log")
  if [[ -n "${ARM_RB:-}" && "$rb_lines" == 0 ]]; then
    log "arm $name: Rb, but the server log has no 'ROUTER_BF16: N routers' line with N above 0 (a plugin without the knob, or no router held as bf16?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_RB:-}" && "$rb_lines" != 0 ]]; then
    log "arm $name: no Rb part, but the server read its routers as bf16 (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_RB:-}" ]] && grep -E 'ROUTER_BF16: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # A predictor part (Ps, Po, Pt, Pp) needs the router predictor: it reads every layer's router from
  # the container (an Rb arm's from the f32 plane its card copy was relaid from, weight_file_bytes),
  # and where it cannot, the read-ahead is off with nothing in the arm's name to say so. Not
  # measured then.
  if [[ -n "${ARM_PRED:-}" ]] && ! grep -qE 'router predictor: [0-9]+ routers loaded' "$OUT/$name.server.log"; then
    log "arm $name: a predictor part, but the server log has no 'router predictor: N routers loaded' line ($(grep -m1 -E 'router predictor' "$OUT/$name.server.log")); not measured"
    stop_server; return 1
  fi
  # The NVMe drive's busy time (io_ticks, ms) and temperature over the requests: read-ahead arms
  # move it from ~45% busy towards saturation.
  local d0 d1 t0 t1 rc temp
  d0=$(awk '{print $10}' /sys/block/nvme0n1/stat 2>/dev/null); t0=$(date +%s%N)
  env AB_CONCURRENCY=${ARM_CONC:-${AB_CONCURRENCY:-1}} \
    python3 -I "$HERE/ab_requests.py" "${ARM_PROMPTS:-$PROMPTS}" "http://127.0.0.1:$PORT" "$n" "$OUT/$name.json" >>"$OUT/window.log" 2>&1
  rc=$?
  log "arm $name: requests exit $rc"
  d1=$(awk '{print $10}' /sys/block/nvme0n1/stat 2>/dev/null); t1=$(date +%s%N)
  temp=$(cat /sys/class/nvme/nvme0/hwmon*/temp1_input 2>/dev/null | head -1)
  [[ -n "$d0" && -n "$d1" ]] && log "arm $name: NVMe busy $(( (d1 - d0) * 100000000 / (t1 - t0 + 1) ))% of the requests' $(( (t1 - t0) / 1000000 )) ms${temp:+, drive at $(( temp / 1000 )) C}"
  stop_server
  grep -E 'host experts|EP scoped|live chain|error|ERR' "$OUT/$name.server.log" | tail -16 | tee -a "$OUT/window.log"
  grep -E 'read-ahead (in pieces|readers)|router predictor \(' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"
  grep -E 'rank 0: on-die reads' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Pa/Pm/Pl/Pe: plan, counts
  grep -E 'post-stop passes: [0-9]' "$OUT/$name.server.log" | tail -1 | tee -a "$OUT/window.log"   # passes after a reply's end (Ss)
  # Every arm's VRAM residents a rank; a Vp arm's packed-form lines (what packed, the scan, the load's checks).
  grep -E 'planned units VRAM=|VRAM packed' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
}

if [[ "$("${DOCKER[@]}" inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)" != running ]]; then
  log "QFN ($NAME) is not running before the window; not starting one"; exit 4
fi
log "waiting for 30 s of QFN idleness"
if ! idle_for 30 600; then log "QFN not idle within 10 min; no window"; exit 3; fi
"${DOCKER[@]}" inspect "$NAME" --format 'before: {{.Id}} {{.State.Status}} {{.State.StartedAt}} {{.Config.Image}}' >>"$OUT/window.log" 2>&1
trap restore EXIT
trap 'exit 130' INT TERM HUP
echo $$ > /home/as/ds41-claude-build/work/qfn_stopped_by   # the QFN watchdog leaves a live owner's stop alone
log "stopping $NAME"
"${DOCKER[@]}" stop -t 60 "$NAME" >>"$OUT/window.log" 2>&1
sleep 5

if [[ "${DEVICE_CHECKS:-1}" == ptime && -x "$HERE/build/ds41_mxfp4_device_check" ]]; then
  log "prefill-shaped moe_gemm_ggml timing on card 0 with $RHOME/kernels/libr4d.so"
  ( export HIP_VISIBLE_DEVICES=0
    timeout 900 "$HERE/build/ds41_mxfp4_device_check" "$RHOME/kernels/libr4d.so" ptime ${PTIME_ARGS:-1 2 4 8 16} >"$OUT/ptime.log" 2>&1
    echo "ptime exit $?" >>"$OUT/ptime.log" )
  grep -E 'prefill|exit' "$OUT/ptime.log" | tee -a "$OUT/window.log"
fi
if [[ "${DEVICE_CHECKS:-1}" == 1 && -x "$HERE/build/ds41_mxfp4_device_check" ]]; then
  # The served plugin's swiglu over every (gate, up) pair, mismatches classified, and the
  # whole host chain against its decode kernels; the table recaptured and compared.
  log "device checks on card 0 with $RHOME/kernels/libr4d.so"
  ( export HIP_VISIBLE_DEVICES=0
    timeout 1200 "$HERE/build/ds41_mxfp4_device_check" "$RHOME/kernels/libr4d.so" swiglu "$OUT/silu_recheck.bin" >"$OUT/swiglu_proof.log" 2>&1
    echo "swiglu exit $?" >>"$OUT/swiglu_proof.log"
    timeout 1200 "$HERE/build/ds41_mxfp4_device_check" "$RHOME/kernels/libr4d.so" check 24 "$SILU" >"$OUT/chain_check.log" 2>&1
    echo "chain exit $?" >>"$OUT/chain_check.log" )
  tail -3 "$OUT/swiglu_proof.log" | tee -a "$OUT/window.log"
  tail -3 "$OUT/chain_check.log" | tee -a "$OUT/window.log"
  if cmp -s "$OUT/silu_recheck.bin" "$SILU"; then log "silu table recaptured identical"; else log "silu table recapture DIFFERS"; fi
fi

# SELFTEST_F32BF16=<r4d_selftest>: before the arms, on card 0 with the H home's libr4d.so (the Rb
# arms'), gemm_nt_f32bf16 byte-compared with gemm_nt_f32 and the host arithmetic (router and
# compressor shapes), then the router (384 x 5120) and compressor (1024 x 5120) GEMVs at one row
# (decode) and 16 (a prefill chunk) timed cold for both weights at every tile form, each form
# byte-compared first (the gate: the halves' rule at or under 14 us at 384, one row). RbN takes
# form N at passes of up to 16 rows, so its form is card-checked at both ends of that range.
# RB_OK is set only when everything compared equal: the check case exits 0 with no FAIL, and the
# sweep compared the rule and forms 1-8 at both shapes and both heights (36 lines) with no FAIL and
# all four runs exit 0. RB_SUM is the checked file's sha256, which an Rb arm's own libr4d.so must have.
RB_OK="" RB_LIB="" RB_SUM="" RB_WHY="no SELFTEST_F32BF16 (an r4d_selftest) given"
if [[ -n "${SELFTEST_F32BF16:-}" && -x "${SELFTEST_F32BF16}" ]]; then
  L4=${RADIANCE_HOME_DIR2:-$RHOME}/kernels/libr4d.so
  log "gemm_nt_f32bf16 checks on card 0 with $L4"
  RB_SUM=$(sha256sum "$L4" 2>/dev/null | cut -d' ' -f1)
  ( export HIP_VISIBLE_DEVICES=0
    # The bf16 halves' tile form is read once a process from this variable; left set, the
    # sweep's "rule" row and its gate reading would time that form instead of the default.
    unset RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM
    timeout 900 "$SELFTEST_F32BF16" "$L4" --case gemm_nt_f32bf16_bf16 >"$OUT/f32bf16_check.log" 2>&1
    echo "check exit $?" >>"$OUT/f32bf16_check.log"
    : >"$OUT/f32bf16_perf.log"
    for n in 384 1024; do
      for m in 1 16; do
        timeout 900 "$SELFTEST_F32BF16" "$L4" --cold 256 --perff32 $n 5120 $m 200 >>"$OUT/f32bf16_perf.log" 2>&1
        echo "perff32 $n $m exit $?" >>"$OUT/f32bf16_perf.log"
      done
    done )
  grep -cE "FAIL" "$OUT/f32bf16_check.log" | sed 's/^/f32bf16 check FAIL lines: /' | tee -a "$OUT/window.log"
  tail -1 "$OUT/f32bf16_check.log" | tee -a "$OUT/window.log"
  grep -E "^gemm_nt|^  (rule|4x1|2x2|8x1|8x4|1x2|1x4)|FAIL|exit" "$OUT/f32bf16_perf.log" | tee -a "$OUT/window.log"
  nfail=$(cat "$OUT/f32bf16_check.log" "$OUT/f32bf16_perf.log" 2>/dev/null | grep -c 'FAIL')
  ncheck=$(grep -cE '^  ok    gemm_nt_f32bf16 at M=' "$OUT/f32bf16_check.log" 2>/dev/null)
  nsweep=$(grep -cE '^  ok    gemm_nt_f32bf16 (at|form) ' "$OUT/f32bf16_perf.log" 2>/dev/null)
  nruns=$(grep -cxE 'perff32 (384|1024) (1|16) exit 0' "$OUT/f32bf16_perf.log" 2>/dev/null)
  nfail=${nfail:-0} ncheck=${ncheck:-0} nsweep=${nsweep:-0} nruns=${nruns:-0}
  if [[ "$nfail" == 0 && "$ncheck" -gt 0 && "$nsweep" == 36 && "$nruns" == 4 && -n "$RB_SUM" ]] &&
     grep -qx 'check exit 0' "$OUT/f32bf16_check.log"; then
    RB_OK=1 RB_LIB=$L4
    log "gemm_nt_f32bf16 card checks passed ($ncheck comparisons, $nsweep in the form sweep) on $L4, sha256 ${RB_SUM:0:16}: Rb arms on it are measured"
  else
    RB_WHY="FAIL lines $nfail, comparisons ok $ncheck, sweep comparisons ok $nsweep of 36, sweep runs exiting 0 $nruns of 4; $(tail -1 "$OUT/f32bf16_check.log" 2>/dev/null)"
    log "gemm_nt_f32bf16 card checks did not pass ($RB_WHY): no Rb arm is measured"
  fi
elif [[ -n "${SELFTEST_F32BF16:-}" ]]; then
  RB_WHY="SELFTEST_F32BF16=$SELFTEST_F32BF16 is not an executable"
  log "$RB_WHY: no Rb arm is measured"
fi

# The Vp arms' gate (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED). VPACK_CHECK=<ds41_vpack_device_check binary>
# runs, once for each libr4d.so a Vp arm serves (the first of the arm's run homes holding one, the one its
# server loads), libr4d's gemv_vp row against the plain gemv row on card 0, byte for byte (`check 48`), and
# both timed with their outputs compared (`time`). A Vp arm runs only on a libr4d.so (by md5) whose check
# printed PASS and exited 0, and whose timed outputs matched; without VPACK_CHECK, or after a failure, it is
# skipped. With RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1 in the window's environment (every server inherits it)
# the check runs again with the live-tail ranges on (`check 48 tail`), and must pass that way too.
declare -A VPACK_PASSED=()   # md5 of a checked libr4d.so: 1 passed, 0 failed
vpack_pass() {   # a check's log: its PASS line and its exit 0
  grep -q "^PASS: $2" "$1" && grep -qx "vpack $3 exit 0" "$1"
}
vpack_check() {   # a libr4d.so: its device check, run once a library; 0 when it passed
  local lib=$1 md5 tag ok=1
  md5=$(md5sum < "$lib" | cut -d' ' -f1)
  if [[ -z "${VPACK_PASSED[$md5]:-}" ]]; then
    tag=vpack-${md5:0:8}
    log "packed VRAM residents: gemv_vp against the plain row on card 0 with $lib (md5 $md5)"
    ( export HIP_VISIBLE_DEVICES=0
      timeout 1200 "$VPACK_CHECK" "$lib" check 48 >"$OUT/$tag.check.log" 2>&1
      echo "vpack check exit $?" >>"$OUT/$tag.check.log"
      if [[ "${RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL:-}" == 1 ]]; then
        timeout 1200 "$VPACK_CHECK" "$lib" check 48 tail >"$OUT/$tag.tail.log" 2>&1
        echo "vpack check exit $?" >>"$OUT/$tag.tail.log"
      fi
      timeout 900 "$VPACK_CHECK" "$lib" time >"$OUT/$tag.time.log" 2>&1
      echo "vpack time exit $?" >>"$OUT/$tag.time.log" )
    tail -5 "$OUT/$tag.check.log" | tee -a "$OUT/window.log"
    vpack_pass "$OUT/$tag.check.log" "gemv_vp over packed experts is bit-identical" check || ok=0
    if [[ "${RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL:-}" == 1 ]]; then
      tail -5 "$OUT/$tag.tail.log" | tee -a "$OUT/window.log"
      vpack_pass "$OUT/$tag.tail.log" "gemv_vp over packed experts is bit-identical" check || ok=0
    fi
    tail -22 "$OUT/$tag.time.log" | tee -a "$OUT/window.log"
    vpack_pass "$OUT/$tag.time.log" "the timed calls' outputs are bit-identical" time || ok=0
    VPACK_PASSED[$md5]=$ok
    if (( ok )); then log "packed VRAM residents: device check PASS on $lib; its Vp arms run"
    else log "packed VRAM residents: the device check did not pass on $lib (see $tag.*.log); its Vp arms are skipped"; fi
  fi
  [[ "${VPACK_PASSED[$md5]}" == 1 ]]
}
vpack_gate() {   # arm name, the libr4d.so it serves; 0 when the arm may run
  local a=$1 lib=$2
  if [[ -z "${VPACK_CHECK:-}" || ! -x "${VPACK_CHECK:-}" ]]; then
    log "arm $a: Vp, but VPACK_CHECK=${VPACK_CHECK:-} is not the packed form's device check; the arm is skipped"
    return 1
  fi
  if [[ ! -f "$lib" ]]; then log "arm $a: Vp, but its run home has no kernels/libr4d.so; the arm is skipped"; return 1; fi
  vpack_check "$lib" && return 0
  log "arm $a: Vp, but the device check did not pass on $lib; the arm is skipped"
  return 1
}
# Up front, before any arm runs: each Vp arm's library (its H / H3 part's home, as the loop below picks it,
# else the run home), so the checks run on idle cards and a failure shows before the window spends its time.
# The loop's gate then finds each library checked (and checks one this missed).
if [[ -n "${VPACK_CHECK:-}" && -x "${VPACK_CHECK:-}" ]]; then
  for a in $ARMS; do
    [[ "-$a-" == *-Vp-* ]] || continue
    h=$RHOME
    [[ "-$a-" == *-H-* ]] && h=${RADIANCE_HOME_DIR2:-$RHOME}
    [[ "-$a-" == *-H3-* ]] && h=${RADIANCE_HOME_DIR3:-$RHOME}
    IFS=: read -r -a homes <<< "$h"
    for hh in "${homes[@]}"; do [[ -f "$hh/kernels/libr4d.so" ]] && { vpack_check "$hh/kernels/libr4d.so"; break; }; done
  done
fi

HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
CANCEL=${CANCEL_FILE:-/home/as/ds41-claude-build/work/cancel_window}
for a in $ARMS; do
  if [[ -e "$CANCEL" ]]; then log "cancel file $CANCEL present; skipping the remaining arms"; break; fi
  # An arm is POLICY[-tN][-gG][-sS][-nP][-mM][-MR][-f][-x]: policy base, verify, greedy or all; N CPU
  # threads; G GB/s the schedule assumes for the CPU; S us the idle pool spins; P the pool's nice
  # value; M MiB of RAM cache (81920); R: passes of at most R tokens compute their missed experts on
  # the CPU from the container's planes; f the mover's direct cache fill; x an access trace;
  # PsK / PoK the next-layer predictor in shadow / on mode keeping K experts; PtK the same read ahead
  # two layers ahead (whole units); PpK read-ahead in pieces (RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces):
  # at layer L each rank's uncached units among layer L+2's top K predicted experts, read in 512 KiB
  # pieces, 4 in flight over the process, none started while a demand read is in flight, handed to
  # their layer only whole (the rest read at once when the layer needs a partly read one); PqN its
  # readers (pieces in flight, 4), PbN its piece size in KiB (512), and Pf0 a partly read unit its
  # layer needs read afresh instead of finished, with PpK; o --profile-ops;
  # H the second run home (RADIANCE_HOME_DIR2); iN background insert threads for missed units;
# pN --host-pool-mib (4096: a rank's pinned expert slots and the file tier's staging); G pin the
# RAM cache's slabs at startup; R / RNN read part of each missed expert from the checkpoint's
# safetensors in $MIRROR (--expert-plane-mirror; share 0.NN, default 0.18) in passes of N+ tokens
# (RmN, --expert-plane-mirror-min-rows, default 2), RdNN share 0.NN in smaller passes (decode;
# --expert-plane-mirror-decode-share, default 0); BN cap the VRAM file buffers at N
# units (the CPU must take every routed unit: all, M16); W warm the RAM cache from the expert
# profile before serving; ZN fail the N-th plane mirror read (tests the fallback to the container);
# A / AN read a layer's missed planes on a worker while the card's ops are issued (passes of N+
# tokens; A is every pass); Q[gnc] the arm's power settings (see power_set); SgK / SgaK prompt-lookup
# speculation, K drafts a step (fixed / adaptive window); Er the token embedding in host RAM; Xp / Xh
# / Xi miss reads on threads started per call / on 4 KiB pages / relaid via scratch (the old ways,
# as A/B switches); Ds / DsN / Dsa DSpark speculation from $DRAFT_BANK at depth 5 / capped at N
# drafts a step (Ds0: the bank loaded, drafting off) / with the adaptive window (the bank pins its
# experts in the host pool: give the arm pN of ~2048); H3 the third run home (RADIANCE_HOME_DIR3); Es
# engram_kv's rows split over the ranks (needs a deepseek41 plugin that has it); Pa the next layer's
# attention weights read into the card's on-die cache during the CPU-row wait (same); PmN those reads
# from the use-order list up to N MiB an issue point, the read that crosses it cut
# (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB), Pl<k1+k2+...> the reads named, in that order (..._SET;
# ekv comp qa kv iw iq qb ik oa ma ob rt sgu mf sdn), Pe an Engram layer's too: its engram_kv by the
# layer before when listed, the rest behind its GEMM (..._ENGRAM); each implies Pa and needs a
# deepseek41 plugin from claude/ds41-pfx; K the RAM cache
# in the CPU-only packed form (RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE; needs all, M16 and B1); ScN
# the RAM cache's profile-hot units get N second chances before eviction; IyN background inserts
# wait (asleep) up to N us for the pool to drain; Xs the pool's swiglu as the scalar loop (the old
# way); F8 the FP8-sourced dense linears read from E4M3 codes
# (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8, gemm_nt_bf16f8; a deepseek41 and libr4d that have it), F8n
# the same with the shared expert left bf16, both with $F8_EXCLUDE as ..._DENSE_FP8_EXCLUDE when set;
# an F8 arm is refused under RADIANCE_TUNE or a tune cache in its home, every other arm runs with the
# knob off; Pf a prefill chunk runs in a step of its own and the decode rows of concurrent requests
# run together (RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE); NK --max-num-seqs K, CK the arm's requests
# sent K at a time (AB_CONCURRENCY), LK --max-model-len K (the KV pool backs about one sequence of K
# tokens in all: 512 holds four 110-token requests, 1024 eight, 2048 sixteen); rN a repeat (ignored).
# the RAM cache's profile-hot units get N second chances before eviction; Ov / Ov2 a launch whose
# first dispatch depends on nothing in flight on its queue goes without the barrier bit, in decode
# passes / in every pass issued live (RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1 / 2; exact, only
# start order changes; the shutdown's "live chain" line counts it); Ct / CtN trace one trunk decode
# pass in N (16) on rank 0: every launch with its dependences into chain-<arm>.ops.csv and every
# kernel's card times into chain-<arm>.stamps.csv (RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE; read them
# with chain_report.py; without Ov it is the dump-only arm: what the check would free, nothing
# changed); rN a repeat (ignored).
# Ss a step of one sequence that its own token can end has its tokens read back before the next
# step is planned, and nothing runs behind one that ended its request: no forward pass after a
# reply's end-of-generation token, which the reply (and the next request) otherwise waits out,
# ~0.26 s when the cache lacks its experts (RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP; exact: the
# passes that run are the ones that ran; the shutdown's "post-stop passes" line counts both).
# the RAM cache's profile-hot units get N second chances before eviction; Ei a routed layer's card
# ops (gate_up, swiglu, down, shared expert) issued ahead of its counts sync, the CPU's share chosen
# at its gather (RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE; needs all, M16 and B1); Fl the counts and
# the CPU's rows fetched in one ranged copy instead of four (RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST;
# needs host experts); rN a repeat (ignored).
# Vp the cards' static expert residents in the packed-scale form, 4.4% smaller, so ~31 more a rank with F8
# (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED; needs a libr4d.so with moe_gemm_ggml_mxfp4_gemv_vp in the arm's
# run home, --deterministic and passes of at most 16 tokens, which every arm here has); runs only on a
# libr4d.so VPACK_CHECK's device check passed (vpack_gate above), skipped otherwise.
# the RAM cache's profile-hot units get N second chances before eviction; MsN a decode pass that
# misses one unit reads the plane mirror's tail in N pieces on N readers
# (RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT, 2-4, designed for 3; needs R and RdNN); Mp a one-token
# pass's miss reads start on the kept readers as soon as the layer's choice is made, before the
# pool's wake (RADIANCE_EXPERIMENTAL_DS41_MISS_POST); U<name> the arm sends every request of
# work/extra-prompts/<name>-receipt.json (chat: 16 chat-framed prompts; long: 64-3500-token
# prompts) instead of $PROMPTS' first AB_REQUESTS (compare texts only between arms with the same
# prompts); rN a repeat (ignored).
# Rb the routers and the compressors' kv|gate projections read as bf16
# (RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16, bit-identical; needs the deepseek41 plugin and libr4d
# built with it; checked in the server log, every other arm runs with the knob at 0), RbN the same
# with tile form N (1-8, r4d_selftest --perff32's names) for the bf16 GEMV at decode; an Rb arm runs
# only on the libr4d.so whose gemm_nt_f32bf16 passed SELFTEST_F32BF16's card checks in this window.
# An arm with a predictor part (Ps, Po, Pt, Pp) is measured only if the router predictor loaded.
  IFS=- read -r -a parts <<< "$a"
  pol=${parts[0]} threads=$HX_THREADS env=() CACHE_MIB=81920 HOST_POOL_MIB=4096 EXTRA_ARGS="" ARM_HOME="$RHOME" ARM_CONC=
  ARM_PROMPTS= ARM_N=
  power="" skip="" ARGS=() f8="" f8x="" rb="" rbf=0 pred=""
  for p in "${parts[@]:1}"; do
    case $p in
      Q[gnc]*) power=${p#Q} ;;
      r[0-9]*) ;;
      t[0-9]*) threads=${p#t} ;;
      g[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_GBPS=${p#g}) ;;
      f)       env+=(RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL=1) ;;
      x)       mkdir -p "$OUT/trace-$a"; env+=(RADIANCE_EXPERIMENTAL_DS41_ACCESS_TRACE=$OUT/trace-$a) ;;
      s[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SPIN_US=${p#s}) ;;
      n[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_NICE=${p#n}) ;;
      m[0-9]*) CACHE_MIB=${p#m} ;;
      M[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=${p#M}) ;;
      Mr[0-9]*) # pool jobs of up to N token rows of one expert (1-32, default 4; above 4, jobs of 4+ rows take the many-row kernel)
               if [[ ${p#Mr} =~ ^[0-9]+$ ]] && (( 10#${p#Mr} >= 1 && 10#${p#Mr} <= 32 )); then
                 env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=${p#Mr})
               else log "arm $a: $p: a job takes 1-32 rows; the arm is skipped"; skip=1; fi ;;
      o)       EXTRA_ARGS="--profile-ops" ;;
      H)       ARM_HOME=${RADIANCE_HOME_DIR2:?H needs RADIANCE_HOME_DIR2} ;;
      H3)      ARM_HOME=${RADIANCE_HOME_DIR3:?H3 needs RADIANCE_HOME_DIR3} ;;
      Es)      env+=(RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1) ;;   # engram_kv rows split over the ranks
      Pa)      env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1) ;;   # next layer's attention weights into the on-die cache
      Pm[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1 RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB=${p#Pm}) ;;   # on-die reads: a budget, use order
      Pl?*)    env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1 RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET=${p#Pl}) ;;   # on-die reads: these, in order
      Pe)      env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1 RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM=1) ;;   # on-die reads at Engram layers
      Rb)      rb=1 ;;                     # routers / compressor kv|gate read as bf16 (gemm_nt_f32bf16)
      Rb[1-8]) rb=1 rbf=${p#Rb} ;;         # the same, the bf16 GEMV at decode in tile form N
      Sc[0-9]) env+=(RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=${p#Sc}) ;;   # profile-hot cached units get N second chances
      Iy[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_INSERT_YIELD_US=${p#Iy}) ;;     # background inserts wait up to N us for an idle pool
      Xs)      env+=(RADIANCE_EXPERIMENTAL_DS41_SWIGLU_SCALAR=1) ;;   # the pool's swiglu as the scalar loop (old way)
      F8)      f8=1 ;;                     # FP8-sourced dense linears from E4M3 codes (gemm_nt_bf16f8)
      F8n)     f8=1 f8x='*_shexp.weight' ;;   # the same, the shared expert left bf16
      i[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_INSERT_THREADS=${p#i}) ;;
      p[0-9]*) HOST_POOL_MIB=${p#p} ;;
      G)       env+=(RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW=1) ;;
      B[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=${p#B}) ;;
      W)       env+=(RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM=1) ;;
      Z[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PLANE_MIRROR_FAULT=${p#Z}) ;;
      A)       env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=1) ;;
      A[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=${p#A}) ;;
      Rm[0-9]*) ARGS+=(--expert-plane-mirror-min-rows "${p#Rm}") ;;
      Rd[0-9]*) ARGS+=(--expert-plane-mirror-decode-share "0.${p#Rd}") ;;
      Sga[0-9]*) ARGS+=(--num-speculative-tokens "${p#Sga}" --spec-ngram "${p#Sga}" --draft-window adaptive) ;;
      Sg[0-9]*) ARGS+=(--num-speculative-tokens "${p#Sg}" --spec-ngram "${p#Sg}") ;;
      R|R[0-9]*)
               if [[ -z "$MIRROR" ]]; then log "arm $a: $p needs MIRROR=<directory of the checkpoint's model-*.safetensors>"; skip=1; fi
               ARGS+=(--expert-plane-mirror "$MIRROR")
               [[ $p != R ]] && ARGS+=(--expert-plane-mirror-share "0.${p#R}") ;;
      Ps[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=shadow RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Ps}) pred=1 ;;
      Po[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=on RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Po}) pred=1 ;;
      Pt[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=on RADIANCE_EXPERIMENTAL_DS41_PREFETCH_AHEAD=2 RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Pt}) pred=1 ;;
      Pp[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Pp}) pred=1 ;;   # read-ahead in pieces, top K
      Pq[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_QD=${p#Pq}) ;;          # ...its readers (4)
      Pb[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_PIECE_KB=${p#Pb}) ;;    # ...its piece size in KiB (512)
      Pf[01])   env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_FINISH=${p#Pf}) ;;      # ...0: partly read units read afresh
      Er)      ARGS+=(--embedding-placement ram) ;;   # the token embedding pinned in host RAM, its VRAM to experts
      Xp)      env+=(RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0) ;;        # miss reads on threads started a call
      Xh)      env+=(RADIANCE_EXPERIMENTAL_DS41_PLANE_HUGEPAGES=0) ;;  # miss buffers on 4 KiB pages
      Xi)      env+=(RADIANCE_EXPERIMENTAL_DS41_INSERT_DIRECT=0) ;;    # missed units relaid via scratch
      Ds|Ds[0-9]|Dsa)
               if [[ -z "$DRAFT_BANK" ]]; then log "arm $a: $p needs DRAFT_BANK=<draft.rad>"; skip=1; fi
               env+=(RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1)
               # host: the bank's 96 expert pairs pinned in the host pool and read in place by the card,
               # never file-tier units (the file-buffer cap needs every file-tier unit to be the CPU's)
               ARGS+=(--draft-bank "$DRAFT_BANK" --draft-bank-expert-placement host
                      --num-speculative-tokens 5 --spec-ngram 0)
               [[ $p == Ds[0-9] ]] && env+=(RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP=${p#Ds})
               [[ $p == Dsa ]] && ARGS+=(--draft-window adaptive) ;;
      K)       env+=(RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1) ;;     # the RAM cache packed for the CPU only
      Vp)      env+=(RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1) ;;      # VRAM residents in the packed-scale form (vpack_gate)
      Lm[0-9]*) ARGS+=(--max-model-len "${p#Lm}") ;;   # a longer context (default 512), e.g. for long-prompt TTFT
      Pf)      env+=(RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1) ;;   # a prefill chunk alone in its step; decode rows batched
      N[0-9]*) ARGS+=(--max-num-seqs "${p#N}") ;;                     # this arm's --max-num-seqs (MAX_SEQS for every arm)
      C[0-9]*) ARM_CONC=${p#C} ;;                                     # this arm's requests sent K at a time (AB_CONCURRENCY)
      L[0-9]*) ARGS+=(--max-model-len "${p#L}") ;;                    # this arm's --max-model-len (KV for N sequences)
      Ov)      env+=(RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=1) ;;    # independent dispatches overlap, decode passes
      Ov2)     env+=(RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP=2) ;;    # ... in every pass issued live
      Ct|Ct[0-9]*)
               ct=${p#Ct}
               env+=(RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE=${ct:-16}
                     RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT=$(cd "$OUT" && pwd)/chain-$a) ;;
      Ei)      env+=(RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1) ;;      # routed card ops ahead of the counts sync
      Ss)      env+=(RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1) ;;   # no pass behind a reply's last token
      Fl)      env+=(RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1) ;;       # counts + CPU rows in one ranged copy
      Ms[0-9]) env+=(RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=${p#Ms}) ;;  # a lone decode miss's mirror tail in N pieces
      Mp)      env+=(RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1) ;;       # miss reads start at the choice, before the pool's wake
      U?*)     ARM_PROMPTS=/home/as/ds41-claude-build/work/extra-prompts/${p#U}-receipt.json ARM_N=999
               [[ -f $ARM_PROMPTS ]] || { log "arm $a: no prompts $ARM_PROMPTS; the arm is skipped"; skip=1; } ;;
      *)     log "arm $a: unknown part $p; the arm is skipped (it would run without what its name says)"; skip=1 ;;
    esac
  done
  # gemm_nt_bf16f8 always takes gemm_bf16_nt_m64's split law; a tuned split for the bf16 rows (a
  # pin, or a tune cache in the arm's home) would make the F8 arm's bytes differ from the others'.
  if [[ -n "$f8" ]]; then
    if [[ -n "${RADIANCE_TUNE+x}" ]]; then log "arm $a: F8 refused: RADIANCE_TUNE is set"; skip=1; fi
    IFS=: read -r -a homes <<< "$ARM_HOME"
    for h in "${homes[@]}"; do
      if compgen -G "$h/tune/*" >/dev/null; then log "arm $a: F8 refused: $h/tune holds a tune cache"; skip=1; fi
    done
    ex=${F8_EXCLUDE:-}
    [[ -n "$f8x" ]] && ex=${ex:+$ex,}$f8x
    env+=(RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1 "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE=$ex")
  else
    env+=(RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=0)
  fi
  # Vp: only on a libr4d.so the packed form's device check passed (vpack_gate), checked here before the
  # arm's first launch on it; the server loads the first of its homes' kernels/libr4d.so.
  if [[ -z "$skip" && " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1 "* ]]; then
    vlib=""
    IFS=: read -r -a homes <<< "$ARM_HOME"
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { vlib=$h/kernels/libr4d.so; break; }; done
    vpack_gate "$a" "${vlib:-${homes[0]}/kernels/libr4d.so}" || skip=1
  fi
  ARM_F8=$f8
  # Every arm states the router knob, so a window's VAR=val cannot turn it on for the others. An Rb
  # arm runs only on the libr4d.so whose gemm_nt_f32bf16 passed the card checks before the arms
  # (SELFTEST_F32BF16): they are its exactness proof, which eight identical greedy replies are not
  # (a 1-ulp logit difference seldom changes a token).
  env+=(RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=${rb:-0} RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM=$rbf)
  if [[ -n "$rb" ]]; then
    IFS=: read -r -a homes <<< "$ARM_HOME"
    lib=""
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
    if [[ -z "$RB_OK" ]]; then
      log "arm $a: Rb refused: gemm_nt_f32bf16's card checks did not pass in this window ($RB_WHY)"; skip=1
    elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$RB_SUM" ]]; then
      log "arm $a: Rb refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the card checks passed on ($RB_LIB, sha256 ${RB_SUM:0:16})"; skip=1
    fi
  fi
  ARM_RB=$rb ARM_PRED=$pred
  # Ss: the read-back runs in an arm only if its name says so, whatever the window's environment.
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=0)
  [[ -n "$skip" ]] && continue
  [[ -n "$power" ]] && power_set "$power"
  case $pol in
    base) arm "$a" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=0 "${env[@]}" ;;
    verify|greedy|all)
      arm "$a" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$threads RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU \
          RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=$pol "${env[@]}" ;;
    *) log "unknown arm $a" ;;
  esac
  power_restore
done
[[ -n "$ARMS" ]] && python3 -I "$HERE/ab_requests.py" --compare "$OUT" $ARMS | tee -a "$OUT/window.log"
log "arms done"
