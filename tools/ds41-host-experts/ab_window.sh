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
#
# Environment: RADIANCE_BIN (the binary), RADIANCE_HOME_DIR (kernels/ and architectures/ for it),
# SILU_TABLE (the card's silu table), ARMS (default "base verify greedy"), HX_THREADS (16),
# VERIFY_REQUESTS (3), DEVICE_CHECKS (1).
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
  local n=${AB_REQUESTS:-8}; [[ "$name" == verify* ]] && n=$VERIFY_REQUESTS
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
  # The NVMe drive's busy time (io_ticks, ms) and temperature over the requests: read-ahead arms
  # move it from ~45% busy towards saturation.
  local d0 d1 t0 t1 rc temp
  d0=$(awk '{print $10}' /sys/block/nvme0n1/stat 2>/dev/null); t0=$(date +%s%N)
  python3 -I "$HERE/ab_requests.py" "$PROMPTS" "http://127.0.0.1:$PORT" "$n" "$OUT/$name.json" >>"$OUT/window.log" 2>&1
  rc=$?
  log "arm $name: requests exit $rc"
  d1=$(awk '{print $10}' /sys/block/nvme0n1/stat 2>/dev/null); t1=$(date +%s%N)
  temp=$(cat /sys/class/nvme/nvme0/hwmon*/temp1_input 2>/dev/null | head -1)
  [[ -n "$d0" && -n "$d1" ]] && log "arm $name: NVMe busy $(( (d1 - d0) * 100000000 / (t1 - t0 + 1) ))% of the requests' $(( (t1 - t0) / 1000000 )) ms${temp:+, drive at $(( temp / 1000 )) C}"
  stop_server
  grep -E 'host experts|EP scoped|error|ERR' "$OUT/$name.server.log" | tail -12 | tee -a "$OUT/window.log"
  grep -E 'read-ahead (in pieces|readers)|router predictor \(' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"
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
# attention weights read into the card's on-die cache during the CPU-row wait (same); K the RAM cache
# in the CPU-only packed form (RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE; needs all, M16 and B1); ScN
# the RAM cache's profile-hot units get N second chances before eviction; rN a repeat (ignored).
  IFS=- read -r -a parts <<< "$a"
  pol=${parts[0]} threads=$HX_THREADS env=() CACHE_MIB=81920 HOST_POOL_MIB=4096 EXTRA_ARGS="" ARM_HOME="$RHOME"
  power="" skip="" ARGS=()
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
      o)       EXTRA_ARGS="--profile-ops" ;;
      H)       ARM_HOME=${RADIANCE_HOME_DIR2:?H needs RADIANCE_HOME_DIR2} ;;
      H3)      ARM_HOME=${RADIANCE_HOME_DIR3:?H3 needs RADIANCE_HOME_DIR3} ;;
      Es)      env+=(RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1) ;;   # engram_kv rows split over the ranks
      Pa)      env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1) ;;   # next layer's attention weights into the on-die cache
      Sc[0-9]) env+=(RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=${p#Sc}) ;;   # profile-hot cached units get N second chances
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
      Ps[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=shadow RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Ps}) ;;
      Po[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=on RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Po}) ;;
      Pt[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=on RADIANCE_EXPERIMENTAL_DS41_PREFETCH_AHEAD=2 RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Pt}) ;;
      Pp[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=${p#Pp}) ;;   # read-ahead in pieces, top K
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
      *)       log "arm $a: unknown part $p" ;;
    esac
  done
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
