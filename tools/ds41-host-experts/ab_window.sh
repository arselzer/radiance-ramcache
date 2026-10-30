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
# DEVICE_CHECKS=draft checks the DSpark draft bank's GEMVs instead (every mode of
# RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV against the host experts, then their times a call), with the
# harness DRAFT_CHECK_BIN and the plugin DRAFT_CHECK_LIB (DRAFT_CHECK_TRIALS, 36).
#
# Arms are named POLICY[-tN][-gG][-sS][-nP][-mM][-MR][-PsK|-PoK|-PpK][-f][-x] (see the loop below), e.g. "base greedy-t20-g65 all-t24".
# An arm with an unknown part is skipped, not run without it; F8 (the FP8-stored dense weights) and
# Rb (the routers read as bf16) are also checked in the server log, and so is a predictor part's
# router predictor. An Rb arm runs only after SELFTEST_F32BF16's card checks passed in the window,
# and a Vp arm only on a libr4d.so VPACK_CHECK passed (below).
#
# Prefix caching (every other arm runs with --no-prefix-cache): Pc the prefix cache on, with
# RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1 (startup's checks after the load stop reserving the
# RAM cache's pinned slabs a second time; without it a G arm has no room for a checkpoint store and
# no ds41 hit can resume); Pc0 the cache on without it (the control); PciN --checkpoint-interval N
# (default 2048: a prompt under 2,048 tokens gets no snapshot to resume from); PcsN
# --checkpoint-slots N (auto otherwise); Pcx a finished sequence publishes only the prefill chunks an
# idle cold prefill cuts alike (RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH: no KV of a short last
# chunk or a decode step under a later hit). PciN, PcsN and Pcx need Pc or Pc0 (an arm with one and
# neither is skipped); every arm without Pc runs with ..._CACHE_HEADROOM_BACKED=0, and without Pcx
# with ..._PREFIX_EXACT_PUBLISH=0. The window prints a Pc arm's prefix-cache startup lines, and at
# the end each request's TTFT and cached prompt tokens (ab_requests.py --ttft), for the receipts
# work/extra-prompts/pfxchat-receipt.json and pfxlong-receipt.json (U parts Upfxchat with Lm4096,
# and Upfxlong with Lm8192: at Lm4096 the pool's trim evicts L1's short last chunk and decode
# blocks before L5, which then tests nothing; a U part's receipt name holds no "-", which splits
# parts). Put the same arm without Pc first: it is the cold reference.
#
# Exact tall passes (RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL): Xt sets the switch (every other arm runs
# with it 0) and BTN the step, --max-num-batched-tokens N (16 otherwise), e.g. ...-M32-...-Xt-BT32; an
# Xt arm must log "exact tall passes" at startup and a plain one must not, or it is not measured. Kld
# runs the arm's server as --kld-record over a corpus instead of serving: every chat_text of the
# receipts KLD_RECEIPTS names (default "long chat", work/extra-prompts/<name>-receipt.json) up to
# KLD_MAX_CHARS characters (6000), as plain text, every position scored, at --max-model-len KLD_LEN
# (2048) unless the arm gives an Lm part. The window's first Kld arm to finish is the reference (an
# Xt arm never is: it is then not compared); every later one's records are compared with it byte for byte (ab_requests.py --kld-compare: every position's
# f16 log-probabilities, its top 16 in f32, and the next token's NLL and the whole row's entropy in
# f32, so one changed logit anywhere in a row shows).
#
# Environment: RADIANCE_BIN (the binary), RADIANCE_HOME_DIR (kernels/ and architectures/ for it),
# SILU_TABLE (the card's silu table), ARMS (default "base verify greedy"), HX_THREADS (16),
# VERIFY_REQUESTS (3), DEVICE_CHECKS (1; `tall`: the exact-tall checks, TALL_LIBR4D and TALL_CHECK),
# SELFTEST_F32BF16 (an r4d_selftest: gemm_nt_f32bf16's checks and timings on card 0 before the
# arms; unset, none, and no Rb arm runs).
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

# RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL (the Xt part): the file stager says at startup which passes
# it takes, after checking every op keeps a 16-token pass's bits in them (it refuses to start
# otherwise). An Xt arm without that line, or a plain arm with it, is not measured.
xt_check() {   # name
  local name=$1 xt_lines
  xt_lines=$(grep -c 'exact tall passes: passes of up to [0-9]* tokens' "$OUT/$name.server.log")
  if [[ -n "${ARM_XT:-}" && "$xt_lines" == 0 ]]; then
    log "arm $name: Xt, but the server log has no 'exact tall passes' line (a radiance without the switch?); not measured"
    return 1
  fi
  if [[ -z "${ARM_XT:-}" && "$xt_lines" != 0 ]]; then
    log "arm $name: no Xt part, but the server runs exact tall passes (the knob set for the window?); not measured"
    return 1
  fi
  [[ -n "${ARM_XT:-}" ]] && grep -m1 -E 'exact tall passes: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  [[ -n "${ARM_BT:-}" ]] && grep -m1 -E 'chunk geometry: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  return 0
}

# A Kld arm: its server records the corpus (--kld-record) instead of serving, then exits. The
# window's first Kld arm that finishes is the reference (KLD_REF), never an Xt arm; a later one's
# records must equal it byte for byte. KLD_TIMEOUT (900 s) bounds a run.
KLD_REF=
kld_finish() {   # name; the run is SERVER_PID
  local name=$1 t=0 rc
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep 5; t=$((t + 5))
    if (( t > ${KLD_TIMEOUT:-900} )); then log "arm $name: kld run still going after ${t}s; stopped"; stop_server; return 1; fi
  done
  wait "$SERVER_PID"; rc=$?; SERVER_PID=
  log "arm $name: kld run exit $rc after ${t}s"
  grep -E 'kld:|error|ERR' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"
  (( rc == 0 )) || return 1
  xt_check "$name" || return 1
  if [[ -z "$KLD_REF" ]]; then
    # The reference must be a 16-token arm's bits: an Xt arm first (the plain one failed or is
    # missing) would let a later Xt arm be compared with another Xt arm.
    if [[ -n "${ARM_XT:-}" ]]; then
      log "arm $name: an Xt arm cannot be the window's kld reference (no plain Kld arm finished before it); not compared"
      return 1
    fi
    KLD_REF=$OUT/kld-$name
    log "arm $name: the window's kld reference"
    return 0
  fi
  python3 -I "$HERE/ab_requests.py" --kld-compare "$KLD_REF" "$OUT/kld-$name" 2>&1 | tee -a "$OUT/window.log"
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
      --num-speculative-tokens 0 --spec-ngram 0 ${ARM_NOPC---no-prefix-cache} --deterministic \
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
  if [[ -n "${ARM_KLD:-}" ]]; then kld_finish "$name"; return; fi
  local t=0
  until curl -s -m 3 "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "arm $name: server exited during startup"; SERVER_PID=; return 1; fi
    sleep 2; t=$((t + 2))
    if (( t > 600 )); then log "arm $name: no server after 600 s"; stop_server; return 1; fi
  done
  log "arm $name: serving after ${t}s"
  # A Pc arm: the prefix cache's startup lines (the snapshot store, the interval, the resume rule).
  [[ -z "${ARM_NOPC---no-prefix-cache}" ]] && \
    grep -E 'host headroom|chunk geometry|prefix resume|prefix cache|checkpoint slots' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
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
  # RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED (the Vp part): the engine names the switch at startup and
  # each rank how many static residents it holds packed. A Vp arm without the switch's line or with
  # no rank holding any packed (a radiance without the switch, a dynamic plan, a DSpark bank), or a
  # plain arm with either line (the switch taken from the window's environment), is not measured.
  local vp_on vp_held
  vp_on=$(grep -c 'VRAM packed experts (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED)' "$OUT/$name.server.log")
  vp_held=$(grep -cE 'holds [1-9][0-9]* static residents' "$OUT/$name.server.log")
  if [[ -n "${ARM_VP:-}" ]] && [[ "$vp_on" == 0 || "$vp_held" == 0 ]]; then
    log "arm $name: Vp, but the server log has $vp_on 'VRAM packed experts (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED)' and $vp_held 'holds N static residents' lines with N above 0 ($(grep -m2 -E 'VRAM packed experts' "$OUT/$name.server.log" | tr '\n' ' ')); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_VP:-}" ]] && [[ "$vp_on" != 0 || "$vp_held" != 0 ]]; then
    log "arm $name: no Vp part, but the server held experts packed (the switch set for the window?); not measured"
    stop_server; return 1
  fi
  # RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP (the Cs part): the scheduler says at startup whether it
  # took it, and the plugin whether it declared the suffix. An arm whose name says Cs and whose
  # server did not take it is not measured.
  if [[ -n "${ARM_CS:-}" ]] && ! grep -q 'consumer skip: on' "$OUT/$name.server.log"; then
    log "arm $name: Cs, but the server did not take it: $(grep -m2 -E 'consumer skip|consumer suffix' "$OUT/$name.server.log" | tr '\n' ' ')"
    stop_server; return 1
  fi
  [[ -n "${ARM_CS:-}" ]] && grep -m2 -E 'consumer skip: on|consumer suffix declared' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  xt_check "$name" || { stop_server; return 1; }
  # DbN (RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK): the engine says at startup that the drafter
  # runs its short block. A Db arm without that line (a plugin without the switch: the full block
  # runs), or another arm with it, is not measured.
  local db_lines; db_lines=$(grep -c "runs its short block of [0-9]* row" "$OUT/$name.server.log")
  if [[ -n "${ARM_DB:-}" && "$db_lines" == 0 ]]; then
    log "arm $name: Db${ARM_DB}, but the server log has no 'runs its short block of N row(s)' line; not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_DB:-}" && "$db_lines" != 0 ]]; then
    log "arm $name: no Db part, but the drafter runs a short block (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_DB:-}" ]] && grep -m2 -E 'runs its short block|short[0-9] query' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX (Fw) and _PREWAKE (WpN): the pool names what it
  # took at startup, and rank 0's file stager how many routers' GEMVs it found for the pre-wake. An
  # arm whose name says either and whose server did not take it, or one without that took it (the
  # knob set for the window), is not measured.
  local fw_on wp_on wp_rt
  fw_on=$(grep -c 'host experts: idle workers sleep on a futex' "$OUT/$name.server.log")
  wp_on=$(grep -cE 'host experts: pre-wake of up to [0-9]+ workers' "$OUT/$name.server.log")
  wp_rt=$(grep -cE 'host experts: pre-wake: an event before [1-9][0-9]* of [0-9]+ routed layers' "$OUT/$name.server.log")
  if [[ -n "${ARM_FW:-}" && "$fw_on" == 0 ]]; then
    log "arm $name: Fw, but the server log has no 'idle workers sleep on a futex' line (a radiance without the switch?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_FW:-}" && "$fw_on" != 0 ]]; then
    log "arm $name: no Fw part, but the pool's workers sleep on a futex (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  if [[ -n "${ARM_WP:-}" ]] && { [[ "$wp_rt" == 0 ]] || ! grep -qE "host experts: pre-wake of up to ${ARM_WP} workers" "$OUT/$name.server.log"; }; then
    log "arm $name: Wp${ARM_WP}, but the server log lacks 'pre-wake of up to ${ARM_WP} workers' or a router event ($(grep -m2 -E 'pre-wake' "$OUT/$name.server.log" | tr '\n' ' ')); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_WP:-}" && "$wp_on" != 0 ]]; then
    log "arm $name: no Wp part, but the server pre-wakes the pool (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_FW:-}${ARM_WP:-}" ]] && grep -E 'idle workers sleep on a futex|pre-wake' "$OUT/$name.server.log" | head -3 | tee -a "$OUT/window.log"
  # Pu / Pu2 (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1 / 2): every rank names at startup what a
  # partly read unit's rest is read from. A Pu arm without such a line with its value (a radiance
  # without the switch, no read-ahead in pieces to finish, ..._PREFETCH_FINISH=0), a Pu2 arm with a
  # rank that has no decode mirror (it would read the rest from the container alone), or another
  # arm with such a line, is not measured. Its shutdown counters are checked after the requests.
  local pu_any pu_lines pu_cont
  pu_any=$(grep -c 'reads the rest of a partly read unit its layer needs' "$OUT/$name.server.log")
  if [[ -n "${ARM_PU:-}" ]]; then
    pu_lines=$(grep -c "reads the rest of a partly read unit its layer needs .*PREFETCH_URGENT=${ARM_PU})" "$OUT/$name.server.log")
    pu_cont=$(grep -c "PREFETCH_URGENT=${ARM_PU}), from the container alone: this rank has no decode mirror" "$OUT/$name.server.log")
    if [[ "$pu_lines" == 0 || "$pu_lines" != "$pu_any" ]]; then
      log "arm $name: Pu${ARM_PU#1}, but the server log has $pu_lines 'reads the rest of a partly read unit ... PREFETCH_URGENT=${ARM_PU})' lines of $pu_any ($(grep -m1 -E 'a partly read unit its layer needs is' "$OUT/$name.server.log")); not measured"
      stop_server; return 1
    fi
    if [[ "$pu_cont" != 0 ]]; then
      log "arm $name: Pu2, but $pu_cont rank(s) have no decode mirror and read the rest from the container alone (MIRROR, R and RdNN?); not measured"
      stop_server; return 1
    fi
    grep -E 'reads the rest of a partly read unit its layer needs' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  elif [[ "$pu_any" != 0 ]]; then
    log "arm $name: no Pu part, but the server reads a partly read unit's rest in the layer's own call (the knob set for the window?); not measured"
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
  # Lq: the quality harness, after the timed requests (which it leaves as every other arm sends
  # them). The window's first Lq arm of a prompt set records the reference (free-running, logprobs)
  # unless LQ_REF names one; every Lq arm then decodes teacher-forced along it (ab_requests.py).
  if [[ -n "${ARM_LQ:-}" ]]; then
    local pset lpref
    pset=$(basename "${ARM_PROMPTS:-$PROMPTS}" .json)
    lpref=${LQ_REF:-$OUT/lpref-$pset.json}
    if [[ ! -f "$lpref" ]]; then
      python3 -I "$HERE/ab_requests.py" --lp "${ARM_PROMPTS:-$PROMPTS}" "http://127.0.0.1:$PORT" "$n" "$lpref" >>"$OUT/window.log" 2>&1
      log "arm $name: the logprob reference for $pset ($lpref): exit $?"
    fi
    python3 -I "$HERE/ab_requests.py" --force "$lpref" "http://127.0.0.1:$PORT" "$OUT/$name.lpforce.json" >>"$OUT/window.log" 2>&1
    log "arm $name: teacher-forced along $lpref: exit $?"
  fi
  stop_server
  # Dk / Dkt (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV): libr4d names the mode at its first draft call
  # and counts the calls when the server exits. A Dk arm without the first line, or whose exit line
  # leaves draft-shaped calls on the old path, or another arm with either line, is not measured:
  # its results are set aside as <arm>.unmeasured.json (the window's comparisons skip it).
  local dk_first dk_exit dk_bad=""
  dk_first=$(grep -c 'libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV: the DSpark draft bank' "$OUT/$name.server.log")
  dk_exit=$(grep -E 'libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV \(' "$OUT/$name.server.log" | tail -1)
  if [[ -n "${ARM_DK:-}" ]]; then
    [[ "$dk_first" == 0 ]] && dk_bad="no 'libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV: the DSpark draft bank' line (a libr4d without the switch, or no draft call)"
    [[ -n "$dk_exit" && "$dk_exit" != *" 0 draft-shaped calls left on the old path"* ]] && dk_bad="draft-shaped calls left on the old path: $dk_exit"
  elif [[ "$dk_first" != 0 || -n "$dk_exit" ]]; then
    dk_bad="no Dk part, but libr4d took RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (the knob set for the window?)"
  fi
  if [[ -n "$dk_bad" ]]; then
    log "arm $name: $dk_bad; not measured (results kept as $name.unmeasured.json)"
    [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
  fi
  # Fw / WpN at shutdown: the pool's line ("host expert pool: ..."). A Wp arm counts only if its
  # router pre-wakes woke workers; an Fw arm only if its pool woke through the futex. Otherwise its
  # results are set aside as <arm>.unmeasured.json (the window's comparisons skip it).
  local pool_line wk_bad=""
  pool_line=$(grep -E 'host expert pool: ' "$OUT/$name.server.log" | tail -1)
  if [[ -n "${ARM_WP:-}" ]] && ! grep -qE 'router pre-wakes: [0-9]+ calls woke [1-9][0-9]* workers' <<< "$pool_line"; then
    wk_bad="Wp${ARM_WP}, but no router pre-wake woke a worker (${pool_line:-no 'host expert pool' line})"
  fi
  if [[ -n "${ARM_FW:-}" && "$pool_line" != *" futex wake "* ]]; then
    wk_bad="Fw, but the pool's line does not say futex wake (${pool_line:-no 'host expert pool' line})"
  fi
  if [[ -n "$wk_bad" && -f "$OUT/$name.json" ]]; then
    log "arm $name: $wk_bad; not measured (results kept as $name.unmeasured.json)"
    mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
  fi
  # Pu / Pu2 at shutdown: some rank's layers read a partly read unit's rest in their own call (Pu2:
  # some of it from the plane mirror), or the switch never acted and the arm is not measured (set
  # aside as <arm>.unmeasured.json, as a Dk arm's).
  if [[ -n "${ARM_PU:-}" ]]; then
    local pu_rest pu_mb pu_bad=""
    pu_rest=$(grep -oE "PREFETCH_URGENT=${ARM_PU}\): [0-9]+ units' rest" "$OUT/$name.server.log" | grep -oE ': [0-9]+' | tr -d ': ' | awk '{s += $1} END {print s + 0}')
    pu_mb=$(grep -E "PREFETCH_URGENT=${ARM_PU}\): [0-9]+ units' rest" "$OUT/$name.server.log" | grep -oE '\([0-9]+ MB of it from the plane mirror' | grep -oE '[0-9]+' | awk '{s += $1} END {print s + 0}')
    [[ "$pu_rest" == 0 ]] && pu_bad="no rank's shutdown line counts a unit's rest read in its layer's own call"
    [[ -z "$pu_bad" && "$ARM_PU" == 2 && "$pu_mb" == 0 ]] && pu_bad="no rank read any of a rest from the plane mirror"
    if [[ -n "$pu_bad" ]]; then
      log "arm $name: Pu${ARM_PU#1}, but $pu_bad; not measured (results kept as $name.unmeasured.json)"
      [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
    fi
  fi
  grep -E 'host experts|EP scoped|live chain|error|ERR' "$OUT/$name.server.log" | tail -16 | tee -a "$OUT/window.log"
  grep -E 'read-ahead (in pieces:|readers)|router predictor \(' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"
  # The per-unit wait for partly read units, every arm with read-ahead in pieces (the Pu parts' gate),
  # and a Pu arm's rests.
  grep -E 'read-ahead in pieces, (partly read units|rests on the kept readers)' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -E 'rank 0: on-die reads' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Pa/Pm/Pl/Pe: plan, counts
  grep -E 'post-stop passes: [0-9]' "$OUT/$name.server.log" | tail -1 | tee -a "$OUT/window.log"   # passes after a reply's end (Ss)
  # Every arm's VRAM residents a rank; a Vp arm's packed-form lines (what packed, the scan, the load's checks).
  grep -E 'planned units VRAM=|VRAM packed' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -E 'route trace' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"   # Rt: per rank, and the files
  grep -E 'consumer skip: [0-9]+ (prefill|cut) passes|rank 0: consumer skip' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Cs: what the cut ran
  grep -E 'prefill hash: seq [0-9]+, [0-9]+ tokens: all ranks' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Kh: a digest a prompt
  grep -E '^libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV' "$OUT/$name.server.log" | tail -2 | tee -a "$OUT/window.log"   # Dk/Dkt: calls taken
  grep -E 'host expert pool: |host experts, pre-wake: |host experts, one-token passes: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Fw/WpN: the wake-up, decode's gather waits
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
# The Xt arms' gate (RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL), like Dk's: an Xt arm runs only on the
# libr4d.so (sha256) the tall checks passed on in this window.
TALL_OK="" TALL_LIB="" TALL_SUM="" TALL_WHY="no exact-tall device checks in this window (DEVICE_CHECKS=tall with TALL_CHECK and TALL_LIBR4D)"
if [[ " ${DEVICE_CHECKS:-1} " == *" tall "* ]]; then
  # RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL on card 0: the twin row and the sparse ops at tall heights
  # (`tall`), and every other height-dependent op with the switch (`sweep`) and without (`sweep0`),
  # each row against the same row alone and in launches of 16. TALL_LIBR4D: the plugin the Xt arms
  # load; TALL_CHECK: a harness built with these modes (both must be named: the window runs from a
  # snapshot directory, so no default exists).
  TL=${TALL_LIBR4D:-} TC=${TALL_CHECK:-}
  if [[ ! -x "$TC" || ! -f "$TL" ]]; then
    TALL_WHY="TALL_CHECK=$TC is not an executable or TALL_LIBR4D=$TL is not a file"
    log "exact-tall device checks: $TALL_WHY; no Xt arm is measured"
  else
    log "exact-tall device checks on card 0 with $TL ($TC)"
    tall_bad=""
    for m in tall sweep sweep0; do
      ( export HIP_VISIBLE_DEVICES=0
        timeout 900 "$TC" "$TL" $m >"$OUT/devcheck-$m.log" 2>&1
        echo "$m exit $?" >>"$OUT/devcheck-$m.log" )
      grep -E '^(PASS|FAIL)|exit|norm rows' "$OUT/devcheck-$m.log" | tail -8 | tee -a "$OUT/window.log"
      grep -qx "$m exit 0" "$OUT/devcheck-$m.log" || tall_bad+=" $m exited non-zero;"
      grep -q '^FAIL' "$OUT/devcheck-$m.log" && tall_bad+=" $m printed FAIL;"
    done
    grep -q "^PASS: tall passes give 16-token passes' bits" "$OUT/devcheck-tall.log" || tall_bad+=" tall has no PASS line;"
    grep -q '^PASS: sweep' "$OUT/devcheck-sweep.log" || tall_bad+=" sweep has no PASS line;"
    if [[ -z "$tall_bad" ]]; then
      TALL_OK=1 TALL_LIB=$TL TALL_SUM=$(sha256sum "$TL" 2>/dev/null | cut -d' ' -f1)
      log "exact-tall device checks passed on $TL, sha256 ${TALL_SUM:0:16}: Xt arms on it are measured"
    else
      TALL_WHY="the exact-tall device checks did not pass on $TL:$tall_bad"
      log "$TALL_WHY no Xt arm is measured"
    fi
  fi
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
# The Dk arms' gate (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV): the target verifies every draft, so a
# wrong draft GEMV only moves acceptance and speed, never a reply; the card check is its exactness
# proof. A Dk arm runs only on the libr4d.so (sha256) this check passed on in the window.
DK_OK="" DK_LIB="" DK_SUM="" DK_WHY="no draft GEMV device check in this window (DEVICE_CHECKS=draft with DRAFT_CHECK_BIN and DRAFT_CHECK_LIB)"
if [[ " ${DEVICE_CHECKS:-1} " == *" draft "* ]]; then
  # RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV (Dk) on card 0: the draft bank's grouped GEMMs in every
  # mode against the host experts, experts in pinned host memory and in VRAM, then each mode's
  # time a call at a 5-row query pass. DRAFT_CHECK_BIN is a ds41_mxfp4_device_check with the draft
  # subcommand, DRAFT_CHECK_LIB a libr4d.so with the switch (both must be named: the window runs
  # from a snapshot directory, so no default exists).
  DCB=${DRAFT_CHECK_BIN:-} DCL=${DRAFT_CHECK_LIB:-}
  if [[ ! -x "$DCB" || ! -f "$DCL" ]]; then
    DK_WHY="DRAFT_CHECK_BIN=$DCB is not an executable or DRAFT_CHECK_LIB=$DCL is not a file"
    log "draft GEMV device check: $DK_WHY; no Dk arm is measured"
  else
    log "draft GEMV device check on card 0 with $DCL"
    ( export HIP_VISIBLE_DEVICES=0
      timeout 900 "$DCB" "$DCL" draft ${DRAFT_CHECK_TRIALS:-36} >"$OUT/draft_check.log" 2>&1
      echo "draft check exit $?" >>"$OUT/draft_check.log" )
    grep -E 'draft GEMV|PASS|FAIL|gate_up|down|exit' "$OUT/draft_check.log" | tail -20 | tee -a "$OUT/window.log"
    if grep -q '^PASS: every draft GEMV mode is bit-identical' "$OUT/draft_check.log" &&
       grep -qx 'draft check exit 0' "$OUT/draft_check.log"; then
      DK_OK=1 DK_LIB=$DCL DK_SUM=$(sha256sum "$DCL" 2>/dev/null | cut -d' ' -f1)
      log "draft GEMV device check passed on $DCL, sha256 ${DK_SUM:0:16}: Dk arms on it are measured"
    else
      DK_WHY="the draft GEMV device check did not pass on $DCL ($(tail -1 "$OUT/draft_check.log" 2>/dev/null))"
      log "$DK_WHY: no Dk arm is measured"
    fi
  fi
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
  # layer needs read afresh instead of finished, with PpK; Pu such a unit finished by its layer's
  # own read instead of urgent pieces: no further piece starts, and the call that reads the layer's
  # other misses on the kept readers reads the rest of each extent too, one large read an extent's
  # rest, while the pieces in flight land (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1; the same
  # bytes at the same offsets), Pu2 the same with the plane mirror's decode share of the rest beside
  # the container's (=2, so a barely started unit is finished too; needs R and RdNN); both need PpK,
  # every other arm runs with the switch 0, and a Pu arm is measured only if every rank names it at
  # startup (Pu2: with its decode mirror) and a shutdown line counts rests read (Pu2: some from the
  # mirror); the "read-ahead in pieces, partly read units" line has the per-unit wait in every
  # arm with PpK; o --profile-ops;
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
# experts in the host pool: give the arm pN of ~2048); Dk the draft bank's grouped GEMMs take the
# live tail and one pass over each expert's weights (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=1: no
# slot routed to another rank runs against the rank's last expert, and an expert's rows share one
# read of its pinned host weights; exact; a libr4d that has it, and a Ds part), Dkt the live tail
# alone (=2) (a Dk/Dkt arm runs only on the libr4d.so DEVICE_CHECKS=draft's check passed on in the
# window, and is measured only if libr4d names the mode at its first draft call and leaves no
# draft-shaped call on the old path; every other arm runs with both switches at 0 and must show
# neither; at the end each Dk arm's draft_n/draft_n_accepted must equal the same arm's without Dk);
# DbN DSpark's query pass as a short block of N rows (2-4), [anchor, noise x (N - 1)],
# on every decoding step (RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=N, --draft-short-from 1;
# with DsK, N = K + 1 drafts what the cap verifies; the replies stay the target's, acceptance may
# move; a deepseek41 plugin that has it, and a Ds part); H3 the third run home (RADIANCE_HOME_DIR3); Es
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
# libr4d.so VPACK_CHECK's device check passed (vpack_gate above), skipped otherwise, and is measured
# only if its server log names the switch and a rank holding N > 0 residents packed; every other arm
# runs with the switch at 0 and must have neither line.
# the RAM cache's profile-hot units get N second chances before eviction; MsN a decode pass that
# misses one unit reads the plane mirror's tail in N pieces on N readers
# (RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT, 2-4, designed for 3; needs R and RdNN); Mp a one-token
# pass's miss reads start on the kept readers as soon as the layer's choice is made, before the
# pool's wake (RADIANCE_EXPERIMENTAL_DS41_MISS_POST); U<name> the arm sends every request of
# work/extra-prompts/<name>-receipt.json (chat: 16 chat-framed prompts; long: 64-3500-token
# prompts; long8k: one ~7K-token document, with LmN of 8448 or more) instead of $PROMPTS' first
# AB_REQUESTS (compare texts only between arms with the same prompts); Cs the exact consumer-suffix
# skip (RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1: a prompt's early prefill passes run layers 21..39
# only where their rows reach its last token or its state; needs a radiance and a deepseek41 from
# claude/ds41-cskip, and the server's "consumer skip: on" line, or the arm is not measured; every
# other arm runs with it off); Kh at the end of each prompt's prefill every rank hashes its KV groups
# and the last row's logits (RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1; steps are not issued ahead
# while it is on, so time a prompt in an arm without it), and the window compares each Kh arm's
# hashes with those of the first Kh arm with the same prompts (ab_requests.py --hashes); Rt the
# routing trace: every EP-routed layer's picks, weights, keys and what the owning
# rank did with them, rank 0's 384 logits on decode passes, each rank's RAM-cache residency, and
# the requests each pass carried, into rtrace-<arm>/rt.rank<r>.bin and rt.steps.bin
# (RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE; exact; read them with rtrace.py); Lq the quality
# harness after the arm's timed requests: the window's first Lq arm of a prompt set records the
# free-running reference with logprobs (lpref-<set>.json, or LQ_REF=<file> names one), and every
# Lq arm decodes teacher-forced along it (<arm>.lpforce.json; ab_requests.py --kl-dir at the end
# compares them: KL and top-1 by prompt family); rN a repeat (ignored).
# Rb the routers and the compressors' kv|gate projections read as bf16
# (RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16, bit-identical; needs the deepseek41 plugin and libr4d
# built with it; checked in the server log, every other arm runs with the knob at 0), RbN the same
# with tile form N (1-8, r4d_selftest --perff32's names) for the bf16 GEMV at decode; an Rb arm runs
# only on the libr4d.so whose gemm_nt_f32bf16 passed SELFTEST_F32BF16's card checks in this window.
# An arm with a predictor part (Ps, Po, Pt, Pp) is measured only if the router predictor loaded.
# Fw the CPU pool's idle workers sleep on a futex and a submit wakes them with one FUTEX_WAKE, no
# lock (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1; exact: only when workers wake changes);
# WpN up to N sleeping workers pre-woken before a one-token layer's CPU work: at the layer's router
# event (an event before each decode router GEMV, waited for ahead of the counts) and in a miss
# read ..._PREWAKE_LEAD_US (60) before its predicted landing; they spin ..._PREWAKE_SPIN_US (100)
# and sleep again if no work came (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=N, 1 .. the
# arm's threads; exact). Every other arm runs with both at 0. Fw must show the pool's futex line at
# startup, WpN its pre-wake line for N workers and the stager's router events, and at shutdown the
# pool's line with router pre-wakes that woke workers (else the arm's results are set aside as
# <arm>.unmeasured.json); an arm without them must show neither startup line.
  IFS=- read -r -a parts <<< "$a"
  xt="" bt="" kld="" dk="" db="" pu=""
  pol=${parts[0]} threads=$HX_THREADS env=() CACHE_MIB=81920 HOST_POOL_MIB=4096 EXTRA_ARGS="" ARM_HOME="$RHOME" ARM_CONC=
  ARM_PROMPTS= ARM_N= ARM_LQ= rt=
  power="" skip="" ARGS=() f8="" f8x="" rb="" rbf=0 pred="" cs="" kh="" pc="" pcopt="" fw="" wp=""
  unset ARM_NOPC   # --no-prefix-cache unless a Pc part sets it empty
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
      Pu|Pu2)   pu=${p#Pu}; pu=${pu:-1}                                          # ...a partly read unit's rest in its layer's own read
                env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=$pu) ;;          # (1; 2: the plane mirror's share of it too)
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
      Dk)      dk=1 ;;                     # draft GEMVs: live tail + one weight pass (DRAFT_GEMV=1)
      Dkt)     dk=2 ;;                     # draft GEMVs: the live tail alone (DRAFT_GEMV=2)
      Db[2-4]) db=${p#Db}                  # DSpark's short query block (DSPARK_SHORT_BLOCK=N)
               ARGS+=(--draft-short-from 1) ;;
      K)       env+=(RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1) ;;     # the RAM cache packed for the CPU only
      Vp)      env+=(RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1) ;;      # VRAM residents in the packed-scale form (vpack_gate)
      Lm[0-9]*) ARGS+=(--max-model-len "${p#Lm}") ;;   # a longer context (default 512), e.g. for long-prompt TTFT
      Pc)      pc=1 ARM_NOPC= ; ARGS+=(--prefix-resume exact)                          # prefix cache on (exact resume, the default, named)
               env+=(RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1) ;;              # ... and pinned slabs not reserved twice
      Pc0)     pc=1 ARM_NOPC= ; ARGS+=(--prefix-resume exact) ;;                      # prefix cache on, the margin as before (the control)
      Pci[0-9]*) pcopt+=" $p"; ARGS+=(--checkpoint-interval "${p#Pci}") ;;                    # snapshot interval (2048)
      Pcs[0-9]*) pcopt+=" $p"; ARGS+=(--checkpoint-slots "${p#Pcs}") ;;                       # snapshot store slots (auto)
      Pcx)     pcopt+=" $p"; env+=(RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1) ;;           # publish only cold-cut chunks
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
      Xt)      xt=1 ;;                                                 # exact tall passes (RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1)
      BT[0-9]*) bt=${p#BT}; ARGS+=(--max-num-batched-tokens "$bt") ;;  # this arm's step: prefill chunks of N
      Kld)     kld=1 ;;                                                # --kld-record over the corpus instead of serving
      Fl)      env+=(RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1) ;;       # counts + CPU rows in one ranged copy
      Ms[0-9]) env+=(RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=${p#Ms}) ;;  # a lone decode miss's mirror tail in N pieces
      Mp)      env+=(RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1) ;;       # miss reads start at the choice, before the pool's wake
      Fw)      fw=1; env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1) ;;   # pool workers sleep on a futex, woken without a lock
      Wp[0-9]*) # pre-wake N pool workers before a decode layer's CPU work (router event, miss read)
               if [[ ${p#Wp} =~ ^[0-9]+$ ]] && (( 10#${p#Wp} >= 1 && 10#${p#Wp} <= 256 )); then
                 wp=$((10#${p#Wp})); env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=$wp)
               else log "arm $a: $p: a pre-wake takes 1-256 workers; the arm is skipped"; skip=1; fi ;;
      U?*)     ARM_PROMPTS=/home/as/ds41-claude-build/work/extra-prompts/${p#U}-receipt.json ARM_N=999
               [[ -f $ARM_PROMPTS ]] || { log "arm $a: no prompts $ARM_PROMPTS; the arm is skipped"; skip=1; } ;;
      Rt)      mkdir -p "$OUT/rtrace-$a"   # the routing trace, exact (rtrace.py reads it)
               rt=1; env+=(RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=$(cd "$OUT" && pwd)/rtrace-$a/rt) ;;
      Lq)      ARM_LQ=1 ;;                 # the quality harness after the timed requests
      Cs)      cs=1 ;;                     # the exact consumer-suffix skip (checked in the server log)
      Kh)      kh=1 ;;                     # KV and logits hashes at each prompt's end
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
  # Every other arm states the switch off, so a window's VAR=val cannot pack its experts (arm() checks
  # the server log both ways).
  if [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1 "* ]]; then ARM_VP=1
  else ARM_VP=; env+=(RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=0); fi
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
  # Pu / Pu2: a partly read unit's rest in its layer's own read only in an arm whose name says so,
  # whatever the window's environment (0: the urgent pieces, as before); arm() checks the log both
  # ways. Without a PpK part there is nothing to finish, and the arm would not test it.
  if [[ -n "$pu" ]]; then
    if [[ "-$a-" != *-Pp[0-9]* ]]; then log "arm $a: Pu${pu#1} needs a PpK part (read-ahead in pieces); the arm is skipped"; skip=1; fi
    [[ "$pu" == 2 && "-$a-" != *-Rd[0-9]* ]] && { log "arm $a: Pu2 needs the decode mirror (R and RdNN); the arm is skipped"; skip=1; }
  else
    env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=0)
  fi
  ARM_PU=$pu
  # Ss: the read-back runs in an arm only if its name says so, whatever the window's environment.
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=0)
  [[ -z "$rt" ]] && env+=(RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=)   # traced only with Rt
  if [[ -n "$cs" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1); else env+=(RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=0); fi
  ARM_CS=$cs
  if [[ -n "$kh" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1); else env+=(RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=0); fi
  # The prefix cache's parts: PciN, PcsN and Pcx act only with the cache on (Pc or Pc0); without it
  # the arm would run cold under their names. Every arm states both prefix switches, so a window's
  # VAR=val cannot turn a cold arm or the Pc0 control into something its name does not say.
  if [[ -n "$pcopt" && -z "$pc" ]]; then
    log "arm $a:$pcopt without Pc or Pc0 (the prefix cache stays off); the arm is skipped"; skip=1
  fi
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=0)
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=0)
  # MrN: arms without it take jobs of the default 4 rows, whatever the window's environment (0 is
  # not this switch's off: any value outside 1-32 turns host experts off).
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS="* ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=4)
  # Dk / Dkt / DbN act on the DSpark drafter only: without a Ds part the arm would not test them.
  if [[ -n "$dk$db" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1 "* ]]; then
    log "arm $a: Dk / Dkt / DbN need a Ds part (the DSpark draft bank); the arm is skipped"; skip=1
  fi
  # Every arm states both switches (0 is off for both), so a window's VAR=val cannot turn them on;
  # arm() checks the logs both ways. A Dk arm runs only on the libr4d.so the draft check passed on.
  env+=(RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=${dk:-0} RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=${db:-0})
  if [[ -n "$dk" && -z "$skip" ]]; then
    IFS=: read -r -a homes <<< "$ARM_HOME"
    lib=""
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
    if [[ -z "$DK_OK" ]]; then
      log "arm $a: Dk refused: $DK_WHY"; skip=1
    elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$DK_SUM" ]]; then
      log "arm $a: Dk refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the draft check passed on ($DK_LIB, sha256 ${DK_SUM:0:16})"; skip=1
    fi
  fi
  ARM_DK=$dk ARM_DB=$db
  if [[ -n "$xt" && -z "$skip" ]]; then
    IFS=: read -r -a homes <<< "$ARM_HOME"
    lib=""
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
    if [[ -z "$TALL_OK" ]]; then
      log "arm $a: Xt refused: $TALL_WHY"; skip=1
    elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$TALL_SUM" ]]; then
      log "arm $a: Xt refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the tall checks passed on ($TALL_LIB, sha256 ${TALL_SUM:0:16})"; skip=1
    fi
  fi
  # Fw / WpN (the CPU pool's wake-up): every arm states both switches, 0 off, so a window's VAR=val
  # cannot turn them on for an arm whose name does not say so; arm() checks the server log both ways.
  # A pre-wake of more workers than the arm's pool has would be cut to its threads: refused instead.
  [[ -n "$fw" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=0)
  [[ -n "$wp" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=0)
  if [[ -n "$wp" && "$pol" != base ]] && (( wp > threads )); then
    log "arm $a: Wp$wp pre-wakes more workers than the pool's $threads threads; the arm is skipped"; skip=1
  fi
  if [[ -n "$fw$wp" && "$pol" == base ]]; then
    log "arm $a: Fw / WpN act on the host experts' pool, which a base arm has not; the arm is skipped"; skip=1
  fi
  ARM_FW=$fw ARM_WP=$wp
  # RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL: on only in an Xt arm, explicitly off in every other one.
  if [[ -n "$xt" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1); else env+=(RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=0); fi
  if [[ -n "$kld" ]]; then
    if [[ ! -s "$OUT/kld-corpus.jsonl" ]] &&
       ! python3 -I "$HERE/ab_requests.py" --kld-corpus "$OUT/kld-corpus.jsonl" "${KLD_MAX_CHARS:-6000}" \
           $(for r in ${KLD_RECEIPTS:-long chat}; do echo /home/as/ds41-claude-build/work/extra-prompts/$r-receipt.json; done) \
           >>"$OUT/window.log" 2>&1; then
      log "arm $a: no kld corpus; the arm is skipped"; skip=1
    fi
    rm -rf "$OUT/kld-$a"
    ARGS=(--max-model-len "${KLD_LEN:-2048}" "${ARGS[@]}" --kld-record "$OUT/kld-$a" --kld-corpus "$OUT/kld-corpus.jsonl")
  fi
  ARM_XT=$xt ARM_BT=$bt ARM_KLD=$kld
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
# Lq arms: each teacher-forced run against its reference (KL, top-1, by prompt family).
compgen -G "$OUT/*.lpforce.json" >/dev/null && python3 -I "$HERE/ab_requests.py" --kl-dir "$OUT" | tee -a "$OUT/window.log"
# Arms by their prompts (the U part; none: the held-out set). Where a window mixes prompt sets, each
# set's replies against its own first arm, and each set's Kh arms' prefill hashes against its first
# Kh arm's, prompt by prompt.
declare -A BY_PROMPTS=() KH_BY_PROMPTS=()
for a in $ARMS; do
  u=$(tr '-' '\n' <<< "$a" | grep -m1 -E '^U.+')
  BY_PROMPTS[${u:-held-out}]+="$a "
  [[ "-$a-" == *-Kh-* ]] && KH_BY_PROMPTS[${u:-held-out}]+="$a "
done
if (( ${#BY_PROMPTS[@]} > 1 )); then
  for u in "${!BY_PROMPTS[@]}"; do
    log "replies with prompts $u"
    python3 -I "$HERE/ab_requests.py" --compare "$OUT" ${BY_PROMPTS[$u]} | tee -a "$OUT/window.log"
  done
fi
for u in "${!KH_BY_PROMPTS[@]}"; do
  log "prefill hashes with prompts $u"
  python3 -I "$HERE/ab_requests.py" --hashes "$OUT" ${KH_BY_PROMPTS[$u]} | tee -a "$OUT/window.log"
done
# Dk / Dkt arms: the draft GEMV is exact, so the drafter proposes the same tokens. Each Dk arm's
# requests must have the draft_n and draft_n_accepted (and the texts) of the same arm without its
# Dk part, when the window ran it (K1 review finding 3).
for a in $ARMS; do
  [[ "-$a-" == *-Dk-* || "-$a-" == *-Dkt-* ]] || continue
  b=$(sed -E 's/-Dkt?(-|$)/\1/' <<< "$a")
  if [[ " $ARMS " != *" $b "* ]]; then log "draft stats: no arm $b in this window to hold $a to"; continue; fi
  log "draft stats: $a against $b"
  python3 -I "$HERE/ab_requests.py" --draft-stats "$OUT" "$b" "$a" | tee -a "$OUT/window.log"
done
[[ " $ARMS" == *-Pc* ]] && python3 -I "$HERE/ab_requests.py" --ttft "$OUT" $ARMS | tee -a "$OUT/window.log"
log "arms done"
