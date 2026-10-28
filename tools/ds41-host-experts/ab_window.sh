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
# Environment: RADIANCE_BIN (the binary), RADIANCE_HOME_DIR (kernels/ and architectures/ for it),
# SILU_TABLE (the card's silu table), ARMS (default "base verify greedy"), HX_THREADS (16),
# VERIFY_REQUESTS (3), DEVICE_CHECKS (1).
set -u
OUT=${1:?output directory}
BIN=${RADIANCE_BIN:?path to the radiance binary}
RHOME=${RADIANCE_HOME_DIR:?radiance home with kernels/ and architectures/}
SILU=${SILU_TABLE:?the card silu table}
ARMS=${ARMS:-"base verify greedy"}
HX_THREADS=${HX_THREADS:-16}
VERIFY_REQUESTS=${VERIFY_REQUESTS:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/as/radiance-setup/research/deepseek-v41-1017-20261024
MODEL=$R/native-models/ds41-native-mxfp4-v3-20261027T121600Z/model.rad
PROFILE=$R/ds41-dev-window-profile-calibration-20261028T034300Z/sealed-decode-profile.json
PROMPTS=$R/ds41-pack-tail-dev-window-tail1-heldout-20261028T085400Z/receipt.json
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

restore() {
  stop_server
  log "restoring $NAME"
  "${DOCKER[@]}" start "$NAME" >>"$OUT/window.log" 2>&1
  local t=0
  until curl -s -m 3 "$URL/stats" >/dev/null 2>&1; do
    sleep 5; t=$((t + 5))
    if (( t > 1200 )); then log "QFN did not come back within 20 min"; return 1; fi
  done
  local r; r=$(curl -s -m 120 "$URL/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"qwen-vllm","prompt":"The capital of France is","max_tokens":4,"temperature":0}')
  log "QFN up after ${t}s; completion: $(echo "$r" | head -c 300)"
  "${DOCKER[@]}" inspect "$NAME" --format '{{.Id}} {{.State.Status}} {{.State.StartedAt}} {{.Config.Image}}' >>"$OUT/window.log" 2>&1
}

arm() {   # name, then env assignments for the server
  local name=$1; shift
  local n=8; [[ "$name" == verify ]] && n=$VERIFY_REQUESTS
  log "arm $name: $*"
  env "$@" RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS=1 RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1 \
      RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2=1 \
    "$BIN" --radiance-home "$RHOME" --kernels libr4d,libref --model "$MODEL" \
      --served-model-name ds41-native-smoke --tp 4 --tp-wire exact --placement expert_tiered \
      --weights-disk-tier --expert-ram-cache-total-mib 81920 --host-pool-mib 4096 \
      --host-headroom-mib 8192 --gpu-headroom-mib 256 --embedding-placement vram \
      --vision-tower-placement vram --ngram-placement disk --max-model-len 512 \
      --max-num-batched-tokens 16 --max-num-seqs 1 --kv-cache-dtype bf16 \
      --num-speculative-tokens 0 --spec-ngram 0 --no-prefix-cache --deterministic \
      --host 127.0.0.1 --port $PORT --expert-profile "$PROFILE" >"$OUT/$name.server.log" 2>&1 &
  SERVER_PID=$!
  local t=0
  until curl -s -m 3 "http://127.0.0.1:$PORT/stats" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "arm $name: server exited during startup"; SERVER_PID=; return 1; fi
    sleep 2; t=$((t + 2))
    if (( t > 600 )); then log "arm $name: no server after 600 s"; stop_server; return 1; fi
  done
  log "arm $name: serving after ${t}s"
  python3 -I "$HERE/ab_requests.py" "$PROMPTS" "http://127.0.0.1:$PORT" "$n" "$OUT/$name.json" >>"$OUT/window.log" 2>&1
  log "arm $name: requests exit $?"
  stop_server
  grep -E 'host experts|EP scoped|error|ERR' "$OUT/$name.server.log" | tail -12 | tee -a "$OUT/window.log"
}

log "waiting for 30 s of QFN idleness"
if ! idle_for 30 600; then log "QFN not idle within 10 min; no window"; exit 3; fi
"${DOCKER[@]}" inspect "$NAME" --format 'before: {{.Id}} {{.State.Status}} {{.State.StartedAt}} {{.Config.Image}}' >>"$OUT/window.log" 2>&1
trap restore EXIT
log "stopping $NAME"
"${DOCKER[@]}" stop -t 60 "$NAME" >>"$OUT/window.log" 2>&1
sleep 5

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
for a in $ARMS; do
  case $a in
    base)   arm base RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=0 ;;
    verify) arm verify "${HX[@]}" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=verify ;;
    greedy) arm greedy "${HX[@]}" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=greedy ;;
    all)    arm all "${HX[@]}" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=all ;;
    *)      log "unknown arm $a" ;;
  esac
done
python3 -I "$HERE/ab_requests.py" --compare "$OUT" $ARMS | tee -a "$OUT/window.log"
log "arms done"
