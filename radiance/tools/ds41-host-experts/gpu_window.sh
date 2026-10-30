#!/usr/bin/env bash
# One GPU window for the ds41 host-expert and GEMV checks. Run it under both research locks,
# compiler first, as the research controllers take them:
#   flock -w 3600 research-compiler.lock flock -w 600 research-heavy-runtime.lock gpu_window.sh OUTDIR
# Waits for 30 s of continuous QFN idleness, stops the QFN container, runs the checks on card 0,
# and always restarts the same container and checks it answers before returning.
set -u
OUT=${1:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-pack-tail-dev-window-tail1-heldout-20261028T085400Z/native-variant-home/kernels/libr4d.so
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

restore() {
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

log "waiting for 30 s of QFN idleness"
if ! idle_for 30 600; then log "QFN not idle within 10 min; no window"; exit 3; fi
"${DOCKER[@]}" inspect "$NAME" --format 'before: {{.Id}} {{.State.Status}} {{.State.StartedAt}} {{.Config.Image}}' >>"$OUT/window.log" 2>&1
trap restore EXIT
log "stopping $NAME"
"${DOCKER[@]}" stop -t 60 "$NAME" >>"$OUT/window.log" 2>&1
sleep 5
rocm-smi --showmeminfo vram >>"$OUT/window.log" 2>&1

cd "$HERE"
run() { local name=$1; shift; log "run $name: $*"; ( timeout 1200 "$@" ) >"$OUT/$name.log" 2>&1; log "$name exit $?"; tail -4 "$OUT/$name.log" | tee -a "$OUT/window.log"; }
export HIP_VISIBLE_DEVICES=0
run cpu_bench ./ds41_host_experts_test bench 32
run swiglu_proof build/ds41_mxfp4_device_check "$PLUGIN" swiglu "$OUT/silu_table_gfx1201.bin"
run chain_check build/ds41_mxfp4_device_check "$PLUGIN" check 24 "$OUT/silu_table_gfx1201.bin"
run gemv_spread build/ds41_mxfp4_device_check build/libr4d_mxfp4_base.so time build/libr4d_mxfp4_spread.so
run gemv_plugin_vs_base build/ds41_mxfp4_device_check "$PLUGIN" time build/libr4d_mxfp4_base.so
log "checks done"
