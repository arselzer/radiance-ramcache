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
# Harness-grade prefix reuse (each needs Pc or Pc0, or the arm is skipped; every arm without one
# runs with its switch off, and is not measured if its server says otherwise): Pce end-of-request
# snapshots (RADIANCE_EXPERIMENTAL_PREFIX_EORS: a decode step completing a position on the resume
# grid past its prompt snapshots the state there, so a conversation's next turn resumes at most 15
# tokens before the reply's end; refused, and the arm skipped, with Pcx or speculation; skipped
# with Xt, which no Kh run has yet covered end to end, unless the window sets PCE_XT=1 for that
# Kh run); PcgN those snapshots on an N-token grid (RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=N: under
# large-chunk prefill the resume grid is the chunk, so Pce alone snapshots a reply every C tokens
# and the next turn re-prefills up to C - 1 of them; with PcgN, N a multiple of 16 that divides C,
# every N tokens past the prompt, so the next turn resumes at most N - 1 tokens before the reply's
# end, off the cold run's chunk grid, exact by ds41's partition invariance under LCP: gate it with
# Kh against the cold arm; it needs Pce and an Lc part, or the arm is skipped); Pcl the
# snapshot store gives up the least recently used conversation first (..._PREFIX_SESSION_LRU); PcbN
# the store as a byte budget of N MiB (..._CHECKPOINT_STORE_MIB, auto otherwise; not with PcsN);
# PcmN --prefix-cache-mib N (the host KV tier: a finished conversation's KV copied to pinned host
# memory and restored on its next turn); Pcr the compressed and index key caches declared
# replicated, so a host-tier entry holds one copy rather than one a rank
# (RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED; a deepseek41 plugin from claude/ds41-eors, run home
# home-eors). The harness receipt is Uharness (work/extra-prompts/harness-receipt.json: two
# interleaved agent sessions, a system prompt of 2-3K tokens and tool outputs of public texts from
# the repository, 6 and 7 turns, final contexts of about 8K and 10K tokens) at Lm32768, whose pool
# holds both without a trim; Uharness1 is its first session alone. Each Pc* arm's startup and
# shutdown lines for these go into window.log, and --ttft at the end reads per-turn TTFT, the
# tokens the cache supplied and identity against the first arm (the cold one), with Kh arms'
# prefill hashes compared prompt by prompt. Those two gate a Pce arm (--hashes: every prompt
# identical; --ttft: every text same); its shutdown line's admissions resumed and each turn's cached
# tokens are advisory, since they move with reply lengths and with how a reply re-tokenizes.
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
# Large-chunk prefill (RADIANCE_EXPERIMENTAL_DS41_LCP=C): LcN sets it and --max-num-batched-tokens N
# (N a multiple of 16 in 64..2048), e.g. ...-M1024-...-B1-...-Lc1024-Mr32; it needs MN (..._MISSES at
# least N), B1 (the CPU takes every routed unit) and an Mr part (Mr32 or Mr64; Mr4 only as a control),
# cannot go with Xt or BT, and leaves
# RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL unset (LCP keeps its pins; every other arm runs with LCP=0).
# AdN the RAM cache admits a chunk's units from the prompt's last N tokens only (..._LCP_ADMIT=N; 0
# none, the chunks read-only), Ada every unit (as at 16-token passes; 128 without either); LpN
# ..._LCP_PLANES (missed units read at once a layer, 48); LiN ..._LCP_INSERTQ (admitted misses queued
# for the cache, 16). An Lc arm must log "large-chunk prefill: passes of up to N tokens" at startup and
# a plain one must not, or it is not measured; an Lc arm is never the Kld reference. `lcp` in
# DEVICE_CHECKS (a list) runs ds41_mxfp4_device_check `lcp LCP_CHECK_C` (1024), `sweep` under LCP and
# `sweep0` on card 0 (TALL_LIBR4D and TALL_CHECK named, as for `tall`), and an Lc arm runs only on the
# libr4d.so (sha256) those passed on, at a chunk of at most LCP_CHECK_C.
#
# MbN: RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=N (libr4d, read by the routed MXFP4 gemv): 1 every call of the
# plain gemv kernel (the decode band to 16 tokens and the twin above), 2 the twin above 16 tokens only;
# the work items follow each expert's row passes, and a pass of dropped slots (the other ranks' rows,
# which mx_range hands the rank's last expert) writes zeros without reading weights. Same bytes by
# construction. Every arm without the part runs with it at 0. `bal` in DEVICE_CHECKS runs
# ds41_mxfp4_device_check `bal LCP_CHECK_C [BAL_ROUTES]` on card 0 (TALL_CHECK and TALL_LIBR4D, as for
# `lcp`): the balanced kernel against the plain one, every output value, and both timed (devcheck-bal.log);
# an Mb arm runs only on the libr4d.so (sha256) that passed it. Lf: RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=1,
# the file stager's per-pass-size time lines at exit (an instrument; no change to what runs).
#
# P1 (prefill Phase 1, RADIANCE_EXPERIMENTAL_DS41_P1=1; core/place/card_expert_stager.h): with an LcN part,
# every routed expert row of a pure prefill pass's routed layer of at least the threshold's rows on the
# cards (residents in place, RAM-cache units, units pinned in the host pool and drive misses through a VRAM
# ring of the file buffers, waves of a few units); a shorter layer pass takes the CPU path exactly as an Lc
# arm without P1 does. Needs BN with N >= 2 (the ring: B6, 12 slots, is the measured one) and an M part of
# min(N_Lc, threshold - 1) or more (the CPU path takes the layer passes below the threshold: M319 at the
# default, M16 at Qm17); the arm's Mr part serves those passes. QmN the threshold (..._P1_MIN_ROWS, 17-4096,
# 320; Qm17 is the card path for every pass above 16 tokens, as before the threshold), QwN units a wave
# (..._P1_WAVE, 2), QiN drive units in flight a rank (..._P1_INFLIGHT, 16), QrN pinned ready slots a rank
# (..._P1_READY, 6), QaN units read ahead (..._P1_AHEAD, 8; Qa0 off), QvN expert-rows a layer recomputed on
# the host (..._P1_VERIFY, 0), Qs chunks aligned to the start (..._P1_ALIGN=start; end by default), QlK the
# lent ring: K of the ring's slots (at most 2N - 2: all but the two the file buffers' alias bases name, B1's
# share) hold the rank's hottest file-tier units as decode residents between the card path's passes, taken
# back by a pass that arms it and reloaded in the background QhMS ms after its last one (..._P1_LEND, 0 off;
# ..._P1_GRACE_MS, 1000; Qh needs Ql or Qe; Q[gnc] stays the power settings), so B6-Ql10 gives decode B1's
# residents; QeK the lent arena: up to K slots of the activation arena's top, above its 256-row level (the
# deepseek41 plugin must come from a tree that declares at a smaller step beside the switch), hold the next
# hottest units the same way, taken back by every step at the full plan (..._P1_LEND_ARENA, 0 off, 1-64; Qe64
# takes what fits: ~42 slots a card at Lc2048, ~18 at Lc1024); with Ql or Qe the CPU path takes every pass
# that does not arm P1, so the M part is N_Lc or more; each needs P1. Every other arm states
# RADIANCE_EXPERIMENTAL_DS41_P1=0 and the Q knobs empty (..._P1_LEND and ..._P1_LEND_ARENA 0). A P1 arm is
# measured only if the server logs "ds41 p1: card experts on" at startup with its threshold and chunk ("a
# pure prefill pass's layer of N to C rows") (and a plain arm only without it), a Ql arm "ds41 p1: lent
# ring: K of the ring's", a Qe arm "ds41 p1: lent arena: N slots (..._P1_LEND_ARENA=K;" (and any other
# arm neither), and runs only on the libr4d.so
# (sha256) on which `p1` in DEVICE_CHECKS passed (ds41_mxfp4_device_check TALL_LIBR4D p1 P1_CHECK_M
# P1_ROUTES on card 0, "p1: PASS" its last line; P1_CHECK_M 4096 by default, P1_ROUTES a p0_tables.py routes
# file, none by default); an Lc2048 arm needs LCP_CHECK_C=2048 as before. The window log takes each P1 arm's
# "ds41 p1 stats, rank" and "ds41 p1 chunk:" lines; a P1 arm whose stats lines count a lease held at a pass
# end, or (Qv) a host-verified expert-row that differs from the card's, is not measured.
#
# Environment: RADIANCE_BIN (the binary), RADIANCE_HOME_DIR (kernels/ and architectures/ for it),
# SILU_TABLE (the card's silu table), ARMS (default "base verify greedy"), HX_THREADS (16),
# VERIFY_REQUESTS (3), DEVICE_CHECKS (1; a list: `swiglu` the default checks, `tall` the exact-tall checks, `lcp` the large-chunk
# ones at LCP_CHECK_C, both with TALL_LIBR4D and TALL_CHECK),
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
# Dv arms' hot set (draft_hotset.py write; RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET); no default. An arm
# takes the first of DRAFT_HOTSET_<set>_db<K>, DRAFT_HOTSET_<set>, DRAFT_HOTSET_db<K>, DRAFT_HOTSET that
# is set, <set> its U part's name (heldout without one) and K its DbK part's (e.g. DRAFT_HOTSET_chat: the
# held-out capture's hot set for the chat prompts and DRAFT_HOTSET_heldout the chat capture's, so neither
# set is scored on its own capture; DRAFT_HOTSET_db2 one taken with the short block the Db2 arms route).
DRAFT_HOTSET=${DRAFT_HOTSET:-}
# Dc arms: DSpark's confidence head as tools/ds41-host-experts/dspark_conf_extract.py writes it
# (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD); no default.
DRAFT_CONF_HEAD=${DRAFT_CONF_HEAD:-}
hotset_for() {   # an arm's name: its hot-set file (above), or empty
  local a=$1 u k v
  u=$(tr '-' '\n' <<< "$a" | grep -m1 -E '^U.+'); u=${u#U}; u=${u:-heldout}
  k=$(tr '-' '\n' <<< "$a" | grep -m1 -E '^Db[0-9]+$'); k=${k#Db}
  for v in ${k:+DRAFT_HOTSET_${u}_db$k} DRAFT_HOTSET_$u ${k:+DRAFT_HOTSET_db$k} DRAFT_HOTSET; do
    [[ $v =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -n "${!v:-}" ]] && { echo "${!v}"; return; }
  done
}
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
  lcp_check "$name"
}

# RADIANCE_EXPERIMENTAL_DS41_LCP (the LcN part): the file stager says at startup that it took passes of
# up to N tokens (after checking every op's bands, the CPU path and the switches beside it; it refuses to
# start otherwise), and the scheduler the chunks and the cache admission. An Lc arm without that line,
# or a plain arm with it, is not measured.
lcp_check() {   # name
  local name=$1 lc_lines p1_lines
  lc_lines=$(grep -c 'large-chunk prefill: passes of up to [0-9]* tokens' "$OUT/$name.server.log")
  if [[ -n "${ARM_LC:-}" ]] && ! grep -q "large-chunk prefill: passes of up to ${ARM_LC} tokens" "$OUT/$name.server.log"; then
    log "arm $name: Lc${ARM_LC}, but the server log has no 'large-chunk prefill: passes of up to ${ARM_LC} tokens' line ($(grep -m2 -E 'large-chunk prefill' "$OUT/$name.server.log" | tr '\n' ' ')); not measured"
    return 1
  fi
  if [[ -z "${ARM_LC:-}" && "$lc_lines" != 0 ]]; then
    log "arm $name: no Lc part, but the server runs large-chunk prefill (the knob set for the window?); not measured"
    return 1
  fi
  if [[ -n "${ARM_LC:-}" ]]; then
    grep -m3 -E 'large-chunk prefill: |chunk geometry: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
    grep -m1 -E 'consumer skip: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  fi
  # P1 (RADIANCE_EXPERIMENTAL_DS41_P1): rank 0's "ds41 p1: card experts on" line at startup, in a P1 arm
  # only.
  p1_lines=$(grep -c 'ds41 p1: card experts on' "$OUT/$name.server.log")
  if [[ -n "${ARM_P1:-}" && "$p1_lines" == 0 ]]; then
    log "arm $name: P1, but the server log has no 'ds41 p1: card experts on' line ($(grep -m2 -E 'ds41 p1|P1' "$OUT/$name.server.log" | tr '\n' ' ')); not measured"
    return 1
  fi
  if [[ -z "${ARM_P1:-}" && "$p1_lines" != 0 ]]; then
    log "arm $name: no P1 part, but the server runs the card path (the knob set for the window?); not measured"
    return 1
  fi
  # ...and the line states the arm's threshold (QmN, 320 without one) and chunk.
  if [[ -n "${ARM_P1:-}" ]] &&
     ! grep -q "ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of ${ARM_QM:-320} to ${ARM_LC} rows" "$OUT/$name.server.log"; then
    log "arm $name: P1, but the card path's line does not state layer passes of ${ARM_QM:-320} to ${ARM_LC} rows ($(grep -m1 'ds41 p1: card experts on' "$OUT/$name.server.log")); not measured"
    return 1
  fi
  # QlK: rank 0's "ds41 p1: lent ring: K of the ring's" line at startup, in a Ql arm only.
  local lend_lines
  lend_lines=$(grep -c 'ds41 p1: lent ring: ' "$OUT/$name.server.log")
  if [[ -n "${ARM_QL:-}" ]] && ! grep -q "ds41 p1: lent ring: ${ARM_QL} of the ring's" "$OUT/$name.server.log"; then
    log "arm $name: Ql${ARM_QL}, but the server log has no 'ds41 p1: lent ring: ${ARM_QL} of the ring's' line ($(grep -m1 'ds41 p1: lent ring' "$OUT/$name.server.log")); not measured"
    return 1
  fi
  if [[ -z "${ARM_QL:-}" && "$lend_lines" != 0 ]]; then
    log "arm $name: no Ql part, but the server lends the ring (the knob set for the window?); not measured"
    return 1
  fi
  # QeK: rank 0's "ds41 p1: lent arena: N slots (..._P1_LEND_ARENA=K;" line at startup, in a Qe arm only.
  local arena_lines
  arena_lines=$(grep -c 'ds41 p1: lent arena: ' "$OUT/$name.server.log")
  if [[ -n "${ARM_QE:-}" ]] &&
     ! grep -qE "ds41 p1: lent arena: [0-9]+ slots \(RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=${ARM_QE};" "$OUT/$name.server.log"; then
    log "arm $name: Qe${ARM_QE}, but the server log has no 'ds41 p1: lent arena: N slots (..._P1_LEND_ARENA=${ARM_QE};' line ($(grep -m1 -E 'ds41 p1: lent arena|LEND_ARENA' "$OUT/$name.server.log")); not measured"
    return 1
  fi
  if [[ -z "${ARM_QE:-}" && "$arena_lines" != 0 ]]; then
    log "arm $name: no Qe part, but the server lends the arena (the knob set for the window?); not measured"
    return 1
  fi
  [[ -n "${ARM_P1:-}" ]] && grep -m2 -E 'ds41 p1: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  [[ -n "${ARM_QE:-}" ]] && grep -m2 -E 'ds41 p1: lent arena: |activation arena: .*lent arena' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
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
  lcp_check "$name" || return 1
  local extra_why
  if ! extra_why=$(vp_server_gate "$OUT/$name.server.log" "${ARM_VP:-}" "$([[ " ${ARGS[*]} " == *" --draft-bank "* ]] && echo 1)") ||
     ! extra_why=$(qsp_gate "$OUT/$name.server.log" "${ARM_QSP:-}" exit); then
    log "arm $name: $extra_why; Kld not measured"; return 1
  fi
  if [[ -z "$KLD_REF" ]]; then
    # The reference must be a 16-token arm's bits: an Xt or Lc arm first (the plain one failed or is
    # missing) would let a later Xt or Lc arm be compared with another one.
    if [[ -n "${ARM_XT:-}${ARM_LC:-}" ]]; then
      log "arm $name: an Xt or Lc arm cannot be the window's kld reference (no plain Kld arm finished before it); not compared"
      return 1
    fi
    KLD_REF=$OUT/kld-$name
    log "arm $name: the window's kld reference"
    return 0
  fi
  python3 -I "$HERE/ab_requests.py" --kld-compare "$KLD_REF" "$OUT/kld-$name" 2>&1 | tee -a "$OUT/window.log"
}

# Qsp must be active on every rank and reduce actual wave work, rather than only enable a switch.
qsp_gate() {   # log, requested switch, start/exit
  local f=$1 on=$2 stage=${3:-start} rank count line spans rows full fallbacks
  if [[ -z "$on" ]]; then
    if grep -q 'wave SwiGLU spans' "$f"; then echo 'no Qsp part, but SwiGLU spans are active'; return 1; fi
    return 0
  fi
  for rank in 0 1 2 3; do
    count=$(grep -cE "ds41 p1: rank $rank's wave SwiGLU spans on" "$f")
    if [[ $count != 1 ]]; then echo "Qsp, but rank $rank lacks one startup marker"; return 1; fi
    [[ "$stage" == exit ]] || continue
    count=$(grep -cE "ds41 p1 stats, rank $rank: .*; wave SwiGLU spans " "$f")
    line=$(sed -nE "s/.*ds41 p1 stats, rank $rank: .*; wave SwiGLU spans ([0-9]+), rows ([0-9]+) of ([0-9]+), full-replay fallbacks ([0-9]+).*/\\1 \\2 \\3 \\4/p" "$f")
    read -r spans rows full fallbacks <<< "$line"
    if [[ $count != 1 || -z "$full" ]] || (( spans == 0 || rows >= full )); then
      echo "Qsp, but rank $rank has no demonstrated work reduction (${line:-no counters})"; return 1
    fi
  done
  return 0
}

# Vp names the switch and actual packed static residents on every rank.
# startup and each rank how many static residents it holds packed. A Vp arm without the switch's line or
# with no rank holding any packed (a radiance without the switch, a dynamic plan), or a plain arm with
# either line (the switch taken from the window's environment), is not measured. With a DSpark bank (a Ds
# part: --draft-bank) the trunk packs from the main container and the bank keeps its block form and its
# own GEMV path (claude/ds41-dsvp; an older radiance refused the switch with a bank, so no rank held any):
# a Vp arm with a bank also needs every rank's 'DSpark bank bound; N trunk weights packed' line, N above 0,
# the engine's check that no draft op took gemv_vp and no bank weight was packed.
# Arguments: the log, the arm's Vp (non-empty: on), its bank (non-empty: bound). 0 when the arm may be
# measured; otherwise 1, the reason on stdout.
vp_server_gate() {
  local f=$1 vp=$2 bank=$3 vp_on vp_held vp_bank rank
  vp_on=$(grep -c 'VRAM packed experts (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED)' "$f")
  vp_held=$(grep -cE 'holds [1-9][0-9]* static residents' "$f")
  vp_bank=$(grep -cE 'VRAM packed experts: rank [0-9]+: DSpark bank bound; [1-9][0-9]* trunk weights packed' "$f")
  if [[ -n "$vp" ]] && [[ "$vp_on" == 0 || "$vp_held" == 0 ]]; then
    echo "Vp, but the server log has $vp_on 'VRAM packed experts (RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED)' and $vp_held 'holds N static residents' lines with N above 0 ($(grep -m2 -E 'VRAM packed experts' "$f" | tr '\n' ' '))"
    return 1
  fi
  # A positive count on just one rank is insufficient, including P1's mixed reader bands.
  # Actual packed static residents prove eligibility and placement both occurred on each rank.
  if [[ -n "$vp" ]]; then
    for rank in 0 1 2 3; do
      if [[ $(grep -cE "VRAM packed experts: rank $rank holds [1-9][0-9]* static residents in the packed-scale form" "$f") != 1 ]]; then
        echo "Vp, but rank $rank did not log exactly one nonzero packed static-resident count"
        return 1
      fi
    done
  fi
  if [[ -n "$vp" && -n "$bank" && "$vp_bank" != 4 ]]; then
    echo "Vp with a DSpark bank, but $vp_bank of the 4 ranks logged 'DSpark bank bound; N trunk weights packed' with N above 0 ($(grep -m2 -E 'VRAM packed experts: (rank [0-9]+: DSpark|a DSpark)' "$f" | tr '\n' ' '))"
    return 1
  fi
  if [[ -z "$vp" ]] && [[ "$vp_on" != 0 || "$vp_held" != 0 ]]; then
    echo "no Vp part, but the server held experts packed (the switch set for the window?)"
    return 1
  fi
  return 0
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
    grep -E 'host headroom|chunk geometry|prefix resume|prefix cache|checkpoint slots|end-of-request snapshots|snapshot store|kv tiers' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # The harness parts (Pce, PcgN, Pcl, PcbN, PcmN, Pcr): each switch's startup line in an arm that
  # names it, and none in an arm that does not (a switch left set for the window), or the arm is not
  # measured.
  local hp_bad=""
  hp_check() {   # part, set in this arm (1 or empty), the line that says it took effect
    local n; n=$(grep -cE "$3" "$OUT/$name.server.log")
    if [[ -n "$2" && "$n" == 0 ]]; then hp_bad+=" $1 asked, but no '$3' line;"; fi
    if [[ -z "$2" && "$n" != 0 ]]; then hp_bad+=" no $1 part, but a '$3' line;"; fi
  }
  hp_check Pce "${ARM_EORS:-}" 'end-of-request snapshots: on'
  hp_check PcgN "${ARM_RGRID:-}" 'end-of-request snapshots: on a [0-9]+-token grid'
  hp_check Pcl "${ARM_SLRU:-}" 'snapshot store: by session'
  hp_check PcbN "${ARM_STOREMIB:-}" 'checkpoint slots: a byte budget'
  hp_check PcmN "${ARM_TIER:-}" "kv tiers: a finished conversation's KV is copied"
  hp_check Pcr "${ARM_KVREP:-}" 'KV_REPLICATED: [0-9]+ paged group'
  if [[ -n "$hp_bad" ]]; then
    log "arm $name: not measured:$hp_bad $(grep -m2 -E 'end-of-request snapshots: refused|RESUME_GRID=[^ ]* refused|KV_REPLICATED refused' "$OUT/$name.server.log" | tr '\n' ' ')"
    stop_server; return 1
  fi
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
  # Pi8 / Pi4, PkN, Pw, NsN, Pv: each switch's startup line, and an arm must have exactly what its name
  # says (a radiance without the switch prints none; a plain arm with one took it from elsewhere).
  # A row a switch: the part, set when the arm has it, the line's pattern.
  local sw_bad="" sw_part sw_on sw_pat sw_n pi_n=${ARM_PSC:+${ARM_PRS:-12}}
  pi_n=${pi_n:-[0-9]+}   # a Pi arm's rescore count; any, for the line no other arm may have
  while IFS='|' read -r sw_part sw_on sw_pat; do
    sw_n=$(grep -cE "$sw_pat" "$OUT/$name.server.log")
    if [[ -n "$sw_on" && "$sw_n" == 0 ]]; then sw_bad+=" $sw_part, but no '$sw_pat' line;"
    elif [[ -z "$sw_on" && "$sw_n" != 0 ]]; then sw_bad+=" no $sw_part part, but a '$sw_pat' line (the switch set for the window?);"; fi
  done <<EOF2
Pi${ARM_PSC:-}|${ARM_PSC:-}|router predictor: [0-9]+ routers loaded \(bf16, [0-9]+ MB\); scored in int${ARM_PSC:-[48]} \([^)]*\), the best $pi_n scored again in bf16
Pk${ARM_PK:-}|${ARM_PK:-}|router predictor: .*one prediction in [0-9]+ checked against bf16 alone
Pw|${ARM_PW:-}|rank 0 posts the read-ahead's router input once a layer's CPU work is submitted
Ns${ARM_NS:-}|${ARM_NS:-}|side store: the planes of up to [0-9]+ units a rank
Pv|${ARM_PV:-}|router predictor: every posted router input is recorded to
Tb|${ARM_TB:-}|two-batch overlap \(RADIANCE_EXPERIMENTAL_SCHED_TBO=1\): a decode step
EOF2
  if [[ -n "$sw_bad" ]]; then
    log "arm $name:$sw_bad not measured"
    stop_server; return 1
  fi
  grep -m1 -E 'router predictor: [0-9]+ routers loaded .*scored in int' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -m1 -E "side store: the planes of up to|rank 0 posts the read-ahead's router input" "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW (the Pmr parts): rank 0's file stager names it at
  # startup with its N. A Pmr arm without that line for its N (a radiance without the switch: decode
  # passes of 2+ rows go without the read-ahead), or another arm with the line, is not measured.
  local pmr_lines; pmr_lines=$(grep -cE 'read-ahead: decode passes of 2-[0-9]+ rows read ahead too' "$OUT/$name.server.log")
  if [[ -n "${ARM_PMR:-}" ]] && ! grep -qE "read-ahead: decode passes of 2-${ARM_PMR} rows read ahead too" "$OUT/$name.server.log"; then
    log "arm $name: Pmr${ARM_PMR}, but the server log has no 'read-ahead: decode passes of 2-${ARM_PMR} rows read ahead too' line ($(grep -m1 -E 'PREFETCH_MULTIROW' "$OUT/$name.server.log")); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_PMR:-}" && "$pmr_lines" != 0 ]]; then
    log "arm $name: no Pmr part, but the server reads ahead in multi-row decode passes (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_PMR:-}" ]] && grep -m1 -E 'read-ahead: decode passes of 2-' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED (the Vp part): vp_server_gate (above) on the server log.
  local vp_why
  if ! vp_why=$(vp_server_gate "$OUT/$name.server.log" "${ARM_VP:-}" "$([[ " ${ARGS[*]} " == *" --draft-bank "* ]] && echo 1)"); then
    log "arm $name: $vp_why; not measured"
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
  # A Cs arm with the prefix cache: the server states which snapshots below a cut prompt's target it
  # makes whole (claude/ds41-csckpt), and it must be what the arm's Ck part says (512 without one).
  # A radiance without the line (writes none below the target) is not measured as anything but Ck0.
  if [[ -n "${ARM_CK:-}" ]]; then
    local ck_ok=""
    if [[ "$ARM_CK" == 0 ]]; then
      grep -q 'consumer skip: on .*CONSUMER_SKIP_CKPT=0)' "$OUT/$name.server.log" && ck_ok=1
    else
      grep -q "consumer skip: complete checkpoints (RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=$ARM_CK)" "$OUT/$name.server.log" && ck_ok=1
    fi
    if [[ -z "$ck_ok" ]]; then
      log "arm $name: Cs with Pc wants ..._CONSUMER_SKIP_CKPT=$ARM_CK, but the server log does not state it ($(grep -m2 -E 'consumer skip: (on|complete)' "$OUT/$name.server.log" | cut -c1-200 | tr '\n' ' ')); not measured"
      stop_server; return 1
    fi
    grep -m1 -E 'consumer skip: complete checkpoints' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  fi
  xt_check "$name" || { stop_server; return 1; }
  local qsp_why
  if ! qsp_why=$(qsp_gate "$OUT/$name.server.log" "${ARM_QSP:-}" start); then
    log "arm $name: $qsp_why; not measured"; stop_server; return 1
  fi
  # In / Is (RADIANCE_EXPERIMENTAL_DS41_INSERT_NT / _SLICE_KIB) and Ph (_HOST_EXPERTS_LEAN_TILES):
  # rank 0 names what it took at startup. An arm whose line is missing or says otherwise (a radiance
  # without the switch, a cache that is not packed), or a plain arm with one, is not measured.
  local ins_line lean_lines want_ins=""
  ins_line=$(grep -m1 -E "background inserts write a missed unit's codes by non-temporal stores" "$OUT/$name.server.log")
  lean_lines=$(grep -c 'host experts: lean tiles:' "$OUT/$name.server.log")
  if [[ -n "${ARM_INS_KIB:-}" ]]; then want_ins="in pieces of ${ARM_INS_KIB} KiB; a one-token pass's insert sleeps before each piece while the pool has submitted jobs unfinished (not for a miss's reservation alone), at most ${ARM_INS_WAIT} us"
  elif [[ -n "${ARM_INS_NT:-}" ]]; then want_ins="by non-temporal stores (RADIANCE_EXPERIMENTAL_DS41_INSERT_NT)"; fi
  if [[ -n "$want_ins" && "$ins_line" != *"$want_ins"* ]]; then
    log "arm $name: In / Is, but the server log does not say it (${ins_line:-no 'background inserts write' line}); not measured"
    stop_server; return 1
  fi
  if [[ -z "$want_ins" && -n "$ins_line" ]]; then
    log "arm $name: no In / Is part, but the server writes its inserts by non-temporal stores (the knob set for the window?); not measured"
    stop_server; return 1
  fi
  if [[ -n "${ARM_LEAN:-}" && "$lean_lines" == 0 ]] || [[ -z "${ARM_LEAN:-}" && "$lean_lines" != 0 ]]; then
    log "arm $name: $([[ -n "${ARM_LEAN:-}" ]] && echo "a Ph part" || echo "no Ph part"), but the server log has $lean_lines 'lean tiles:' line(s); not measured"
    stop_server; return 1
  fi
  [[ -n "$ins_line" ]] && echo "$ins_line" | tee -a "$OUT/window.log"
  [[ -n "${ARM_LEAN:-}" ]] && grep -m1 'host experts: lean tiles:' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
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
  # WlN: the pool's pre-wake line names the miss-read trigger's lead (60 without the part; Wl0: none).
  if [[ -n "${ARM_WP:-}" ]]; then
    local want_lead="${ARM_WL:-60} us before a miss read's predicted end"
    [[ "${ARM_WL:-}" == 0 ]] && want_lead="not in miss reads"
    if ! grep -qF "before a one-token layer's CPU work, at its router and ${want_lead}" "$OUT/$name.server.log"; then
      log "arm $name: Wp${ARM_WP}${ARM_WL:+-Wl$ARM_WL}, but the pool's pre-wake line does not say '${want_lead}' ($(grep -m1 -E 'pre-wake of up to' "$OUT/$name.server.log")); not measured"
      stop_server; return 1
    fi
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
    # One a rank (--tp 4): a rank that did not take the switch would read its rests as before.
    if [[ "$pu_lines" != 4 || "$pu_lines" != "$pu_any" ]]; then
      log "arm $name: Pu${ARM_PU#1}, but the server log has $pu_lines 'reads the rest of a partly read unit ... PREFETCH_URGENT=${ARM_PU})' lines of $pu_any, not one on each of the 4 ranks ($(grep -m1 -E 'a partly read unit its layer needs is' "$OUT/$name.server.log")); not measured"
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
  # DvN (RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM): the engine names N and the hot set, every rank its final
  # placement (N hot pairs a stage static in VRAM, the rest parked) and its three masked correction
  # planes. A Dv arm without all of them, or another arm with any "draft VRAM" line, is not measured.
  local dv_on dv_place dv_mask dv_any
  dv_on=$(grep -c "draft VRAM: RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=${ARM_DV:-none} from" "$OUT/$name.server.log")
  dv_place=$(grep -cE "draft VRAM final placement: rank=[0-3] .*hot=${ARM_DV:-none}/${ARM_DV:-none}/${ARM_DV:-none} " "$OUT/$name.server.log")
  dv_mask=$(grep -cE 'draft VRAM: rank [0-3]: the router correction planes of stages 40/41/42 masked' "$OUT/$name.server.log")
  dv_any=$(grep -c 'draft VRAM' "$OUT/$name.server.log")
  if [[ -n "${ARM_DV:-}" ]] && [[ "$dv_on" == 0 || "$dv_place" != 4 || "$dv_mask" != 4 ]]; then
    log "arm $name: Dv${ARM_DV}, but the server log has $dv_on switch line(s), $dv_place of 4 'draft VRAM final placement' and $dv_mask of 4 masked-plane lines (a radiance without the switch?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_DV:-}" && "$dv_any" != 0 ]]; then
    log "arm $name: no Dv part, but the server placed the draft bank in VRAM (the switch set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_DV:-}" ]] && grep -E 'draft VRAM|draft-bank experts: RADIANCE' "$OUT/$name.server.log" | head -16 | tee -a "$OUT/window.log"
  # Xd (--expert-stats): the startup line counts the trunk's 40 routed layers and the drafter's 3; an
  # arm without Xd must not count anything.
  local xd_lines; xd_lines=$(grep -cE 'expert stats: [0-9]+ routed layers' "$OUT/$name.server.log")
  if [[ -n "${ARM_XD:-}" ]] && ! grep -qE 'expert stats: 43 routed layers' "$OUT/$name.server.log"; then
    log "arm $name: Xd, but the server log has no 'expert stats: 43 routed layers' line ($(grep -m1 'expert stats' "$OUT/$name.server.log")); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_XD:-}" && "$xd_lines" != 0 ]]; then
    log "arm $name: no Xd part, but the server counts expert selections (--expert-stats from the window?); not measured"
    stop_server; return 1
  fi
  # Dg (RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE): the scheduler says at startup that the acceptance
  # gate is on, with its threshold, window and probe. A Dg arm without that line (a radiance without
  # the switch), or another arm with it, is not measured.
  local dg_lines; dg_lines=$(grep -c 'DSpark acceptance gate: on' "$OUT/$name.server.log")
  if [[ -n "${ARM_DG:-}" && "$dg_lines" == 0 ]]; then
    log "arm $name: Dg, but the server log has no 'DSpark acceptance gate: on' line (a radiance without the switch?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_DG:-}" && "$dg_lines" != 0 ]]; then
    log "arm $name: no Dg part, but the scheduler ran the acceptance gate (the switch set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_DG:-}" ]] && grep -m1 'DSpark acceptance gate: on' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # Dc / Dl / Dx / Dm (draft_conf.h, planner.h): each says so at startup (Dc also its head, Dm a
  # placement line for each codebook); an arm with the part and without its line, or the other way
  # round, is not measured.
  local want got pat
  for pat in "Dc:DSpark confidence head: " "Dl:DSpark load-aware drafting: on" "Dx:DSpark context pass deferred (" \
             "Dm:placement: 'dspark.markov_w" "Dpc:DSpark read-only proposal capture: on" \
             "Ds with NK:DSpark draft bank with up to "; do
    want=${pat%%:*}; got=$(grep -cF "${pat#*:}" "$OUT/$name.server.log")
    local flag; case $want in Dc) flag=${ARM_DC:-};; Dl) flag=${ARM_DL:-};; Dx) flag=${ARM_DX:-};; Dm) flag=${ARM_DM:-};;
                              Dpc) flag=${ARM_DPC:-};; "Ds with NK") flag=${ARM_SOLO:-};; esac
    if [[ -n "$flag" && "$got" == 0 ]]; then
      log "arm $name: $want, but the server log has no '${pat#*:}' line (a radiance or plugin without the switch?); not measured"
      stop_server; return 1
    fi
    if [[ -z "$flag" && "$got" != 0 ]]; then
      log "arm $name: no $want part, but the server log has '${pat#*:}' (the switch set for the window?); not measured"
      stop_server; return 1
    fi
  done
  grep -E "DSpark (confidence controller|confidence head|load-aware drafting|context pass deferred) \(|DSpark confidence head: |DSpark read-only proposal capture: on|placement: 'dspark.markov_w|DSpark draft bank with up to " "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # Df (RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8): the plugin names, on each rank, the drafter's dense
  # weights it reads from E4M3 codes. A Df arm without such a line with N above 0, or another arm
  # with one, is not measured.
  local df_lines; df_lines=$(grep -cE 'DSpark drafter: [1-9][0-9]* dense weights read by gemm_nt_bf16f8' "$OUT/$name.server.log")
  if [[ -n "${ARM_DF:-}" && "$df_lines" == 0 ]]; then
    log "arm $name: Df, but the server log has no 'DSpark drafter: N dense weights read by gemm_nt_bf16f8' line (a plugin without the switch, or F8 off?); not measured"
    stop_server; return 1
  fi
  if [[ -z "${ARM_DF:-}" && "$df_lines" != 0 ]]; then
    log "arm $name: no Df part, but the drafter read its dense weights from E4M3 codes (the switch set for the window?); not measured"
    stop_server; return 1
  fi
  [[ -n "${ARM_DF:-}" ]] && grep -E 'DSpark drafter: [0-9]+ dense weights' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
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
  local qsp_exit_why
  if ! qsp_exit_why=$(qsp_gate "$OUT/$name.server.log" "${ARM_QSP:-}" exit); then
    log "arm $name: $qsp_exit_why; not measured (results kept as $name.unmeasured.json)"
    [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
  fi
  # P1 at exit: each rank's "ds41 p1 stats, rank" line counts the leases its passes still held at their
  # end (released there: a source not let go in order) and, with a Qv part, the expert-rows the host
  # recomputed and how many differ from the card's. A P1 arm with a lease at a pass end or a differing
  # row on any rank is not measured (results kept as <arm>.unmeasured.json), with the reason.
  if [[ -n "${ARM_P1:-}" ]]; then
    local p1_bad="" p1_ranks=0 p1_leases=0 p1_vrows=0 p1_differ=0 p1l pl pv pd
    while IFS= read -r p1l; do
      pl=$(sed -nE 's/.*leases at pass end ([0-9]+).*/\1/p' <<< "$p1l")
      pv=$(sed -nE 's/.*verify rows ([0-9]+), differ ([0-9]+).*/\1/p' <<< "$p1l")
      pd=$(sed -nE 's/.*verify rows ([0-9]+), differ ([0-9]+).*/\2/p' <<< "$p1l")
      p1_ranks=$((p1_ranks + 1)); p1_leases=$((p1_leases + ${pl:-0}))
      p1_vrows=$((p1_vrows + ${pv:-0})); p1_differ=$((p1_differ + ${pd:-0}))
      [[ -z "$pl" || -z "$pd" ]] && p1_bad="a 'ds41 p1 stats, rank' line without its leases or verify counts ($p1l)"
    done < <(grep -E 'ds41 p1 stats, rank [0-9]+: ' "$OUT/$name.server.log")
    (( p1_leases > 0 )) && p1_bad="${p1_bad:+$p1_bad; }$p1_leases leases still held at a pass end over $p1_ranks ranks"
    (( p1_differ > 0 )) && p1_bad="${p1_bad:+$p1_bad; }the host verify: $p1_differ of $p1_vrows expert-rows differ from the card's over $p1_ranks ranks"
    if [[ -n "$p1_bad" ]]; then
      log "arm $name: P1, but $p1_bad; not measured (results kept as $name.unmeasured.json)"
      [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
    fi
  fi
  # Pmr with a drafting Ds part (DSpark's 2+ row verify passes) at shutdown: the predictor's
  # multi-row line, and some rank's "by pass size" line with a "2 rows:" entry that planned units;
  # otherwise the switch never acted on a verify pass and the arm is set aside (mrra review B3).
  if [[ -n "${ARM_PMR:-}" && "-$name-" =~ -Ds([1-9]|a)[0-9]*- ]]; then
    local pmr_bad="" pmr2
    grep -q 'router predictor, multi-row decode passes: [1-9]' "$OUT/$name.server.log" ||
      pmr_bad="no 'router predictor, multi-row decode passes' line with posts"
    # Any pass of 2..9 rows: a DsK arm's verify passes have K + 1 rows.
    pmr2=$(grep -oE '[2-9] rows: [0-9]+ planned' "$OUT/$name.server.log" | grep -oE '[0-9]+ planned' | grep -oE '^[0-9]+' | awk '{s += $1} END {print s + 0}')
    [[ -z "$pmr_bad" && "$pmr2" == 0 ]] && pmr_bad="no rank planned a unit in a pass of 2-9 rows ('K rows: N planned')"
    if [[ -n "$pmr_bad" ]]; then
      log "arm $name: Pmr${ARM_PMR}, but $pmr_bad; not measured (results kept as $name.unmeasured.json)"
      [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
    fi
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
    local pu_rest pu_mir pu_bad=""
    pu_rest=$(grep -oE "PREFETCH_URGENT=${ARM_PU}\): [0-9]+ units' rest" "$OUT/$name.server.log" | grep -oE ': [0-9]+' | tr -d ': ' | awk '{s += $1} END {print s + 0}')
    # Pu2's mirror share as a count of rests: the MB, even with one decimal, can round to 0 in a
    # verify arm, whose rests fall mostly in the down extent, which has no mirror part.
    pu_mir=$(grep -E "PREFETCH_URGENT=${ARM_PU}\): [0-9]+ units' rest" "$OUT/$name.server.log" | grep -oE "[0-9]+ units' rests with a part from it" | grep -oE '^[0-9]+' | awk '{s += $1} END {print s + 0}')
    [[ "$pu_rest" == 0 ]] && pu_bad="no rank's shutdown line counts a unit's rest read in its layer's own call"
    [[ -z "$pu_bad" && "$ARM_PU" == 2 && "$pu_mir" == 0 ]] && pu_bad="no rank read any of a rest from the plane mirror"
    if [[ -n "$pu_bad" ]]; then
      log "arm $name: Pu${ARM_PU#1}, but $pu_bad; not measured (results kept as $name.unmeasured.json)"
      [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
    fi
  fi
  # In / Is / Ph at shutdown: every rank's "background inserts" line counts the pieces its inserts
  # wrote by non-temporal stores (an In or Is arm: above 0 on every rank; a plain arm: 0; an Is arm's
  # pieces after a wait are only noted: one-token passes' inserts wait, but a DSpark verify pass or a
  # batched step never does, and a large cache leaves few inserts), and the pool's "lean tiles" line the tiles claimed and the
  # lanes reused (Ph: both above 0; a plain arm has no such line). Otherwise not measured.
  local ins_bad="" ins_ranks=0 ins_zero=0 ins_pieces=0 ins_waited=0 l np nw lean_line
  while IFS= read -r l; do
    np=$(sed -nE 's/.*non-temporal: ([0-9]+) pieces.*/\1/p' <<< "$l")
    nw=$(sed -nE 's/.*waited before ([0-9]+) pieces.*/\1/p' <<< "$l")
    ins_ranks=$((ins_ranks + 1)); ins_pieces=$((ins_pieces + ${np:-0})); ins_waited=$((ins_waited + ${nw:-0}))
    (( ${np:-0} == 0 )) && ins_zero=$((ins_zero + 1))
  done < <(grep -E "host experts' background inserts: " "$OUT/$name.server.log")
  if [[ -n "${ARM_INS_NT:-}${ARM_INS_KIB:-}" ]]; then
    (( ins_ranks == 0 || ins_zero > 0 )) && ins_bad="$ins_zero of $ins_ranks ranks' 'background inserts' lines show no non-temporal piece"
    [[ -n "${ARM_INS_KIB:-}" ]] && (( ins_waited == 0 )) && log "arm $name: Is, but no piece waited for an idle pool (passes of several rows, or few inserts)"
  elif (( ins_pieces > 0 )); then
    ins_bad="no In / Is part, but $ins_pieces pieces went by non-temporal stores"
  fi
  lean_line=$(grep -m1 -E 'host expert pool, lean tiles: ' "$OUT/$name.server.log")
  if [[ -n "${ARM_LEAN:-}" ]]; then
    [[ "$lean_line" =~ lean\ tiles:\ ([0-9]+)\ tiles\ claimed.*reused\ on\ ([0-9]+) ]] &&
      (( BASH_REMATCH[1] > 0 && BASH_REMATCH[2] > 0 )) ||
      ins_bad="${ins_bad:+$ins_bad; }Ph, but the pool's lean tiles line is ${lean_line:-missing} (no tile claimed or no lanes reused)"
  elif [[ -n "$lean_line" ]]; then
    ins_bad="${ins_bad:+$ins_bad; }no Ph part, but the pool ran lean tiles"
  fi
  if [[ -n "$ins_bad" ]]; then
    log "arm $name: $ins_bad; not measured (results kept as $name.unmeasured.json)"
    [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
  fi
  grep -E "host experts' background inserts: |host expert pool, lean tiles: " "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  # Dv at exit: every rank tallies its draft routing histograms; a selection of a parked expert would
  # name weights the card never loaded. Measured only if all four ranks report none.
  if [[ -n "${ARM_DV:-}" ]]; then
    local dv_exit dv_held
    dv_exit=$(grep -cE 'draft VRAM: rank [0-3] at exit: ' "$OUT/$name.server.log")
    dv_held=$(grep -cE 'draft VRAM: rank [0-3] at exit: .*\(none: the mask held\)' "$OUT/$name.server.log")
    grep -E 'draft VRAM: rank [0-3] at exit: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
    if [[ "$dv_exit" != 4 || "$dv_held" != 4 ]]; then
      log "arm $name: Dv${ARM_DV}: $dv_held of 4 ranks report at exit that no parked expert was routed to ($dv_exit exit lines); not measured (results kept as $name.unmeasured.json)"
      [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
    fi
  fi
  # Xd: the capture, written at shutdown; a file without draft passes captured nothing.
  if [[ -n "${ARM_XD:-}" ]]; then
    local xd_sum xd_rc
    xd_sum=$(python3 -I "$HERE/draft_hotset.py" summary "$OUT/xstats-$name.json" 2>&1); xd_rc=$?
    echo "$xd_sum" | tee -a "$OUT/window.log"
    (( xd_rc == 0 )) || log "arm $name: Xd, but xstats-$name.json holds no draft capture (exit $xd_rc)"
  fi
  grep -E 'host experts|EP scoped|live chain|error|ERR' "$OUT/$name.server.log" | tail -16 | tee -a "$OUT/window.log"
  # (Pmr: a rank's "by pass size" line beside its totals, and the predictor's multi-row line.)
  grep -E 'read-ahead (in pieces:|in pieces by pass size|readers)|router predictor( \(|, multi-row)' "$OUT/$name.server.log" | tail -11 | tee -a "$OUT/window.log"
  # The per-unit wait for partly read units, every arm with read-ahead in pieces (the Pu parts' gate),
  # and a Pu arm's rests.
  grep -E 'read-ahead in pieces, (partly read units|rests on the kept readers)' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -E 'router predictor (scoring|check|record)|side store:|unit inserts refused' "$OUT/$name.server.log" | tail -10 | tee -a "$OUT/window.log"   # Pi/Pk, Pv, Ns
  grep -E 'rank 0: on-die reads' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Pa/Pm/Pl/Pe: plan, counts
  grep -E 'post-stop passes: [0-9]' "$OUT/$name.server.log" | tail -1 | tee -a "$OUT/window.log"   # passes after a reply's end (Ss)
  # Every arm's VRAM residents a rank; a Vp arm's packed-form lines (what packed, the scan, the load's checks).
  grep -E 'planned units VRAM=|VRAM packed' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -E 'route trace' "$OUT/$name.server.log" | tail -6 | tee -a "$OUT/window.log"   # Rt: per rank, and the files
  grep -E 'consumer skip: [0-9]+ (prefill|cut) passes|rank 0: consumer skip' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Cs: what the cut ran
  grep -E 'ds41 p1 stats, rank|ds41 p1 chunk:' "$OUT/$name.server.log" | tail -24 | tee -a "$OUT/window.log"   # P1: each rank's totals, rank 0's chunks
  # Tb: the steps each rank issued as two micro-batches and the routed layers rank stagers ran as two
  # slices. A Tb arm that sent requests together (CK, K > 1) and split no step did not do what its
  # name says: its results are set aside as <arm>.unmeasured.json.
  grep -E 'two-batch overlap: [0-9]+ (decode steps|routed layers)' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  if [[ -n "${ARM_TB:-}" && "${ARM_CONC:-${AB_CONCURRENCY:-1}}" -gt 1 ]] &&
     ! grep -qE 'rank 0: two-batch overlap: [1-9][0-9]* decode steps' "$OUT/$name.server.log"; then
    log "arm $name: Tb, but rank 0 split no decode step; not measured (results kept as $name.unmeasured.json)"
    [[ -f "$OUT/$name.json" ]] && mv "$OUT/$name.json" "$OUT/$name.unmeasured.json"
  fi
  grep -E 'prefill hash: seq [0-9]+, [0-9]+ tokens: all ranks' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Kh: a digest a prompt
  grep -E '^libr4d: RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV' "$OUT/$name.server.log" | tail -2 | tee -a "$OUT/window.log"   # Dk/Dkt: calls taken
  grep -E 'host expert pool: |host experts, pre-wake: |host experts, one-token passes: ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"   # Fw/WpN: the wake-up, decode's gather waits
  # Dg: the gate's totals, the decode steps by kind (the measured break-even), each sequence's line.
  grep -E 'DSpark acceptance gate: ' "$OUT/$name.server.log" | grep -v 'DSpark acceptance gate: on' | tee -a "$OUT/window.log"
  # Dc / Dl / Dx / a Ds part with NK: their shutdown report (totals, calibration, context passes, steps by
  # kind and of two or more sequences, each request).
  grep -E 'DSpark (confidence controller|load-aware drafting|context pass deferred|controls|draft bank with several sequences): ' "$OUT/$name.server.log" | tee -a "$OUT/window.log"
  grep -E 'end-of-request snapshots: [0-9]+ reserved' "$OUT/$name.server.log" | tail -1 | tee -a "$OUT/window.log"   # Pce/Pcl: stored, hits, store
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
# The Lc arms' gate (RADIANCE_EXPERIMENTAL_DS41_LCP), like Xt's: an Lc arm runs only on the
# libr4d.so (sha256) the large-chunk checks passed on in this window, at a chunk no taller than they
# checked (LCP_CHECK_C).
LCP_OK="" LCP_LIB="" LCP_SUM="" LCP_C=0
LCP_WHY="no large-chunk prefill device checks in this window (DEVICE_CHECKS=lcp with TALL_CHECK and TALL_LIBR4D)"
if [[ " ${DEVICE_CHECKS:-1} " == *" lcp "* ]]; then
  # RADIANCE_EXPERIMENTAL_DS41_LCP on card 0: the twin and the sparse ops at chunk heights up to
  # LCP_CHECK_C (`lcp`, which also sweeps every other height-dependent op to 2048 rows, gemm_nt by the
  # row the table resolves, the router's column split and gemm_nt_bf16f8 at N 1280 and 4096 among
  # them), the sweep under LCP alone, and
  # without any switch (`sweep0`), each row against the same row alone and in launches of 16.
  # TALL_LIBR4D: the plugin the Lc arms load; TALL_CHECK: a harness built with the lcp mode (both must
  # be named, as for `tall`). The logs are devcheck-lcp-*.log, apart from `tall`'s.
  TL=${TALL_LIBR4D:-} TC=${TALL_CHECK:-} LC=${LCP_CHECK_C:-1024}
  if [[ ! -x "$TC" || ! -f "$TL" ]]; then
    LCP_WHY="TALL_CHECK=$TC is not an executable or TALL_LIBR4D=$TL is not a file"
    log "large-chunk prefill device checks: $LCP_WHY; no Lc arm is measured"
  elif ! [[ $LC =~ ^[0-9]+$ ]] || (( 10#$LC < 64 || 10#$LC > 2048 || 10#$LC % 16 )); then
    LCP_WHY="LCP_CHECK_C=$LC is not a multiple of 16 in 64..2048"
    log "large-chunk prefill device checks: $LCP_WHY; no Lc arm is measured"
  else
    LC=$((10#$LC))
    log "large-chunk prefill device checks on card 0 with $TL ($TC), chunks of up to $LC"
    lcp_bad=""
    for m in "lcp $LC" sweep sweep0; do
      f=${m%% *}
      ( export HIP_VISIBLE_DEVICES=0
        unset RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL RADIANCE_EXPERIMENTAL_DS41_LCP
        [[ $f == sweep ]] && export RADIANCE_EXPERIMENTAL_DS41_LCP=$LC
        timeout 1800 "$TC" "$TL" $m >"$OUT/devcheck-lcp-$f.log" 2>&1
        echo "$f exit $?" >>"$OUT/devcheck-lcp-$f.log" )
      grep -E '^(PASS|FAIL)|exit|norm rows' "$OUT/devcheck-lcp-$f.log" | tail -10 | tee -a "$OUT/window.log"
      grep -qx "$f exit 0" "$OUT/devcheck-lcp-$f.log" || lcp_bad+=" $f exited non-zero;"
      grep -q '^FAIL' "$OUT/devcheck-lcp-$f.log" && lcp_bad+=" $f printed FAIL;"
    done
    grep -q "^PASS: lcp $LC: " "$OUT/devcheck-lcp-lcp.log" || lcp_bad+=" lcp has no PASS line;"
    grep -q "^PASS: tall passes give 16-token passes' bits" "$OUT/devcheck-lcp-lcp.log" || lcp_bad+=" lcp's chunk heights have no PASS line;"
    grep -q '^PASS: sweep' "$OUT/devcheck-lcp-sweep.log" || lcp_bad+=" sweep has no PASS line;"
    # The FP8-stored dense weights' gemm_nt_bf16f8 (one skinny row at every height, N >= 128) is in the
    # sweep under LCP: a harness from before its case (lcp review M1) is not taken for one that checked it.
    for f in lcp sweep; do
      grep -q 'gemm_nt_bf16f8 N=1280 K=5120' "$OUT/devcheck-lcp-$f.log" && grep -q 'gemm_nt_bf16f8 N=4096 K=1024' "$OUT/devcheck-lcp-$f.log" ||
        lcp_bad+=" $f did not sweep gemm_nt_bf16f8 (a harness without its case?);"
    done
    if [[ -z "$lcp_bad" ]]; then
      LCP_OK=1 LCP_LIB=$TL LCP_C=$LC LCP_SUM=$(sha256sum "$TL" 2>/dev/null | cut -d' ' -f1)
      log "large-chunk prefill device checks passed on $TL, sha256 ${LCP_SUM:0:16}: Lc arms of up to $LC tokens on it are measured"
    else
      LCP_WHY="the large-chunk prefill device checks did not pass on $TL:$lcp_bad"
      log "$LCP_WHY no Lc arm is measured"
    fi
  fi
fi
# The Mb arms' gate (RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE): an Mb arm runs only on the libr4d.so (sha256)
# whose `bal` check passed in this window (the balanced gemv and twin give the plain kernel's bytes).
BAL_OK="" BAL_LIB="" BAL_SUM=""
BAL_WHY="no balanced-gemv device check in this window (DEVICE_CHECKS=bal with TALL_CHECK and TALL_LIBR4D)"
if [[ " ${DEVICE_CHECKS:-1} " == *" bal "* ]]; then
  TL=${TALL_LIBR4D:-} TC=${TALL_CHECK:-} LC=${LCP_CHECK_C:-1024}
  if [[ ! -x "$TC" || ! -f "$TL" ]]; then
    BAL_WHY="TALL_CHECK=$TC is not an executable or TALL_LIBR4D=$TL is not a file"
    log "balanced gemv device check: $BAL_WHY; no Mb arm is measured"
  else
    log "balanced gemv device check on card 0 with $TL ($TC), up to $LC tokens${BAL_ROUTES:+, routes $BAL_ROUTES}"
    ( export HIP_VISIBLE_DEVICES=0
      unset RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE
      timeout 1800 "$TC" "$TL" bal "$LC" ${BAL_ROUTES:-} >"$OUT/devcheck-bal.log" 2>&1
      echo "bal exit $?" >>"$OUT/devcheck-bal.log" )
    grep -E '^(PASS|FAIL)|exit|^bal: ' "$OUT/devcheck-bal.log" | tail -6 | tee -a "$OUT/window.log"
    if grep -qx "bal exit 0" "$OUT/devcheck-bal.log" && grep -q '^PASS: bal: ' "$OUT/devcheck-bal.log" &&
       ! grep -q '^FAIL' "$OUT/devcheck-bal.log"; then
      BAL_OK=1 BAL_LIB=$TL BAL_SUM=$(sha256sum "$TL" 2>/dev/null | cut -d' ' -f1)
      log "balanced gemv device check passed on $TL, sha256 ${BAL_SUM:0:16}: Mb arms on it are measured"
    else
      BAL_WHY="the balanced gemv device check did not pass on $TL (devcheck-bal.log)"
      log "$BAL_WHY; no Mb arm is measured"
      fi
    fi
fi
# The P1 arms' gate (RADIANCE_EXPERIMENTAL_DS41_P1), like Lc's: a P1 arm runs only on the libr4d.so
# (sha256) on which ds41_mxfp4_device_check `p1` passed in this window (the P1 row against the gemv in
# passes of 16, the marks, the live tail and the wave composition; the check sets LCP=2048 and P1=1 itself).
P1_OK="" P1_LIB="" P1_SUM=""
P1_WHY="no prefill Phase 1 device check in this window (DEVICE_CHECKS=p1 with TALL_CHECK and TALL_LIBR4D)"
if [[ " ${DEVICE_CHECKS:-1} " == *" p1 "* ]]; then
  TL=${TALL_LIBR4D:-} TC=${TALL_CHECK:-} PM=${P1_CHECK_M:-4096} PR=${P1_ROUTES:-}
  if [[ ! -x "$TC" || ! -f "$TL" ]]; then
    P1_WHY="TALL_CHECK=$TC is not an executable or TALL_LIBR4D=$TL is not a file"
    log "prefill Phase 1 device check: $P1_WHY; no P1 arm is measured"
  elif ! [[ $PM =~ ^[0-9]+$ ]] || (( 10#$PM < 17 || 10#$PM > 4096 )); then
    P1_WHY="P1_CHECK_M=$PM is not in 17..4096"
    log "prefill Phase 1 device check: $P1_WHY; no P1 arm is measured"
  elif [[ -n "$PR" && ! -f "$PR" ]]; then
    P1_WHY="P1_ROUTES=$PR is not a file"
    log "prefill Phase 1 device check: $P1_WHY; no P1 arm is measured"
  else
    log "prefill Phase 1 device check on card 0 with $TL ($TC), M up to $PM${PR:+, routes $PR}"
    ( export HIP_VISIBLE_DEVICES=0
      unset RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1
      timeout 1800 "$TC" "$TL" p1 $((10#$PM)) $PR >"$OUT/devcheck-p1.log" 2>&1
      echo "p1 exit $?" >>"$OUT/devcheck-p1.log" )
    grep -E '^p1( time)?:|exit' "$OUT/devcheck-p1.log" | tail -14 | tee -a "$OUT/window.log"
    p1_bad=""
    grep -qx "p1 exit 0" "$OUT/devcheck-p1.log" || p1_bad+=" p1 exited non-zero;"
    [[ "$(grep -E '^p1: (PASS|FAIL)' "$OUT/devcheck-p1.log" | tail -1)" == "p1: PASS"* ]] || p1_bad+=" its last p1 line is not PASS;"
    if [[ -z "$p1_bad" ]]; then
      P1_OK=1 P1_LIB=$TL P1_SUM=$(sha256sum "$TL" 2>/dev/null | cut -d' ' -f1)
      log "prefill Phase 1 device check passed on $TL, sha256 ${P1_SUM:0:16}: P1 arms on it are measured"
    else
      P1_WHY="the prefill Phase 1 device check did not pass on $TL:$p1_bad"
      log "$P1_WHY no P1 arm is measured"
    fi
  fi
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
# launch_window.sh runs a snapshot copy of this script, whose directory has no build/: TALL_CHECK (the same
# ds41_mxfp4_device_check) is used when given.
MX_CHECK=${TALL_CHECK:-$HERE/build/ds41_mxfp4_device_check}
if [[ ( "${DEVICE_CHECKS:-1}" == 1 || " ${DEVICE_CHECKS:-1} " == *" swiglu "* ) && ! -x "$MX_CHECK" ]]; then
  log "swiglu and chain checks asked for, but $MX_CHECK is not an executable: not run"
fi
if [[ ( "${DEVICE_CHECKS:-1}" == 1 || " ${DEVICE_CHECKS:-1} " == *" swiglu "* ) && -x "$MX_CHECK" ]]; then
  # The served plugin's swiglu over every (gate, up) pair, mismatches classified, and the
  # whole host chain against its decode kernels; the table recaptured and compared.
  log "device checks on card 0 with $RHOME/kernels/libr4d.so"
  ( export HIP_VISIBLE_DEVICES=0
    timeout 1200 "$MX_CHECK" "$RHOME/kernels/libr4d.so" swiglu "$OUT/silu_recheck.bin" >"$OUT/swiglu_proof.log" 2>&1
    echo "swiglu exit $?" >>"$OUT/swiglu_proof.log"
    timeout 1200 "$MX_CHECK" "$RHOME/kernels/libr4d.so" check 24 "$SILU" >"$OUT/chain_check.log" 2>&1
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
  # arm with PpK; PmrN / Pmr the read-ahead in pieces in
  # decode passes of 2..N rows too (Pmr: N = 6; a DSpark verify pass, or several sequences' tokens):
  # rank 0 posts every row's router input, each row predicts layer L+2's top K, each rank reads the
  # union under the same live cap (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=N; needs a Pp part;
  # reads only; measured only if the server names it at startup with its N, and every other arm runs
  # with it 0 and must not name it); o --profile-ops;
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
# experts in the host pool: give the arm pN of ~2048); a Ds part goes with NK / CK / Pf like any
# other: with K above 1 a sequence drafts only in a step that carries it alone while no other decodes,
# and a step of two or more decoding sequences verifies no draft (the plain batch); a drafting Ds arm
# with K above 1 is measured only with the scheduler's "DSpark draft bank with up to K sequences" line
# (a radiance that has it), and every other arm must not have it; Dk the draft bank's grouped GEMMs take the
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
# move; a deepseek41 plugin that has it, and a Ds part); DvN the bank's hot set in VRAM
# (RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=N with ..._DRAFT_HOTSET=$DRAFT_HOTSET or a variable by prompt set
# and short block, hotset_for): each rank keeps the first N experts (1-32) of each of its stage lists as
# static VRAM residents charged to the weights pool (the trunk gives up 3N residents a rank), loads no other
# draft expert, pins nothing in the host pool (so the arm runs at B's p256 and m98304), and the draft
# router is masked to the kept experts (-inf in the correction planes); a Ds part, and with Dkt the live
# tail serves the VRAM weights; measured only if the server names the switch, every rank its final
# placement (N/N/N hot) and its masked planes, and at exit every rank its draft selections with none on
# a parked expert (otherwise <arm>.unmeasured.json); every other arm runs with ..._DRAFT_VRAM=0 and must
# have no "draft VRAM" line; Xd the draft-routing capture: --expert-stats into xstats-<arm>.json (a Ds
# part; the startup line must count 43 routed layers and the file draft passes, draft_hotset.py
# summary), the input of draft_hotset.py write; Dg / DgTT[wW][rR] DSpark's per-sequence
# acceptance gate (RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE=1, or 0.TT,W,R: a sequence drafts nothing
# while the mean drafts accepted by its last W verifies, 8 by default, is under 0.TT, 0.78 by default
# (DgTTT: T.TT), and drafts one step in R, 8, of a closed run as a probe; the context pass still
# runs; a radiance that has it, a Ds part other than Ds0; measured only with the scheduler's
# "DSpark acceptance gate: on" line, and every other arm runs with it 0 and must not have the line);
# Df the drafter's dense linears from E4M3 codes as F8 reads the trunk's
# (RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8; needs F8, a Ds part and a deepseek41 plugin that has it;
# measured only with its "DSpark drafter: N dense weights read by gemm_nt_bf16f8" lines, every other
# arm runs with it 0 and must not have them; at the end each Df arm's draft_n/draft_n_accepted must
# equal the same arm's without Df); Dc / DcTT[rR] DSpark's confidence controller
# (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF=1 or 0.TT[,R], with ..._CONF_HEAD=$DRAFT_CONF_HEAD: a sequence
# drafts while the confidence head's p for its newest draft is at least 0.TT, 0.55 by default, and below
# it only one step in R, 8, as a probe), Dl / DlN load-aware drafting (..._DSPARK_LOAD=1 or N: no
# drafts while 2, or N, sequences decode), Dx the drafter's context pass deferred while nothing drafts
# (..._DSPARK_CTX_DEFER=1: the trunk keeps every sequence's taps in the drafter cache's slots, and a
# pass runs only for a drafting sequence, over what it owes: from its cache's end, or the window below
# its anchor, in passes of 16; with NK, steps of several sequences run none and the one left alone
# drafts at once), Dm the drafter's
# Markov codebooks pinned in host memory with the embedding (..._DSPARK_CODEBOOK_RAM=1; needs Er): each
# needs a Ds part (Dc / Dl / Dx not Ds0, Dc not with Dg) and a radiance (Dc, Dx: and a deepseek41
# plugin) that has it; measured only with its startup line, and every other arm runs with it 0 and must
# not have it; their shutdown report goes to window.log.
# Dpc is read-only proposal probability/score/margin capture (DSPARK_PROB_CAPTURE), beside a drafting
# Ds part only, never Ds0. It writes JSON events to the arm's server log; candidate_p is normalized
# over the existing Markov-conditioned candidate set, not full vocabulary or target acceptance.
# H3 the third run home (RADIANCE_HOME_DIR3); Es
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
# tokens in all: 512 holds four 110-token requests, 1024 eight, 2048 sixteen); Tb a decode step of two or
# more sequences issued as two micro-batches, each layer's MoE gather of one behind the other's card
# work (RADIANCE_EXPERIMENTAL_SCHED_TBO; a radiance and deepseek41 from claude/ds41-tbo); rN a repeat (ignored).
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
# runs with the switch at 0 and must have neither line. With a Ds part the trunk's residents pack and
# the bank's draft experts (Dv's hot set too) stay block_mxfp4 on their own GEMV path (Dk / Dkt), and
# every rank must log the engine's bank check (arm() above).
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
# other arm runs with it off); Ck / CkN with Cs and Pc: which snapshots below a cut prompt's target
# the cut makes whole (RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT: none a band of 512, the
# default; Ck every one its idle prefill writes (all); CkN those within N tokens below the target and a shared head; Ck0
# none, the behaviour before claude/ds41-csckpt), checked in the server's policy line, which a Cs Pc
# arm must have (a radiance from claude/ds41-csckpt); Kh at the end of each prompt's prefill every rank hashes its KV groups
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
# arm's threads; exact); WlN with WpN the miss-read trigger's lead in us (..._PREWAKE_LEAD_US, 60 in
# every arm without it; Wl0: no miss-read trigger, the router's alone). Every other arm runs with
# both at 0. Fw must show the pool's futex line at
# startup, WpN its pre-wake line for N workers and the stager's router events, and at shutdown the
# pool's line with router pre-wakes that woke workers (else the arm's results are set aside as
# <arm>.unmeasured.json); an arm without them must show neither startup line.
# In the background inserts write a missed unit's codes into the packed RAM cache by non-temporal
# stores (RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=1; needs K); Is / IsN the same in pieces of N KiB
# (256), a one-token pass's insert sleeping before each piece while the pool has submitted jobs
# unfinished (not while it holds only a miss's reservation), at most ..._INSERT_WAIT_US an insert
# (RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB; needs K), IwN that wait
# (4000 us; needs Is); Ph the pool's lean tiles: a worker reuses its last tile's x lanes on a tile of
# the same job and projection, and claims look only at the slots holding a job
# (RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES=1). All exact (the cache's bytes, every row).
# Every arm states the three switches (0 without In / Is / Ph, the wait empty, i.e. its default,
# without Iw), and is measured only if the server log shows what its name says: the startup line
# ("background inserts write ... by non-temporal stores", "in pieces of N KiB"; "lean tiles:") and
# at shutdown every rank's "background inserts" line with non-temporal pieces (Is: some after a
# wait), the "lean tiles" line with tiles claimed and lanes reused; a plain arm none of it (results
# otherwise kept as <arm>.unmeasured.json).
# Pi8 / Pi4 the predictor scores every expert from an int8 / int4 copy of the routers first (one
# scale an expert, 79 / 39 MB beside the bf16 copy) and its best 12 again in bf16, Pi8rN / Pi4rN its
# best N (4-64) (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8|int4, ..._PREFETCH_RESCORE, 12 in
# every arm without rN; needs a predictor part; only reads ahead change, never a reply; offline on
# the container's routers int8 at 12 gave bf16's top 4 every time, int4 needs ~24: router_score_eval);
# PkN with Pi8 / Pi4 one prediction in N is
# also scored in bf16 alone after it was handed over and compared (..._PREFETCH_SCORE_CHECK=N; the
# shutdown's "router predictor check" line: the same set, the same order, where bf16's picks ranked):
# a precision arm, not a timed one; Pw at an early-issued layer (Ei) rank 0 posts the read-ahead's
# router input once the layer's CPU work is submitted instead of before its choice
# (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=1; needs Ei and PpK); NsN up to N units a rank whose
# inserts the packed RAM cache refuses (7 experts never fit its raw-round room) keep their planes,
# and their later misses read nothing (RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=N, ~18 MiB a unit; needs
# K). Each is measured only if its startup line is in the server log (Pi: "scored in int8|int4" in
# the predictor's line; Pk: "checked against bf16 alone" there; Pw: "rank 0 posts the read-ahead's
# router input once a layer's CPU work is submitted"; Ns: "side store: the planes of up to N units"),
# and every arm without the part runs with its switch at 0 and must not have the line. Pv records
# every router input the predictor is posted into pred-<arm>.bin (..._PREFETCH_RECORD; ~10 KB an
# input, ~190 MB an arm of the held-out prompts; an instrument: router_score_eval scores the file
# offline in bf16, int8 and int4 against the container's routers); needs a predictor part.
  IFS=- read -r -a parts <<< "$a"
  xt="" bt="" kld="" dk="" db="" pu="" ins_nt="" ins_kib="" ins_wait="" lean="" lc="" ad="" dv="" xd="" hs="" dg="" df="" pmr="" tb=""
  mb="" lf="" dc="" dl="" dx="" dm="" dpc="" ds="" nseq=""
  p1="" q1=() qm="" ql="" qg="" qe="" qsp=""
  pol=${parts[0]} threads=$HX_THREADS env=() CACHE_MIB=81920 HOST_POOL_MIB=4096 EXTRA_ARGS="" ARM_HOME="$RHOME" ARM_CONC=
  ARM_PROMPTS= ARM_N= ARM_LQ= rt=
  power="" skip="" ARGS=() f8="" f8x="" rb="" rbf=0 pred="" cs="" ck="" kh="" pc="" pcopt="" fw="" wp="" wl=""
  psc="" prs=12 pk="" pw="" ns="" pv=""
  hpe="" hpg="" hpl="" hpb="" hpm="" hpr=""
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
      Mr[0-9]*) # pool jobs of up to N token rows of one expert (1-64, default 4; above 4, jobs of 4+ rows take the many-row kernel)
               if [[ ${p#Mr} =~ ^[0-9]+$ ]] && (( 10#${p#Mr} >= 1 && 10#${p#Mr} <= 64 )); then
                 env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=${p#Mr})
               else log "arm $a: $p: a job takes 1-64 rows; the arm is skipped"; skip=1; fi ;;
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
      Sga[0-9]*) ARGS+=(--num-speculative-tokens "${p#Sga}" --spec-ngram "${p#Sga}" --draft-window adaptive)
                 env+=(RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1) ;;   # the plugin leaves its drafter out
      Sg[0-9]*) ARGS+=(--num-speculative-tokens "${p#Sg}" --spec-ngram "${p#Sg}")
                env+=(RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE=1) ;;
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
               ds=$p
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
      Dv[0-9]*) dv=${p#Dv}                 # the bank's hot set in VRAM, router masked (DRAFT_VRAM=N)
               if ! [[ $dv =~ ^[0-9]+$ ]] || (( 10#$dv < 1 || 10#$dv > 32 )); then
                 log "arm $a: $p: a stage keeps 1-32 experts a rank; the arm is skipped"; skip=1
               fi ;;
      Xd)      xd=1 ;;                     # the draft-routing capture (--expert-stats xstats-<arm>.json)
      Dg|Dg[0-9]*)                         # DSpark's acceptance gate (DSPARK_GATE): defaults, or DgTT[wW][rR]
               if [[ $p == Dg ]]; then dg=1
               elif [[ $p =~ ^Dg([0-9]{2,3})(w([0-9]+))?(r([0-9]+))?$ ]]; then
                 t=${BASH_REMATCH[1]}
                 if (( ${#t} == 2 )); then t=0.$t; else t=${t:0:1}.${t:1}; fi
                 dg="$t,${BASH_REMATCH[3]:-8},${BASH_REMATCH[5]:-8}"
                 # R = 1 is refused by the engine (consecutive probes would share one probe step).
                 (( 10#${BASH_REMATCH[5]:-8} >= 2 )) || { log "arm $a: $p: a probe one step in R needs R >= 2; the arm is skipped"; skip=1; }
               else log "arm $a: $p: the gate is Dg or DgTT[wW][rR] (threshold 0.TT or T.TT, W verifies, a probe one step in R); the arm is skipped"; skip=1; fi ;;
      Df)      df=1 ;;                     # the drafter's dense linears from E4M3 codes (DSPARK_FP8; needs F8)
      Dc|Dc[0-9]*)                         # DSpark's confidence controller (DSPARK_CONF): defaults, or DcTT[rR]
               if [[ $p == Dc ]]; then dc=1
               elif [[ $p =~ ^Dc([0-9]{2})(r([0-9]+))?$ ]]; then
                 dc="0.${BASH_REMATCH[1]}${BASH_REMATCH[3]:+,${BASH_REMATCH[3]}}"
                 (( 10#${BASH_REMATCH[3]:-8} >= 2 )) || { log "arm $a: $p: a probe one step in R needs R >= 2; the arm is skipped"; skip=1; }
               else log "arm $a: $p: the controller is Dc or DcTT[rR] (threshold 0.TT, a probe one step in R); the arm is skipped"; skip=1; fi
               [[ -f "$DRAFT_CONF_HEAD" ]] || { log "arm $a: $p needs DRAFT_CONF_HEAD=<dspark_conf_extract.py's file>; the arm is skipped"; skip=1; } ;;
      Dl|Dl[0-9]*) dl=${p#Dl}; dl=${dl:-1}  # load-aware drafting (DSPARK_LOAD): none from 2 (Dl) or N decoding sequences
               if ! [[ $dl =~ ^[0-9]+$ ]] || ! { [[ $dl == 1 ]] || (( 10#$dl >= 2 && 10#$dl <= 64 )); }; then
                 log "arm $a: $p: DlN takes 2-64; the arm is skipped"; skip=1
               fi ;;
      Dx)      dx=1 ;;                     # the drafter's context pass deferred while nothing drafts (DSPARK_CTX_DEFER)
      Dpc)     dpc=1 ;;                    # read-only candidate-set probability/score/margin capture (DSPARK_PROB_CAPTURE)
      Dm)      dm=1 ;;                     # the drafter's Markov codebooks pinned with the embedding (DSPARK_CODEBOOK_RAM; needs Er)
      K)       env+=(RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1) ;;     # the RAM cache packed for the CPU only
      Vp)      env+=(RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1) ;;      # VRAM residents in the packed-scale form (vpack_gate)
      Lm[0-9]*) ARGS+=(--max-model-len "${p#Lm}") ;;   # a longer context (default 512), e.g. for long-prompt TTFT
      Pc)      pc=1 ARM_NOPC= ; ARGS+=(--prefix-resume exact)                          # prefix cache on (exact resume, the default, named)
               env+=(RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1) ;;              # ... and pinned slabs not reserved twice
      Pc0)     pc=1 ARM_NOPC= ; ARGS+=(--prefix-resume exact) ;;                      # prefix cache on, the margin as before (the control)
      Pcn)     pc=1 ARM_NOPC= ; ARGS+=(--prefix-resume nearest)                        # prefix cache on, a hit resumes at the deepest snapshot
               env+=(RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1) ;;              # ... (Kh against a cold arm checks it stays exact)
      Pci[0-9]*) pcopt+=" $p"; ARGS+=(--checkpoint-interval "${p#Pci}") ;;                    # snapshot interval (2048)
      Pcs[0-9]*) pcopt+=" $p"; ARGS+=(--checkpoint-slots "${p#Pcs}") ;;                       # snapshot store slots (auto)
      Pcx)     pcopt+=" $p"; env+=(RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1) ;;           # publish only cold-cut chunks
      Pce)     pcopt+=" $p"; hpe=1; env+=(RADIANCE_EXPERIMENTAL_PREFIX_EORS=1) ;;              # end-of-request snapshots
      Pcg[0-9]*) pcopt+=" $p"; hpg=${p#Pcg}; env+=(RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=${p#Pcg}) ;;   # ...on an N-token grid
      Pcl)     pcopt+=" $p"; hpl=1; env+=(RADIANCE_EXPERIMENTAL_PREFIX_SESSION_LRU=1) ;;       # snapshot store LRU by session
      Pcb[0-9]*) pcopt+=" $p"; hpb=1; env+=(RADIANCE_EXPERIMENTAL_CHECKPOINT_STORE_MIB=${p#Pcb}) ;;   # snapshot store as N MiB
      Pcm[0-9]*) pcopt+=" $p"; hpm=1; ARGS+=(--prefix-cache-mib "${p#Pcm}") ;;                 # host KV tier of N MiB
      Pcr)     pcopt+=" $p"; hpr=1; env+=(RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=1) ;;       # index/compressed caches replicated
      Pf)      env+=(RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1) ;;   # a prefill chunk alone in its step; decode rows batched
      Tb)      tb=1 ;;                                                 # two-batch overlap of concurrent decode steps
      N[0-9]*) ARGS+=(--max-num-seqs "${p#N}"); nseq=${p#N} ;;        # this arm's --max-num-seqs (MAX_SEQS for every arm)
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
      Lc[0-9]*) lc=${p#Lc}                                             # large-chunk prefill: chunks of N (RADIANCE_EXPERIMENTAL_DS41_LCP)
               if [[ $lc =~ ^[0-9]+$ ]] && (( 10#$lc >= 64 && 10#$lc <= 2048 && 10#$lc % 16 == 0 )); then
                 lc=$((10#$lc)); ARGS+=(--max-num-batched-tokens "$lc"); env+=(RADIANCE_EXPERIMENTAL_DS41_LCP=$lc)
               else log "arm $a: $p: a chunk is a multiple of 16 in 64..2048; the arm is skipped"; skip=1; fi ;;
      P1)      p1=1 ;;                                                  # prefill Phase 1: card experts in large chunks (needs Lc)
      Qsp)     qsp=1 ;;                                                 # each P1 wave's covering sorted-slot SwiGLU span
      Qw[0-9]*|Qi[0-9]*|Qr[0-9]*|Qa[0-9]*|Qv[0-9]*|Qm[0-9]*|Ql[0-9]*|Qh[0-9]*|Qe[0-9]*)   # ...wave / in flight / ready / read-ahead / verify / threshold / lent slots / its hold / lent arena slots
               qn=${p:2}
               case ${p:0:2} in
                 Qw) qk=WAVE qlo=1 qhi=16 ;; Qi) qk=INFLIGHT qlo=1 qhi=64 ;; Qr) qk=READY qlo=1 qhi=32 ;;
                 Qa) qk=AHEAD qlo=0 qhi=64 ;; Qm) qk=MIN_ROWS qlo=17 qhi=4096 ;; Ql) qk=LEND qlo=1 qhi=64 ;;
                 Qh) qk=GRACE_MS qlo=0 qhi=60000 ;; Qe) qk=LEND_ARENA qlo=1 qhi=64 ;; *) qk=VERIFY qlo=0 qhi=4096 ;;
               esac
               if [[ $qn =~ ^[0-9]+$ ]] && (( 10#$qn >= qlo && 10#$qn <= qhi )); then
                 q1+=(RADIANCE_EXPERIMENTAL_DS41_P1_$qk=$((10#$qn)))
                 [[ $qk == MIN_ROWS ]] && qm=$((10#$qn))
                 [[ $qk == LEND ]] && ql=$((10#$qn))
                 [[ $qk == GRACE_MS ]] && qg=$((10#$qn))
                 [[ $qk == LEND_ARENA ]] && qe=$((10#$qn))
               else log "arm $a: $p: ${p:0:2} takes $qlo-$qhi; the arm is skipped"; skip=1; fi ;;
      Qs)      q1+=(RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN=start) ;;       # ...chunks aligned to the start
      Ad[0-9]*) ad=${p#Ad}; env+=(RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=$((10#$ad))) ;;   # admit the last N tokens' units
      Ada)     ad=all; env+=(RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=all) ;;       # admit every unit (as at 16 rows)
      Lp[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_LCP_PLANES=${p#Lp}) ;;        # missed units read at once a layer
      Li[0-9]*) env+=(RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ=${p#Li}) ;;       # admitted misses queued for the cache
      Mb1|Mb2) mb=${p#Mb} ;;                                           # balanced routed gemv (..._MOE_BALANCE)
      Lf)      lf=1 ;;                                                 # per-pass-size time lines (..._LCP_PROFILE)
      Fl)      env+=(RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1) ;;       # counts + CPU rows in one ranged copy
      Ms[0-9]) env+=(RADIANCE_EXPERIMENTAL_DS41_MIRROR_SPLIT=${p#Ms}) ;;  # a lone decode miss's mirror tail in N pieces
      Mp)      env+=(RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1) ;;       # miss reads start at the choice, before the pool's wake
      Fw)      fw=1; env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1) ;;   # pool workers sleep on a futex, woken without a lock
      Wp[0-9]*) # pre-wake N pool workers before a decode layer's CPU work (router event, miss read)
               if [[ ${p#Wp} =~ ^[0-9]+$ ]] && (( 10#${p#Wp} >= 1 && 10#${p#Wp} <= 256 )); then
                 wp=$((10#${p#Wp})); env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=$wp)
               else log "arm $a: $p: a pre-wake takes 1-256 workers; the arm is skipped"; skip=1; fi ;;
      Wl[0-9]*) # ...its miss-read trigger N us before the read's predicted landing (Wl0: none)
               if [[ ${p#Wl} =~ ^[0-9]+$ ]] && (( 10#${p#Wl} <= 100000 )); then
                 wl=$((10#${p#Wl})); env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US=$wl)
               else log "arm $a: $p: a pre-wake lead takes 0-100000 us; the arm is skipped"; skip=1; fi ;;
      U?*)     ARM_PROMPTS=/home/as/ds41-claude-build/work/extra-prompts/${p#U}-receipt.json ARM_N=999
               [[ -f $ARM_PROMPTS ]] || { log "arm $a: no prompts $ARM_PROMPTS; the arm is skipped"; skip=1; } ;;
      Rt)      mkdir -p "$OUT/rtrace-$a"   # the routing trace, exact (rtrace.py reads it)
               rt=1; env+=(RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=$(cd "$OUT" && pwd)/rtrace-$a/rt) ;;
      Lq)      ARM_LQ=1 ;;                 # the quality harness after the timed requests
      Cs)      cs=1 ;;                     # the exact consumer-suffix skip (checked in the server log)
      Ck|Cka)  ck=all ;;                   # Cs with Pc: every snapshot below a cut prompt's target made whole (without a Ck part: 512, the default)
      Ck[0-9]*) ck=${p#Ck} ;;              # ... only those within N tokens below it and a shared head (Ck0: none, as before)
      Kh)      kh=1 ;;                     # KV and logits hashes at each prompt's end
      In)      ins_nt=1 ;;                 # background inserts' codes by non-temporal stores (INSERT_NT)
      Is)      ins_kib=256 ;;              # ... in 256 KiB pieces, a decode insert waiting for an idle pool
      Is[0-9]*|Iw[0-9]*)                   # ... in pieces of N KiB (INSERT_SLICE_KIB); at most N us of waiting (INSERT_WAIT_US)
               if [[ ${p:2} =~ ^[0-9]+$ ]] && (( 10#${p:2} <= 100000 )); then
                 if [[ $p == Is* ]]; then ins_kib=$((10#${p:2})); else ins_wait=$((10#${p:2})); fi
               else log "arm $a: $p: a number of KiB / us up to 100000; the arm is skipped"; skip=1; fi ;;
      Ph)      lean=1 ;;                   # the pool's lean tiles (HOST_EXPERTS_LEAN_TILES)
      Pi8|Pi4|Pi8r[0-9]*|Pi4r[0-9]*)      # the predictor scores in int8 / int4 first, its best 12 (rN: N) again in bf16
               psc=${p:2:1}
               if [[ $p == Pi?r* ]]; then
                 if [[ ${p#Pi?r} =~ ^[0-9]+$ ]] && (( 10#${p#Pi?r} >= 4 && 10#${p#Pi?r} <= 64 )); then prs=$((10#${p#Pi?r}))
                 else log "arm $a: $p: the rescore count takes 4-64; the arm is skipped"; skip=1; fi
               fi ;;
      Pk[0-9]*) # ...one prediction in N also scored in bf16 alone and compared (a precision arm)
               if [[ ${p#Pk} =~ ^[0-9]+$ ]]; then pk=$((10#${p#Pk})); else log "arm $a: $p: PkN takes a number; the arm is skipped"; skip=1; fi ;;
      Pw)      pw=1 ;;                     # Ei: the read-ahead's post after the layer's CPU work is submitted
      Pv)      pv=1 ;;                     # the predictor's inputs recorded (an instrument)
      Ns[0-9]*) # side store: up to N units a rank the packed cache refuses keep their planes
               if [[ ${p#Ns} =~ ^[0-9]+$ ]]; then ns=$((10#${p#Ns})); else log "arm $a: $p: NsN takes a number; the arm is skipped"; skip=1; fi ;;
      Pmr|Pmr[0-9]*)                       # read-ahead in pieces in decode passes of 2..N rows too (Pmr: 6)
               pmr=${p#Pmr}; pmr=${pmr:-6}
               if ! [[ $pmr =~ ^[0-9]+$ ]] || (( 10#$pmr < 2 || 10#$pmr > 16 )); then
                 log "arm $a: $p: N is 2-16 rows; the arm is skipped"; skip=1
               else pmr=$((10#$pmr)); fi ;;
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
  # Pmr: the multi-row read-ahead acts only on the read-ahead in pieces (a Pp part); without one the
  # server refuses the switch. Every arm states it (0 is off), so a window's VAR=val cannot turn it on
  # for the others; arm() checks the server log both ways.
  if [[ -n "$pmr" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces "* ]]; then
    log "arm $a: Pmr needs a Pp part (the read-ahead in pieces); the arm is skipped"; skip=1
  fi
  env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=${pmr:-0})
  ARM_PMR=$pmr
  # Tb: stated by every arm (0 off), so a window's VAR=val cannot split another arm's steps; arm()
  # checks deepseek41's startup line both ways.
  env+=(RADIANCE_EXPERIMENTAL_SCHED_TBO=${tb:-0})
  ARM_TB=$tb
  # Ss: the read-back runs in an arm only if its name says so, whatever the window's environment.
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=0)
  # Pi8 / Pi4, PkN, Pw, NsN, Pv: each acts only with what it changes (a predictor; Pi; Ei and the
  # read-ahead in pieces; the packed cache; a predictor); without it the arm would run under a name
  # it does not do, so it is skipped. Every arm states the five switches (0 or empty is off for
  # each) and the rescore count, so a window's VAR=val cannot change an arm its name does not cover.
  if [[ -n "$psc$pv" && -z "$pred" ]]; then log "arm $a: Pi$psc / Pv without a predictor part (PpK, PsK, PoK, PtK); the arm is skipped"; skip=1; fi
  if [[ -n "$pk" && ( -z "$psc" || "$pk" == 0 ) ]]; then log "arm $a: Pk$pk needs Pi8 or Pi4 and N of 1 or more; the arm is skipped"; skip=1; fi
  if [[ -n "$pw" ]] && [[ " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1 "* ||
                          " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces "* ]]; then
    log "arm $a: Pw needs Ei and PpK (the read-ahead in pieces); the arm is skipped"; skip=1
  fi
  if [[ -n "$ns" ]] && [[ "$ns" == 0 || " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1 "* ]]; then
    log "arm $a: Ns$ns needs K (the packed RAM cache) and N of 1 or more; the arm is skipped"; skip=1
  fi
  if [[ -n "$psc" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int$psc)
  else env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=0); fi
  env+=(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RESCORE=$prs RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE_CHECK=${pk:-0}
        RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=${pw:-0} RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=${ns:-0}
        RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD=${pv:+$(cd "$OUT" && pwd)/pred-$a.bin})
  ARM_PSC=$psc ARM_PRS=$prs ARM_PK=$pk ARM_PW=$pw ARM_NS=$ns ARM_PV=$pv
  [[ -z "$rt" ]] && env+=(RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=)   # traced only with Rt
  if [[ -n "$cs" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1); else env+=(RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=0); fi
  ARM_CS=$cs
  # Ck / CkN: which snapshots below a cut prompt's target are made whole; it acts only under Cs with
  # the prefix cache. Every arm states it (512, the engine's default, without a Ck part), so a
  # window's VAR=val cannot change what a Cs Pc arm's name says; arm() checks the server's policy
  # line.
  if [[ -n "$ck" && ( -z "$cs" || -z "$pc" ) ]]; then
    log "arm $a: a Ck part (..._CONSUMER_SKIP_CKPT=$ck) acts only with Cs and Pc (or Pc0); the arm is skipped"; skip=1
  fi
  env+=(RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=${ck:-512})
  ARM_CK=
  [[ -n "$cs" && -n "$pc" ]] && ARM_CK=${ck:-512}
  if [[ -n "$kh" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1); else env+=(RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=0); fi
  # The prefix cache's parts: PciN, PcsN and Pcx act only with the cache on (Pc or Pc0); without it
  # the arm would run cold under their names. Every arm states both prefix switches, so a window's
  # VAR=val cannot turn a cold arm or the Pc0 control into something its name does not say.
  if [[ -n "$pcopt" && -z "$pc" ]]; then
    log "arm $a:$pcopt without Pc or Pc0 (the prefix cache stays off); the arm is skipped"; skip=1
  fi
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=0)
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1 "* ]] || env+=(RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=0)
  # The harness parts: combinations the engine refuses are skipped here, and every arm states each
  # switch, so a window's VAR=val cannot turn one on for an arm whose name does not say it (arm()
  # also checks the server's lines both ways).
  if [[ -n "$hpe" && " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=1 "* ]]; then
    log "arm $a: Pce with Pcx (exact publish keeps out the decode-written blocks an end-of-request snapshot stands on; the engine refuses it); the arm is skipped"; skip=1
  fi
  if [[ -n "$hpe" && ( " ${ARGS[*]} " == *" --num-speculative-tokens "[1-9]* || " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1 "* ) ]]; then
    log "arm $a: Pce with speculation (the engine refuses it); the arm is skipped"; skip=1
  fi
  # Pce with exact tall passes: the engine takes it (the tall checks are (A) up to their reach), but
  # no window has yet shown it end to end (a Kh arm's prefill hashes of resumed turns against a
  # cold arm's at Xt-BT32), so it is not measured; PCE_XT=1 runs it, for that evidence alone.
  if [[ -n "$hpe" && -n "$xt" && "${PCE_XT:-}" != 1 ]]; then
    log "arm $a: Pce with Xt (no end-to-end evidence yet that a turn resumed from a tall-pass run's decode snapshot is a cold run's; PCE_XT=1 runs it for a Kh check); the arm is skipped"; skip=1
  fi
  if [[ -n "$hpb" && " ${ARGS[*]} " == *" --checkpoint-slots "* ]]; then
    log "arm $a: PcbN with PcsN (a count given takes no byte budget); the arm is skipped"; skip=1
  fi
  # PcgN: the grid of Pce's snapshots under large-chunk prefill, whose chunk C is the resume grid; N a
  # multiple of 16 below C that divides it (the engine refuses anything else, and LCP's absence).
  if [[ -n "$hpg" ]]; then
    if [[ -z "$hpe" ]]; then
      log "arm $a: PcgN needs Pce (it sets the grid end-of-request snapshots are taken on); the arm is skipped"; skip=1
    elif [[ -z "$lc" ]]; then
      log "arm $a: PcgN needs an Lc part (large-chunk prefill's pins are the partition invariance it stands on; the engine refuses it otherwise); the arm is skipped"; skip=1
    elif ! [[ $hpg =~ ^[0-9]+$ ]] || (( 10#$hpg < 16 || 10#$hpg % 16 || 10#$hpg >= lc || lc % 10#$hpg )); then
      log "arm $a: Pcg$hpg: the grid is a multiple of 16 below the chunk $lc that divides it; the arm is skipped"; skip=1
    fi
  fi
  [[ -n "$hpe" ]] || env+=(RADIANCE_EXPERIMENTAL_PREFIX_EORS=0)
  [[ -n "$hpg" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=0)
  [[ -n "$hpl" ]] || env+=(RADIANCE_EXPERIMENTAL_PREFIX_SESSION_LRU=0)
  [[ -n "$hpb" ]] || env+=(RADIANCE_EXPERIMENTAL_CHECKPOINT_STORE_MIB=)
  [[ -n "$hpr" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=0)
  ARM_EORS=$hpe ARM_RGRID=$hpg ARM_SLRU=$hpl ARM_STOREMIB=$hpb ARM_TIER=$hpm ARM_KVREP=$hpr
  # MrN: arms without it take jobs of the default 4 rows, whatever the window's environment (0 is
  # not this switch's off: any value outside 1-32 turns host experts off).
  [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS="* ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=4)
  # Dk / Dkt / DbN / Dg / Df act on the DSpark drafter only: without a Ds part the arm would not test
  # them. Dg gates drafts, so not with Ds0 (nothing drafts; the engine refuses the pair); Df extends
  # the trunk's FP8 to the drafter, so it needs F8 (without it the plugin keeps the drafter bf16).
  if [[ -n "$dk$db$dg$df" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1 "* ]]; then
    log "arm $a: Dk / Dkt / DbN / Dg / Df need a Ds part (the DSpark draft bank); the arm is skipped"; skip=1
  fi
  if [[ -n "$dg" && " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP=0 "* ]]; then
    log "arm $a: Dg with Ds0 (no draft to gate); the arm is skipped"; skip=1
  fi
  if [[ -n "$df" && -z "$f8" ]]; then
    log "arm $a: Df needs F8 (the drafter's FP8 extends the trunk's); the arm is skipped"; skip=1
  fi
  # Dc / Dl / Dx / Dm (core/sched/draft_conf.h, place/planner.h) act on the DSpark drafter only; the
  # first three decide about drafts, so not with Ds0, and Dc not with Dg (both decide the query pass;
  # the engine refuses the pair); Dm moves the codebooks where the embedding goes, so it needs Er.
  if [[ -n "$dc$dl$dx$dm" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1 "* ]]; then
    log "arm $a: Dc / Dl / Dx / Dm need a Ds part (the DSpark draft bank); the arm is skipped"; skip=1
  fi
  if [[ -n "$dc$dl$dx" && " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP=0 "* ]]; then
    log "arm $a: Dc / Dl / Dx with Ds0 (nothing drafts); the arm is skipped"; skip=1
  fi
  if [[ -n "$dc" && -n "$dg" ]]; then log "arm $a: Dc and Dg both decide the query pass; the arm is skipped"; skip=1; fi
  if [[ -n "$dm" && "-$a-" != *-Er-* ]]; then log "arm $a: Dm needs Er (the codebooks go where the embedding goes); the arm is skipped"; skip=1; fi
  if [[ -n "$dpc" && ( -z "$ds" || "$ds" == Ds0 ) ]]; then
    log "arm $a: Dpc needs a drafting Ds part (not Ds0); the arm is skipped"; skip=1
  fi
  env+=(RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF=${dc:-0} "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD=${dc:+$DRAFT_CONF_HEAD}"
        RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD=${dl:-0} RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=${dx:-0}
        RADIANCE_EXPERIMENTAL_DS41_DSPARK_CODEBOOK_RAM=${dm:-0}
        RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE=${dpc:-0})
  ARM_DC=$dc ARM_DL=$dl ARM_DX=$dx ARM_DM=$dm ARM_DPC=$dpc
  # A drafting Ds part under --max-num-seqs above 1 (an NK part, or MAX_SEQS for the window): the
  # scheduler drafts for a sequence alone and says so at startup (arm() checks the line both ways).
  ARM_SOLO=
  if [[ -n "$ds" && "$ds" != Ds0 ]] && (( 10#${nseq:-${MAX_SEQS:-1}} > 1 )); then ARM_SOLO=1; fi
  # Every arm states these switches (0 is off for each), so a window's VAR=val cannot turn them on;
  # arm() checks the logs both ways. A Dk arm runs only on the libr4d.so the draft check passed on.
  env+=(RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=${dk:-0} RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=${db:-0}
        RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE=${dg:-0} RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=${df:-0})
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
  ARM_DK=$dk ARM_DB=$db ARM_DG=$dg ARM_DF=$df
  # DvN / Xd act on the DSpark drafter only: without a Ds part the arm would not test them. A Dv arm's
  # hot set: DRAFT_HOTSET_<name> for its U<name> part when set, else DRAFT_HOTSET.
  if [[ -n "$dv$xd" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_DSPARK_RAD_BANK=1 "* ]]; then
    log "arm $a: Dv / Xd need a Ds part (the DSpark draft bank); the arm is skipped"; skip=1
  fi
  if [[ -n "$dv" ]]; then
    hs=$(hotset_for "$a")
    if [[ ! -f "$hs" ]]; then
      log "arm $a: Dv needs a hot-set file (DRAFT_HOTSET, or DRAFT_HOTSET_<set>[_db<K>] / DRAFT_HOTSET_db<K>; draft_hotset.py write), not '${hs}'; the arm is skipped"; skip=1
    fi
  fi
  # Every arm states the switch (0 is off), so a window's VAR=val cannot put the bank in VRAM.
  env+=(RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=${dv:-0} "RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET=${dv:+$hs}")
  [[ -n "$xd" ]] && ARGS+=(--expert-stats "$(cd "$OUT" && pwd)/xstats-$a.json")
  ARM_DV=$dv ARM_XD=$xd ARM_HS=$hs
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
  # RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE and ..._LCP_PROFILE: stated in every arm (0 off).
  env+=(RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=${mb:-0} RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=${lf:-0})
  if [[ -n "$mb" && -z "$skip" ]]; then
    IFS=: read -r -a homes <<< "$ARM_HOME"
    lib=""
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
    if [[ -z "$BAL_OK" ]]; then
      log "arm $a: Mb refused: $BAL_WHY"; skip=1
    elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$BAL_SUM" ]]; then
      log "arm $a: Mb refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the bal check passed on ($BAL_LIB, sha256 ${BAL_SUM:0:16})"; skip=1
    fi
  fi
  if [[ -n "$lc" && -z "$skip" ]]; then
    IFS=: read -r -a homes <<< "$ARM_HOME"
    lib=""
    for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
    if [[ -z "$LCP_OK" ]]; then
      log "arm $a: Lc refused: $LCP_WHY"; skip=1
    elif (( lc > LCP_C )); then
      log "arm $a: Lc refused: the large-chunk checks ran at chunks of up to $LCP_C (LCP_CHECK_C), not $lc"; skip=1
    elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$LCP_SUM" ]]; then
      log "arm $a: Lc refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the large-chunk checks passed on ($LCP_LIB, sha256 ${LCP_SUM:0:16})"; skip=1
    fi
  fi
  # RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL: on only in an Xt arm, explicitly off in every other one but an
  # Lc arm, where it is unset (LCP keeps its pins; the core refuses EXACT_TALL=0 beside it).
  # RADIANCE_EXPERIMENTAL_DS41_LCP: an Lc arm's chunk, 0 in every other arm.
  if [[ -n "$lc" ]]; then
    [[ -n "$xt$bt" ]] && { log "arm $a: Lc goes with neither Xt nor BT (it sets the step itself); the arm is skipped"; skip=1; }
    mm=$(grep -oE ' RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=[0-9]+' <<< " ${env[*]}" | tail -1 | cut -d= -f2)
    bn=$(grep -oE ' RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=[0-9]+' <<< " ${env[*]}" | tail -1 | cut -d= -f2)
    if [[ -n "$p1" ]]; then
      # P1: the ring is the file buffers (B2 or more); the CPU path takes every layer pass below the threshold
      # (QmN, 320 without one): its misses up to min(C, N - 1) rows (M16 at Qm17, M319 at the default).
      qmin=${qm:-320}
      qcpu=$(( qmin - 1 < lc ? qmin - 1 : lc ))
      (( ${bn:-0} >= 2 )) || { log "arm $a: P1 needs B2 or more (the ring: 2N slots; B6 is the measured one); the arm is skipped"; skip=1; }
      (( ${mm:-0} >= qcpu )) || { log "arm $a: P1 at the threshold $qmin needs M$qcpu or more (the CPU path takes the layer passes below it, up to $qcpu rows); the arm is skipped"; skip=1; }
      # QlK lends K of the ring's 2N slots to decode residents: at most the 2N - 2 past the file buffers' alias bases.
      [[ -n "$ql" ]] && (( ql > 2 * ${bn:-0} - 2 )) && { log "arm $a: Ql$ql lends more than the $(( 2 * ${bn:-0} - 2 )) slots B${bn:-0}'s ring has past the file buffers' alias bases; the arm is skipped"; skip=1; }
      # With lent slots (QlK, QeK) the CPU path takes every pass that does not arm P1 (the copy path refuses while
      # a slot is lent): its misses up to the chunk.
      [[ -n "$ql$qe" ]] && (( ${mm:-0} < lc )) && { log "arm $a: Ql / Qe need M$lc or more (with lent slots the CPU path takes every pass that does not arm P1); the arm is skipped"; skip=1; }
    else
      [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=1 "* ]] ||
        { log "arm $a: Lc needs B1 (the CPU takes every routed unit); the arm is skipped"; skip=1; }
      (( ${mm:-0} >= lc )) || { log "arm $a: Lc$lc needs M$lc or more (the CPU computes a chunk's misses); the arm is skipped"; skip=1; }
      # The pool's job size is not a matter of bits, but 4-row jobs (the default) take a chunk about twice
      # the CPU time of the many-row kernel's: an Lc arm names it (Mr32 or Mr64; Mr4 for a control).
      [[ " ${env[*]} " == *" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS="* ]] ||
        { log "arm $a: Lc needs an Mr part (Mr32 or Mr64: a chunk's experts in many-row jobs); the arm is skipped"; skip=1; }
    fi
    env+=(RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=)
  elif [[ -n "$xt" ]]; then env+=(RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1 RADIANCE_EXPERIMENTAL_DS41_LCP=0)
  else env+=(RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=0 RADIANCE_EXPERIMENTAL_DS41_LCP=0); fi
  if [[ -n "$ad" && -z "$lc" ]]; then log "arm $a: Ad$ad needs Lc (it acts on large chunks only); the arm is skipped"; skip=1; fi
  # Qsp is default-off; its oracle must include the new full/span wave comparisons.
  env+=(RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN=${qsp:-0})
  if [[ -n "$qsp" ]]; then
    [[ -n "$p1" ]] || { log "arm $a: Qsp needs P1; the arm is skipped"; skip=1; }
    if ! grep -qE '^p1: PASS: .*full/span SwiGLU waves' "$OUT/devcheck-p1.log" 2>/dev/null; then
      log "arm $a: Qsp needs the full/span SwiGLU card oracle; the arm is skipped"; skip=1
    fi
  fi
  # P1 (RADIANCE_EXPERIMENTAL_DS41_P1): with Lc, on the libr4d.so the `p1` device check passed on; every
  # other arm states the switch 0 and the Q knobs empty, so a window's VAR=val cannot turn them on.
  if [[ -n "$p1" ]]; then
    [[ -n "$lc" ]] || { log "arm $a: P1 needs an Lc part (the chunk the cards take); the arm is skipped"; skip=1; }
    if [[ -z "$skip" ]]; then
      IFS=: read -r -a homes <<< "$ARM_HOME"
      lib=""
      for h in "${homes[@]}"; do [[ -f "$h/kernels/libr4d.so" ]] && { lib=$h/kernels/libr4d.so; break; }; done
      if [[ -z "$P1_OK" ]]; then
        log "arm $a: P1 refused: $P1_WHY"; skip=1
      elif [[ -z "$lib" || "$(sha256sum "$lib" 2>/dev/null | cut -d' ' -f1)" != "$P1_SUM" ]]; then
        log "arm $a: P1 refused: its libr4d.so (${lib:-none under $ARM_HOME}) is not the one the p1 check passed on ($P1_LIB, sha256 ${P1_SUM:0:16})"; skip=1
      fi
    fi
    env+=(RADIANCE_EXPERIMENTAL_DS41_P1=1 RADIANCE_EXPERIMENTAL_DS41_P1_WAVE= RADIANCE_EXPERIMENTAL_DS41_P1_INFLIGHT=
          RADIANCE_EXPERIMENTAL_DS41_P1_READY= RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD= RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY=
          RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN= RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS= RADIANCE_EXPERIMENTAL_DS41_P1_LEND=0
          RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS= RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=0 "${q1[@]}")
    [[ -n "$qg" && -z "$ql$qe" ]] && { log "arm $a: Qh$qg sets the lent slots' grace, which needs a QlK or QeK part; the arm is skipped"; skip=1; }
  else
    (( ${#q1[@]} )) && { log "arm $a: the Q parts (${q1[*]}) need P1; the arm is skipped"; skip=1; }
    env+=(RADIANCE_EXPERIMENTAL_DS41_P1=0 RADIANCE_EXPERIMENTAL_DS41_P1_WAVE= RADIANCE_EXPERIMENTAL_DS41_P1_INFLIGHT=
          RADIANCE_EXPERIMENTAL_DS41_P1_READY= RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD= RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY=
          RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN= RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS= RADIANCE_EXPERIMENTAL_DS41_P1_LEND=0
          RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS= RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=0)
  fi
  [[ -n "$ad" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=)
  # Fw / WpN (the CPU pool's wake-up): every arm states both switches, 0 off, so a window's VAR=val
  # cannot turn them on for an arm whose name does not say so; arm() checks the server log both ways.
  # A pre-wake of more workers than the arm's pool has would be cut to its threads: refused instead.
  [[ -n "$fw" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=0)
  [[ -n "$wp" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=0)
  # WlN: the miss-read trigger's lead, 60 (the default) in every arm without the part.
  [[ -n "$wl" ]] || env+=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US=60)
  if [[ -n "$wl" && -z "$wp" ]]; then
    log "arm $a: Wl$wl sets the pre-wake's miss-read lead, which needs a WpN part; the arm is skipped"; skip=1
  fi
  if [[ -n "$wp" && "$pol" != base ]] && (( wp > threads )); then
    log "arm $a: Wp$wp pre-wakes more workers than the pool's $threads threads; the arm is skipped"; skip=1
  fi
  if [[ -n "$fw$wp" && "$pol" == base ]]; then
    log "arm $a: Fw / WpN act on the host experts' pool, which a base arm has not; the arm is skipped"; skip=1
  fi
  ARM_FW=$fw ARM_WP=$wp ARM_WL=$wl
  # In / Is / Iw / Ph: every arm states the switches, so a window's VAR=val cannot turn them on for
  # an arm whose name does not say so; arm() checks the server log both ways.
  if [[ -n "$ins_kib" ]] && (( ins_kib <= 0 )); then log "arm $a: Is0 slices nothing; the arm is skipped"; skip=1; fi
  if [[ -n "$ins_wait" && -z "$ins_kib" ]]; then log "arm $a: Iw needs an Is part (the wait is the pieces'); the arm is skipped"; skip=1; fi
  if [[ -n "$ins_nt$ins_kib" && " ${env[*]} " != *" RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1 "* ]]; then
    log "arm $a: In / Is write into the packed RAM cache only (a K part); the arm is skipped"; skip=1
  fi
  env+=(RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=${ins_nt:-0} RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB=${ins_kib:-0}
        RADIANCE_EXPERIMENTAL_DS41_INSERT_WAIT_US=${ins_wait} RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES=${lean:-0})
  ARM_INS_NT=$ins_nt ARM_INS_KIB=$ins_kib ARM_INS_WAIT=${ins_wait:-4000} ARM_LEAN=$lean
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
  ARM_XT=$xt ARM_BT=$bt ARM_KLD=$kld ARM_LC=$lc ARM_P1=$p1 ARM_QM=${qm:-320} ARM_QL=$ql ARM_QE=$qe ARM_QSP=$qsp
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
# DSpark arms (a Ds part): drafts verified and accepted, the acceptance and its ratio r to the first
# arm with the same Ds and Db parts in each prompt set (put the pinned-bank arm first, so a Dv arm's masked
# draft is read against the bank's unmasked one at its verify cap), tokens a step and decode tokens/s.
ds_arms=""
for a in $ARMS; do [[ "-$a-" =~ -Ds[0-9a]?- && -f "$OUT/$a.json" ]] && ds_arms+="$a "; done
[[ -n "$ds_arms" ]] && python3 -I "$HERE/ab_requests.py" --acceptance "$OUT" $ds_arms | tee -a "$OUT/window.log"
# Xd arms: each capture's share of draft selections the hot set its prompts' Dv arms use keeps at N = 8,
# 16, 24 (held-out coverage when that hot set was written from the other prompt set's capture).
for a in $ARMS; do
  [[ "-$a-" == *-Xd-* && -f "$OUT/xstats-$a.json" ]] || continue
  h=$(hotset_for "$a")
  [[ -f "$h" ]] && python3 -I "$HERE/draft_hotset.py" cover "$h" "$OUT/xstats-$a.json" 8 16 24 | tee -a "$OUT/window.log"
  # A Dv arm with Xd: its capture, taken under the mask, must select only the first N of each list.
  n=$(tr '-' '\n' <<< "$a" | grep -m1 -E '^Dv[0-9]+$'); n=${n#Dv}
  [[ -n "$n" && -f "$h" ]] && python3 -I "$HERE/draft_hotset.py" check "$h" "$n" "$OUT/xstats-$a.json" | tee -a "$OUT/window.log"
done
# Df arms: gemm_nt_bf16f8 gives gemm_nt's bytes, so the drafter proposes the same tokens; each Df
# arm's draft_n and draft_n_accepted must equal the same arm's without Df, as for Dk.
for a in $ARMS; do
  [[ "-$a-" == *-Df-* ]] || continue
  b=$(sed -E 's/-Df(-|$)/\1/' <<< "$a")
  if [[ " $ARMS " != *" $b "* ]]; then log "draft stats: no arm $b in this window to hold $a to"; continue; fi
  log "draft stats: $a against $b"
  python3 -I "$HERE/ab_requests.py" --draft-stats "$OUT" "$b" "$a" | tee -a "$OUT/window.log"
done
[[ " $ARMS" == *-Pc* ]] && python3 -I "$HERE/ab_requests.py" --ttft "$OUT" $ARMS | tee -a "$OUT/window.log"
# The read-only capture must preserve the draft/acceptance counts of its otherwise identical arm.
for a in $ARMS; do
  [[ "-$a-" == *-Dpc-* ]] || continue
  b=$(sed -E 's/-Dpc(-|$)/\1/' <<< "$a")
  if [[ " $ARMS " != *" $b "* ]]; then log "draft stats: no arm $b in this window to hold $a to"; continue; fi
  log "draft stats: read-only capture $a against $b"
  python3 -I "$HERE/ab_requests.py" --draft-stats "$OUT" "$b" "$a" | tee -a "$OUT/window.log"
done
log "arms done"
