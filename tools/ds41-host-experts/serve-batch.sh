#!/usr/bin/env bash
# The measured best four-card batch configuration of DeepSeek-V4.1-Flash (README "Batch jobs: quickstart"):
# four 16 GB RDNA4 cards at TP4, no fifth card, many concurrent streams of short prompts that share long heads.
# The environment and flags are the window harness's for the arm (ab_window.sh, 12 November)
#
#   all-t40-f-M2048-H-p256-m229376-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-
#   P1-Qe64-Mb1-Rb-Vp-Cs-mA-Qd64-Qa0-Pc-Pch32-Pcs128-L294912-Pf-N256-C256-Ba2048-Bs-Qy60-Xqagent3
#
# taken from its stubbed dry run, with each variable resolved to its last assignment and each flag to its last value
# (the harness passes a few twice; the server keeps the last). What differs, on purpose:
#   - the shared-head floor is RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=32 (--shared-head), the pending default; the
#     measured rows used 64 (Pch64). 32 ran in the same windows' Pch32 arms.
#   - the placement profile: the arm's (agent3) holds private captures and is not shipped. --profile names yours;
#     without it the server ranks experts by the container's own activation profile (unmeasured for batches).
#   - the plane mirror (R, Rd13) only with --mirror; the measured arm had one on a second drive.
#   - hybrid decode (Qy60, RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=60, the share adapting) is the default: exact and +28%
#     decode at 256 streams (window ab-20261113T014719Z); --hybrid 0 turns it off (the arm without Qy).
#   - --max-seq-len N (SlN) only when given.
#   - --served-model-name ds41 (a name only; --served-name sets another) and --radiance-home only when given
#     (RADIANCE_HOME otherwise).
#
# Usage: serve-batch.sh --model MODEL.rad --silu-table FILE [--streams 256] [--port 8000] [--reserve-mib N]
#                       [--ctx-per-stream 1152] [--profile FILE] [--mirror DIR] [--shared-head 32] [--served-name ds41]
#                       [--radiance-home DIR] [--hybrid 60] [--max-seq-len N] [--dry-run]
#                       [-- more radiance flags, e.g. --expert-stats FILE to record a profile; outside the measured set]
#   RADIANCE_BIN (default: radiance on PATH, else build/bin/radiance), RADIANCE_HOME as for any radiance run.
# --dry-run prints the environment and the command and starts nothing. Otherwise the script becomes the server (exec:
# same pid), beside a memory guard that stops it (SIGINT) if MemAvailable falls below MEMGUARD_MIB (3072; 0 off), as the
# window harness does: an exhausted host can hit an amdgpu bug on the OOM path that panics the kernel.
set -u
die() { echo "serve-batch.sh: $*" >&2; exit 2; }
EXTRA=() NAME=ds41 MODEL="" SILU="" STREAMS=256 PORT=8000 RESERVE="" CTX=1152 PROFILE="" MIRROR="" SHEAD=32 DRY=0 HOME_DIR="" HYBRID=60 SEQLEN=""
while (( $# )); do
  case $1 in
    --model) MODEL=${2:?}; shift ;;
    --silu-table) SILU=${2:?}; shift ;;
    --streams) STREAMS=${2:?}; shift ;;
    --port) PORT=${2:?}; shift ;;
    --reserve-mib) RESERVE=${2:?}; shift ;;
    --ctx-per-stream) CTX=${2:?}; shift ;;
    --profile) PROFILE=${2:?}; shift ;;
    --mirror) MIRROR=${2:?}; shift ;;
    --shared-head) SHEAD=${2:?}; shift ;;
    --radiance-home) HOME_DIR=${2:?}; shift ;;
    --served-name) NAME=${2:?}; shift ;;
    --hybrid) HYBRID=${2:?}; shift ;;
    --max-seq-len) SEQLEN=${2:?}; shift ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '$1' (--help)" ;;
  esac
  shift
done
num() { [[ $2 =~ ^[0-9]+$ ]] && (( 10#$2 >= $3 )) || die "$1 takes a whole number of at least $3, not '$2'"; }
[[ -n "$MODEL" ]] || die "--model MODEL.rad is required"
[[ -n "$SILU" ]] || die "--silu-table FILE is required: the card's silu table (ds41_mxfp4_device_check LIBR4D.so swiglu FILE)"
num --streams "$STREAMS" 1; num --port "$PORT" 1; num --ctx-per-stream "$CTX" 1; num --shared-head "$SHEAD" 0
[[ -z "$RESERVE" ]] || num --reserve-mib "$RESERVE" 1024
num --hybrid "$HYBRID" 0; (( 10#$HYBRID <= 400 )) || die "--hybrid takes 0 (off) or a share of 1-400 percent, not '$HYBRID'"
[[ -z "$SEQLEN" ]] || num --max-seq-len "$SEQLEN" 64
STREAMS=$((10#$STREAMS)) CTX=$((10#$CTX))
if (( ! DRY )); then
  [[ -f "$MODEL" ]] || die "no model container at $MODEL"
  [[ -s "$SILU" ]] || die "no silu table at $SILU"
  [[ -z "$PROFILE" || -f "$PROFILE" ]] || die "no profile at $PROFILE"
  [[ -z "$MIRROR" || -d "$MIRROR" ]] || die "no mirror directory at $MIRROR"
fi
BIN=${RADIANCE_BIN:-$(command -v radiance || echo build/bin/radiance)}

# The arm's environment (133 assignments). The empty ones are set empty, as the harness sets them, so that a value
# in the caller's environment cannot change the configuration.
ENV=(
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=40
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU="$SILU"
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY=all
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=2048
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW=1
  RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS=6
  RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=2
  RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN=1
  RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1
  RADIANCE_EXPERIMENTAL_DS41_CACHE_SECOND_CHANCE=2
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K=4
  RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB=56
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM=1
  RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX=1
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS=32
  RADIANCE_EXPERIMENTAL_DS41_LCP=2048
  RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=1
  RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=1
  RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED=1
  RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD="$SHEAD"
  RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1
  RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8=1
  RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE=
  RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1
  RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16_FORM=0
  RADIANCE_EXPERIMENTAL_DS41_N128=0
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=0
  RADIANCE_EXPERIMENTAL_SCHED_TBO=0
  RADIANCE_EXPERIMENTAL_SCHED_BATCH_ADMIT=2048
  RADIANCE_EXPERIMENTAL_SCHED_BATCH_SPLIT=1
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RESCORE=12
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE_CHECK=0
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=0
  RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=7
  RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD=
  RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=
  RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP=1
  RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP_CKPT=512
  RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=0
  RADIANCE_EXPERIMENTAL_PREFIX_EXACT_PUBLISH=0
  RADIANCE_EXPERIMENTAL_PREFIX_EORS=0
  RADIANCE_EXPERIMENTAL_DS41_RESUME_GRID=0
  RADIANCE_EXPERIMENTAL_PREFIX_SESSION_LRU=0
  RADIANCE_EXPERIMENTAL_CHECKPOINT_STORE_MIB=
  RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD=
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_CODEBOOK_RAM=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY=off
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_PREFIX_POLICY=off
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE=0
  RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8=0
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM=0
  RADIANCE_EXPERIMENTAL_DS41_DRAFT_HOTSET=
  RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=1
  RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=0
  RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=
  RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN=0
  RADIANCE_EXPERIMENTAL_DS41_P1=1
  RADIANCE_EXPERIMENTAL_DS41_P1_WAVE=
  RADIANCE_EXPERIMENTAL_DS41_P1_INFLIGHT=
  RADIANCE_EXPERIMENTAL_DS41_P1_READY=
  RADIANCE_EXPERIMENTAL_DS41_P1_AHEAD=0
  RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY=
  RADIANCE_EXPERIMENTAL_DS41_P1_ALIGN=
  RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND=0
  RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=64
  RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=64
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=$((10#$HYBRID))   # QyP: hybrid decode, the CPU's share P%; 0 off
  RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT=            # empty: the share adapts (Qy, not Qyf)
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_HALF_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_MARGIN_PCT=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_PRIOR_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=0
  RADIANCE_EXPERIMENTAL_DS41_P1_LEND_REST_MS=
  RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=0
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US=60
  RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=0
  RADIANCE_EXPERIMENTAL_DS41_INSERT_SLICE_KIB=256
  RADIANCE_EXPERIMENTAL_DS41_INSERT_WAIT_US=
  RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES=0
  RADIANCE_EXPERIMENTAL_DS41_AR_ROWS=0
  RADIANCE_EXPERIMENTAL_DS41_AG_ROWS=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_DEVICE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_POLICY=
  RADIANCE_EXPERIMENTAL_DS41_AUX_DEADLINE_US=2000
  RADIANCE_EXPERIMENTAL_DS41_AUX_HEADROOM_MIB=1024
  RADIANCE_EXPERIMENTAL_DS41_AUX_BACKOFF=64
  RADIANCE_EXPERIMENTAL_DS41_AUX_FORCE_LATE=0
  RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_ALLOW=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FOREIGN_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_MIN_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FLOOR_MIB=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_POLL_MS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_RELEASE_POLLS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_REACQUIRE_S=
  RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FILL_MBPS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_RATE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MAX=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_EVERY=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_HALFLIFE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_PRIOR=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MARGIN=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_SPARE=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MBPS=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_CHECK=
  RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_DRY=
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS=1
  RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE=1
  RADIANCE_EXPERIMENTAL_DS41_MXFP4_PACK_SSE2=1
)
# mAN: the RAM budget's reserve (the engine's default 10240 MiB without it).
[[ -n "$RESERVE" ]] && ENV+=(RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB=$((10#$RESERVE)))

# The KV pool: --max-model-len is the pool the streams share, streams x tokens a stream (prompt plus reply). The
# measured arms: 256 x 1152 = 294912 and 128 x 1152 = 147456 for prompts of about 1K tokens and 128-token replies.
POOL=$((STREAMS * CTX))
ARGS=(
  --kernels libr4d,libref --model "$MODEL" --served-model-name "$NAME"
  --tp 4 --tp-wire exact --placement expert_tiered --weights-disk-tier
  --expert-ram-cache-total-mib 229376        # a cap: the RAM budget sizes the cache from what startup leaves
  --host-pool-mib 256 --host-headroom-mib 8192 --gpu-headroom-mib 256
  --embedding-placement ram --vision-tower-placement vram --ngram-placement disk
  --max-model-len "$POOL" --max-num-batched-tokens 2048 --max-num-seqs "$STREAMS"
  --kv-cache-dtype bf16 --num-speculative-tokens 0 --spec-ngram 0 --deterministic
  --prefix-resume exact --checkpoint-slots 128
  --host 127.0.0.1 --port "$PORT"
)
[[ -n "$HOME_DIR" ]] && ARGS=(--radiance-home "$HOME_DIR" "${ARGS[@]}")
[[ -n "$PROFILE" ]] && ARGS+=(--expert-profile "$PROFILE")
[[ -n "$SEQLEN" ]] && ARGS+=(--max-seq-len $((10#$SEQLEN)))
[[ -n "$MIRROR" ]] && ARGS+=(--expert-plane-mirror "$MIRROR" --expert-plane-mirror-decode-share 0.13)
ARGS+=("${EXTRA[@]}")

if (( DRY )); then
  printf 'env %s\n' "${ENV[@]}"
  printf 'cmd'; printf ' %q' "$BIN" "${ARGS[@]}"; echo
  exit 0
fi
# The memory guard is its own background process watching this pid, which `exec` then hands to the server: the
# process a caller starts and signals is radiance itself (a SIGINT reaches it, and nothing of it outlives a SIGKILL).
GUARD=${MEMGUARD_MIB:-3072}
LPID=$$
if (( GUARD > 0 )); then
  ( while kill -0 "$LPID" 2>/dev/null; do
      avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
      if (( avail < GUARD )); then
        echo "serve-batch.sh: MemAvailable ${avail} MiB below ${GUARD} MiB; stopping the server" >&2
        kill -INT "$LPID"; break
      fi
      sleep 1
    done ) < /dev/null &
fi
exec env "${ENV[@]}" "$BIN" "${ARGS[@]}"
