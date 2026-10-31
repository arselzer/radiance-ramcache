#!/usr/bin/env bash
# Fake-only host fixtures: no model, device, real docker, protected writes or dlopen.
# Source-prepared only; run under build.lock after coherent build.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HERE/ab_window.sh}
source "$HERE/ab_window_window_kv.sh"
for fn in kv_cohort_key kv_server_gate kv_parts; do eval "$(sed -n "/^$fn() {/,/^}/p" "$W")"; done
T=$(mktemp -d /tmp/ab-window-kv-test-XXXXXX);trap 'rm -rf "$T"' EXIT
bad=0
expect(){ local want=$1 label=$2 rc;shift 2;"$@" >/dev/null 2>&1;rc=$?;
  if [[ $rc == "$want" ]];then echo "ok $label";else echo "FAIL $label got=$rc expected=$want";bad=1;fi; }
OUT=$T/out;mkdir -p "$OUT";log(){ echo "$*" >>"$OUT/window.log"; }
R=$T/research;mkdir -p "$R"
touch "$R/research-compiler.lock" "$R/research-heavy-runtime.lock"
FAKE_STATE=exited FAKE_OWNER=$$ FAKE_LOCKED=1
fake_docker(){ echo "$FAKE_STATE"; }
DOCKER=(fake_docker);NAME=fake-qfn
cat(){ if [[ $# == 1 && $1 == /home/as/ds41-claude-build/work/qfn_stopped_by ]];then echo "$FAKE_OWNER";else command cat "$@";fi; }
flock(){ [[ $FAKE_LOCKED != 1 ]]; }
KV_WINDOW_ACTIVE=1 KV_PROOFS_OPEN=1 KV_WINDOW_TOKEN=fake-fresh-token
expect 0 'owned stopped two-lock scope' window_kv_scope
KV_WINDOW_ACTIVE=0;expect 1 'outside window' window_kv_scope;KV_WINDOW_ACTIVE=1
KV_PROOFS_OPEN=0;expect 1 'proof stage closed' window_kv_scope;KV_PROOFS_OPEN=1
FAKE_STATE=running;expect 1 'QFN still running' window_kv_scope;FAKE_STATE=exited
FAKE_OWNER=1;expect 1 'wrong stop owner' window_kv_scope;FAKE_OWNER=$$
FAKE_LOCKED=0;expect 1 'research locks absent' window_kv_scope;FAKE_LOCKED=1
expect 0 'Kvwin requires Kvfp4 valid' kv_parts greedy-Kvfp4-Kvwin
for arm in greedy-Kvwin greedy-Kvfp4-Kvwin-Kvwin greedy-Kvfp4-Kvunknown;do expect 1 "refuse $arm" kv_parts "$arm";done
H=$T/home;mkdir -p "$H/architectures" "$H/kernels"
LIB=$H/kernels/libr4d.so;echo fake-lib >"$LIB";echo fake-ref >"$H/kernels/libref.so";echo fake-arch >"$H/architectures/deepseek41.so"
BIN=$T/engine;printf '#!/bin/false\n' >"$BIN";chmod +x "$BIN"
KV_CODEC_CHECK=$T/base-codec KV_PATHS_CHECK=$T/base-paths KV_FAST_CHECK=$T/base-fast
for f in "$KV_CODEC_CHECK" "$KV_PATHS_CHECK" "$KV_FAST_CHECK";do cp "$BIN" "$f";done
WINDOW_KV_CODEC_CHECK=$T/window-codec WINDOW_KV_PATHS_CHECK=$T/window-paths
command cat >"$T/fake-checker" <<'EOF'
#!/usr/bin/env bash
set -u
[[ ${HIP_VISIBLE_DEVICES:-} =~ ^[0-3]$ && -z ${ROCR_VISIBLE_DEVICES+x}${CUDA_VISIBLE_DEVICES+x}${GPU_DEVICE_ORDINAL+x}${HSA_VISIBLE_DEVICES+x}${R4D_SELFTEST_ONE_CARD+x}${RADIANCE_TUNE+x} ]] || exit 2
[[ -z ${RADIANCE_EXPERIMENTAL_DS41_FAKE_LEAK+x} && ${RADIANCE_EXPERIMENTAL_DS41_WINDOW_PACKED:-} == 1 && ${RADIANCE_EXPERIMENTAL_DS41_KV_PACKED:-} == 1 && ${RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST:-} =~ ^[01]$ ]] || exit 2
[[ $# == 1 && -f $1 ]] || exit 2
echo 'PASS window-fp8-domain groups=1146 raw=0 bad=0 pair=2'
for i in 0 1 2 3;do echo "PASS window-fp8-original-producer profile=$i groups=65536 raw=0 bad=0 scale_bad=0 byte_guard_bad=0 pair=1";done
echo 'PASS ds41_window_codec_exact'
if [[ $0 == *window-paths ]];then
  echo 'PASS window packed producers cases=84 raw=0 byte_errors=0 forms=7 max_M=4096'
  echo 'PASS window packed readers cases=1152 byte_errors=0 forms=32767'
  echo 'PASS ds41_window_packed_paths_exact producers=84 readers=1152 max_M=4096 raw=0 byte_errors=0'
fi
EOF
for f in "$WINDOW_KV_CODEC_CHECK" "$WINDOW_KV_PATHS_CHECK";do cp "$T/fake-checker" "$f";chmod +x "$f";done
KV_COHORT_MANIFEST=$T/base-manifest WINDOW_KV_COHORT_MANIFEST=$T/window-manifest
{
  echo ds41-kv-cohort-v1;echo source_commit 0000000000000000000000000000000000000001
  for key in engine architecture libref library codec_checker paths_checker;do
    case $key in engine)f=$BIN;;architecture)f=$H/architectures/deepseek41.so;;libref)f=$H/kernels/libref.so;;library)f=$LIB;;codec_checker)f=$KV_CODEC_CHECK;;paths_checker)f=$KV_PATHS_CHECK;;esac
    echo "$key $(sha256sum "$f"|cut -d' ' -f1)"
  done
  echo "fast_checker $(sha256sum "$KV_FAST_CHECK"|cut -d' ' -f1)"
} >"$KV_COHORT_MANIFEST"
{
  echo ds41-window-kv-cohort-v1;echo source_commit 0000000000000000000000000000000000000001
  echo "packed_cohort $(sha256sum "$KV_COHORT_MANIFEST"|cut -d' ' -f1)"
  echo "codec_checker $(sha256sum "$WINDOW_KV_CODEC_CHECK"|cut -d' ' -f1)"
  echo "paths_checker $(sha256sum "$WINDOW_KV_PATHS_CHECK"|cut -d' ' -f1)"
} >"$WINDOW_KV_COHORT_MANIFEST"
expect 0 'explicit full cohort authority' window_kv_cohort_key "$H" "$LIB" 0
cp "$WINDOW_KV_COHORT_MANIFEST" "$T/manifest-good"
echo source_commit 0000000000000000000000000000000000000001 >>"$WINDOW_KV_COHORT_MANIFEST"
expect 1 'duplicate manifest key' window_kv_cohort_key "$H" "$LIB" 0
cp "$T/manifest-good" "$WINDOW_KV_COHORT_MANIFEST"
declare -A WINDOW_KV_PASSED=() WINDOW_KV_RECEIPT=()
export HIP_VISIBLE_DEVICES=3 ROCR_VISIBLE_DEVICES=3 RADIANCE_EXPERIMENTAL_DS41_FAKE_LEAK=1
expect 0 'fake oracle gets clean card/profile environment in all12scopes' window_kv_check "$H" "$LIB" 0
expect 0 'same-window gate accepts fresh receipt' window_kv_gate "$H" "$LIB" 0
expect 0 'same-profile optional FP4-fast combined oracle' window_kv_check "$H" "$LIB" 1
expect 0 'optional fast gate requires both fresh mode receipts' window_kv_gate "$H" "$LIB" 1
sum=$(sha256sum "$LIB"|cut -d' ' -f1);cohort=$(window_kv_cohort_key "$H" "$LIB" 0);key=$sum:$cohort:0
receipt=${WINDOW_KV_RECEIPT[$key]};cp "$receipt" "$T/receipt-good"
KV_WINDOW_TOKEN=stale;expect 1 'stale window receipt' window_kv_gate "$H" "$LIB" 0;KV_WINDOW_TOKEN=fake-fresh-token
sed -i '/^window-kv begin card=3 profile=lcp /d' "$receipt"
expect 1 'missing card/profile scope' window_kv_gate "$H" "$LIB" 0;cp "$T/receipt-good" "$receipt"
sed -i 's/raw=0 byte_errors=0 forms=7/raw=1 byte_errors=0 forms=7/' "$receipt"
expect 1 'Raw signal never tolerated' window_kv_gate "$H" "$LIB" 0;cp "$T/receipt-good" "$receipt"
echo changed >>"$LIB";expect 1 'changed full library SHA' window_kv_gate "$H" "$LIB" 0
f=$T/server
{
for r in 0 1 2 3;do
  echo "radiance: deepseek41: KV_PACKED rank=$r zero=0 fast=0 latent_format=1 row_bytes=288 groups=2 index_format=2 row_bytes=68 groups=2; BF16 reader math, packed FP8 window/F32 tail"
  echo "radiance: deepseek41: WINDOW_PACKED rank=$r zero=0 format=3 row_bytes=528 page_bytes=8448 block_size=16 groups=1; BF16 reader math, plain index-window/F32 tail"
  echo "INFO ds41 WINDOW_PACKED admitted rank=$r format=3 row_bytes=528 page_bytes=8448 groups=1 layers=43 producers=43 stores=43 readers=43"
  echo "[r4d] ds41 WINDOW_PACKED active rank=$r role=store format=3 row_bytes=528 page_bytes=8448"
  echo "[r4d] ds41 WINDOW_PACKED active rank=$r role=attention format=3 row_bytes=528 page_bytes=8448 decoder=reference M=1"
  echo "[r4d] ds41 KV_PACKED active rank=$r role=entry_store format=1 row_bytes=288"
  echo "[r4d] ds41 KV_PACKED active rank=$r role=entry_store format=2 row_bytes=68"
  echo "[r4d] ds41 KV_PACKED active rank=$r role=index_score format=2 row_bytes=68 dispatch=original decoder=reference M=1"
  echo "[r4d] ds41 KV_PACKED active rank=$r role=attention format=1 row_bytes=288 dispatch=original decoder=reference M=1"
done
} >"$f"
expect 0 'actual four-rank admitted+store+reader proof' kv_server_gate "$f" 1 0 exit 4 1
expect 1 'per-arm off rejects leaked WINDOW marker' kv_server_gate "$f" 1 0 exit 4 0
cp "$f" "$T/server-good";sed -i '/WINDOW_PACKED active rank=3 role=attention/d' "$f"
expect 1 'missing actual reader rank' kv_server_gate "$f" 1 0 exit 4 1
cp "$T/server-good" "$f";grep 'WINDOW_PACKED active rank=1 role=store' "$f" >>"$T/duplicate";command cat "$T/duplicate" >>"$f"
expect 1 'duplicate actual store rank' kv_server_gate "$f" 1 0 exit 4 1
grep -Fq 'RADIANCE_EXPERIMENTAL_DS41_WINDOW_PACKED=${kvwin:-0}' "$W" || bad=1
grep -Fq 'RADIANCE_EXPERIMENTAL_DS41_WINDOW_PACKED=0' "$W" || bad=1
grep -Fq '${ARM_KVFP4:-0} == 1' "$W" || bad=1 # Kvwin requiresKvfp4, therefore existing KLD-reference guard applies.
exit "$bad"
