#!/usr/bin/env bash
# Fake-only source fixtures. No GPU, real server, model, service or dlopen. Run under build.lock.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-kv-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
expect() { local want=$1 label=$2 rc; shift 2; "$@" >/dev/null 2>&1; rc=$?;
  [[ $rc == "$want" ]] && ok "$label" || bad "$label: got$rc expected$want"; }
for fn in kv_cohort_key kv_server_gate kv_tool_receipt_ok kv_receipt_ok kv_check kv_gate kv_parts ar_rows_parse ag_rows_parse; do
  eval "$(sed -n "/^$fn() {/,/^}/p" "$W")"
  declare -F "$fn" >/dev/null || { bad "missing $fn"; exit 1; }
done
OUT=$T/out; mkdir -p "$OUT"
log() { echo "LOG $*" >>"$OUT/window.log"; }
H=$T/home; mkdir -p "$H/kernels" "$H/architectures"
LIB=$H/kernels/libr4d.so
echo fake-lib >"$LIB"; echo fake-ref >"$H/kernels/libref.so"; echo fake-arch >"$H/architectures/deepseek41.so"
BIN=$T/engine; echo '#!/bin/false' >"$BIN"; chmod +x "$BIN"
export KV_TEST_CALLS=$T/calls KV_TEST_LIB=$LIB
cat >"$T/checker-source" <<'EOF'
#!/usr/bin/env bash
set -u
[[ ${HIP_VISIBLE_DEVICES:-} == 0 && -z ${ROCR_VISIBLE_DEVICES+x}${CUDA_VISIBLE_DEVICES+x}${GPU_DEVICE_ORDINAL+x}${HSA_VISIBLE_DEVICES+x}${R4D_SELFTEST_ONE_CARD+x} ]] || { echo 'FAIL inherited cardmask'; exit 1; }
[[ -z ${RADIANCE_EXPERIMENTAL_DS41_FAKE_LEAK+x}${RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE+x}${RADIANCE_TUNE+x} ]] || { echo 'FAIL inherited knob'; exit 1; }
[[ ${RADIANCE_EXPERIMENTAL_DS41_KV_PACKED:-} == 1 && ${RADIANCE_EXPERIMENTAL_DS41_P1:-} == 0 ]] || exit 2
profile=unpinned
if [[ ${RADIANCE_EXPERIMENTAL_DS41_LCP:-} == 2048 ]]; then
  profile=lcp; [[ ${RADIANCE_EXPERIMENTAL_DS41_N128:-} == 1 && -z ${RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL+x} ]] || exit 2
elif [[ ${RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL:-} == 1 ]]; then
  profile=exact; [[ ${RADIANCE_EXPERIMENTAL_DS41_N128:-} == 0 && -z ${RADIANCE_EXPERIMENTAL_DS41_LCP+x} ]] || exit 2
else
  [[ ${RADIANCE_EXPERIMENTAL_DS41_N128:-} == 0 && -z ${RADIANCE_EXPERIMENTAL_DS41_LCP+x}${RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL+x} ]] || exit 2
fi
tool=$(basename "$0"); mode=${RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST:-}
[[ $mode == 0 || $mode == 1 ]] || exit 2
echo "$tool $profile $mode" >>"$KV_TEST_CALLS"
if [[ $tool == fast ]]; then
  [[ $# == 0 && $mode == 1 ]] || exit 2
  echo 'PASS ds41 fast codec vs reference: normal=131072 pair=131072 rows=256 metadata=1 bad=0'
else
  [[ $# == 1 && $1 == "$KV_TEST_LIB" ]] || exit 2
  echo 'PASS codec-domain: 382 groups, normal378 pair3 half-ftz1 raw0 bad0; actual minimum half bits +0000 -8000'
  echo 'PASS producer+packed-store: 0 uint16/guard differences'
  echo 'PASS producer+packed-store coverage: cases=20 kinds=1,2 blocks=1,2 M=1,7,33,257,512 exhaustive_bf16=65536'
  if [[ $tool == paths ]]; then
    decoder=reference; [[ $mode == 1 ]] && decoder=integer-fast
    echo "[r4d] ds41 KV_PACKED active rank=0 role=index_score format=2 row_bytes=68 dispatch=original decoder=$decoder M=1"
    echo "[r4d] ds41 KV_PACKED active rank=0 role=attention format=1 row_bytes=288 dispatch=original decoder=$decoder M=1"
    echo 'PASS packed fused rope/norm old+n128: 28 cases 0 byte failures'
    echo 'PASS packed fused coverage: cases=28 max_M=4096 heights=1,7,33,49,257,2048,4096 norm_forms=normal,old,n128 rope=normal'
    echo 'PASS packed reader normal/scalar6/decode7/WMMA: 504 cases 0 byte failures'
  fi
fi
[[ -z ${KV_MUTATE:-} ]] || printf '\n# mutated by FAKE checker\n' >>"$KV_MUTATE"
exit 0
EOF
for tool in codec paths fast; do cp "$T/checker-source" "$T/$tool"; chmod +x "$T/$tool"; done
KV_CODEC_CHECK=$T/codec KV_PATHS_CHECK=$T/paths KV_FAST_CHECK=$T/fast KV_COHORT_MANIFEST=$T/cohort
manifest() {
  {
    echo ds41-kv-cohort-v1
    echo "source_commit 0000000000000000000000000000000000000001"
    for field in engine architecture libref library codec_checker paths_checker fast_checker; do
      case $field in
        engine) file=$BIN ;; architecture) file=$H/architectures/deepseek41.so ;; libref) file=$H/kernels/libref.so ;;
        library) file=$LIB ;; codec_checker) file=$KV_CODEC_CHECK ;; paths_checker) file=$KV_PATHS_CHECK ;; fast_checker) file=$KV_FAST_CHECK ;;
      esac
      echo "$field $(sha256sum "$file" | cut -d' ' -f1)"
    done
  } >"$KV_COHORT_MANIFEST"
}
manifest
SUM=$(sha256sum "$LIB" | cut -d' ' -f1); COHORT=$(kv_cohort_key "$H" "$LIB" 0)
expect 0 'explicit coherent files match full hashes' kv_cohort_key "$H" "$LIB" 0
cp "$KV_COHORT_MANIFEST" "$T/cohort-good"
for mode in duplicate unknown truncated wrong_hash; do
  cp "$T/cohort-good" "$KV_COHORT_MANIFEST"
  case $mode in
    duplicate) echo 'source_commit 0000000000000000000000000000000000000001' >>"$KV_COHORT_MANIFEST" ;;
    unknown) echo 'inferred_capability yes' >>"$KV_COHORT_MANIFEST" ;;
    truncated) sed -i '/libref /d' "$KV_COHORT_MANIFEST" ;;
    wrong_hash) sed -i 's/^architecture .*/architecture 0000000000000000000000000000000000000000000000000000000000000000/' "$KV_COHORT_MANIFEST" ;;
  esac
  expect 1 "refuse $mode manifest" kv_cohort_key "$H" "$LIB" 0
done
cp "$T/cohort-good" "$KV_COHORT_MANIFEST"
sed -i 's/^fast_checker .*/fast_checker none/' "$KV_COHORT_MANIFEST"
expect 0 'initial packed cohort may explicitly omit optional fast checker' kv_cohort_key "$H" "$LIB" 0
expect 1 'no declared fast checker cannot authorize Kvfast' kv_cohort_key "$H" "$LIB" 1
cp "$T/cohort-good" "$KV_COHORT_MANIFEST"
fake_docker() { echo "${FAKE_STATE:-exited}"; }
DOCKER=(fake_docker) NAME=fake KV_WINDOW_TOKEN=fresh-window KV_WINDOW_ACTIVE=0 KV_PROOFS_OPEN=1
declare -A KV_PASSED=() KV_RECEIPT=()
expect 1 'no checker outside window' kv_check "$H" "$LIB" 0
[[ ! -e $KV_TEST_CALLS ]] || bad 'outside-window fake checker ran'
KV_WINDOW_ACTIVE=1 FAKE_STATE=running
expect 1 'no checker beside running QFN' kv_check "$H" "$LIB" 0
[[ ! -e $KV_TEST_CALLS ]] || bad 'running-QFN fake checker ran'
FAKE_STATE=exited KV_PROOFS_OPEN=0
expect 1 'no checker after proof phase closes' kv_check "$H" "$LIB" 0
KV_PROOFS_OPEN=1
expect 1 'fast needs initial packed reference receipts first' kv_check "$H" "$LIB" 1
export HIP_VISIBLE_DEVICES=99 ROCR_VISIBLE_DEVICES=99 CUDA_VISIBLE_DEVICES=99 GPU_DEVICE_ORDINAL=99 HSA_VISIBLE_DEVICES=99 R4D_SELFTEST_ONE_CARD=1
export RADIANCE_EXPERIMENTAL_DS41_FAKE_LEAK=1 RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE=1 RADIANCE_TUNE=fake
export RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1 RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=1
echo stale >"$OUT/kv-packed-$SUM-$COHORT-fast0.log"
expect 0 'fresh packed proof clears masks/knobs and replaces stale disk receipt' kv_check "$H" "$LIB" 0
expect 0 'fast proof runs separate numeric and FAST1 packed readers in all profiles' kv_check "$H" "$LIB" 1
[[ $(wc -l <"$KV_TEST_CALLS") == 12 ]] || bad 'missing profile/tool process'
for profile in unpinned exact lcp; do for item in 'codec 0' 'paths 0' 'fast 1' 'paths 1'; do
  read -r tool mode <<< "$item"
  [[ $(grep -Fxc "$tool $profile $mode" "$KV_TEST_CALLS") == 1 ]] || bad "coverage $tool/$profile/$mode"
done; done
expect 0 'same-window cached proof revalidates receipt' kv_check "$H" "$LIB" 0
[[ $(wc -l <"$KV_TEST_CALLS") == 12 ]] || bad 'unnecessary repeated checker'
FAKE_STATE=running
expect 1 'cached proof cannot run beside running QFN' kv_check "$H" "$LIB" 0
expect 1 'cached receipt cannot authorize an arm beside running QFN' kv_gate arm "$H" "$LIB" 0
FAKE_STATE=exited
KV_PROOFS_OPEN=0
expect 0 'packed arm revalidates without executing a checker' kv_gate arm "$H" "$LIB" 0
expect 0 'fast arm requires both fresh receipts' kv_gate arm "$H" "$LIB" 1
F=${KV_RECEIPT[$SUM:$COHORT:0]} FF=${KV_RECEIPT[$SUM:$COHORT:1]}
for kind in raw fail missing_profile missing_case missing_fused_height legacy_producer duplicate wrong_sha stale exit wrong_decoder; do
  cp "$F" "$T/bad.log"
  case $kind in
    raw) sed -i '0,/raw0/s/raw0/raw1/' "$T/bad.log" ;;
    fail) echo 'FAIL fake failure' >>"$T/bad.log" ;;
    missing_profile) sed -i '/kv begin profile=exact tool=codec/d' "$T/bad.log" ;;
    missing_case) sed -i 's/: 504 cases/: 503 cases/' "$T/bad.log" ;;
    missing_fused_height) sed -i 's/heights=1,7,33,49,257,2048,4096/heights=1,7,33,49,257,2048/' "$T/bad.log" ;;
    legacy_producer) sed -i '/producer+packed-store coverage:/d' "$T/bad.log" ;;
    duplicate) head -1 "$F" >>"$T/bad.log" ;;
    wrong_sha) sed -i "s/$SUM/0000000000000000000000000000000000000000000000000000000000000000/g" "$T/bad.log" ;;
    stale) sed -i 's/token=fresh-window/token=old-window/g' "$T/bad.log" ;;
    exit) sed -i 's/^kv exit 0$/kv exit 1/' "$T/bad.log" ;;
    wrong_decoder) sed -i 's/decoder=reference/decoder=integer-fast/g' "$T/bad.log" ;;
  esac
  expect 1 "reject $kind proof receipt" kv_receipt_ok "$T/bad.log" "$SUM" "$COHORT" 0
done
cp "$FF" "$T/bad.log"; sed -i 's/decoder=integer-fast/decoder=reference/g' "$T/bad.log"
expect 1 'FAST1 needs actual fast-reader markers' kv_receipt_ok "$T/bad.log" "$SUM" "$COHORT" 1
saved_token=$KV_WINDOW_TOKEN; KV_WINDOW_TOKEN=other-window
expect 1 'in-memory stale-window receipt refuses too' kv_gate stale "$H" "$LIB" 0
KV_WINDOW_TOKEN=$saved_token
for file in "$BIN" "$H/architectures/deepseek41.so" "$H/kernels/libref.so" "$LIB" "$KV_CODEC_CHECK" "$KV_PATHS_CHECK" "$KV_FAST_CHECK"; do
  cp "$file" "$T/saved-file"; echo '# changed' >>"$file"
  expect 1 "changed cohort file refused: $(basename "$file")" kv_gate changed "$H" "$LIB" 0
  cp "$T/saved-file" "$file"
done
# A fake mutation DURING otherwise-successful processes invalidates the whole fresh proof.
KV_PASSED=() KV_RECEIPT=() KV_PROOFS_OPEN=1
cp "$LIB" "$T/lib-good"; export KV_MUTATE=$LIB
expect 1 'pre/post full-SHA mutation refuses' kv_check "$H" "$LIB" 0
unset KV_MUTATE; cp "$T/lib-good" "$LIB"

emit_server() { # mode0/1, modern0/1
  local mode=$1 modern=$2 r field decoder
  decoder=reference; [[ $mode == 1 ]] && decoder=integer-fast
  for r in 0 1 2 3; do
    field=; [[ $modern == 1 ]] && field=" fast=$mode"
    echo "radiance: deepseek41: KV_PACKED rank=$r zero=0$field latent_format=1 row_bytes=288 groups=2 index_format=2 row_bytes=68 groups=2; BF16 reader math, plain window/F32 tail"
    echo "[r4d] ds41 KV_PACKED active rank=$r role=entry_store format=1 row_bytes=288"
    echo "[r4d] ds41 KV_PACKED active rank=$r role=entry_store format=2 row_bytes=68"
    field=; [[ $modern == 1 ]] && field=" decoder=$decoder"
    echo "[r4d] ds41 KV_PACKED active rank=$r role=index_score format=2 row_bytes=68 dispatch=original$field M=73"
    echo "[r4d] ds41 KV_PACKED active rank=$r role=attention format=1 row_bytes=288 dispatch=original$field M=73"
  done
}
echo plain >"$T/plain.log"; emit_server 0 0 >"$T/old.log"; emit_server 0 1 >"$T/ref.log"; emit_server 1 1 >"$T/fast.log"
expect 0 'plain arm stays plain' kv_server_gate "$T/plain.log" 0 0 exit 4
expect 0 'initial codec cohort has exact startup and four rank paths' kv_server_gate "$T/old.log" 1 0 exit 4
expect 0 'modern reference decoder complete' kv_server_gate "$T/ref.log" 1 0 exit 4
expect 0 'fast decoder complete' kv_server_gate "$T/fast.log" 1 1 exit 4
expect 1 'old plugin cannot establish fast capability' kv_server_gate "$T/old.log" 1 1 exit 4
expect 1 'plain arm cannot log packed paths' kv_server_gate "$T/ref.log" 0 0 exit 4
expect 1 'wrong requested decoder refuses' kv_server_gate "$T/fast.log" 1 0 exit 4
grep '^radiance:' "$T/ref.log" >"$T/start.log"
expect 0 'startup proof before requests needs only startup geometry' kv_server_gate "$T/start.log" 1 0 start 4
expect 1 'startup alone is insufficient at completion' kv_server_gate "$T/start.log" 1 0 exit 4
for kind in missing_rank duplicate duplicate_path missing_store zero_rank wrong_bytes wrong_format zero_rows mixed_startup; do
  cp "$T/ref.log" "$T/server-bad.log"
  case $kind in
    missing_rank) sed -i '/rank=3 /d' "$T/server-bad.log" ;;
    duplicate) head -1 "$T/ref.log" >>"$T/server-bad.log" ;;
    duplicate_path) grep 'rank=0 role=index_score ' "$T/ref.log" >>"$T/server-bad.log" ;;
    missing_store) sed -i '/rank=2 role=entry_store format=2 /d' "$T/server-bad.log" ;;
    zero_rank) sed -i 's/rank=1 zero=0/rank=1 zero=1/' "$T/server-bad.log" ;;
    wrong_bytes) sed -i 's/row_bytes=68/row_bytes=80/g' "$T/server-bad.log" ;;
    wrong_format) sed -i 's/index_format=2/index_format=1/g' "$T/server-bad.log" ;;
    zero_rows) sed -i 's/M=73/M=0/g' "$T/server-bad.log" ;;
    mixed_startup) sed -i 's/rank=1 zero=0 fast=0/rank=1 zero=0/' "$T/server-bad.log" ;;
  esac
  expect 1 "reject server $kind" kv_server_gate "$T/server-bad.log" 1 0 exit 4
done
for a in base-Kvfp4 base-Kvfp4-Kvfast base-Kvfast-Kvfp4; do expect 0 "parse $a" kv_parts "$a"; done
for a in base-Kvfast base-Kvfp4-Kvfp4 base-Kvfp4-Kvfast-Kvfast base-Kvgarbage; do expect 1 "reject $a" kv_parts "$a"; done

# KLD reference rule on fake completed processes. No actual server/python command.
(
  eval "$(sed -n '/^kld_finish() {/,/^}/p' "$W")"
  kill(){ return 1; }; wait(){ return 0; }; xt_check(){ return 0; }; lcp_check(){ return 0; }
  n128_server_gate(){ return 0; }; vp_server_gate(){ return 0; }; qsp_gate(){ return 0; }; python3(){ return 0; }
  OUT=$T/kld;mkdir -p "$OUT"; cp "$T/ref.log" "$OUT/packed.server.log";cp "$T/plain.log" "$OUT/plain.server.log"
  SERVER_PID=fake KLD_REF= ARM_KVFP4=1 ARM_KVFAST=0 ARGS=()
  ARM_XT= ARM_LC= ARM_N128= ARM_AR_ROWS= ARM_AG_ROWS=
  expect 1 'packed KLD cannot become its own reference' kld_finish packed
  [[ -z $KLD_REF ]] || bad 'packed KLD reference escaped'
  ARM_KVFP4=0 SERVER_PID=fake
  expect 0 'plain KLD can establish reference' kld_finish plain
  [[ $KLD_REF == "$OUT/kld-plain" ]] || bad 'plain reference not recorded'
  (( fail == 0 ))
) || fail=1

# Main loop only; every side effect and gate is a fake. Existing independent AR/AG gates must run.
out=$(
  set +u
  log(){ echo "LOG $*"; };power_set(){ :; };power_restore(){ :; }
  kv_gate(){ echo "KV_GATE $1 mode=$4"; [[ $1 != *-r9 ]]; }
  ar_rows_gate(){ echo "AR_GATE $1"; };ag_rows_gate(){ echo "AG_GATE $1"; };ar_ag_shared_gate(){ echo "SHARED_GATE $1"; }
  arm(){ echo "ARM $1";shift;printf ' ENV %s\n' "$@"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H PROMPTS=$T/prompts.json CANCEL=/nonexistent/cancel
  TALL_OK=1 TALL_SUM=$SUM TALL_LIB=$LIB TALL_WHY=fake
  echo '{}' >"$PROMPTS";unset RADIANCE_TUNE
  ARMS='base-r0 base-Kvfp4 base-Kvfp4-Kvfast base-Kvfast base-Kvfp4-Kvfp4 base-Kvfp4-Kvfast-Kvfast base-Kvgarbage base-Kvfp4-Ar1280-Ag1024-BT1280 base-Kvfp4-BT4096-Xt base-Kvfp4-BT4096 base-BT4096 base-Kvfp4-r9 base-r1'
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR>s&&/^done$/{print NR;exit}' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
)
section(){ awk -v arm="ARM $1" '$0==arm{on=1;next}/^ARM /{on=0}on{print}' <<<"$out"; }
for a in base-r0 base-r1 base-BT4096; do
  body=$(section "$a")
  [[ $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=0'* && $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=0'* ]] || bad "global flags leaked into $a"
done
body=$(section base-Kvfp4)
[[ $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1'* && $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=0'* ]] || bad 'Kvfp4 lacks explicit1/0'
body=$(section base-Kvfp4-Kvfast)
[[ $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1'* && $body == *'RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=1'* ]] || bad 'Kvfast lacks explicit1/1'
grep -Fqx 'ARM base-Kvfp4-BT4096-Xt' <<<"$out" || bad 'pinned4096 bound refused despite full-height oracle'
for a in base-Kvfast base-Kvfp4-Kvfp4 base-Kvfp4-Kvfast-Kvfast base-Kvgarbage base-Kvfp4-BT4096 base-Kvfp4-r9; do
  grep -Fqx "ARM $a" <<<"$out" && bad "invalid/refused arm launched $a"
done
for gate in KV AR AG SHARED; do
  grep -Fq "${gate}_GATE base-Kvfp4-Ar1280-Ag1024-BT1280" <<<"$out" || bad "existing $gate guard bypassed"
done
(( fail )) && exit 1
echo 'ab_window_kv_test: all fake checks passed'
