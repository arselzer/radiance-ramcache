#!/usr/bin/env bash
# HOST-only F5 arm contracts. Extracts the actual arm parser; every physical action is stubbed.
# Run under work/build.lock. No server/model/GPU is opened.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-f5-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
mkdir -p "$T/home/kernels" "$T/extra" "$T/out"
echo fake > "$T/home/kernels/libr4d.so"
echo '{"max_context_tokens":48045,"benchmark_requests":[{"name":"long48k"}]}' > "$T/extra/long48k-receipt.json"
echo '{"benchmark_requests":[{"name":"long8k"}]}' > "$T/extra/long8k-receipt.json"
echo '{"benchmark_requests":[]}' > "$T/heldout.json"
R=all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs
POST=$R-N8-C8-Lm131072-Pf-Ba-F5post
WARM=$R-N8-C1-Lm131072-Pf-Ba-Ulong8k-F5warm
PLAIN=$R-Lm131072-Ulong8k
BADU=$POST-Ulong8k
BADN=${POST/-N8-/-N1-}
BADK=$POST-Kh
BADC=${WARM/-C1-/-C4-}
SHORT=${POST/-Lm131072-/-Lm32768-}
DUP=$POST-F5warm
(
  set +u
  unset AB_REQUEST_TIMEOUT_S AB_CONCURRENCY RADIANCE_TUNE
  log() { echo "LOG $*"; }
  power_set() { :; }; power_restore() { :; }; vpack_gate() { return 0; }
  arm() {
    echo "ARM $1"
    echo "MAIN ${ARM_PROMPTS:-$PROMPTS}|${ARM_N:-8}|${ARM_CONC:-1}"
    echo "BEFORE ${ARM_BEFORE_PROMPTS:-}|${ARM_BEFORE_N:-}|${ARM_BEFORE_CONC:-}"
    echo "OBSERVER ${ARM_OBS_CONC:-} TIMEOUT ${ARM_TIMEOUT:-}"
  }
  HX_THREADS=16 SILU=/dev/null RHOME=$T/home RADIANCE_HOME_DIR2=$T/home MIRROR=$T EXTRA_PROMPTS=$T/extra
  PROMPTS=$T/heldout.json OUT=$T/out CANCEL_FILE=/nonexistent/f5cancel CANCEL=$CANCEL_FILE
  SUM=$(sha256sum "$RHOME/kernels/libr4d.so" | cut -d' ' -f1); LIB=$RHOME/kernels/libr4d.so
  TALL_OK=1 TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=2048
  BAL_OK=1 BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=2048 P1_OK=1 P1_SUM=$SUM P1_LIB=$LIB
  ARMS="$PLAIN $POST $WARM $BADU $BADN $BADK $BADC $SHORT $DUP"
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) > "$T/loop" 2>&1
section() { awk -v a="ARM $1" '$0 == a { p=1; next } /^ARM / { p=0 } p' "$T/loop"; }
sec=$(section "$POST")
if [[ "$sec" == *"MAIN $T/heldout.json|8|8"* && "$sec" == *"BEFORE $T/extra/long48k-receipt.json|1|1"* &&
      "$sec" == *"OBSERVER 8 TIMEOUT 900"* ]]; then ok "48K C1 then original held-out N8"; else bad "post contract: $sec"; fi
sec=$(section "$WARM")
if [[ "$sec" == *"MAIN $T/extra/long8k-receipt.json|1|1"* && "$sec" == *"BEFORE $T/heldout.json|8|8"* &&
      "$sec" == *"OBSERVER 8 TIMEOUT 900"* ]]; then ok "held-out N8 then original Ulong8k C1"; else bad "warm contract: $sec"; fi
sec=$(section "$PLAIN")
[[ "$sec" == *"BEFORE ||"* ]] && ok "ordinary arm has no first phase" || bad "ordinary arm changed: $sec"
for a in "$BADU" "$BADN" "$BADK" "$BADC" "$SHORT" "$DUP"; do
  [[ -z "$(section "$a")" ]] && ok "invalid sequence skipped: ${a##*-}" || bad "invalid sequence launched: $a"
done
grep -q 'first phase needs 48045 tokens' "$T/loop" && ok "first-phase context checked" || bad "no first-phase context guard"
# The actual client command receives the prefix as argv, not by changing AB_CONCURRENCY for
# both phases. Existing no-prefix invocation keeps exactly its old positional arguments.
mkdir -p "$T/stub"
cat > "$T/stub/ab_requests.py" <<'PY'
import json, os, sys
json.dump({"argv":sys.argv[1:],"concurrency":os.environ["AB_CONCURRENCY"]}, open(os.environ["STUB_OUT"],"w"))
PY
cmd=$(awk '/^arm\(\) \{/ { p=1 } p && /^  env AB_CONCURRENCY=/ { print; getline; print; exit }' "$W")
(
  set +u
  HERE=$T/stub OUT=$T/out PORT=1 n=8 name=post ARM_PROMPTS=$T/heldout.json PROMPTS=$T/heldout.json
  ARM_CONC=8 ARM_TIMEOUT=900 before_args=(--before "$T/extra/long48k-receipt.json" 1 1)
  export STUB_OUT=$T/client.json
  eval "$cmd"
)
if python3 -I -c 'import json,sys
d=json.load(open(sys.argv[1]))
assert d["concurrency"]=="8"
assert d["argv"]==["--before",sys.argv[2]+"/extra/long48k-receipt.json","1","1",sys.argv[2]+"/heldout.json","http://127.0.0.1:1","8",sys.argv[2]+"/out/post.json"]' "$T/client.json" "$T"; then
  ok "real client invocation preserves both phase concurrencies"
else bad "real client invocation"; fi
exit "$fail"
