#!/usr/bin/env bash
# serve_batch_test.sh [ab_window.sh] -- serve-batch.sh against the window harness, with no GPU, no server and no model:
#   1. the harness's own arm loop (every action stubbed, as ab_window_ba_test.sh) builds the measured four-card batch
#      arm, and arm()'s own launch line (taken from ab_window.sh, its `env` captured) gives the server's environment
#      and flags; serve-batch.sh --dry-run gives the same environment, each variable at its last assignment, and the
#      same flags, each at its last value, apart from the port, --radiance-home and --served-model-name;
#   2. --streams 128 gives the 128-stream arm's pool (L147456, N128), --reserve-mib the mAN variable, --shared-head
#      the PchN floor, and no --profile / --mirror leaves those flags out;
#   3. a missing --silu-table or --model, and a reserve under 1024 MiB, are refused.
# Exit 0 when every check passes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HERE/ab_window.sh}
SB=$HERE/serve-batch.sh
T=$(mktemp -d /tmp/serve-batch-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

ARM=all-t40-f-M2048-H-p256-m229376-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs-mA-Qd64-Qa0-Pc-Pch32-Pcs128-L294912-Pf-N256-C256-Ba2048-Bs-Xqprof
mkdir -p "$T/home/kernels" "$T/mirror" "$T/profiles"; echo "{}" > "$T/profiles/prof.json"
echo "not a library" > "$T/home/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
# arm()'s launch, its `env` a capture: one argument a line.
LAUNCH=$(awk '/^  env "\$@" RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS/ { p = 1 } p { print } p && /server\.log" 2>&1 &$/ { exit }' "$W" |
         sed -e '1s/^  env /  capture /' -e '$s/ >"\$OUT\/\$name\.server\.log" 2>&1 &$//')
[[ $LAUNCH == *capture* && $LAUNCH == *'"${ARGS[@]}"' ]] || bad "arm()'s launch line not found in $W"
(
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  capture() { printf '%s\n' "$@"; }
  arm() { local name=$1; shift; echo "ARM $name"; eval "$LAUNCH"; }
  HX_THREADS=16 SILU=SILU RHOME=$T/home RADIANCE_HOME_DIR2=$T/home MIRROR=$T/mirror PROMPTS=$T/prompts.json
  BIN=BIN MODEL=MODEL PORT=18089 PROFILE=SEALED EXTRA_PROFILES=$T/profiles
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$T/home/kernels/libr4d.so" | cut -d' ' -f1); LIB=$T/home/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  ARMS=$ARM
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) > "$T/loop.txt" 2>&1
grep -q "^ARM $ARM$" "$T/loop.txt" && ok "the harness launches the arm" ||
  { bad "the harness did not launch the arm: $(grep -m3 -E 'skip|unknown' "$T/loop.txt")"; }
awk -v a="ARM $ARM" '$0 == a { p = 1; next } p' "$T/loop.txt" > "$T/harness.txt"

"$SB" --model MODEL --silu-table SILU --profile "$T/profiles/prof.json" --mirror "$T/mirror" --dry-run > "$T/sb.txt" ||
  bad "serve-batch.sh --dry-run failed"

python3 -I - "$T/harness.txt" "$T/sb.txt" <<'EOF' || fail=1
import shlex, sys
NOVAL = {"--weights-disk-tier", "--deterministic", "--no-prefix-cache"}
def resolve(words):
    env, order, i = {}, [], 0
    while i < len(words) and "=" in words[i] and words[i].startswith("RADIANCE_"):
        k, v = words[i].split("=", 1)
        if k not in env: order.append(k)
        env[k] = v; i += 1
    binary, flags, i = words[i], {}, i + 1
    while i < len(words):
        k = words[i]
        if k in NOVAL: flags[k] = None; i += 1
        else: flags[k] = words[i + 1]; i += 2
    return [(k, env[k]) for k in order], binary, flags
h = resolve([l for l in open(sys.argv[1]).read().split("\n") if l != ""])
lines = open(sys.argv[2]).read().splitlines()
words = [l[4:] for l in lines if l.startswith("env ")] + shlex.split([l for l in lines if l.startswith("cmd ")][0][4:])
s = resolve(words)
bad = 0
if h[0] == s[0]: print("ok   the environment: %d variables, the same values in the same order" % len(h[0]))
else:
    bad = 1
    hd, sd = dict(h[0]), dict(s[0])
    for k in sorted(set(hd) | set(sd)):
        if hd.get(k) != sd.get(k): print("FAIL env %s: harness %r, serve-batch %r" % (k, hd.get(k), sd.get(k)))
    if hd == sd: print("FAIL the environment's order differs")
allowed = {"--port", "--radiance-home", "--served-model-name"}
for k in sorted(set(h[2]) | set(s[2])):
    if k in allowed: continue
    if h[2].get(k, "<absent>") != s[2].get(k, "<absent>"):
        bad = 1; print("FAIL flag %s: harness %r, serve-batch %r" % (k, h[2].get(k, "<absent>"), s[2].get(k, "<absent>")))
if not bad: print("ok   the flags: %d, the same values (port, --radiance-home and the served name aside)" % len(h[2]))
sys.exit(bad)
EOF

# ---- 2. the parameters
d128=$("$SB" --model M --silu-table S --streams 128 --dry-run)
[[ $d128 == *"--max-model-len 147456 "* && $d128 == *"--max-num-seqs 128 "* ]] && ok "128 streams: L147456, N128" ||
  bad "128 streams: $(grep '^cmd' <<< "$d128")"
dres=$("$SB" --model M --silu-table S --reserve-mib 20480 --shared-head 64 --port 9000 --dry-run)
grep -qx 'env RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB=20480' <<< "$dres" && ok "--reserve-mib: the mAN variable" ||
  bad "--reserve-mib"
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=64' <<< "$dres" && ok "--shared-head 64: Pch64's floor" ||
  bad "--shared-head"
[[ $dres == *"--port 9000"* ]] && ok "--port" || bad "--port"
d0=$("$SB" --model M --silu-table S --dry-run)
if grep -q 'RAM_RESERVE_MIB' <<< "$d0" || [[ $d0 == *--expert-profile* || $d0 == *--expert-plane-mirror* ]]; then
  bad "the defaults add a reserve, a profile or a mirror"
else ok "the defaults: no reserve variable, no --expert-profile, no mirror"; fi
grep -qx 'env RADIANCE_EXPERIMENTAL_PREFIX_SHARED_HEAD=32' <<< "$d0" && ok "the shared-head floor's default: 32" ||
  bad "the shared-head default"

# ---- 3. refusals
"$SB" --model M --dry-run > /dev/null 2>&1 && bad "no --silu-table accepted" || ok "no --silu-table refused"
"$SB" --silu-table S --dry-run > /dev/null 2>&1 && bad "no --model accepted" || ok "no --model refused"
"$SB" --model M --silu-table S --reserve-mib 512 --dry-run > /dev/null 2>&1 && bad "a 512 MiB reserve accepted" ||
  ok "a reserve under 1024 MiB refused"
"$SB" --model "$T/none.rad" --silu-table S > /dev/null 2>&1 && bad "a missing container accepted" ||
  ok "a missing container refused before any start"

(( fail == 0 )) && echo "serve_batch_test: all passed"
exit $fail
