#!/usr/bin/env bash
# ab_window_lh_test.sh [ab_window.sh] -- ab_window.sh's Lh part (LhS: the lent slots follow live decode
# routing, RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=S) with no GPU, no server and no model:
#   1. lcp_check on synthetic server logs: an Lh arm is measured only with rank 0's "ds41 p1: lent heat: on
#      (..._P1_LEND_HEAT=S;" line naming its S, and an arm without the part only without such a line;
#   2. the arm loop, every action stubbed (as ab_window_p1_test.sh): P1 at Lc2048 with Qe64 and Lh2 (Lh8; Lh1
#      beside Ql10 and Qe64) is launched with the switch at its S, the lent arena on and the two tunables
#      stated empty; the same arm without Lh, launched right after an Lh arm, the Ql arm and the adopted
#      large-chunk arm state the switch 0 (and the tunables empty); Lh with the lent ring alone, with P1
#      alone, without P1, Lh0, Lh9, Lh2x, Lhx and a second Lh part are skipped;
#   3. the adaptive residency parts: LsN (the lent slab, ..._P1_LEND_SLAB=N), La (the lent slots' read-back
#      audit, ..._P1_LEND_AUDIT=1) and LhhR (the heat's half-life, ..._P1_LEND_HEAT_HALF_ROWS=R): lcp_check
#      on their startup lines (an Ls arm only with its "lent slab: N slots (..._P1_LEND_SLAB=N;" line, an La
#      arm only with "lent audit: on", an Lh arm only with the half-life it asks for: Lhh's, 2000 beside Ls,
#      else 100); lend_exit_gate on the shutdown stats lines (a slab that did not hold all its units at
#      bring-up, an audit that read nothing back or found a slot differing or unreadable, a heat that never
#      counted a half-life of rows: each set aside with its reason); and the arm loop (each arm states the
#      three variables; Ls without P1 or with an M part below the chunk, La with nothing lent, Lhh without
#      Lh, and values out of range are skipped; Lh beside Ls needs no Qe).
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-lh-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
HEAT=RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT

# ---- 1. the server-log gate
eval "$(sed -n '/^lcp_check() {/,/^}/p' "$W")"
declare -F lcp_check >/dev/null || { echo "FAIL no lcp_check in $W"; exit 1; }
OUT=$T/logs
mkdir -p "$OUT"
log() { echo "LOG: $*" >> "$OUT/window.log"; }
lc_line='I[0] large-chunk prefill: passes of up to 2048 tokens (RADIANCE_EXPERIMENTAL_DS41_LCP=2048): ...'
p1_line="I[0] ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of 320 to 2048 rows on the cards (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=320; ...)"
arena_line="I[0] ds41 p1: lent arena: 42 slots (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=64; 42 fit) in the activation arena's top, ..."
# The engine's line (FileStager::p1_lend_init, ds41p1::kLogLendHeat).
heat_line() { echo "I[0] ds41 p1: lent heat: on (${HEAT}=$1; half-life 100 decode rows, ${HEAT}_HALF_ROWS; margin 150%, ${HEAT}_MARGIN_PCT): each rank's lent slots (rank 0: 42) follow live decode routing, at most $1 handed to another unit a pass; ..."; }
mk() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.server.log"; }
# The rest's line (FileStager::p1_lend_init, ds41p1::kLogLendRest): 5000 ms by default beside the heat or the slab.
REST=RADIANCE_EXPERIMENTAL_DS41_P1_LEND_REST_MS
rest_line() { echo "I[0] ds41 p1: lent rest: $1 ms (${REST}=$1${2:-, the default beside the heat or the slab}): from a pass that takes the lent slots back ..."; }
mk arena "$lc_line" "$p1_line" "$arena_line"
mk heat2 "$lc_line" "$p1_line" "$arena_line" "$(heat_line 2)" "$(rest_line 5000)"
mk heat22 "$lc_line" "$p1_line" "$arena_line" "$(heat_line 22)" "$(rest_line 5000)"
mk plain "$lc_line"
expect() {   # want(0/1) what log P1 QE LH
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=$4 ARM_QM=320 ARM_QL= ARM_QE=$5 ARM_LH=$6 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect 0 "Qe64 with Lh2, its line"                                   heat2  1 64 2
expect 1 "Qe64 with Lh2, no lent-heat line"                          arena  1 64 2
expect 1 "Lh4 on a server that took 2"                               heat2  1 64 4
expect 1 "Lh2 on a server whose line says 22"                        heat22 1 64 2
expect 1 "no Lh part on a server whose lent slots follow routing"    heat2  1 64 ""
expect 0 "Qe64 without Lh, no lent-heat line"                        arena  1 64 ""
expect 0 "the plain Lc arm"                                          plain  "" "" ""
# An Lh arm's line goes to the window log.
: > "$OUT/window.log"
ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL= ARM_QE=64 ARM_LH=2 lcp_check heat2 >/dev/null 2>&1
grep -q "ds41 p1: lent heat: on (${HEAT}=2;" "$OUT/window.log" && ok "the window log takes the lent-heat line" ||
  bad "the window log has no lent-heat line"

# ---- 3a. Ls / La / Lhh at startup (lcp_check)
SLAB=RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB
AUDIT=RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT
# The engine's lines (FileStager::p1_lend_init: ds41p1::kLogLendSlab, kLogLendAudit, kLogLendHeat).
slab_line() { echo "I[0] ds41 p1: lent slab: $1 slots (${SLAB}=$1; 2.24 GiB of VRAM a card that no pass writes, for which the plan gave up 134 static residents, coldest first) hold rank 0's $1 hottest file-tier units of that plan (profile share 0.0412) as decode residents from bring-up on ($1 loaded in 3.1 s, ...); ..."; }
audit_line="I[0] ds41 p1: lent audit: on (${AUDIT}=1): every lent slot is copied back from the card and compared, ..."
heat_half() { echo "I[0] ds41 p1: lent heat: on (${HEAT}=$1; half-life $2 decode rows, ${HEAT}_HALF_ROWS; margin 150%, ${HEAT}_MARGIN_PCT): each rank's lent slots (rank 0: 170) follow live decode routing, at most $1 handed to another unit a pass; ..."; }
mk slab128 "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)" "$(rest_line 5000)"
mk slab128h "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)" "$(heat_half 2 2000)" "$(rest_line 5000)"
mk slab128h100 "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)" "$(heat_half 2 100)" "$(rest_line 5000)"
mk audit "$lc_line" "$p1_line" "$arena_line" "$audit_line"
mk heat500 "$lc_line" "$p1_line" "$arena_line" "$(heat_half 2 500)" "$(rest_line 5000)"
mk all3 "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)" "$audit_line" "$(heat_half 4 5000)" "$(rest_line 5000)"
expect3() {   # want(0/1) what log LH LS LA LHH
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL= ARM_QE=64 ARM_LH=$4 ARM_LS=$5 ARM_LA=$6 ARM_LHH=$7 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect3 0 "Ls128, its line"                                           slab128 "" 128 "" ""
expect3 1 "Ls128, no lent-slab line"                                  arena   "" 128 "" ""
expect3 1 "Ls64 on a server whose slab has 128 slots"                 slab128 "" 64 "" ""
expect3 1 "Ls12 on a server whose slab has 128 slots"                 slab128 "" 12 "" ""
expect3 1 "no Ls part on a server that lends a slab"                  slab128 "" "" "" ""
expect3 0 "La, its line"                                              audit   "" "" 1 ""
expect3 1 "La, no lent-audit line"                                    arena   "" "" 1 ""
expect3 1 "no La part on a server that reads its lent slots back"     audit   "" "" "" ""
expect3 0 "Lh2 beside Ls128: the half-life is 2000 rows"              slab128h 2 128 "" ""
expect3 1 "Lh2 beside Ls128 on a server whose half-life is 100 rows"  slab128h100 2 128 "" ""
expect3 0 "Lh2 beside Ls128 with Lhh100"                              slab128h100 2 128 "" 100
expect3 0 "Lh2 with Lhh500, its half-life"                            heat500 2 "" "" 500
expect3 1 "Lh2 with Lhh500 on a server whose half-life is 100 rows"   heat2   2 "" "" 500
expect3 1 "Lh2 without Lhh on a server whose half-life is 500 rows"   heat500 2 "" "" ""
expect3 1 "Lh2 with Lhh50 on a server whose half-life is 500 rows"    heat500 2 "" "" 50
expect3 0 "Ls128 with La, Lh4 and Lhh5000, their lines"               all3    4 128 1 5000
expect3 1 "the same without La in the arm's name"                     all3    4 128 "" 5000
: > "$OUT/window.log"
ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL= ARM_QE=64 ARM_LH=4 ARM_LS=128 ARM_LA=1 ARM_LHH=5000 lcp_check all3 >/dev/null 2>&1
grep -q "ds41 p1: lent slab: 128 slots (${SLAB}=128;" "$OUT/window.log" && grep -q "ds41 p1: lent audit: on (${AUDIT}=1)" "$OUT/window.log" &&
  ok "the window log takes the lent-slab and lent-audit lines" || bad "the window log lacks the lent-slab or lent-audit line"
grep -q "ds41 p1: lent rest: 5000 ms (${REST}=5000" "$OUT/window.log" && ok "the window log takes the lent-rest line" ||
  bad "the window log lacks the lent-rest line"

# ---- 3a'. Lr at startup: the rest the arm runs with (the part's, else 5000 beside Lh or Ls, else 0)
mk slab128norest "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)"
mk slab128r2000 "$lc_line" "$p1_line" "$arena_line" "$(slab_line 128)" "$(rest_line 2000 "")"
mk arenar2000 "$lc_line" "$p1_line" "$arena_line" "$(rest_line 2000 "")"
expectr() {   # want(0/1) what log LH LS LR
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL= ARM_QE=64 ARM_LH=$4 ARM_LS=$5 ARM_LA= ARM_LHH= ARM_LR=$6 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expectr 1 "Ls128 without Lr on a server with no lent-rest line (an older radiance)" slab128norest "" 128 ""
expectr 0 "Ls128 with Lr2000, its line"                                          slab128r2000 "" 128 2000
expectr 1 "Ls128 with Lr2000 on a server resting 5000 ms"                        slab128 "" 128 2000
expectr 1 "Ls128 without Lr on a server resting 2000 ms"                         slab128r2000 "" 128 ""
expectr 0 "Ls128 with Lr0, no lent-rest line"                                    slab128norest "" 128 0
expectr 1 "Ls128 with Lr0 on a server that rests"                                slab128 "" 128 0
expectr 0 "Qe64 with Lr2000 alone, its line"                                     arenar2000 "" "" 2000
expectr 1 "Qe64 without Lr on a server that rests (the knob set for the window?)" arenar2000 "" "" ""

# ---- 3b. Ls / La / Lhh at shutdown (lend_exit_gate)
eval "$(sed -n '/^lend_exit_gate() {/,/^}/p' "$W")"
declare -F lend_exit_gate >/dev/null || { echo "FAIL no lend_exit_gate in $W"; exit 1; }
# The engine's shutdown line (CardExpertPath::stats_line, FileStager::p1_stats_line): its lent parts.
stat() {   # rank, the slab's part, the audit's, the heat's half-life part (each may be empty)
  echo "I[$1] ds41 p1 stats, rank $1: 3 chunks, 120 card-path layers; units RAM 0 / drive 0 / host pool 0 (...); leases at pass end 0; verify rows 0, differ 0 (0 samples skipped); lent ring: 41 slots (41 in the activation arena), 41 residents now; 3 bursts took them back (123 residents withdrawn), 4 returns, 420 reloaded (RAM cache 0, drive 420 at 9.1 ms a unit), 0 failed, 0 overtaken, 2 passes held back by the grace${RL-; lent reloads: 420 from the RAM cache (380 relaid from the packed form at 6.2 ms a unit), 40 from the drive, 12 passes held back by the rest (30 loads from the RAM cache in it)}$2$3; lent heat: 30 retargets (units published at a slot another unit gave up), 30 residents withdrawn for them, 900 passes declined (100 in a long prefill or its rest, 600 lent or in the grace, 200 a slot busy, 0 draft bank, 0 candidate, 0 no fence); touched experts 9000: VRAM residents 7000, slab 0, ring 2000; admission orders moved 0$4"
}
sl() { echo "; lent slab: $1 slots, $2 residents now ($3 at bring-up)"; }
au() { echo "; lent audit: $1 slots read back ($2 differ, $3 not read) at 7.4 ms a slot, $4 slots lost, retargets ${5:-not stopped}"; }
hh() { echo "; heat half-life $1 decode rows, $2 counted"; }
gate() {   # want(0/1) what log LS LA LHH [a word the reason must hold]
  local want=$1 what=$2 why rc
  why=$(lend_exit_gate "$OUT/$3.server.log" "$4" "$5" "$6"); rc=$?
  if [[ $rc != "$want" ]]; then bad "$what: exit $rc, want $want ($why)"
  elif [[ -n "${7:-}" && "$why" != *"$7"* ]]; then bad "$what: the reason '$why' does not say '$7'"
  else ok "$what${why:+: $why}"; fi
}
mk xgood "$(stat 0 "$(sl 128 128 128)" "$(au 300 0 0 0)" "$(hh 2000 5000)")" "$(stat 1 "$(sl 128 127 128)" "$(au 280 0 0 0)" "$(hh 2000 1500)")" \
         "$(stat 2 "$(sl 128 128 128)" "$(au 310 0 0 0)" "$(hh 2000 1900)")" "$(stat 3 "$(sl 128 128 128)" "$(au 1 0 0 0)" "$(hh 2000 100)")"
mk xprime "$(stat 0 "$(sl 128 128 128)" "" "")" "$(stat 1 "$(sl 128 128 128)" "" "")" "$(stat 2 "$(sl 128 128 127)" "" "")" "$(stat 3 "$(sl 128 128 128)" "" "")"
mk xslab64 "$(stat 0 "$(sl 64 64 64)" "" "")" "$(stat 1 "$(sl 64 64 64)" "" "")" "$(stat 2 "$(sl 64 64 64)" "" "")" "$(stat 3 "$(sl 64 64 64)" "" "")"
mk xnone "$(stat 0 "" "" "")" "$(stat 1 "" "" "")" "$(stat 2 "" "" "")" "$(stat 3 "" "" "")"
mk x3ranks "$(stat 0 "$(sl 128 128 128)" "$(au 300 0 0 0)" "$(hh 2000 5000)")" "$(stat 1 "$(sl 128 128 128)" "$(au 300 0 0 0)" "$(hh 2000 5000)")" \
           "$(stat 2 "$(sl 128 128 128)" "$(au 300 0 0 0)" "$(hh 2000 5000)")"
mk xdiffer "$(stat 0 "" "$(au 300 0 0 0)" "")" "I[1] E ds41 p1: lent audit: rank 1 arena slot 7 (lent slot 7), unit 2211: the slot read back after its copy differs from the bytes copied at offset 4660 (...)" \
           "$(stat 1 "" "$(au 290 1 0 1 stopped)" "")" "$(stat 2 "" "$(au 300 0 0 0)" "")" "$(stat 3 "" "$(au 300 0 0 0)" "")"
mk xunread "$(stat 0 "" "$(au 300 0 0 0)" "")" "$(stat 1 "" "$(au 300 0 0 0)" "")" "$(stat 2 "" "$(au 300 0 0 0)" "")" "$(stat 3 "" "$(au 300 0 2 2 stopped)" "")"
mk xidle "$(stat 0 "" "$(au 300 0 0 0)" "")" "$(stat 1 "" "$(au 0 0 0 0)" "")" "$(stat 2 "" "$(au 300 0 0 0)" "")" "$(stat 3 "" "$(au 300 0 0 0)" "")"
mk xcold "$(stat 0 "" "" "$(hh 2000 1999)")" "$(stat 1 "" "" "$(hh 2000 1500)")" "$(stat 2 "" "" "$(hh 2000 0)")" "$(stat 3 "" "" "$(hh 2000 100)")"
mk xhalf "$(stat 0 "" "" "$(hh 2000 5000)")" "$(stat 1 "" "" "$(hh 100 5000)")" "$(stat 2 "" "" "$(hh 2000 5000)")" "$(stat 3 "" "" "$(hh 2000 5000)")"
gate 0 "Ls128 + La + Lhh2000: every rank's slab in at bring-up, slots read back, none differing, one rank warm" xgood 128 1 2000
gate 0 "an arm without the three parts, whatever the lines say"                         xgood "" "" ""
gate 0 "an arm without the three parts and no lent part in the lines"                   xnone "" "" ""
gate 1 "Ls128, a rank's slab one unit short at bring-up"                                xprime 128 "" "" "held 127 of its 128 units at bring-up"
gate 1 "Ls128 on a server whose slabs have 64 slots"                                    xslab64 128 "" "" "has 64 slots"
gate 1 "Ls128, no lent-slab part in the stats lines"                                    xnone 128 "" "" "no 'lent slab"
gate 1 "Ls128, a rank with no stats line"                                               x3ranks 128 "" "" "rank 3"
gate 1 "La, a slot differing on rank 1: set aside, with its counts"                     xdiffer "" 1 "" "1 differ"
gate 1 "La, slots that could not be read back on rank 3"                                xunread "" 1 "" "and 2 could not be read"
gate 1 "La, a rank that read nothing back"                                              xidle "" 1 "" "read no lent slot back"
gate 1 "La, no lent-audit part in the stats lines"                                      xnone "" 1 "" "no 'lent audit"
gate 0 "La, every rank read slots back and none differ"                                 xgood "" 1 ""
gate 1 "Lhh2000, no rank counted 2000 decode rows"                                      xcold "" "" 2000 "never could act"
gate 1 "Lhh2000, a rank ran with a half-life of 100 rows"                               xhalf "" "" 2000 "half-life of 100"
gate 1 "Lhh2000, no half-life part in the stats lines"                                  xnone "" "" 2000 "no 'heat half-life"
gate 0 "Lhh2000, one rank counted its half-life"                                        xgood "" "" 2000
RL=""; mk xoldrl "$(stat 0 "$(sl 128 128 128)" "" "")" "$(stat 1 "$(sl 128 128 128)" "" "")" "$(stat 2 "$(sl 128 128 128)" "" "")" "$(stat 3 "$(sl 128 128 128)" "" "")"; unset RL
gate 1 "Ls128 on a radiance whose lender has no RAM-cache reload part"                  xoldrl 128 "" "" "no 'lent reloads"
# The differing slot's own log line is quoted in the reason.
why=$(lend_exit_gate "$OUT/xdiffer.server.log" "" 1 "")
[[ "$why" == *"unit 2211"* ]] && ok "the reason quotes the audit's line (slot, unit, offset)" || bad "the reason does not quote the audit's line: $why"

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
pre=all-t40-f
mid=H-p256-m196608-G-R
post=W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc1024
post2k=${post%-Lc1024}-Lc2048
REF=$pre-M1024-$mid-B1-$post-Cs-Mb1-Lm4096
RING=$pre-M1024-$mid-B6-$post-P1-Ql10-Cs-Mb1-Lm4096
BIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
LH2=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Cs-Mb1-Lm4096
LH8=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh8-Cs-Mb1-Lm4096
LH1BOTH=$pre-M2048-$mid-B6-$post2k-P1-Ql10-Qe64-Lh1-Cs-Mb1-Lm4096
LHRING=$pre-M1024-$mid-B6-$post-P1-Ql10-Lh2-Cs-Mb1-Lm4096
LHP1=$pre-M2048-$mid-B6-$post2k-P1-Lh2-Cs-Mb1-Lm4096
LHNOP1=$pre-M1024-$mid-B1-$post-Lh2-Cs-Mb1-Lm4096
LH0=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh0-Cs-Mb1-Lm4096
LH9=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh9-Cs-Mb1-Lm4096
LH2X=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2x-Cs-Mb1-Lm4096
LHX=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lhx-Cs-Mb1-Lm4096
LHTWICE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lh3-Cs-Mb1-Lm4096
# The adaptive residency parts.
LS=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls128-Cs-Mb1-Lm4096
LSH=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls128-Lh4-La-Cs-Mb1-Lm4096
LSONLY=$pre-M2048-$mid-B6-$post2k-P1-Ls256-Lh2-Lhh500-Cs-Mb1-Lm4096
LAQE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-La-Cs-Mb1-Lm4096
LHH=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lhh100000000-Cs-Mb1-Lm4096
LS0=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls0-Cs-Mb1-Lm4096
LS257=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls257-Cs-Mb1-Lm4096
LS12X=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls12x-Cs-Mb1-Lm4096
LSTWICE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls64-Ls128-Cs-Mb1-Lm4096
LSNOP1=$pre-M1024-$mid-B1-$post-Ls64-Cs-Mb1-Lm4096
LSLOWM=$pre-M1024-$mid-B6-$post2k-P1-Ls64-Cs-Mb1-Lm4096
LANONE=$pre-M2048-$mid-B6-$post2k-P1-La-Cs-Mb1-Lm4096
LHHNOLH=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lhh500-Cs-Mb1-Lm4096
LHH0=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lhh0-Cs-Mb1-Lm4096
LHHBIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lhh100000001-Cs-Mb1-Lm4096
LHHTWICE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lh2-Lhh500-Lhh600-Cs-Mb1-Lm4096
# The rest after a long prefill (LrMS).
LR=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Ls128-Lh2-Lr2000-Cs-Mb1-Lm4096
LR0=$pre-M2048-$mid-B6-$post2k-P1-Ls128-Lr0-Cs-Mb1-Lm4096
LRBIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lr600001-Cs-Mb1-Lm4096
LRX=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lr5s-Cs-Mb1-Lm4096
LRNONE=$pre-M2048-$mid-B6-$post2k-P1-Lr1000-Cs-Mb1-Lm4096
LRTWICE=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Lr10-Lr20-Cs-Mb1-Lm4096
# BIG right after the Lh arms, and BIG2 (the same arm under another name) after the Ls / La / Lhh arms: a
# switch an arm set must not reach the next one.
BIG2=$BIG-r2
ARMS="$REF $LH2 $LH8 $LH1BOTH $BIG $RING $LHRING $LHP1 $LHNOP1 $LH0 $LH9 $LH2X $LHX $LHTWICE"
ARMS="$ARMS $LS $LSH $LSONLY $LAQE $LHH $BIG2 $LS0 $LS257 $LS12X $LSTWICE $LSNOP1 $LSLOWM $LANONE $LHHNOLH $LHH0 $LHHBIG $LHHTWICE"
ARMS="$ARMS $LR $LR0 $LRBIG $LRX $LRNONE $LRTWICE"
run_loop() {   # every model action stubbed
 (
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; echo "  ARM_LH: ${ARM_LH}"; echo "  ARM_LS: ${ARM_LS}"; echo "  ARM_LA: ${ARM_LA}"; echo "  ARM_LHH: ${ARM_LHH}"; echo "  ARM_LR: ${ARM_LR}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE
  # The window's own environment tries to turn the switch on and retune it for every arm: each arm states all three.
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=4 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_HALF_ROWS=7
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT_MARGIN_PCT=900
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB=7 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=1
  export RADIANCE_EXPERIMENTAL_DS41_P1_LEND_REST_MS=7
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=4096 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
 ) 2>&1
}
out=$(run_loop)
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
# The last value an arm states for a variable (later entries win in the arm's environment); "unstated" if none.
val() { local l; l=$(grep -E "^    $2=" <<< "$1" | tail -1); [[ -n "$l" ]] && echo "${l#*=}" || echo unstated; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
launched() {   # what arm heat [lend_arena [lend]]
  local sec
  sec=$(section "$2")
  if [[ -z "$sec" ]]; then bad "$1: arm $2 not launched: $(grep -m3 "arm $2" "$T/loop.txt")"; return; fi
  if grep -q "LOG: arm $2: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "$1: $(grep -m1 "LOG: arm $2: " "$T/loop.txt")"; fi
  has "$1" "$sec" ${HEAT} "$3"
  has "$1" "$sec" ${HEAT}_HALF_ROWS ""
  has "$1" "$sec" ${HEAT}_MARGIN_PCT ""
  has "$1" "$sec" ${SLAB} 0      # an arm without Ls / La states both off, whatever the window's environment says
  has "$1" "$sec" ${AUDIT} 0
  has "$1" "$sec" ${REST} ""     # the engine's default, whatever the window's environment says
  [[ -n "${4:-}" ]] && has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA "$4"
  [[ -n "${5:-}" ]] && has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND "$5"
  local lh
  lh=$(grep -m1 '^  ARM_LH: ' <<< "$sec" | cut -d' ' -f4)
  local want_lh=$3; [[ $want_lh == 0 ]] && want_lh=""
  if [[ "$lh" == "$want_lh" ]]; then ok "$1: lcp_check will ask for '${want_lh}'"; else bad "$1: ARM_LH is '$lh', want '$want_lh'"; fi
}
launched "P1+Qe64+Lh2" "$LH2" 2 64 0
launched "P1+Qe64+Lh8" "$LH8" 8 64 0
launched "P1+Ql10+Qe64+Lh1" "$LH1BOTH" 1 64 10
launched "P1+Qe64, right after the Lh arms" "$BIG" 0 64 0
launched "P1+Ql10" "$RING" 0 0 10
launched "the adopted large-chunk arm" "$REF" 0 0 0
sec=$(section "$LH2")
has "P1+Qe64+Lh2" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
has "P1+Qe64+Lh2" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
has "the adopted large-chunk arm" "$(section "$REF")" RADIANCE_EXPERIMENTAL_DS41_P1 0
skipped() {   # arm, the reason's words
  if [[ -n "$(section "$1")" ]]; then bad "arm $1 was launched"
  elif grep -q "LOG: arm $1: .*$2.*skipped" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $1: .*$2" "$T/loop.txt" | sed "s|LOG: arm $1: ||")"
  else bad "arm $1 not skipped for '$2': $(grep -m2 "LOG: arm $1: " "$T/loop.txt" | tr '\n' ' ')"; fi
}
skipped "$LHRING" "needs the lent arena"
skipped "$LHP1" "needs the lent arena"
skipped "$LHNOP1" "needs the lent arena"
skipped "$LH0" "Lh takes 1-8"
skipped "$LH9" "Lh takes 1-8"
skipped "$LH2X" "Lh takes 1-8"
skipped "$LHX" "unknown part Lhx"
skipped "$LHTWICE" "duplicate Lh part"

# ---- 3c. Ls / La / Lhh in the arm loop
launched3() {   # what arm heat slab audit half lend_arena
  local sec v
  sec=$(section "$2")
  if [[ -z "$sec" ]]; then bad "$1: arm $2 not launched: $(grep -m3 "arm $2" "$T/loop.txt")"; return; fi
  if grep -q "LOG: arm $2: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "$1: $(grep -m1 "LOG: arm $2: " "$T/loop.txt")"; fi
  has "$1" "$sec" ${HEAT} "$3"
  has "$1" "$sec" ${SLAB} "$4"
  has "$1" "$sec" ${AUDIT} "$5"
  has "$1" "$sec" ${HEAT}_HALF_ROWS "$6"
  has "$1" "$sec" ${HEAT}_MARGIN_PCT ""
  has "$1" "$sec" ${REST} "${8:-}"
  has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA "$7"
  has "$1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  for v in "LH:$3" "LS:$4" "LA:$5" "LHH:$6"; do
    local got want=${v#*:}
    [[ $want == 0 ]] && want=""
    got=$(grep -m1 "^  ARM_${v%%:*}: " <<< "$sec" | cut -d' ' -f4)
    if [[ "$got" == "$want" ]]; then ok "$1: the gates will ask for ARM_${v%%:*}='${want}'"; else bad "$1: ARM_${v%%:*} is '$got', want '$want'"; fi
  done
}
launched3 "P1+Qe64+Ls128"            "$LS"     0 128 0 ""          64
launched3 "P1+Qe64+Ls128+Lh4+La"     "$LSH"    4 128 1 ""          64
launched3 "P1+Ls256+Lh2+Lhh500"      "$LSONLY" 2 256 0 500         0
launched3 "P1+Qe64+La"               "$LAQE"   0 0   1 ""          64
launched3 "P1+Qe64+Lh2+Lhh100000000" "$LHH"    2 0   0 100000000   64
launched3 "P1+Qe64, right after them" "$BIG2"  0 0   0 ""          64
skipped "$LS0" "Ls takes 1-256"
skipped "$LS257" "Ls takes 1-256"
skipped "$LS12X" "Ls takes 1-256"
skipped "$LSTWICE" "duplicate Ls part"
skipped "$LSNOP1" "needs P1"
skipped "$LSLOWM" "Ls needs M2048"
skipped "$LANONE" "needs a lent part"
skipped "$LHHNOLH" "needs an LhS part"
skipped "$LHH0" "Lhh takes 1-100000000"
skipped "$LHHBIG" "Lhh takes 1-100000000"
skipped "$LHHTWICE" "duplicate Lhh part"
launched3 "P1+Qe64+Ls128+Lh2+Lr2000" "$LR"     2 128 0 ""          64 2000
launched3 "P1+Ls128+Lr0"             "$LR0"    0 128 0 ""          0  0
for a in "$LR:2000" "$LR0:0" "$LS:"; do
  got=$(grep -m1 '^  ARM_LR: ' <<< "$(section "${a%:*}")" | cut -d' ' -f4)
  [[ "$got" == "${a##*:}" ]] && ok "lcp_check will ask for a rest of '${a##*:}'" || bad "ARM_LR is '$got', want '${a##*:}'"
done
skipped "$LRBIG" "Lr takes 0-600000"
skipped "$LRX" "Lr takes 0-600000"
skipped "$LRNONE" "needs a lent part"
skipped "$LRTWICE" "duplicate Lr part"

(( fail )) && { echo "ab_window_lh_test: FAILED"; echo "$out" | tail -40; exit 1; }
echo "ab_window_lh_test: all checks passed"
