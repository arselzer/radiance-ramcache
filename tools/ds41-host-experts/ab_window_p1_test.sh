#!/usr/bin/env bash
# ab_window_p1_test.sh [ab_window.sh] -- ab_window.sh's P1 parts (P1, QmN, QlK, QeK, QhMS) with no GPU, no
# server and no model:
#   1. lcp_check on synthetic server logs: a P1 arm is measured only when the card path's startup line
#      states its threshold and chunk, a Ql arm only with its lent-ring line, a Qe arm only with its
#      lent-arena line naming its K, and an arm without a part only without its line;
#   2. the arm loop, every action stubbed (as ab_window_vp_test.sh): the adopted large-chunk arm with P1
#      on top of it (B6, Ql10, M1024, Mr32, Mb1, Cs) is launched with the switches it names and the rest
#      stated off; the adopted arm itself states P1 off; P1 at the default threshold with M16, Ql12 on
#      B6's ring, Qh without Ql and Ql without P1 are skipped; Qm17 with M16 is launched; P1 at Lc2048 with
#      Qe64 and M2048 is launched with the lent arena on, Qh beside Qe alone too; Qe at M1024 beside Lc2048
#      and Qe without P1 are skipped; Lc4096 with explicit Pci4096 works cold or cached, while
#      a missing/short interval, another prefix-only part without Pc, or Lc4112 are refused;
#   3. balanced-kernel receipts state their actual requested-height coverage; an M2048 balance
#      oracle cannot authorize Mb+Lc4096 even when the independent LCP oracle covered 4096;
#   4. SgK / SgaK (prompt-lookup speculation) beside Lc, P1 and Cs: sg_gate on synthetic logs (an Sg arm
#      is measured only with the file stager's, the scheduler's, the card path's and the consumer skip's
#      admission lines naming its K, an arm without the part only without them, and at exit only with no
#      pass outside the decode band and no verify pass over 16 rows); in the arm loop the adopted fast
#      arm with Sg8, Sga8 or Sg15 on top is launched with the speculation's arguments and every other
#      switch as without it, with N2 Sg7 too, and Sg16, N2 with Sg8 and N8 with Sg2 (more than 16 decode
#      rows a step) are skipped with the reason;
#   5. QdN (decode passes on the cards, ..._P1_DECODE_ROWS): qd_gate on synthetic logs (a Qd arm is measured only
#      with rank 0's decode line naming its N and chunk, and at exit only when every rank's stats line counts its
#      decode passes and some rank armed one; an arm without the part only without the line and with none armed),
#      alone and through lcp_check; in the arm loop the fast arm with Qd200 is launched with the switch at 200
#      and every arm without the part states it 0, while Qd above the chunk, Qd16, Qd beside Tb and Qd without
#      P1 are skipped with the reason;
#   6. QyP / QyfP (hybrid decode, ..._P1_HYBRID): qy_gate on synthetic logs (a Qy arm is measured only with rank 0's
#      hybrid line naming its share and mode, and at exit only when every rank's stats line counts its hybrid passes
#      and some rank split one; an arm without the part only without the line and with none split), alone and
#      through lcp_check; in the arm loop Qd64-Qy25 and Qd64-Qyf30 on the fast arm are launched with the switch (and
#      the held share) and every other switch as the fast arm, every arm without the part states it 0 and the adapt
#      switch empty, while Qy without Qd, beside Sg8, Qv8 or Ax, Qy0, two Qy parts and Qy without P1 are skipped
#      with the reason.
#   7. Ctp / CtpN (the prefill chain trace, ..._CHAIN_TRACE_PREFILL): the fast arm with Ctp traces one armed
#      prefill pass in 4 into chain-<arm>, with Qd64 Ct16 Ctp2 both traces at their N into the same files;
#      Ctp without P1 and Ctp0 are skipped with the reason; arms without the part do not set it; and
#      chain_report.py's selftest (the --split classes on a synthetic prefill pass, old files as decode).
# Exit 0 when every check passes.
set -u
W=${1:-$(dirname "$0")/ab_window.sh}
T=$(mktemp -d /tmp/ab-p1-test-XXXXXX)
trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# ---- 1. the server-log gate
eval "$(sed -n '/^lcp_check() {/,/^}/p' "$W")"
declare -F lcp_check >/dev/null || { echo "FAIL no lcp_check in $W"; exit 1; }
eval "$(sed -n '/^qd_gate() {/,/^}/p' "$W")"
declare -F qd_gate >/dev/null || { echo "FAIL no qd_gate in $W"; exit 1; }
eval "$(sed -n '/^qy_gate() {/,/^}/p' "$W")"
declare -F qy_gate >/dev/null || { echo "FAIL no qy_gate in $W"; exit 1; }
OUT=$T/logs
mkdir -p "$OUT"
log() { echo "LOG: $*" >> "$OUT/window.log"; }
lc_line='I[0] large-chunk prefill: passes of up to 1024 tokens (RADIANCE_EXPERIMENTAL_DS41_LCP=1024): ...'
p1_line() { echo "I[0] ds41 p1: card experts on: every routed expert row of a pure prefill pass's layer of $1 to 1024 rows on the cards (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS=$1; ...)"; }
lend_line() { echo "I[0] ds41 p1: lent ring: $1 of the ring's 12 slots (10 of them past the file buffers' alias bases; RADIANCE_EXPERIMENTAL_DS41_P1_LEND=$1) hold ..."; }
arena_line() { echo "I[0] ds41 p1: lent arena: $2 slots (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=$1; $2 fit) in the activation arena's top, ..."; }
mk() { local n=$1; shift; printf '%s\n' "$@" > "$OUT/$n.server.log"; }
mk p1def "$lc_line" "$(p1_line 320)"
mk p1lend "$lc_line" "$(p1_line 320)" "$(lend_line 10)"
mk p1q17 "$lc_line" "$(p1_line 17)"
mk plain "$lc_line"
mk p1arena "$lc_line" "$(p1_line 320)" "$(arena_line 64 18)"
mk p1both "$lc_line" "$(p1_line 320)" "$(lend_line 10)" "$(arena_line 64 18)"
expect() {   # want(0/1) what log LC P1 QM QL [QE]
  local want=$1 what=$2 rc
  ARM_LC=$4 ARM_P1=$5 ARM_QM=$6 ARM_QL=$7 ARM_QE=${8:-} lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
expect 0 "P1 at the default threshold, its line"                      p1def  1024 1 320 ""
expect 1 "P1 at Qm17 on a server that took 320"                       p1def  1024 1 17  ""
expect 0 "P1 at Qm17, its line"                                       p1q17  1024 1 17  ""
expect 0 "P1 with Ql10, both lines"                                   p1lend 1024 1 320 10
expect 1 "P1 with Ql10, no lent-ring line"                            p1def  1024 1 320 10
expect 1 "P1 with Ql8 on a server that lent 10"                       p1lend 1024 1 320 8
expect 1 "P1 without Ql on a server that lends (the knob from the window)" p1lend 1024 1 320 ""
expect 1 "no P1 part on a server that runs the card path"             p1def  1024 "" 320 ""
expect 0 "the plain Lc arm"                                           plain  1024 "" 320 ""
expect 0 "P1 with Qe64, its line"                                     p1arena 1024 1 320 "" 64
expect 1 "P1 with Qe64, no lent-arena line"                           p1def  1024 1 320 "" 64
expect 1 "P1 with Qe32 on a server that took 64"                      p1arena 1024 1 320 "" 32
expect 1 "P1 without Qe on a server that lends the arena"             p1arena 1024 1 320 ""
expect 0 "P1 with Ql10 and Qe64, both lines"                          p1both 1024 1 320 10 64
expect 1 "P1 with Qe64 alone on a server that lends the ring too"     p1both 1024 1 320 "" 64

# SgK beside Lc: the admission's startup lines and the exit counts (sg_gate), alone and through lcp_check.
eval "$(sed -n '/^sg_gate() {/,/^}/p' "$W")"
declare -F sg_gate >/dev/null || { echo "FAIL no sg_gate in $W"; exit 1; }
sg_stager() { echo "I[0] large-chunk prefill: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step is a decode pass of at most $(( $1 + 1 )) rows, inside the decode band of 16 (...)"; }
sg_sched() { echo "I large-chunk prefill: prompt-lookup speculation admitted (--num-speculative-tokens $1, --max-num-seqs 1, no draft head): a step is one prompt's pure prefill chunk, ..."; }
sg_p1() { echo "I[0] ds41 p1: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step of at most $(( $1 + 1 )) decode rows never arms the card path; ..."; }
sg_cs() { echo "I consumer skip: prompt-lookup speculation admitted (--num-speculative-tokens $1, no draft head): a verify step of at most $(( $1 + 1 )) decode rows runs every layer on every row ..."; }
sg_exit() { echo "I large-chunk prefill, rank $1: prompt-lookup verify passes $2, $(( $2 * 5 )) rows (at most $3 a pass); passes outside the decode band $4"; }
mk sgall "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgnocs "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_p1 8)"
mk sgnop1 "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_sched 8)" "$(sg_cs 8)"
mk sgnosched "$lc_line" "$(p1_line 320)" "$(sg_stager 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgnostager "$lc_line" "$(p1_line 320)" "$(sg_sched 8)" "$(sg_p1 8)" "$(sg_cs 8)"
mk sgrefused "E[0] RADIANCE_EXPERIMENTAL_DS41_LCP=2048 (large-chunk prefill) refused: speculation is on (a window of 8): verify steps and DSpark's drafter are not covered"
for r in 0 1 2 3; do sg_exit $r 40 9 0; done > "$OUT/sgexit-good.server.log"
{ sg_exit 0 40 9 0; sg_exit 1 40 9 1; sg_exit 2 40 9 0; sg_exit 3 40 9 0; } > "$OUT/sgexit-outside.server.log"
{ sg_exit 0 40 9 0; sg_exit 1 40 17 0; } > "$OUT/sgexit-tall.server.log"
echo "I large-chunk prefill, rank 0: prompt-lookup verify passes 40 (an older line)" > "$OUT/sgexit-odd.server.log"
: > "$OUT/sgexit-none.server.log"
sgexpect() {   # want(0/1) what log K P1 CS stage
  local want=$1 what=$2 rc
  sg_gate "$OUT/$3.server.log" "$4" "$5" "$6" "$7" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
sgexpect 0 "Sg8 with P1 and Cs, every admission line"                 sgall      8 1 1 start
sgexpect 0 "Sg8 with P1, no Cs part: the consumer skip's line is not asked for" sgnocs 8 1 "" start
sgexpect 1 "Sg8 with Cs, no consumer skip admission line"             sgnocs     8 1 1 start
sgexpect 1 "Sg8 with P1, no card path admission line"                 sgnop1     8 1 1 start
sgexpect 1 "Sg8, no scheduler admission line"                         sgnosched  8 1 1 start
sgexpect 1 "Sg8, no file stager admission line"                       sgnostager 8 1 1 start
sgexpect 1 "Sg7 on a server that admitted depth 8"                    sgall      7 1 1 start
sgexpect 1 "Sg8 on a server that refused the speculation"             sgrefused  8 1 1 start
sgexpect 1 "no Sg part on a server that took the speculation"         sgall      "" 1 1 start
sgexpect 0 "no Sg part, no admission line"                            p1def      "" 1 1 start
sgexpect 0 "Sg8 at exit: every rank's verify passes inside the band"  sgexit-good 8 1 1 exit
sgexpect 1 "Sg8 at exit: a rank with a pass outside the band"         sgexit-outside 8 1 1 exit
sgexpect 1 "Sg8 at exit: a verify pass of 17 rows"                    sgexit-tall 8 1 1 exit
sgexpect 1 "Sg8 at exit: a line without its counts"                   sgexit-odd 8 1 1 exit
sgexpect 0 "Sg8 at exit: no verify pass ran (nothing to refuse)"      sgexit-none 8 1 1 exit
# ...and through lcp_check, which names the arm's K only beside an Lc part.
sglc() {   # want(0/1) what log LC SG CS
  local want=$1 what=$2 rc
  ARM_LC=$4 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_SG=$5 ARM_CS=$6 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
sglc 0 "lcp_check: the P1 arm with Sg8 and Cs, its lines"            sgall  1024 8 1
sglc 1 "lcp_check: the P1 arm with Sg8, no consumer skip line"       sgnocs 1024 8 1
sglc 1 "lcp_check: the P1 arm without Sg on a server that took it"   sgall  1024 "" 1
sglc 0 "lcp_check: the P1 arm without Sg, as before"                 p1def  1024 "" 1

# ---- 5a. QdN: qd_gate on synthetic logs, alone and through lcp_check.
qd_line() { echo "I[0] ds41 p1: decode passes on the cards: every routed expert row of a pure decode pass of $1 to ${2:-1024} rows on the cards, as a prefill pass's layer of the threshold's rows (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=$1; ...)"; }
qd_stats() { echo "I[$1] ds41 p1 stats, rank $1: passes 40, ...; leases at pass end 0, verify rows 0, differ 0; touched experts 900: ...; decode passes armed $2, rows $(( $2 * 256 )) (at most 256 a pass), issue 20.500 s; decode passes of 17 to 199 rows on the CPU path 3"; }
mk qd200 "$lc_line" "$(p1_line 320)" "$(qd_line 200)"
mk qd200x2048 "$lc_line" "$(p1_line 320)" "$(qd_line 200 2048)"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qd_stats $r 12; done; } > "$OUT/qdexit.server.log"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qd_stats $r 0; done; } > "$OUT/qdidle.server.log"
{ cat "$OUT/qd200.server.log"; qd_stats 0 12; echo "I[1] ds41 p1 stats, rank 1: passes 40, leases at pass end 0, verify rows 0, differ 0"; } > "$OUT/qdodd.server.log"
{ cat "$OUT/p1def.server.log"; for r in 0 1 2 3; do qd_stats $r 5; done; } > "$OUT/qdstray.server.log"
qdexpect() {   # want(0/1) what log QD LC stage
  local want=$1 what=$2 rc
  qd_gate "$OUT/$3.server.log" "$4" "$5" "$6" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
qdexpect 0 "Qd200, its line"                                        qd200      200 1024 start
qdexpect 1 "Qd256 on a server that took 200"                        qd200      256 1024 start
qdexpect 1 "Qd200 at Lc2048 on a server whose line names 1024"      qd200      200 2048 start
qdexpect 0 "Qd200 at Lc2048, its line"                              qd200x2048 200 2048 start
qdexpect 1 "Qd200, no decode line"                                  p1def      200 1024 start
qdexpect 1 "no Qd part on a server that arms decode passes"         qd200      ""  1024 start
qdexpect 0 "no Qd part, no decode line"                             p1def      ""  1024 start
qdexpect 0 "Qd200 at exit: ranks armed decode passes"               qdexit     200 1024 exit
qdexpect 1 "Qd200 at exit: no rank armed one (the switch never acted)" qdidle  200 1024 exit
qdexpect 1 "Qd200 at exit: a stats line without its decode counts"  qdodd      200 1024 exit
qdexpect 1 "Qd200 at exit: no stats line"                           qd200      200 1024 exit
qdexpect 1 "no Qd part at exit, but a rank armed decode passes"     qdstray    ""  1024 exit
qdexpect 0 "no Qd part at exit, none armed"                         p1def      ""  1024 exit
qdlc() {   # want(0/1) what log QD
  local want=$1 what=$2 rc
  ARM_LC=1024 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_QD=$4 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
qdlc 0 "lcp_check: the P1 arm with Qd200, its line"                 qd200 200
qdlc 1 "lcp_check: the P1 arm with Qd200, no decode line"           p1def 200
qdlc 1 "lcp_check: the P1 arm without Qd on a server that arms decode" qd200 ""
qdlc 0 "lcp_check: the P1 arm without Qd, as before"                p1def ""

# ---- 6a. QyP / QyfP: qy_gate on synthetic logs, alone and through lcp_check.
qy_line() { echo "I[0] ds41 p1: hybrid decode on: in a decode pass armed for the cards (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=64), each routed layer gives the CPU pool the RAM-cache units with the fewest rows while the pool's predicted time stays within $1% of the card's predicted link time for the rest (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=$1; pool 78.0 GB/s, link 20.0 GB/s, the share $2); those units ..."; }
qy_adapt="adapts per rank from each layer's outcome, at most that"
qy_held="held (..._P1_HYBRID_ADAPT=0)"
qy_stats() { echo "I[$1] ds41 p1 stats, rank $1: passes 40, ...; leases at pass end 0, verify rows 0, differ 0; touched experts 900: VRAM residents 1, slab 2, ring 3, CPU pool 4; ...; decode passes armed 12, rows 3072 (at most 256 a pass), issue 20.500 s; decode passes of 17 to 63 rows on the CPU path 3; hybrid: passes split $2 of 12 armed, layers 400, units CPU 900 / card 3000 (fell back 0, layers without rows 0), CPU rows 2000, ..., share 25.0% (lowest 12.0%)"; }
mk qy25 "$lc_line" "$(p1_line 320)" "$(qd_line 64 2048)" "$(qy_line 25 "$qy_adapt")"
mk qyf30 "$lc_line" "$(p1_line 320)" "$(qd_line 64 2048)" "$(qy_line 30 "$qy_held")"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qy_stats $r 12; done; } > "$OUT/qyexit.server.log"
{ cat "$OUT/qy25.server.log"; for r in 0 1 2 3; do qy_stats $r 0; done; } > "$OUT/qyidle.server.log"
{ cat "$OUT/qy25.server.log"; qy_stats 0 12; qd_stats 1 12; } > "$OUT/qyodd.server.log"
{ cat "$OUT/qd200.server.log"; for r in 0 1 2 3; do qy_stats $r 5; done; } > "$OUT/qystray.server.log"
qyexpect() {   # want(0/1) what log QY QYF stage
  local want=$1 what=$2 rc
  qy_gate "$OUT/$3.server.log" "$4" "$5" "$6" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want"; fi
}
qyexpect 0 "Qy25, its line"                                          qy25    25 ""  start
qyexpect 1 "Qy30 on a server that took 25"                            qy25    30 ""  start
qyexpect 1 "Qyf25 on a server whose share adapts"                     qy25    25 1   start
qyexpect 0 "Qyf30, its held line"                                     qyf30   30 1   start
qyexpect 1 "Qy30 on a server that holds the share"                    qyf30   30 ""  start
qyexpect 1 "Qy25, no hybrid line"                                     qd200   25 ""  start
qyexpect 1 "no Qy part on a server that splits"                       qy25    "" ""  start
qyexpect 0 "no Qy part, no hybrid line"                               qd200   "" ""  start
qyexpect 0 "Qy25 at exit: ranks split passes"                         qyexit  25 ""  exit
qyexpect 1 "Qy25 at exit: no rank split one (the switch never acted)" qyidle  25 ""  exit
qyexpect 1 "Qy25 at exit: a stats line without its hybrid counts"     qyodd   25 ""  exit
qyexpect 1 "Qy25 at exit: no stats line"                              qy25    25 ""  exit
qyexpect 1 "no Qy part at exit, but a rank split passes"              qystray "" ""  exit
qyexpect 0 "no Qy part at exit, none split"                           qdexit  "" ""  exit
qylc() {   # want(0/1) what log QY QYF
  local want=$1 what=$2 rc
  ARM_LC=2048 ARM_P1=1 ARM_QM=320 ARM_QL="" ARM_QE="" ARM_QD=64 ARM_QY=$4 ARM_QYF=$5 lcp_check "$3" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$what"; else bad "$what: exit $rc, want $want ($(tail -1 "$OUT/window.log"))"; fi
}
mk qy25lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')" "$(qd_line 64 2048)" "$(qy_line 25 "$qy_adapt")"
mk qd64lc "${lc_line//1024/2048}" "$(p1_line 320 | sed 's/320 to 1024/320 to 2048/')" "$(qd_line 64 2048)"
qylc 0 "lcp_check: the P1 arm with Qd64-Qy25, its line"              qy25lc 25 ""
qylc 1 "lcp_check: the P1 arm with Qd64-Qy25, no hybrid line"        qd64lc 25 ""
qylc 1 "lcp_check: the P1 arm without Qy on a server that splits"    qy25lc "" ""
qylc 0 "lcp_check: the P1 arm with Qd64 without Qy, as before"       qd64lc "" ""

# Qsp's startup and useful-work gates, independent of ordinary P1 startup admission.
eval "$(sed -n '/^qsp_gate() {/,/^}/p' "$W")"
declare -F qsp_gate >/dev/null || { echo "FAIL no qsp_gate in $W"; exit 1; }
: > "$T/qsp-good.log"
for rank in 0 1 2 3; do
  echo "ds41 p1: rank $rank's wave SwiGLU spans on (test)" >> "$T/qsp-good.log"
  echo "ds41 p1 stats, rank $rank: test; wave SwiGLU spans 12, rows 20 of 200, full-replay fallbacks 0" >> "$T/qsp-good.log"
done
sed '/rank 3/d' "$T/qsp-good.log" > "$T/qsp-missing.log"
sed 's/spans 12/spans 0/' "$T/qsp-good.log" > "$T/qsp-idle.log"
sed 's/rows 20 of 200/rows 200 of 200/' "$T/qsp-good.log" > "$T/qsp-full.log"
: > "$T/qsp-off.log"
qexpect() {
  local want=$1 label=$2 rc; shift 2
  qsp_gate "$@" >/dev/null 2>&1; rc=$?
  if [[ $rc == "$want" ]]; then ok "$label"; else bad "$label: exit $rc, want $want"; fi
}
qexpect 0 'Qsp every-rank startup' "$T/qsp-good.log" 1 start
qexpect 0 'Qsp every-rank work reduction' "$T/qsp-good.log" 1 exit
qexpect 1 'Qsp missing rank' "$T/qsp-missing.log" 1 start
qexpect 1 'Qsp enabled without actual waves' "$T/qsp-idle.log" 1 exit
qexpect 1 'Qsp no reduced work' "$T/qsp-full.log" 1 exit
qexpect 1 'Qsp unrequested activation' "$T/qsp-good.log" '' start
qexpect 0 'Qsp explicitly off' "$T/qsp-off.log" '' exit

# ---- the balance receipt: an old generic PASS is not requested-height evidence
eval "$(sed -n '/^bal_receipt_bound() {/,/^}/p' "$W")"
declare -F bal_receipt_bound >/dev/null || { echo "FAIL no bal_receipt_bound in $W"; exit 1; }
receipt() {   # want status, description, requested C, expected maximum, receipt line(s)
  local want=$1 what=$2 requested=$3 maximum=$4 got rc
  printf '%s\n' "$5" > "$T/bal.log"
  got=$(bal_receipt_bound "$T/bal.log" "$requested"); rc=$?
  if [[ $rc == "$want" && ( $rc != 0 || $got == "$maximum" ) ]]; then ok "$what"
  else bad "$what: status $rc/max '$got', wanted $want/'$maximum'"; fi
}
receipt 0 'the 2048 receipt keeps its actual bound' 2048 2048 \
  'bal: byte coverage max_M=2048 requested_C=2048 heights=1,6,16,17,64,256,512,1024,2048'
receipt 0 'the 3072 boundary has an actual receipt' 3072 3072 \
  'bal: byte coverage max_M=3072 requested_C=3072 heights=1,6,16,17,64,256,512,1024,2048,3072'
receipt 0 'an intermediate requested C is actually present' 4000 4000 \
  'bal: byte coverage max_M=4000 requested_C=4000 heights=1,6,16,17,64,256,512,1024,2048,3072,4000'
receipt 0 'the 4096 receipt covers its upper boundary' 4096 4096 \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096'
receipt 1 'a generic legacy PASS has no actual bound' 4096 '' \
  "PASS: bal: the balanced gemv and twin give the plain kernel's bytes at 1..4096 tokens"
receipt 1 'a short maximum cannot certify the requested C' 4096 '' \
  'bal: byte coverage max_M=2048 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048'
receipt 1 'a claimed bound missing from the tested list refuses' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072'
receipt 1 'a receipt for another C refuses' 3072 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096'
receipt 1 'duplicate tested heights refuse' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,2048,3072,4096'
receipt 1 'a trailing empty tested height refuses' 4096 '' \
  'bal: byte coverage max_M=4096 requested_C=4096 heights=1,6,16,17,64,256,512,1024,2048,3072,4096,'

# ---- 2. the arm loop, stubbed
H=$T/home
mkdir -p "$H/kernels"
echo "not a library" > "$H/kernels/libr4d.so"
echo '{}' > "$T/prompts.json"
pre=all-t40-f
mid=H-p256-m196608-G-R
post=W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc1024
REF=$pre-M1024-$mid-B1-$post-Cs-Mb1-Lm4096
NEW=$pre-M1024-$mid-B6-$post-P1-Ql10-Cs-Mb1-Lm4096
NEWG=$pre-M1024-$mid-B6-$post-P1-Ql10-Qh3000-Cs-Mb1-Lm4096
SHORTM=$pre-M16-$mid-B6-$post-P1-Cs-Mb1-Lm4096
Q17=$pre-M16-$mid-B6-$post-P1-Qm17-Cs-Mb1-Lm4096
OVERL=$pre-M1024-$mid-B6-$post-P1-Ql12-Cs-Mb1-Lm4096
GNOL=$pre-M1024-$mid-B6-$post-P1-Qh3000-Cs-Mb1-Lm4096
LNOP=$pre-M1024-$mid-B1-$post-Ql10-Cs-Mb1-Lm4096
post2k=${post%-Lc1024}-Lc2048
BIG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
BIGG=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Qh3000-Cs-Mb1-Lm4096
BIGM=$pre-M1024-$mid-B6-$post2k-P1-Qe64-Cs-Mb1-Lm4096
ENOP=$pre-M1024-$mid-B1-$post-Qe64-Cs-Mb1-Lm4096
post4k=${post%-Lc1024}-Lc4096
COLD4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci4096-Cs-Mb1-Lm8192
PC4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pc-Pci4096-Cs-Mb1-Lm8192
NOINT4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Cs-Mb1-Lm8192
LOWINT4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci2048-Cs-Mb1-Lm8192
NOPC4K=$pre-M4096-$mid-B6-$post4k-P1-Qe64-Pci4096-Pce-Cs-Mb1-Lm8192
OVER4K=$pre-M4112-$mid-B6-${post%-Lc1024}-Lc4112-P1-Qe64-Pci8192-Cs-Mb1-Lm8192
# The fast arm the windows run (Lc2048, P1, Qe64, Mb1, Rb, Vp, Cs) and prompt-lookup speculation on top of it.
FAST=$pre-M2048-$mid-B6-$post2k-P1-Qe64-Mb1-Rb-Vp-Cs
SG8=$FAST-Sg8
SGA8=$FAST-Sga8
SG15=$FAST-Sg15
SG16=$FAST-Sg16
N2SG7=$FAST-N2-Sg7
N2SG8=$FAST-N2-Sg8
N8SG2=$FAST-Sg2-N8
SG0=$FAST-Sg0
QD200=$FAST-Qd200
QD4096=$FAST-Qd4096
QD16=$FAST-Qd16
QDTB=$FAST-Qd200-Tb
QDNOP=$REF-Qd200
QY25=$FAST-Qd64-Qy25
QYF30=$FAST-Qd64-Qyf30
QYNOQD=$FAST-Qy25
QYSG=$FAST-Qd64-Qy25-Sg8
QYV=$FAST-Qd64-Qy25-Qv8
QYAX=$FAST-Qd64-Qy25-Ax
QY0=$FAST-Qd64-Qy0
QYTWO=$FAST-Qd64-Qy25-Qy30
QYNOP=$REF-Qd64-Qy25
CTP=$FAST-Ctp
CTP2=$FAST-Qd64-Ct16-Ctp2
CTPNOP=$REF-Ctp
CTP0=$FAST-Ctp0
ARMS="$REF $NEW $NEWG $SHORTM $Q17 $OVERL $GNOL $LNOP $BIG $BIGG $BIGM $ENOP $COLD4K $PC4K $NOINT4K $LOWINT4K $NOPC4K $OVER4K"
ARMS="$ARMS $FAST $SG8 $SGA8 $SG15 $SG16 $N2SG7 $N2SG8 $N8SG2 $SG0"
ARMS="$ARMS $QD200 $QD4096 $QD16 $QDTB $QDNOP"
ARMS="$ARMS $QY25 $QYF30 $QYNOQD $QYSG $QYV $QYAX $QY0 $QYTWO $QYNOP"
ARMS="$ARMS $CTP $CTP2 $CTPNOP $CTP0"
run_loop() {   # actual balance-oracle bound; every model action remains stubbed
 (
  set +u
  log() { echo "LOG: $*"; }
  power_set() { :; }; power_restore() { :; }
  vpack_gate() { return 0; }
  arm() { echo "ARM $1"; shift; printf '    %s\n' "$@"; echo "  ARGS: ${ARGS[*]}"; echo "  ARM_SG: ${ARM_SG:-}"; echo "  ARM_QD: ${ARM_QD:-}"; echo "  ARM_QY: ${ARM_QY:-}"; echo "  ARM_QYF: ${ARM_QYF:-}"; }
  HX_THREADS=16 SILU=/dev/null RHOME=$H RADIANCE_HOME_DIR2=$H MIRROR=$T
  PROMPTS=$T/prompts.json
  unset RADIANCE_TUNE
  SUM=$(sha256sum "$H/kernels/libr4d.so" | cut -d' ' -f1); LIB=$H/kernels/libr4d.so
  TALL_OK=1 TALL_WHY=test TALL_SUM=$SUM TALL_LIB=$LIB DK_OK=1 DK_WHY=test DK_SUM=$SUM DK_LIB=$LIB
  RB_OK=1 RB_WHY=test RB_SUM=$SUM RB_LIB=$LIB LCP_OK=1 LCP_WHY=test LCP_SUM=$SUM LCP_LIB=$LIB LCP_C=4096
  BAL_OK=1 BAL_WHY=test BAL_SUM=$SUM BAL_LIB=$LIB BAL_MAX_M=$1 P1_OK=1 P1_WHY=test P1_SUM=$SUM P1_LIB=$LIB
  OUT=$T/out; mkdir -p "$OUT"; CANCEL_FILE=/nonexistent/cancel; CANCEL=$CANCEL_FILE
  eval "$(sed -n '/^hotset_for() {/,/^}/p' "$W")"
  HX=(RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS=$HX_THREADS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU=$SILU)
  s=$(grep -n '^for a in \$ARMS; do' "$W" | head -1 | cut -d: -f1)
  e=$(awk -v s="$s" 'NR > s && /^done$/ { print NR; exit }' "$W")
  eval "$(sed -n "${s},${e}p" "$W")"
) 2>&1
}
out=$(run_loop 4096)
echo "$out" > "$T/loop.txt"
section() { awk -v a="ARM $1" '$0 == a { p = 1; next } /^ARM / { p = 0 } p' "$T/loop.txt"; }
# The last value an arm states for a variable (later entries win in the arm's environment).
val() { grep -oE "^    $2=.*" <<< "$1" | tail -1 | cut -d= -f2-; }
has() {   # what section var want
  local got
  got=$(val "$2" "$3")
  if [[ "$got" == "$4" ]]; then ok "$1: $3=$4"; else bad "$1: $3 is '${got}', want '$4'"; fi
}
sec=$(section "$REF")
if [[ -z "$sec" ]]; then bad "the adopted arm was not launched: $(grep -m3 "arm $REF" "$T/loop.txt")"
else
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 1024
  has adopted "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
fi
for a in "$NEW" "$NEWG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Ql10
  [[ "$a" == *-Qh3000-* ]] && n=P1+Ql10+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 10
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS ""
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS 6
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 1024
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS 32
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP 1
  [[ "$sec" == *"--max-num-batched-tokens 1024"* ]] && ok "$n: chunks of 1024" || bad "$n: no --max-num-batched-tokens 1024"
done
has "P1+Ql10+Qh3000" "$(section "$NEWG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
has "P1+Ql10" "$(section "$NEW")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS ""
sec=$(section "$Q17")
[[ -n "$sec" ]] && has "Qm17 with M16" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS 17 || bad "the Qm17 M16 arm was not launched"
for a in "$BIG" "$BIGG"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  n=P1+Lc2048+Qe64
  [[ "$a" == *-Qh3000-* ]] && n=P1+Lc2048+Qe64+Qh3000
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 64
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND 0
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 2048
  has "$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 2048
  [[ "$sec" == *"--max-num-batched-tokens 2048"* ]] && ok "$n: chunks of 2048" || bad "$n: no --max-num-batched-tokens 2048"
done
has "P1+Lc2048+Qe64+Qh3000" "$(section "$BIGG")" RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS 3000
for a in "$COLD4K" "$PC4K"; do
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "4K arm $a was not launched"; continue; fi
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_LCP 4096
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES 4096
  has "LCP4096" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA 64
  [[ "$sec" == *"--checkpoint-interval 4096"* ]] && ok "LCP4096: explicit interval 4096" || bad "LCP4096: missing interval 4096"
done
for a in "$SHORTM" "$OVERL" "$GNOL" "$LNOP" "$BIGM" "$ENOP" "$NOINT4K" "$LOWINT4K" "$NOPC4K" "$OVER4K"; do
  if [[ -n "$(section "$a")" ]]; then bad "arm $a was launched"
  elif grep -q "LOG: arm $a: .*skipped" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $a: " "$T/loop.txt" | sed "s|LOG: arm $a: ||")"
  else bad "arm $a neither launched nor skipped"; fi
done
for a in "$REF" "$NEW" "$NEWG" "$Q17" "$BIG" "$BIGG" "$COLD4K" "$PC4K"; do
  if grep -q "LOG: arm $a: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "arm $a: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
done

# Prompt-lookup speculation on the fast arm: launched with the speculation's arguments and the drafter left
# out, every other switch as the fast arm states it; skipped where a step of decode rows passes 16.
fast=$(section "$FAST")
[[ -n "$fast" ]] || bad "the fast arm was not launched: $(grep -m3 "arm $FAST" "$T/loop.txt")"
[[ "$fast" == *"  ARM_SG: "$'\n'* || "$fast" == *"  ARM_SG: " ]] && ok "the fast arm names no speculation" || bad "the fast arm's ARM_SG is not empty"
[[ "$fast" != *"--num-speculative-tokens"* ]] && ok "the fast arm adds no speculation argument" || bad "the fast arm passes --num-speculative-tokens"
sgarm() {   # arm, K, --max-num-seqs or empty, adaptive (non-empty)
  local a=$1 k=$2 n=$3 ad=$4 sec v
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "arm $a not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; return; fi
  if grep -q "LOG: arm $a: .*\(skipped\|refused\)" "$T/loop.txt"; then bad "arm $a: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
  [[ "$sec" == *"--num-speculative-tokens $k --spec-ngram $k"* ]] && ok "Sg$k${n:+ N$n}: $k drafts by prompt lookup" || bad "Sg$k${n:+ N$n}: no --num-speculative-tokens $k --spec-ngram $k"
  if [[ -n "$ad" ]]; then
    [[ "$sec" == *"--draft-window adaptive"* ]] && ok "Sga$k: the adaptive window" || bad "Sga$k: no --draft-window adaptive"
  fi
  [[ -z "$n" || "$sec" == *"--max-num-seqs $n"* ]] && ok "Sg$k${n:+ N$n}: its sequences" || bad "Sg$k N$n: no --max-num-seqs $n"
  grep -qx "  ARM_SG: $k" <<< "$sec" && ok "Sg$k${n:+ N$n}: the arm is gated on its admission lines (ARM_SG=$k)" || bad "Sg$k${n:+ N$n}: ARM_SG is '$(grep '  ARM_SG:' <<< "$sec")'"
  has "Sg$k${n:+ N$n}" "$sec" RADIANCE_EXPERIMENTAL_DS41_LOOKUP_ALONE 1
  # Every switch the fast arm states, the speculating arm states alike.
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED RADIANCE_EXPERIMENTAL_DS41_MOE_BALANCE \
           RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16 RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS \
           RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE RADIANCE_EXPERIMENTAL_DS41_SKIP_POST_STOP \
           RADIANCE_EXPERIMENTAL_PREFIX_EORS RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW; do
    has "Sg$k${n:+ N$n} as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
  [[ "$sec" == *"--max-num-batched-tokens 2048"* ]] && ok "Sg$k${n:+ N$n}: chunks of 2048" || bad "Sg$k${n:+ N$n}: no --max-num-batched-tokens 2048"
}
sgarm "$SG8" 8 "" ""
sgarm "$SGA8" 8 "" 1
sgarm "$SG15" 15 "" ""
sgarm "$N2SG7" 7 2 ""
for a in "$SG16" "$N2SG8" "$N8SG2"; do
  if [[ -n "$(section "$a")" ]]; then bad "arm $a was launched"
  elif grep -q "LOG: arm $a: Sg[0-9]* beside Lc2048: .*more than the 16 decode rows a step.*skipped" "$T/loop.txt"; then
    ok "skipped: $(grep -m1 "LOG: arm $a: " "$T/loop.txt" | sed "s|LOG: arm $a: ||" | cut -c1-110)..."
  else bad "arm $a neither launched nor skipped for its rows: $(grep -m1 "LOG: arm $a: " "$T/loop.txt")"; fi
done
# Sg0 is no speculation: launched, not gated.
sec=$(section "$SG0")
if [[ -z "$sec" ]]; then bad "the Sg0 arm was not launched: $(grep -m3 "arm $SG0" "$T/loop.txt")"
else grep -qx "  ARM_SG: " <<< "$sec" && ok "Sg0: no speculation, no gate" || bad "Sg0: ARM_SG is '$(grep '  ARM_SG:' <<< "$sec")'"; fi

# ---- 5b. QdN in the arm loop: Qd200 on the fast arm states the switch at 200 and every other switch as the fast
# arm; every arm without the part states it 0; Qd above the chunk, Qd16, Qd beside Tb and Qd without P1 skipped.
sec=$(section "$QD200")
if [[ -z "$sec" ]]; then bad "the Qd200 arm was not launched: $(grep -m3 "arm $QD200" "$T/loop.txt")"
else
  has "Qd200" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 200
  grep -qx "  ARM_QD: 200" <<< "$sec" && ok "Qd200: the arm is gated on its decode line (ARM_QD=200)" || bad "Qd200: ARM_QD is '$(grep '  ARM_QD:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_SCHED_TBO; do
    has "Qd200 as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
fi
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8"; do
  has "no Qd part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 0
done
grep -qx "  ARM_QD: " <<< "$fast" && ok "the fast arm names no Qd" || bad "the fast arm's ARM_QD is '$(grep '  ARM_QD:' <<< "$fast")'"
qdskip() {   # arm, the reason's pattern
  if [[ -n "$(section "$1")" ]]; then bad "arm $1 was launched"
  elif grep -q "LOG: arm $1: $2" "$T/loop.txt"; then ok "skipped: $(grep -m1 "LOG: arm $1: $2" "$T/loop.txt" | sed "s|LOG: arm $1: ||" | cut -c1-100)"
  else bad "arm $1 not skipped for '$2': $(grep -m1 "LOG: arm $1: " "$T/loop.txt")"; fi
}
qdskip "$QD4096" "Qd4096 is above the chunk Lc2048.*skipped"
qdskip "$QD16" "Qd16: Qd takes 17-4096.*skipped"
qdskip "$QDTB" "Qd200 with Tb.*skipped"
qdskip "$QDNOP" "the Q parts (RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS=200) need P1.*skipped"

# ---- 6b. QyP / QyfP in the arm loop.
for spec in "$QY25:25:" "$QYF30:30:1"; do
  IFS=: read -r a n f <<< "$spec"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Qy$n arm was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID "$n"
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT "${f:+0}"
  has "Qy${f:+f}$n" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 64
  grep -qx "  ARM_QY: $n" <<< "$sec" && ok "Qy${f:+f}$n: the arm is gated on its hybrid line (ARM_QY=$n)" || bad "Qy$n: ARM_QY is '$(grep '  ARM_QY:' <<< "$sec")'"
  grep -qx "  ARM_QYF: $f" <<< "$sec" && ok "Qy${f:+f}$n: ARM_QYF='$f'" || bad "Qy$n: ARM_QYF is '$(grep '  ARM_QYF:' <<< "$sec")'"
  for v in RADIANCE_EXPERIMENTAL_DS41_LCP RADIANCE_EXPERIMENTAL_DS41_P1 RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA \
           RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES \
           RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS RADIANCE_EXPERIMENTAL_SCHED_TBO RADIANCE_EXPERIMENTAL_DS41_P1_VERIFY; do
    has "Qy${f:+f}$n as the fast arm" "$sec" "$v" "$(val "$fast" "$v")"
  done
done
for a in "$REF" "$NEW" "$BIG" "$FAST" "$SG8" "$QD200"; do
  has "no Qy part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID 0
  has "no Qy part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID_ADAPT ""
done
grep -qx "  ARM_QY: " <<< "$fast" && ok "the fast arm names no Qy" || bad "the fast arm's ARM_QY is '$(grep '  ARM_QY:' <<< "$fast")'"
qdskip "$QYNOQD" "Qy25 splits the decode passes a Qd part arms, and there is none.*skipped"
qdskip "$QYSG" "Qy25 beside Sg8 (speculation.*skipped"
qdskip "$QYV" "Qy25 beside Qv8 (the host verify.*skipped"
qdskip "$QYAX" "Qy25 beside the aux expert worker.*skipped"
qdskip "$QY0" "Qy0: Qy takes 1-400.*skipped"
qdskip "$QYTWO" "two Qy parts.*skipped"
qdskip "$QYNOP" "the Q parts (.*RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=25.*) need P1.*skipped"
# ---- 7. Ctp: the prefill chain trace's switch and output prefix; skipped without P1 or at 0.
for x in "$CTP 4" "$CTP2 2"; do
  read -r a n <<< "$x"
  sec=$(section "$a")
  if [[ -z "$sec" ]]; then bad "the Ctp arm $a was not launched: $(grep -m3 "arm $a" "$T/loop.txt")"; continue; fi
  has "Ctp ($n)" "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_PREFILL "$n"
  [[ "$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT)" == */chain-$a ]] && ok "Ctp ($n): traced into chain-<arm>" ||
    bad "Ctp ($n): CHAIN_TRACE_OUT is '$(val "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT)'"
  has "Ctp ($n) keeps P1" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1 1
done
sec=$(section "$CTP2")
has "Ct16 beside Ctp2" "$sec" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE 16
has "Qd64 beside Ctp2" "$sec" RADIANCE_EXPERIMENTAL_DS41_P1_DECODE_ROWS 64
has "Ctp alone sets no decode trace" "$(section "$CTP")" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE ""
for a in "$FAST" "$QD200" "$REF"; do
  has "no Ctp part ($a)" "$(section "$a")" RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_PREFILL ""
done
qdskip "$CTPNOP" "Ctp4 traces armed card-path prefill passes, which need P1.*skipped"
qdskip "$CTP0" "Ctp0: CtpN takes a positive N.*skipped"
if pyout=$(python3 -I "$(dirname "$W")/chain_report.py" --selftest 2>&1) && grep -q "^selftest ran" <<< "$pyout"; then
  ok "chain_report.py --selftest (the --split classes of a prefill pass)"
else bad "chain_report.py --selftest: $(tail -5 <<< "$pyout")"; fi

short_out=$(run_loop 2048)
for a in "$COLD4K" "$PC4K"; do
  if grep -qxF "ARM $a" <<< "$short_out"; then bad "Mb+Lc4096 launched after only the balance M2048 oracle"
  elif grep -qF "arm $a: Mb refused: its row bound 4096 exceeds the balance oracle's actual M2048" <<< "$short_out"; then
    ok 'Mb+Lc4096 refused above BAL_MAX_M despite the independent LCP4096 receipt'
  else bad "no actual-bound refusal for arm $a"; fi
done
grep -qxF "ARM $BIG" <<< "$short_out" && ok 'Mb+Lc2048 still runs on its M2048 balance receipt' ||
  bad 'the shorter balance receipt incorrectly stopped Mb+Lc2048'

(( fail )) && { echo "ab_window_p1_test: FAILED (loop output in $T/loop.txt, removed on exit)"; exit 1; }
echo "ab_window_p1_test: all checks passed"
