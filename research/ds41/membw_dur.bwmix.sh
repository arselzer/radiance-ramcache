#!/usr/bin/env bash
# CPU streaming alone, the four cards' host->card copies alone, then both at once (QFN stopped).
W=/home/as/ds41-claude-build/work
M=$W/membw/membw_dur
X=(env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES $W/p0/bin/ds41_p0_bench $W/p0/bin/libr4d.so copy)
echo "== CPU alone, 40 threads"; $M 40 8
echo "== cards alone"; "${X[@]}" --units 512
echo "== both: CPU 40 threads for 60 s, the copy bench from 5 s"
$M 40 60 > $W/bwmix/both-cpu.txt & pid=$!
sleep 5
echo "copy start $(date +%s.%N)"; "${X[@]}" --units 1536; echo "copy end $(date +%s.%N)"
wait $pid; cat $W/bwmix/both-cpu.txt
