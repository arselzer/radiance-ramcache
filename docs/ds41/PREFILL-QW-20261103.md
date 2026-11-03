# P1 wave size at Qa0: 3 November 2026 screening

Qw4 reduced time to first token for the 6,817-token prompt by3.44–5.08% against
separate Qw2 controls, but was0.47% slower at3,086 tokens. This is a workload-specific
screening result. The controller retains `performance_eligible=false` and
`adopted=false`; it does not qualify a general recommendation or a decode improvement.

| Prompt tokens | First Qw2 TTFT(s) | Qw4 TTFT(s) | Last Qw2 TTFT(s) |
|---:|---:|---:|---:|
| 82 | 2.074391 | 2.091227 | 2.101448 |
| 3,086 | 8.084278 | 8.122578 | 8.084362 |
| 6,817 | 11.441712 | 11.048631 | 11.639860 |

The short request is mixed (+0.81%/-0.49% latency against the separate controls).
For the longest prompt Qw4 gives617.08 native prompt tokens/s, against595.89 and
585.74. No controls were pooled and no sample was excluded. This experiment does
not establish why the longest prompt benefits; cumulative overlapping waits are
not exclusive timing or physical SSD-byte measurements.

All five fresh servers used the identical current43 engine and all five plugins
from its one coherent build. Qa0 was fixed throughout; only Qw2/Qw4 changed.
The first two servers ran three full-hash requests each, then three servers ran
quiet2/4/2 with profiling and hash capture off. Every arm retained the original
82/3086/6817-token prompts, cap16 and original10Hz scheduler observer. All15
replies, prompt IDs and generated counts matched; all6 hash requests matched
the complete retained group/logit/40-layer/rank words and capture scopes.

Before any model request, the actual current43 checker and libr4d ran a fresh
full P1 threshold2048 native gate on all four selected9070XTs:116cases per card,
including positive livewave4,170 projection checks and30 full-span waves. Older
compiled or pre-move results were not substituted. No engine, GPU kernel,
checker or plugin was rebuilt for this window.

The model configuration was TP4/192GiB expert cache/rawBF16 KV/MAX8192/M2048/B6/
Mr32/Lc2048/P1/Qm320/Qe64/Mb1/Rb/Vp/Cs, no prefix or drafting. Prefix, lifecycle,
cancellation,96GiB, other context lengths and native decode throughput remain
outside this experiment. Variable RAM/VRAM budgets and defaults are unchanged.

The physical build lock and both research locks covered the window. All children
exited ordinary0/unforced/reaped; raw/masked physical maps and empty foreign KFD
checks passed. The original production container and image returned the normal
answer at07:32:47UTC after raw command0 at07:31:56. Full post-validation passed.

Evidence:

- Private summary: `/home/as/ds41-claude-build/work/autonomous-20261101/P1_Qw_Qa0_A/summary.json`.
- Summary SHA256: `ead03c7ec3a5c0ce88153dc1fb9e0ed7b551a15eae4fd2bdfb13de2786f831ec`,2975196bytes.
- Sealed source: `186b37667f4513cbf748bd2991b173895cf840236018820f3700419726a4a4d9`.
- Current engine source: `43a1d1d21a65f229d5a1ed4533fbb869ac0a04ac`.
- ROOT preparation:16 source guards,AST11/bash1, complete prior/reference and
  static ELF/provider/namespace checks; ROOT79632 full execution and post PASS.

The stable e184 Qa0 timing result is a separate experiment with its own controls.
The current43 broader Qa0 prefix/cancel/N4/N8 aggregate76replies+2accepted cancels
also passed, but supplies functional evidence rather than quiet timings. Neither
study transfers whole-build or untested lifecycle qualifications to Qw4.
