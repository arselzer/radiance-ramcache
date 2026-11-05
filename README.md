# Radiance expert RAM cache: DeepSeek-V4.1-Flash (ds41)

Experimental work on running DeepSeek-V4.1-Flash with its native MXFP4 experts (40 layers of 384 routed experts) on four 16 GB RX 9070 XT cards, whose VRAM holds only part of the routed experts. The engine keeps the others in a RAM cache and on one NVMe drive. The CPU computes every routed expert the cards do not hold, with the same arithmetic as the cards' MXFP4 GEMV, so replies stay byte-identical to the base engine's.

This branch is a research checkpoint, not a serving deployment: exported source snapshots plus the reports and A/B evidence. The engine with its full git history is on [main](https://github.com/arselzer/radiance-ramcache/tree/main), a radiance fork on upstream 1.16.0 (see its RAMCACHE.md); the history of the exported engine commits is on [ds41-stable](https://github.com/arselzer/radiance-ramcache/tree/ds41-stable) and the default-off prototypes on [ds41-research](https://github.com/arselzer/radiance-ramcache/tree/ds41-research).

On 30 Oct the box went from 128 GB to 256 GB of RAM, so the RAM cache now holds 192 GiB of experts and drive reads are rare. Unless a row says otherwise, measurements use the 8 held-out prompts: 64 tokens each, greedy, four cards (TP4). The original results below come from A/B windows in `evidence/ds41/`; the November prefill reports separately identify their custom guarded windows and exported timing records.

The latest recommended configuration uses runtime `f84b23db` with coherent `libr4d` `92bd7719`.
The source export includes documentation updates over that runtime; no rebuild occurred during export.

| Workload | Four RX 9070 XT (tok/s) | With optional expert worker (tok/s) |
|---|---:|---:|
| Decode, 1 stream | 28–30 | 33.9 |
| Decode, 4 streams | 41–44 | 51.5 |
| Decode, 8 streams | 45–47 | 57.6–58.2 |
| Decode, 8 streams, chat | 38.4–38.9 | 45.8–46.5 |
| Decode, prefix-cached agent turns | 25–26 | 29.4–29.6 |
| Prefill, 6,817-token prompt | 515 | unmeasured |
| Prefill, 47,917-token prompt | 776 | unmeasured |
| Prefill, 99,987-token prompt | 661 | unmeasured |

Decode rates total the streams, using N times the mean server-reported rate. Prefill uses the
server's prompt-processing rate; all three rows use admit-all at the tested 192 GiB cache budget.
The 100K ABBA comparison measured 416→661 tok/s and TTFT 240.46→151.32 s, with all 84 original
replies/counts and 14 full prefill hashes exact. Subsequent N8 wave time stayed neutral.
Smaller/dynamic caches retain the general admission fallback; the worker combination is unmeasured.

The latest DSpark prototype measured 24.92→29.19 tok/s (+17.14%) on its matched 32K/single-stream
code workload, passing all 96 original replies/counts. The valid earlier 29.79 result used a
different profile and metric: this is not a new absolute record. A separate matched current-profile
PLAIN bracket measured accepted 24.83 versus prototype OFF 24.65 (-0.75%, within observed drift).
Qa0 improved warmed 6.8K TTFT by 4.12%; the cold 48K comparison remains inconclusive due to control drift.

[Detailed results and qualification limits](radiance/docs/DS41-RESULTS-20261105.md) distinguish
model tests from the newer CPU component batches (7/21/46 suites), whose GPU/model gates remain open.
[Controller extracts](research/ds41/qualification-results-20261105.json) retain full local receipt hashes.

The imported baseline preserves official MXFP4 expert codes/scales and passed all 1,119 declared
other-neural-tensor conversion mappings plus seven integer controls. Two ISTA IQ4_NL Engram tables
remain additional quantization; reply exactness does not establish independent official-runner equivalence.

The ds41 plugin also declares its DSML reply format: `/v1/chat/completions` returns tool calls in `message.tool_calls` and thinking in `message.reasoning_content`. Across `ab-20261031T111018Z` and `ab-20261031T115207Z`, all 16 chat and 13 harness replies stayed byte-identical with registration off/on, with identical prompts and generated token counts; the base arm matched the merged base on 8/8 held-out replies. A thinking-on calculator request parsed both fields and produced the same 116 token IDs through raw `/completion`; the weather request also matched all 67 IDs and chose empty reasoning (`evidence/ds41/ab-20261031T115207Z/dsml-probe.json`). The research container has no chat template, so chat serving needs `--override-chat-template`; the final window used the saved text-only adapter `evidence/ds41/ab-20261031T115207Z/dsml-chat-template.jinja.txt`, whose tested request renderings match the official V4.1 encoder byte for byte. The first window stopped at501 for the missing template, and the second stopped on an overly strict empty-reasoning assertion. All three windows restored QFN and checked a completion.

The latest follow-up checkpoint adds measured opt-in controls, with all serving replies and corpus comparisons matching their merged-base references:

| Follow-up | Result | Evidence |
|---|---|---|
| P1 speculative read-ahead disabled (Qa0), stable build | Independent 8/0/8 comparison: 3,086-token TTFT 8.71/7.65/8.92 s (12.22–14.25% lower); 6,817-token 11.85/10.82/11.83 s (8.52–8.67% lower). Short 82-token request 1.8–3.5% slower; default 8 unchanged. All 15 replies and 6 full hash checks exact. | [Stable Qa0 report](docs/ds41/PREFILL-QA0-20261103.md), [timing records](research/ds41/ds41_prefill_qa0_stable_20261103.json) |
| P1 wave size at Qa0, current research build | Qw4 longest-prompt TTFT 11.05 s against 11.44/11.64 s Qw2 controls; 3K 0.47% slower. All 15 replies, 6 hash checks and fresh four-card native gates pass. Diagnostic screening only; unadopted. | [Qw report](docs/ds41/PREFILL-QW-20261103.md) |
| Frozen adaptive DSpark verification | Code +4.7% versus fixed2; chat +9.9% versus fixed1. Held-out +3.4% / +7.7%; query/observer/wait and required telemetry included. One global TRAIN-selected policy, no refit; single-stream, 192 GiB cache. | `ab-20261031T165338Z`; frozen fit/cost/metadata in `research/ds41/ds41_policy_*` |
| Optional 4K prefill chunks | 3,086-token TTFT 7.33→5.69 s; 6,817-token 17.12→12.69 s (~537 tok/s). Long/prefix/96 GiB checks passed; the five-prompt sum was neutral and the first tiny request slower. General recommendation remains 2K until concurrency/combination gates complete. | `ab-20261031T162629Z` |
| P1 plus packed static experts | Actual packed residents and corrected long/96 GiB gate passed. Reverse N4/N8 pairs 38.31→39.06 / 41.35→42.47 tok/s; previous N8 pair neutral. Small, noisy gain. | `ab-20261031T144623Z` |
| Bank lending | Correct full-byte restoration and solo drafting resume after actual N4 at 192/96 GiB; mixed workload slower: 56.62→60.29 / 79.01→82.91 s. Full cold lifecycle also slower; keep experimental. | `ab-20261031T180730Z` |
| Collective slices; span/n128 | Fresh byte/corpus/logical-hash gates passed. Collective long/prefix TTFT improved ~4% / ~1.4%, N8 throughput fell ~1.1%. Span/n128 timing differences were ~0–1.2%, insufficient for a performance recommendation. | `ab-20261031T182505Z`, `ab-20261031T185127Z` |

New research switches remain opt-in. The 4K window initially kept Rb/Vp/N128 off; the adaptive policy still needs its separate smaller-cache and concurrent-solo composition gates. Packed KV, physical KV growth, larger arena lending and further bank optimizations remain separate unvalidated source work, outside this exported engine checkpoint. Failed windows and invalid legs are retained and identified in the night log.

Hardware: four RX 9070 XT 16 GB cards (gfx1201, PCIe 4.0 x16 host paths) and an AMD EPYC 7443 (24 cores). Since the owner's 2 November slot move, the four primary cards span four CPU root buses; older windows had two cards on one quadrant. An optional R9700 is excluded from the four-card November comparisons. Memory is 8 x 32 GB DDR4-2666 RDIMM: 247 GiB visible. The earlier CPU measurement was ~121 GB/s, and ~140 GB/s together with the cards' host-to-card copies (`research/ds41/membw_dur.bwmix.out.txt`). One Samsung 990 Pro NVMe (~6.4 GB/s O_DIRECT) holds the 388 GB model container. The box also runs a production service (QFN). Each window waits for it to be idle, stops it, runs fresh servers for its configurations, and checks replies against its retained references. QFN is restored and checked at the end.

On Linux with Python 3.10+, both checks below read files only. They need no build, model, GPU or network:

```sh
python3 tools/verify_snapshot.py
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json evidence/ds41/ab-20261030T205440Z/*.json
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261031T104524Z/base.json evidence/ds41/ab-20261031T100852Z/*.json
```

The second compares every held-out reply of that window with the base engine's (24 of 24) and skips the code and chat arms, whose prompts the base run did not have. The third does the same for the upstream sync against the merged base engine (32 of 32).

Nothing was rebuilt for this export. Each window ran a build of the engine branch at that time, and `docs/ds41/NIGHT-LOG.md` names the commit for most of them. The post-upgrade decode windows ran builds of `claude/ds41-lcpfast` and `claude/ds41-dsvp`, both merged in the exported engine; the Phase 1 adoption window ran a build of `claude/ds41-int5` (merged in the exported engine) with its own deepseek41 plugin; the first Phase 1 window, `ab-20261030T201710Z`, ran a build of `claude/ds41-p1`; the DSpark-beside-concurrency window `ab-20261030T234957Z` ran a build of `claude/ds41-dsconc`, and the lent-arena windows `ab-20261031T011849Z` to `ab-20261031T030023Z` builds of `claude/ds41-p1big` with its own plugin, and the upstream-sync windows `ab-20261031T100852Z` and `ab-20261031T104524Z` a build of `claude/ds41-sync` with every plugin from it (all merged). The host-expert optimizations use opt-in `RADIANCE_EXPERIMENTAL_*` switches or flags. The DSML chat reply format is registered by default.

- `radiance/`: exact engine `d316a3b3` (1866 files), including the latest documentation over runtime `f84b23db`; all paths, bytes and modes pinned.
- `docs/ds41/`: 47 reports, including the night log, recovered workflow research and current prefill/precision plans; `provenance/` pins every engine file.
- `evidence/ds41/`: 149 A/B windows, including interrupted and failed experiments with their original status preserved; selected server logs are compressed.
- `research/ds41/`: the small measurement programs and outputs the reports cite.
- `tools/`: `verify_snapshot.py` (snapshot verification), `ds41_reply_check.py` (reply identity check) and `ds41_refresh.py`. The refresh tool rebuilds this branch from an engine commit, the report folder and a window list.

The engine is Radiance ([upstream](https://codeberg.org/StillDeadcode/radiance)), from the measured 1.15-based line with the deepseek41 architecture work and this fork’s exact optimizations. The separate `main` branch follows upstream 1.16. Licensing and notices remain unchanged.

These are excluded:

- model, container and tokenizer payloads
- research binaries and captures: access and routing traces, KLD logit records, SiLU table captures
- all other server logs
- private serving configuration and credentials

`evidence/ds41/index.json` lists every file left out of a window, with its reason. Paths in the reports and logs refer to the author's machine.

See [STATUS.md](STATUS.md) for what is and is not validated.

Three larger historical windows remain fully available in the [prior snapshot index](research/ds41/archived-window-index-20261105.json), with file hashes and immutable links. Their results and failures remain part of the research record. The current export keeps the existing 100 MB evidence bound.
