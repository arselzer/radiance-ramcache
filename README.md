# Radiance expert RAM cache: DeepSeek-V4.1-Flash (ds41)

Experimental work on running DeepSeek-V4.1-Flash with its native MXFP4 experts (40 layers of 384 routed experts) on four 16 GB RX 9070 XT cards, whose VRAM holds only part of the routed experts. The engine keeps the others in a RAM cache and on one NVMe drive. The CPU computes every routed expert the cards do not hold, with the same arithmetic as the cards' MXFP4 GEMV, so replies stay byte-identical to the base engine's.

This branch is a research checkpoint, not a serving deployment. The GLM 5.3 checkpoint is on [main](https://github.com/arselzer/radiance-ramcache/tree/main).

On 30 Oct the box went from 128 GB to 256 GB of RAM, so the RAM cache now holds 192 GiB of experts and drive reads are rare. Unless a row says otherwise, measurements use the 8 held-out prompts: 64 tokens each, greedy, four cards (TP4). Every number comes from an A/B window in `evidence/ds41/`:

| Measure | Result | Window |
|---|---|---|
| Decode, one stream | recommended configuration 31.24 tok/s with the 192 GiB cache (20.87 at the earlier 96 GiB; 19.85 on 128 GB). Base engine 5.03 tok/s (128 GB, not re-measured) | `ab-20261030T194552Z`, `ab-20261030T205440Z`; base `ab-20261028T223812Z` |
| Concurrent requests | 38.7 tok/s total decode at 4 streams, 42.2-42.9 at 8 (decode-only settings with packed VRAM residents); with the recommended prefill settings 37.2-38.0 and 41.2-41.5. Two-batch overlap of the steps: -3 to -5% (left off) | `ab-20261030T194552Z`, `ab-20261030T205440Z`, `ab-20261030T231526Z`, `ab-20261031T030023Z` |
| DSpark speculative decoding, one stream | code prompts +12.2 to +12.5% (29.75-29.79 against 26.48-26.52 tok/s, 89% of drafts accepted), held-out +9.3%, chat +1.2 to +1.5%. Beside concurrent requests it drafts only while one decodes alone: 4 streams 37.1 against 38.8 tok/s in total (-4.5%, the bank's VRAM) | `ab-20261030T205440Z`, `ab-20261030T213623Z`, `ab-20261030T234957Z` |
| Prefill, recommended (Phase 1: every routed expert of long layer passes on the cards, 2,048-token chunks, lent arena) | 82 / 272 / 933 tokens unchanged (1.9 / 3.8 / 4.2 s); 1,890 tokens 5.0 s, **3,086 7.2 s (429 tok/s)**, **6,817 17.2 s (396 tok/s)**; prefill hashes identical (5 of 5); held-out decode 29.45 tok/s (1,024-token chunks: 29.91, with 7.5 / 8.6 / 26.7 s prefill) | `ab-20261031T022949Z`, `ab-20261031T011849Z`, `ab-20261030T231526Z` |
| Prefill, large chunks without Phase 1 | 933 / 1,890 / 3,086 tokens in 10.5 / 18.5 / 28.0 s; 6,817 tokens in 51.2 s. Phase 1 at 2,048-token chunks was faster still (6.6 s at 3,086) but cost decode residents | `ab-20261030T231526Z`, `ab-20261030T201710Z` |
| Harness turns (2 sessions, 13 turns up to 9.7K tokens) | summed time to first token 135.1 s cold; 82.5 s with the prefix cache and end-of-request snapshots on the chunk grid; **72.7 s with snapshots on a 16-token grid** (`..._RESUME_GRID=16`: a turn resumes within 15 tokens of the previous reply's end; short follow-ups 0.3-0.6 s instead of 3.3-3.7 s). Prefill hashes identical to the cold run on 13 of 13 turns; decode unchanged | `ab-20261031T090154Z`, `ab-20261031T034433Z` |
| Replies | all 72 held-out replies of the four post-upgrade windows byte-identical to the base engine's; code, chat and long-prompt replies identical across every arm of their window | the windows above |
| Upstream sync (upstream main v1.15.0 + 12 merged, 31 Oct) | speeds unchanged: decode 31.09 / 29.29 tok/s (decode-only / recommended), 37.6 / 41.4 total at 4 / 8 streams, 3,086-token prefill 7.21 s, harness turns 72.9 s. Exact on the merged build: KLD records 1,512 / 1,512 positions, prefill hashes 13 / 13, CPU rows 29,668 against the card's with no difference. Upstream's own rounding changed (its RMS norm sums over a fixed partition), so the reference is now the merged base engine: 32 of 32 held-out replies identical to it; it differs from the pre-sync base in 3 of 8 replies (same top token at 98.9% of KLD positions) | `ab-20261031T100852Z`, `ab-20261031T104524Z` |

The ds41 plugin also declares its DSML reply format: `/v1/chat/completions` returns tool calls in `message.tool_calls` and thinking in `message.reasoning_content`. Across `ab-20261031T111018Z` and `ab-20261031T115207Z`, all 16 chat and 13 harness replies stayed byte-identical with registration off/on, with identical prompts and generated token counts; the base arm matched the merged base on 8/8 held-out replies. A thinking-on calculator request parsed both fields and produced the same 116 token IDs through raw `/completion`; the weather request also matched all 67 IDs and chose empty reasoning (`evidence/ds41/ab-20261031T115207Z/dsml-probe.json`). The research container has no chat template, so chat serving needs `--override-chat-template`; the final window used the saved text-only adapter `evidence/ds41/ab-20261031T115207Z/dsml-chat-template.jinja.txt`, whose tested request renderings match the official V4.1 encoder byte for byte. The first window stopped at501 for the missing template, and the second stopped on an overly strict empty-reasoning assertion. All three windows restored QFN and checked a completion.

The latest follow-up checkpoint adds measured opt-in controls, with all serving replies and corpus comparisons matching their merged-base references:

| Follow-up | Result | Evidence |
|---|---|---|
| Frozen adaptive DSpark verification | Code +4.7% versus fixed2; chat +9.9% versus fixed1. Held-out +3.4% / +7.7%; query/observer/wait and required telemetry included. One global TRAIN-selected policy, no refit; single-stream, 192 GiB cache. | `ab-20261031T165338Z`; frozen fit/cost/metadata in `research/ds41/ds41_policy_*` |
| Optional 4K prefill chunks | 3,086-token TTFT 7.33→5.69 s; 6,817-token 17.12→12.69 s (~537 tok/s). Long/prefix/96 GiB checks passed; the five-prompt sum was neutral and the first tiny request slower. General recommendation remains 2K until concurrency/combination gates complete. | `ab-20261031T162629Z` |
| P1 plus packed static experts | Actual packed residents and corrected long/96 GiB gate passed. Reverse N4/N8 pairs 38.31→39.06 / 41.35→42.47 tok/s; previous N8 pair neutral. Small, noisy gain. | `ab-20261031T144623Z` |
| Bank lending | Correct full-byte restoration and solo drafting resume after actual N4 at 192/96 GiB; mixed workload slower: 56.62→60.29 / 79.01→82.91 s. Full cold lifecycle also slower; keep experimental. | `ab-20261031T180730Z` |
| Collective slices; span/n128 | Fresh byte/corpus/logical-hash gates passed. Collective long/prefix TTFT improved ~4% / ~1.4%, N8 throughput fell ~1.1%. Span/n128 timing differences were ~0–1.2%, insufficient for a performance recommendation. | `ab-20261031T182505Z`, `ab-20261031T185127Z` |

New research switches remain opt-in. The 4K window initially kept Rb/Vp/N128 off; the adaptive policy still needs its separate smaller-cache and concurrent-solo composition gates. Packed KV, physical KV growth, larger arena lending and further bank optimizations remain separate unvalidated source work, outside this exported engine checkpoint. Failed windows and invalid legs are retained and identified in the night log.

Hardware: four RX 9070 XT 16 GB cards (gfx1201, PCIe 4.0 x16 each; two of them share one CPU quadrant) and an AMD EPYC 7443 (24 cores). Memory is 8 x 32 GB DDR4-2666 RDIMM: 247 GiB visible. The CPU reads ~121 GB/s, and ~140 GB/s together with the cards' host-to-card copies (`research/ds41/membw_dur.bwmix.out.txt`). One Samsung 990 Pro NVMe (~6.4 GB/s O_DIRECT) holds the 388 GB model container. The box also runs a production service (QFN). Each window waits for it to be idle and stops it. It then runs every configuration ("arm") as its own server on a cold RAM cache and compares every reply with the window's first arm. QFN is always restored at the end.

On Linux with Python 3.10+, both checks below read files only. They need no build, model, GPU or network:

```sh
python3 tools/verify_snapshot.py
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json evidence/ds41/ab-20261030T205440Z/*.json
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261031T104524Z/base.json evidence/ds41/ab-20261031T100852Z/*.json
```

The second compares every held-out reply of that window with the base engine's (24 of 24) and skips the code and chat arms, whose prompts the base run did not have. The third does the same for the upstream sync against the merged base engine (32 of 32).

Nothing was rebuilt for this export. Each window ran a build of the engine branch at that time, and `docs/ds41/NIGHT-LOG.md` names the commit for most of them. The post-upgrade decode windows ran builds of `claude/ds41-lcpfast` and `claude/ds41-dsvp`, both merged in the exported engine; the Phase 1 adoption window ran a build of `claude/ds41-int5` (merged in the exported engine) with its own deepseek41 plugin; the first Phase 1 window, `ab-20261030T201710Z`, ran a build of `claude/ds41-p1`; the DSpark-beside-concurrency window `ab-20261030T234957Z` ran a build of `claude/ds41-dsconc`, and the lent-arena windows `ab-20261031T011849Z` to `ab-20261031T030023Z` builds of `claude/ds41-p1big` with its own plugin, and the upstream-sync windows `ab-20261031T100852Z` and `ab-20261031T104524Z` a build of `claude/ds41-sync` with every plugin from it (all merged). The host-expert optimizations use opt-in `RADIANCE_EXPERIMENTAL_*` switches or flags. The DSML chat reply format is registered by default.

- `radiance/`: the exact tracked tree of engine commit `e184c6b7` (1,849 files) from branch `claude/ds41-decode-impl`. Its switches, the recommended configuration and the A/B tools are described in `radiance/tools/ds41-host-experts/README.md`.
- `docs/ds41/`: 31 reports: the review, the plan, the night log, research, designs, implementation reports and the upstream sync (`UPSTREAM-SYNC-20261031.md`). `provenance/` pins every engine file.
- `evidence/ds41/`: 127 A/B windows, each with its log, per-arm results and memory traces. Server logs (compressed) are kept for the thirty-four windows the numbers above and the prefill exactness come from.
- `research/ds41/`: the small measurement programs and outputs the reports cite.
- `tools/`: `verify_snapshot.py` (snapshot verification), `ds41_reply_check.py` (reply identity check) and `ds41_refresh.py`. The refresh tool rebuilds this branch from an engine commit, the report folder and a window list.

The engine is Radiance ([upstream](https://codeberg.org/StillDeadcode/radiance)). It contains upstream main through `b6acbd69` (2026-10-31, v1.15.0 and 12 commits), the 80 commits of the deepseek41 architecture pull request 266 (at `bd65d4e5`), and 325 commits of this work. Upstream licensing and notices remain (`LICENSE`, `radiance/LICENSE`, `radiance/NOTICE`).

These are excluded:

- model, container and tokenizer payloads
- research binaries and captures: access and routing traces, KLD logit records, SiLU table captures
- all other server logs
- private serving configuration and credentials

`evidence/ds41/index.json` lists every file left out of a window, with its reason. Paths in the reports and logs refer to the author's machine.

See [STATUS.md](STATUS.md) for what is and is not validated.
