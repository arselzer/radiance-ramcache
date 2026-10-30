# Radiance expert RAM cache: DeepSeek-V4.1-Flash (ds41)

Experimental work on running DeepSeek-V4.1-Flash with its native MXFP4 experts (40 layers of 384 routed experts) on four 16 GB RX 9070 XT cards, whose VRAM holds only part of the routed experts. The engine keeps the others in a RAM cache and on one NVMe drive. The CPU computes every routed expert the cards do not hold, with the same arithmetic as the cards' MXFP4 GEMV, so replies stay byte-identical to the base engine's.

This branch is a research checkpoint, not a serving deployment. The GLM 5.3 checkpoint is on [main](https://github.com/arselzer/radiance-ramcache/tree/main).

Unless a row says otherwise, measurements use the 8 held-out prompts: 64 tokens each, greedy, one at a time, four cards (TP4). Every number comes from an A/B window in `evidence/ds41/`:

| Measure | Result | Window |
|---|---|---|
| Decode, one stream | base engine 5.03 tok/s; recommended configuration 18.93 / 18.88 / 18.98 tok/s in three rounds (~18.9) | `ab-20261028T223812Z` (base), `ab-20261030T090127Z` |
| Replies | all 88 replies of that window's 11 arms byte-identical to the base engine's | same two |
| Prefill, one 3,086-token prompt | 99.3 s, then 63.1 s (~49 tok/s) with the exact consumer skip and 16-row CPU jobs | `ab-20261029T220909Z` |
| Follow-up turns on a ~3,100-token document | 97.6-102.3 s cold, 0.42-2.75 s with the prefix cache (3,072-3,152 tokens reused) | `ab-20261029T232708Z` |
| Four concurrent requests | ~20 tok/s aggregate decode (20.0 / 20.2), ~4.4 tok/s a stream, replies identical; configuration of 29 Oct | `ab-20261029T153750Z` |

The window before, `ab-20261030T081259Z`, tested six new switches. All six together gave 18.30 / 18.57 / 18.74 tok/s, and all 211 of its replies were identical too. Five of the six are in the recommended configuration (`radiance/tools/ds41-host-experts/README.md`). The timed pre-wake was left out.

Hardware: four RX 9070 XT 16 GB cards (gfx1201, each on its own PCIe 4.0 x16 port) and an AMD EPYC 7443 (24 cores). Memory is 8 x 16 GB DDR4-2133 RDIMM: 123 GiB visible, with a read ceiling measured at ~100 GB/s. One Samsung 990 Pro NVMe (~6.4 GB/s O_DIRECT) holds the 388 GB model container. The box also runs a production service (QFN). Each window waits for it to be idle and stops it. It then runs every configuration ("arm") as its own server on a cold RAM cache and compares every reply with the window's first arm. QFN is always restored at the end.

On Linux with Python 3.10+, both checks below read files only. They need no build, model, GPU or network:

```sh
python3 tools/verify_snapshot.py
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json evidence/ds41/ab-20261030T090127Z/*.json
```

Nothing was rebuilt for this export. Each window ran a build of the engine branch at that time, and `docs/ds41/NIGHT-LOG.md` names the commit for most of them. Both decode windows above ran the same radiance binary, built from `c10ba3a0`, with the same `libr4d.so`. It differs from the exported `d243fba8` only in five lines of `tools/ds41-host-experts/README.md`, the recommended configuration. Every ds41 change is an opt-in `RADIANCE_EXPERIMENTAL_*` switch or flag, off by default.

- `radiance/`: the exact tracked tree of engine commit `d243fba8` (1,428 files) from branch `claude/ds41-decode-impl`. Its switches, the recommended configuration and the A/B tools are described in `radiance/tools/ds41-host-experts/README.md`.
- `docs/ds41/`: 26 reports: the review, the plan, the night log, research, designs and implementation reports. `provenance/` pins every engine file.
- `evidence/ds41/`: 84 A/B windows, each with its log, per-arm results and memory traces. Server logs (compressed) are kept for the seven windows the numbers above and the prefill exactness come from.
- `research/ds41/`: the small measurement programs and outputs the reports cite.
- `tools/`: `verify_snapshot.py` (snapshot verification), `ds41_reply_check.py` (reply identity check) and `ds41_refresh.py`. The refresh tool rebuilds this branch from an engine commit, the report folder and a window list.

The engine is Radiance ([upstream](https://codeberg.org/StillDeadcode/radiance)). It contains upstream main through `4374bea1` (2026-10-28), the 80 commits of the deepseek41 architecture pull request 266, and 172 commits of this work. Upstream licensing and notices remain (`LICENSE`, `radiance/LICENSE`, `radiance/NOTICE`).

These are excluded:

- model, container and tokenizer payloads
- research binaries and captures: access and routing traces, KLD logit records, SiLU table captures
- all other server logs
- private serving configuration and credentials

`evidence/ds41/index.json` lists every file left out of a window, with its reason. Paths in the reports and logs refer to the author's machine.

See [STATUS.md](STATUS.md) for what is and is not validated.
