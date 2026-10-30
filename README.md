# Radiance expert RAM cache: DeepSeek-V4.1-Flash (ds41)

Experimental work on running DeepSeek-V4.1-Flash with its native MXFP4 experts (40 layers of 384 routed experts) on four 16 GB RX 9070 XT cards, whose VRAM holds only part of the routed experts. The engine keeps the others in a RAM cache and on one NVMe drive. The CPU computes every routed expert the cards do not hold, with the same arithmetic as the cards' MXFP4 GEMV, so replies stay byte-identical to the base engine's.

This branch is a research checkpoint, not a serving deployment. The GLM 5.3 checkpoint is on [main](https://github.com/arselzer/radiance-ramcache/tree/main).

On 30 Oct the box went from 128 GB to 256 GB of RAM, so the RAM cache now holds 192 GiB of experts and drive reads are rare. Unless a row says otherwise, measurements use the 8 held-out prompts: 64 tokens each, greedy, four cards (TP4). Every number comes from an A/B window in `evidence/ds41/`:

| Measure | Result | Window |
|---|---|---|
| Decode, one stream | recommended configuration 31.24 tok/s with the 192 GiB cache (20.87 at the earlier 96 GiB; 19.85 on 128 GB). Base engine 5.03 tok/s (128 GB, not re-measured) | `ab-20261030T194552Z`, `ab-20261030T205440Z`; base `ab-20261028T223812Z` |
| Concurrent requests | 38.7 tok/s total decode at 4 streams, 42.2 at 8 | `ab-20261030T194552Z`, `ab-20261030T205440Z` |
| DSpark speculative decoding, one stream | code prompts +12.2 to +12.5% (29.75-29.79 against 26.48-26.52 tok/s, 89% of drafts accepted), held-out +9.3%, chat +1.2 to +1.5%. The draft bank needs a single sequence | `ab-20261030T205440Z`, `ab-20261030T213623Z` |
| Prefill, recommended large-chunk mode | 3,086 tokens in 27.8-28.1 s (110 tok/s); 6,817 tokens in 51.2-51.7 s (132 tok/s) | `ab-20261030T194552Z`, `ab-20261030T201710Z` |
| Prefill with every expert on the cards (unmerged branch `claude/ds41-p1`) | 3,086 tokens in 6.6 s (468 tok/s); 6,817 tokens in 17.3 s (394 tok/s); KLD records identical over 3,397 positions, prefill hashes 5 of 5. Not adopted yet: an 82-token prompt takes 4.2 s against 1.9 s, and decode is 5-12% slower in that configuration | `ab-20261030T201710Z` |
| Follow-up turns on a ~3,100-token document | 97.6-102.3 s cold, 0.42-2.75 s with the prefix cache (128 GB) | `ab-20261029T232708Z` |
| Replies | all 72 held-out replies of the four post-upgrade windows byte-identical to the base engine's; code, chat and long-prompt replies identical across every arm of their window | the windows above |

Hardware: four RX 9070 XT 16 GB cards (gfx1201, PCIe 4.0 x16 each; two of them share one CPU quadrant) and an AMD EPYC 7443 (24 cores). Memory is 8 x 32 GB DDR4-2666 RDIMM: 247 GiB visible. The CPU reads ~121 GB/s, and ~140 GB/s together with the cards' host-to-card copies (`research/ds41/membw_dur.bwmix.out.txt`). One Samsung 990 Pro NVMe (~6.4 GB/s O_DIRECT) holds the 388 GB model container. The box also runs a production service (QFN). Each window waits for it to be idle and stops it. It then runs every configuration ("arm") as its own server on a cold RAM cache and compares every reply with the window's first arm. QFN is always restored at the end.

On Linux with Python 3.10+, both checks below read files only. They need no build, model, GPU or network:

```sh
python3 tools/verify_snapshot.py
python3 tools/ds41_reply_check.py evidence/ds41/ab-20261028T223812Z/base.json evidence/ds41/ab-20261030T205440Z/*.json
```

The second compares every held-out reply of that window with the base engine's (24 of 24) and skips the code and chat arms, whose prompts the base run did not have.

Nothing was rebuilt for this export. Each window ran a build of the engine branch at that time, and `docs/ds41/NIGHT-LOG.md` names the commit for most of them. The post-upgrade decode windows ran builds of `claude/ds41-lcpfast` and `claude/ds41-dsvp`, both merged in the exported engine; the Phase 1 prefill window ran a build of the unmerged `claude/ds41-p1`. Every ds41 change is an opt-in `RADIANCE_EXPERIMENTAL_*` switch or flag, off by default.

- `radiance/`: the exact tracked tree of engine commit `7bfa66ad` (1,443 files) from branch `claude/ds41-decode-impl`. Its switches, the recommended configuration and the A/B tools are described in `radiance/tools/ds41-host-experts/README.md`.
- `docs/ds41/`: 28 reports: the review, the plan, the night log, research, designs and implementation reports. `provenance/` pins every engine file.
- `evidence/ds41/`: 100 A/B windows, each with its log, per-arm results and memory traces. Server logs (compressed) are kept for the eleven windows the numbers above and the prefill exactness come from.
- `research/ds41/`: the small measurement programs and outputs the reports cite.
- `tools/`: `verify_snapshot.py` (snapshot verification), `ds41_reply_check.py` (reply identity check) and `ds41_refresh.py`. The refresh tool rebuilds this branch from an engine commit, the report folder and a window list.

The engine is Radiance ([upstream](https://codeberg.org/StillDeadcode/radiance)). It contains upstream main through `4374bea1` (2026-10-28), the 80 commits of the deepseek41 architecture pull request 266, and 223 commits of this work. Upstream licensing and notices remain (`LICENSE`, `radiance/LICENSE`, `radiance/NOTICE`).

These are excluded:

- model, container and tokenizer payloads
- research binaries and captures: access and routing traces, KLD logit records, SiLU table captures
- all other server logs
- private serving configuration and credentials

`evidence/ds41/index.json` lists every file left out of a window, with its reason. Paths in the reports and logs refer to the author's machine.

See [STATUS.md](STATUS.md) for what is and is not validated.
