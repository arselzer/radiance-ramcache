# Status

The current source is engine `d243fba8`. This is a research checkpoint for DeepSeek-V4.1-Flash on four RX 9070 XT cards with CPU-computed experts. Its evidence runs through window `ab-20261030T093449Z`.

| Check | Result |
|---|---|
| Source snapshot | `radiance/` equals engine commit `d243fba8` file for file: 1,428 files, matched by SHA256 pins and by git blob ids and modes against the engine repository. `tools/verify_snapshot.py` passes. |
| Decode, 8 held-out prompts, one stream | In `ab-20261030T090127Z`, the recommended configuration ran 18.93 / 18.88 / 18.98 tok/s. Without the futex wake it ran 18.86 / 18.84 / 18.93, and without that window's five new switches 17.94 / 17.90 / 17.86. All six switches of `ab-20261030T081259Z` together ran 18.30 / 18.57 / 18.74 tok/s. The base engine ran 5.03 tok/s in `ab-20261028T223812Z`. |
| Replies against the base engine | Byte-identical to `ab-20261028T223812Z/base.json`: 88 of 88 in `ab-20261030T090127Z`, 211 of 211 in `ab-20261030T081259Z`, and 48 of 48 in `ab-20261029T153750Z` (up to 8 concurrent requests). Checked with `tools/ds41_reply_check.py`. Every comparison table in the seven headline windows reads `same`, and none reads `DIFF`. |
| CPU rows against the card's | `POLICY=verify` arms found 0 differences: 31,670 rows in `ab-20261028T233549Z`, 34,618 rows of the many-row kernel in `ab-20261029T200958Z`, and 29,629 rows with the read-ahead finish in `ab-20261030T081259Z`. |
| Card checks | These passed on each window's own `libr4d.so`, before its arms ran: packed VRAM residents (`gemv_vp`), bf16 routers (`gemm_nt_f32bf16`), the DSpark draft GEMV, and the tall-pass sweeps. The logs are in the window directories. |
| Prefill exactness | Consumer skip: KV-group and last-row-logit hashes match with and without the skip, in 5/5 and 1/1 prompts in `ab-20261029T200958Z` and 5/5 with the prefix cache on in `ab-20261029T220909Z`. 32-token passes are byte-identical to 16-token passes over 1,512 positions in `ab-20261029T200958Z`. |
| Prefill, 3,086-token prompt | 63.1 s with the skip and 16-row CPU jobs, against 99.3 s without, in `ab-20261029T220909Z`. Plain arms took 99.2-99.5 s in `ab-20261029T200958Z`. Faster prefill needs the large-chunk mode or expert compute on the cards, and neither is in this snapshot. |
| Prefix cache | Long-document follow-up turns: 97.6-102.3 s cold, 0.42-2.75 s cached. Questions on one 1,211-token text: 38 s cold, 0.69-0.84 s at a 16-token checkpoint interval. Replies match the cold arm (`ab-20261029T232708Z`). With the consumer skip on, a turn that branches inside an earlier prompt loses its hit, because no checkpoint is written below the cut. |
| Concurrency | 4 requests: 20.0 / 20.2 tok/s aggregate decode. 8 requests: 20.6 tok/s. Aggregate decode is decode rows over decode-step time, from each arm's `/stats`. Replies match running alone (`ab-20261029T153750Z`). This used the configuration of 29 Oct, before the later decode switches. |
| Stability | 120 of 120 replies identical over 15 rounds on one server, with server RSS flat from minute 8 (`ab-20261029T062836Z`, configuration of that time). |
| DSpark speculative decoding | Experimental and off by default. On the best settings it was a net loss or a tie: depth 1 gave +1% and chat -1.8% (`ab-20261029T220909Z`); a 2-row short block gave 16.80 against 16.78 tok/s without the draft bank (`ab-20261030T070039Z`). `ab-20261030T093449Z` captured draft routing on a build of the unmerged VRAM hot-set branch. |
| Exact tall prefill passes (32 tokens) | Byte-identical, but slower alone (+9% time to first token at 3,086 tokens). Off. |
| Not in this snapshot | Merged after `d243fba8` and not yet measured: complete checkpoints under the consumer skip. Still on unmerged branches: large-chunk prefill, end-of-request prefix snapshots, and the DSpark VRAM hot set and acceptance gate. Exact prefill at 200+ tok/s is a design only (`docs/ds41/PREFILL-200-DESIGN.md`). All are unvalidated. |
| Builds | Not rebuilt for this export. Each window ran a build of the engine branch of its time; `docs/ds41/NIGHT-LOG.md` names most commits. `ab-20261030T081259Z` and `ab-20261030T090127Z` ran the same radiance binary (the run snapshots' SHA256 match), built from `c10ba3a0`, with the same `libr4d.so`. `c10ba3a0` differs from `d243fba8` only in five README lines. |
| Other prompts and models, long-context decode, unattended serving | Unvalidated. Output quality is established only as byte identity with the base engine on the measured prompts. |

No deployment, driver change or hardware change is part of this export.

The windows ran on a box shared with a production service (QFN). Each window logs its stop and its restoration, with a test completion.
