# ds41 A/B windows

Each directory is one GPU window run by `radiance/tools/ds41-host-experts/ab_window.sh` and named by its start time (UTC). A window:

1. waits for the box's production container (QFN) to be idle, then stops it;
2. runs each arm as its own radiance server on the four cards, with a cold RAM cache and the same requests in the same order;
3. compares every reply with the first arm that sent the same request;
4. always restores QFN and logs a test completion.

`docs/ds41/NIGHT-LOG.md` says what each window was for and what it found.

What a window directory holds:

| File | Content |
|---|---|
| `window.log` | The window's log: each arm's switches, per-request times, shutdown counters, the device checks run before the arms, and the comparison tables at the end (`same` / `DIFF` per request, `pdif` where an earlier differing reply was sent back) |
| `<arm>.json` | `ab_requests.py`'s result for one arm: every request's reply text, timings and the server's `/stats` (`<arm>.unmeasured.json`: an arm whose gates failed) |
| `<arm>.mem` | The box's available memory (MiB) every 5 s while the arm ran |
| other `*.log`, `kld-corpus.jsonl` | Device checks run in the window (chain check, SwiGLU proof, gemv_vp, gemm_nt_f32bf16, draft GEMV, tall-pass sweeps) and the KLD byte gate's corpus |
| `<arm>.server.log.xz` | The arm's server log, xz-compressed. Kept only for the windows listed below |

An arm's name is its policy followed by its parts, for example `all-t40-...-F8-...-Rb-Vp-Ss`. `ab_window.sh` and `radiance/tools/ds41-host-experts/README.md` list the parts. A few examples: `t40` is 40 CPU threads, `m98304` the RAM cache in MiB, `F8` FP8-stored dense weights, `Cs` the consumer skip, and `-r2` a repeat.

Server logs are kept for the windows behind the top-level README's numbers and the prefill exactness:

| Window | Why |
|---|---|
| `ab-20261028T223812Z` | Base engine decode, 5.03 tok/s |
| `ab-20261030T090127Z` | The recommended configuration's decode: 18.93 / 18.88 / 18.98 tok/s, with every reply the base engine's |
| `ab-20261030T081259Z` | The six switches tried before it: 18.30 / 18.57 / 18.74 tok/s together, and the read-ahead finish's 29,629 verified rows |
| `ab-20261029T220909Z` | 3,086-token prefill: 99.3 s without, and 63.1 s with, the consumer skip and 16-row CPU jobs |
| `ab-20261029T200958Z` | The consumer skip's exactness: per-rank prefill hashes and the 32-token byte gate |
| `ab-20261029T232708Z` | Prefix-cache follow-up turns |
| `ab-20261029T153750Z` | Four concurrent requests, ~20 tok/s aggregate decode |

The following are left out. `index.json` lists each file with its size and reason:

- all other server logs
- the KLD logit records (2 x 391 MB)
- the routing traces (129 MB)
- the per-rank access traces (10.6 MB)
- two SiLU table re-captures (0.5 MB)

Some windows failed, partly or wholly. They are kept as they are:

- `ab-20261028T220743Z` was cancelled while stopping QFN (`NIGHT-LOG.md`, 22:08).
- `ab-20261028T220855Z` found QFN not idle within 10 minutes and ran nothing.
- In `ab-20261029T031236Z`, both arms failed at startup: the window was launched from a shell that hid the cards (03:12).
- In `ab-20261029T082738Z`, one arm hit the memory guard and the rest could not start (08:29).
- In `ab-20261029T102740Z` and `ab-20261029T104356Z`, the control arm ran but the DSpark arms exited during startup (10:27).

`ab-20261029T010905Z` ran a card timing (`ptime.log`) and no arms.

`ab-20261030T093449Z` (window G) ran a build of the unmerged DSpark VRAM hot-set branch, which is not in `radiance/`.

Windows after `ab-20261030T093449Z` are not in this export. `NIGHT-LOG.md` already describes some of them, and `python3 tools/ds41_refresh.py --add-window <window>` adds them.

Paths in the logs refer to the author's machine.
