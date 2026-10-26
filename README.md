# Radiance expert RAM cache

Work in progress on an expert RAM-cache tier for Radiance, focused on GLM 5.3 Flash and models whose experts exceed system RAM.

This checkpoint selects Core V7, Engine fixture V9 and Components V13. Actual startup and one model step passed on four RX 9070 XT GPUs through the complete-store RAM-cache path: one raw input token produced one sampled token, `154822` (`[gMASK]`). The same QFN container was restored and passed an independent tiny inference check. Matching HOST Release/ASan/TSan, full-data prewalk and thirteen portable HOST programs also passed. This does not establish chat readiness, subsequent decode, quality or performance.

Model and logit dimensions stay at 154,880 rows. The private GLM profile admits the actual 154,856 tokenizer IDs and masks the 24 undefined tail rows through the existing sampler mask operation. Stock/default paths retain a zero opt-in value. The repair changes no model arithmetic, kernel schema or public ABI.

The cache keeps a bounded set of expert records in RAM and reads cold records from disk. A RAM hit avoids disk I/O and still uploads the expert to its GPU consumer. Persistent GPU expert caching, prefill overlap and cache-admission policies are being investigated; no performance change is claimed here.

- `radiance/`: exactly 1,535 pinned Core V7 source files.
- `research/`: immutable fixtures, source/cache implementations and portable HOST checks.
- `evidence/`: selected original receipts/logs with each run's scope retained.
- `docs/`: designs, independent reviews, source overlays and tested manifests.

The source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`. Upstream licensing and notices remain. Model weights, native-store payloads, tokenizer payloads, serving configuration, API keys and compiled research libraries are excluded.

On Linux, use Python 3.10+, CMake 3.24+ and a C++20 compiler:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The portable entrypoint now builds thirteen HOST checkers, adding the actual sampler staging/mask tests to the retained twelve. All thirteen passed Release execution against the exact source-identical existing Core V7 at tested manifest `8b966d37` (3,212 files); this final export adds documentation/evidence only. The default builds without executing; `--run-host-checks` requests CPU execution. It requires no model download, GPU or API server. The vocabulary binary's separate actual-model CPU mode remains manual. See [the portable guide](research/portable/README.txt).

The validated private manual profile uses one prompt token and one output token, maxTok1/seq1/spec0, exact wire, context512 and no fixed resident experts. Normal prompts, 262K, MTP, quality, throughput and deployment are not validated. The archived Runner V5 requires the reviewed Core V7/fixture V9 pair, focused vocabulary evidence, three HOST modes and a fresh Core V7 prewalk/HOME. No maintenance or GPU action is invoked by public tools.

The published Core V6 checkpoint remains preserved: its CPU/prewalk/portable passes stand, its GPU attempt failed before source admission, and the original QFN container was restored. Six archived V6 files reconstruct that exact Core over V7; applying the earlier two-file overlay then reconstructs V5. See [STATUS.md](STATUS.md) and [the candidate scope](docs/GLM53_VERIFIED_MAIN_V7_PUBLICATION_SCOPE.txt). The 840.326-second child duration includes full data verification, loading, startup, one model step and healthy teardown; it is not a token-rate benchmark. The first focused ASan executor attempt failed because LeakSanitizer cannot run under ptrace; its failed receipt is preserved alongside the passing unchanged host rerun.
