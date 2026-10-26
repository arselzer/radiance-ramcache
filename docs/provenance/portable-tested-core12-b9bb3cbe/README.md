# Radiance expert RAM cache

Unpublished Core12 / Fixture17 validation candidate. The source now contains the
four-token prefill/declaration repair and sixteen portable HOST checks. Private
Release/ASan/TSan checks and actual four-rank declaration, metadata and complete
source prewalk passed. A formatted-prompt GPU trial is running; its outcome and
fresh public sixteen-check build are pending. No useful-text result is claimed.

The following records the preceding published Core8 checkpoint. Its exact
overwritten files and manifest are preserved in `docs/provenance/core8-before-core12/`.

Work in progress on an expert RAM-cache tier for Radiance, focused on GLM 5.3 Flash and models whose experts exceed system RAM.

This checkpoint selects Core V8, Engine fixture V10 and Components V14. An isolated four-GPU test passed with one raw input token (`a`, ID64) and two sampled outputs, `[154822, 154822]` (`[gMASK]`, `[gMASK]`). The second output came from an actual recurrent decode using the first output as input. The same QFN container was restored and passed an independent four-token inference check. This establishes one bounded decode; normal prompts, chat, quality, sixteen-output GPU generation and performance remain unvalidated.

Matching full HOST Release/ASan/TSan suites, focused vocabulary/bounded-decode checks, actual metadata and complete-data prewalk passed. The public entrypoint now adds the bounded-decode HOST checker as its fourteenth program. A fresh build from this published source and execution of all fourteen public programs **passed**. The tested manifest `b0fef123` and overwritten metadata are archived; the final delta adds evidence and documentation only. The preceding Core V7 public thirteen-program build/runtime remains preserved with its own tested manifest.

The cache keeps a bounded set of expert records in RAM and reads cold records from disk. A RAM hit avoids a payload reread and still verifies/uploads the selected expert to a transient GPU consumer. This checkpoint adds no persistent GPU expert cache, checksum shortcut, changed kernel or cache policy. The actual trial used the unchanged defaults: 64 MiB HOT per rank and 64 KiB read chunks.

- `radiance/`: exactly 1,535 pinned Core V8 source files.
- `research/`: immutable fixtures, source/cache implementations and portable HOST checks.
- `evidence/`: original receipts/logs, retaining the scope of each run.
- `docs/`: designs, reviews, exact copy maps and reconstructible source history.

The source is based on Radiance commit `5a9ce4912bb4ae5315e4a177f9e7d5b10d035766`. Upstream licensing and notices remain. Model/native-store/tokenizer payloads, compiled research binaries, private serving configuration and credentials are excluded.

On Linux, use Python 3.10+, CMake 3.24+ and a C++20 compiler:

```sh
python3 tools/verify_snapshot.py
python3 tools/build_host.py --mode release --run-host-checks
```

The default builds without executing. `--run-host-checks` explicitly requests fourteen CPU programs and requires no model download, GPU or API server. The vocabulary checker's actual-model CPU mode, archived manual Engine modes and Runner V9 are never invoked by public tools. Matching source/private ABI/compiler/cache safeguards remain; see [the portable guide](research/portable/README.txt).

The actual tested profile is one prompt token, two outputs, maxTok1/seq1/spec0, exact wire, context512, BF16 KV and no fixed resident experts. The model/logits retain 154,880 rows; the actual 154,856-token prefix is admitted and the 24 undefined tail rows remain masked through the existing sampler path. Stock defaults keep the opt-in disabled. Code and HOST checks also bound generation through sixteen outputs and explicit cache-budget options; those extensions have no additional GPU validation here.

Startup took 832.387 seconds, including eager complete-source verification and loading. The two model/sampler/collect frames took 5.102 and 4.596 seconds in this trial. These are recorded single-run durations, with no benchmark or speedup claim. The generated `[gMASK]` IDs are valid tokens, not useful-text or model-quality evidence.

The preceding Core V7 checkpoint (`d04b562`, manifest `ea0b8f81`) and all earlier raw archives/evidence remain intact. A four-file overlay reconstructs exact Core V7 over V8; inherited overlays then reconstruct V6 and V5. See [STATUS.md](STATUS.md), [the current scope](docs/GLM53_VERIFIED_MAIN_V8_PUBLICATION_SCOPE.txt) and [the current evidence](evidence/glm53-main-verified-v8/VALIDATION_SUMMARY.json). No serving deployment or configuration change is part of this source export.
