RunnerV3: explicit reviewed Core/fixture identity; no maintenance behavior change

verify_build now requires explicit fixture_revision/fixture_directory for the
reviewed Corev6/fixturev7 pair. The supplied directory must be the canonical
absolute direct research-root child with its exact revision name, not a symlink
or aliased path. Corev6 with fixturev6, missing fields, or future unreviewed pairs
refuses BEFORE any test-window preparation/stop. Legacy fallback is limited to
Core<=v5; explicit legacy fields must match the same Core revision.

Fixture CORE_SOURCE_PINS must equal the complete build-receipt Core map; its
SOURCE_RECEIPT prototype inventory plus receipt itself must equal the build's
full fixture map. Both actual file sets are closed (no extras/symlinks), all
existing source/artifact SHA and archive/header source-home/manual/module hashes
remain checked. No old private-header/archive substitution is admitted.

ALL Docker/stop/restore/physical resource/signal/process-group/health/preparation/
window bodies are BYTE-EXACT RunnerV2. Only verify_build changed. Fresh V3 prepare
is required because existing source-receipt digest checks remain unchanged; V2
preparation/source/evidence are preserved. This source-only revision has not run
any compiler/tokenizer/model/Docker/maintenance action; parent owns all execution.
Current CoreV6 fixturev7 source pins83d57401; previous notes below are historical.

Parent-owned single maintenance test runner

This package is authored and reviewed source. The author has not imported or run
runner.py, its commands, a tokenizer, compiler, model, GPU or Docker operation.
Only Python AST parsing and content hashes are used to freeze this package.
The parent owns every execution below. Nothing runs on import. There is no
service deployment, image pull/recreate, configuration edit, driver reset,
compose operation, source-proof generation or full store packing in this code.

The authorized window is ONE combined actual startup/transport + one-token
GLM child followed by restoration of the SAME existing radiance-flashnext
container. It is not two independent model runs or an inference benchmark.
The input is literal one-byte ASCII a, without a newline, chat template, BOS,
EOS or parse_special. Generation is exactly one token. Context512/maxTok1/
seq1/spec0/exact-wire/empty fixed residents are the closed research profile.
A result with one arbitrary sampled token establishes execution, not accuracy,
model quality, long prompts, normal chat, useful prefill speed or production fit.

Stage 1: preparation while the existing server stays running

Use actual successful matching V4 HOST/AQL Release build receipts. OUTPUT must
be a NEW immediate child of the research root. Absolute argument paths are
recommended; the receipt records resolved paths. The command below is a template,
not an executed result:

python3 glm53-manual-maintenance-runner-v2-20261025/runner.py prepare \
  --host-build-receipt ABS_HOST_RELEASE_BUILD_RECEIPT \
  --aql-build-receipt ABS_AQL_RELEASE_BUILD_RECEIPT \
  --output ABS_NEW_PREPARATION_DIRECTORY

The prepare command verifies reviewed source and all build artifact hashes,
then uses research-compiler.lock and the existing guarded-heavy launcher to
build only matching HOST rad-tokcheck. It creates prompt-a.txt as b'a', runs
rad-tokcheck --user --turns 0 --repeat 1, and requires actual tokenizer count1,
reference-identical comparison and exit0. Static metadata suggests token64;
rad-tokcheck prints the count rather than IDs, so the runtime receipt does not
claim that it observed ID64. Tokenizer/generation/template metadata are pinned.
No model payload is read by preparation. Building the tool may execute normal
CMake dependencies, whose source/archive hashes are rechecked afterward.

Preparation creates exactly these six real library copies in OUTPUT/exclusive-home:
  architectures/glm5next_cache.so (the matching AQL build, never a HOST/old module)
  kernels/libavx.so
  kernels/libr4d.so
  kernels/libref.so
  kernels/spechub.so
  quantizers/libquant.so
The five shipped plugins match GLM53_SHIPPED_PLUGIN_HASHES.json. The experimental
architecture is the only architecture. Libraries are chmod0444. There is no
colon-merged production HOME, tuning-data fabrication or serving-path write.

OUTPUT/home-receipt.json schema is glm53-manual-research-home-v1:
  passed: true
  home: ABS_OUTPUT/exclusive-home
  files: {the exact six relative filenames: SHA256}
  revision: matching Core revision, currently v4
  aql_build_receipt: ABS
  aql_build_receipt_sha256: SHA256
  source_architecture: ABS matching built experimental module
  shipped_pin_manifest_sha256: SHA256
  production_changes: false
  GPU_kernel_execution: false
OUTPUT/receipt.json schema glm53-manual-preparation-v1 additionally binds the
HOST/AQL build receipts, actual tokenizer executable/metadata/logs, exact prompt,
manual executable, source receipt and home receipt. The prewalk wrapper must
hash OUTPUT/home-receipt.json, not OUTPUT/receipt.json, for home_receipt_sha256.

Stage 2: all gates complete before stopping anything

The parent must first finish matching HOST Release/ASanUBSan/TSan validation,
and the actual AQL complete graph/Sampler/ordinary-loader prewalk. The runner
requires all three distinct HOST modes, completed nonempty command/artifact
sets, the digest-bound exact successful build for each mode, current source and
artifact hashes, actual complete assembly, actual full-data Reader/provider PASS
and a successful complete-prewalk gate. It does not replace these checks with
a source-only quote. No kernel/model execution is claimed by these gates.

The prewalk gate schema is glm53-verified-component-run-v1:
  passed: true
  mode: complete-prewalk
  component_build_receipt: ABS
  component_build_receipt_sha256: SHA256
  core_build_receipt: ABS matching AQL Core build
  core_build_receipt_sha256: SHA256
  result_file: ABS actual component JSON
  result_sha256: SHA256
  home_receipt_sha256: SHA256 of Stage1 home-receipt.json
  assembly_receipt_sha256: SHA256
  source_pass_receipt_sha256: SHA256
  GPU_kernel_execution: false
  production_changes: false
The runner also validates the raw component result (actual4graphs,
193536 omissions, actual data size/bundle ID, cap refusals, bounded staging,
no loader execute/SourceFactory/Session/Engine activation). The parent remains
responsible for producing this gate from real current results, never mocks.

Stage 3: explicit authorized maintenance window

python3 glm53-manual-maintenance-runner-v2-20261025/runner.py run-window \
  --preparation-receipt ABS_PREPARATION/receipt.json \
  --assembly-receipt ABS_FULL_ASSEMBLY_RECEIPT \
  --source-pass-receipt ABS_ACTUAL_COMPLETE_READER_PROVIDER_PASS \
  --prewalk-receipt ABS_PARENT_VALIDATED_PREWALK_GATE \
  --host-validation ABS_MATCHING_RELEASE_HOST_VALIDATION \
  --host-validation ABS_MATCHING_ASAN_HOST_VALIDATION \
  --host-validation ABS_MATCHING_TSAN_HOST_VALIDATION \
  --output ABS_NEW_WINDOW_EVIDENCE_DIRECTORY \
  --test-timeout 3600

The default Docker command is sudo -n /usr/bin/docker. The optional
--no-sudo-docker is for an already authorized direct Docker connection; the
runner does not grant group membership, request passwords or change privileges.

research-heavy-runtime.lock is held continuously through fresh status-ok/
in_flight0 samples, stop, sole child, restoration and stable health. The runner
captures only full container ID, image ID, name and filtered State scalars.
It NEVER requests raw inspect, Docker Config/Env/Cmd/HostConfig, labels, mounts
or health-log bodies. ID+image+name mean the same original container and image;
this is not independent model/config attestation or a check for concurrent
HostConfig updates. The parent supplies the trusted serving baseline and
supervisor review. Stop/start by captured full ID preserve that container;
there is no name-based restart or fallback to a newly created container.

Restoration is armed before issuing stop, since a timed-out/negative Docker
client may still have changed daemon state. Both an acknowledged stopped
boundary and the original exact ID/image/name are required before model launch.
Source/receipt/checker/HOME hashes are rechecked under the lock before stop and
again immediately before the sole four-GPU child. The original ID must remain
fully exited immediately before and periodically during that child. The child
uses ROCR_VISIBLE_DEVICES=0,1,2,3 and HIP_VISIBLE_DEVICES=0,1,2,3 explicitly,
CUDA_VISIBLE_DEVICES=-1, OMP1, nice19 and idle I/O priority. RADIANCE_* overrides
are removed; actual HOME is the explicit manual argument.

The combined child deadline defaults3600s (allowed60..7200), log cap64MiB,
heartbeat15s, container-stop guard5s. Heartbeats show only child PID, elapsed,
log path and /proc RSS/I/O counters. Only newly owned session/process groups
are terminated. Logged jobs keep their direct child unreaped via WNOWAIT until
owned descendant TERM/KILL cleanup, avoiding reuse of a reaped group leader ID.
SIGINT/TERM/HUP prevent further model/compiler launches; cleanup cannot be
interrupted by another such signal. The actual sole manual child command is:
  nice -n19 ionice -c3 MANUAL --one-token MODEL HOME BUNDLE PROOF 7279810219167007678 a
Known Core negative retirement paths exit88. An actual exit0 plus exact final
Engine JSON is required, including real ordinary job/upload/drain closure to
the prewalk. Startup and one-token are reported as one combined experiment.

Finally always attempts original-ID restoration when stop might have been
issued, including exit88, error, timeout or handled signal. It is bounded900s,
starts only from a confirmed exited boundary, requires a new StartedAt, and
requires40s consecutive actual healthy samples. A timed-out stop is treated
as ambiguous: recovery repeats/inspects stop until an exited boundary exists,
then starts even if a stop client failed. An early currently-running/healthy
sample cannot prove recovery while an outstanding stop might finish later.
The receipt records best-effort health separately and overall passed remains
false for ambiguous original stop acknowledgement. Evidence-write failures are
recorded best-effort and cannot prevent physical restore attempts. Child and
restore results are independent; the overall PASS requires both plus no signal.

This is not an independent external recovery guardian. SIGKILL/power loss,
Docker daemon loss, unkillable kernel/device work, fatal GPU/driver faults,
physical free failures and outside concurrent administrative mutation can defeat
recovery. Such cases must not be described as proven successful restoration.
The runner will not reset drivers, reboot, alter configuration or silently
substitute a different container. Ordinary caught failures have bounded owned
child cleanup and finally-based restoration. Parent should retain its outer
owned-process-group containment and operational observation during the window.

V2 resource-gate delta (V1 and its preparation remain unchanged)

V2 preparation produces its own source-bound receipt; use that receipt with V2.
Rebuilding the already existing HOST tokenizer target is normally a cheap no-op.
The sole runtime change is a passive physical-resource gate after acknowledged
original-container stop and all final source/hash verification, immediately
before the four-GPU child. It polls for at most60s using monotonic time and up to
31 bounded samples. Each sample first verifies that the original exact container
is still fully stopped. The only new physical inputs are the vendor/total/used
fields under /sys/class/drm/card*/device and /proc/meminfo MemAvailable.
No GPU-query subprocess, process/driver action, reset or memory operation exists.

Admission requires exactly four unique canonical AMD0x1002 PCI devices:
0000:c3:00.0,0000:83:00.0,0000:03:00.0,0000:06:00.0. Each must report exactly
17095983104 VRAM bytes total and at most67108864 bytes used (64MiB), while system
MemAvailable must be at least34359738368 bytes (32GiB). DRM card numbers are
recorded but identity uses canonical BDF/device directories. Integer byte fields
and Linux kB units are parsed without truncation. Extra/missing/duplicate AMD
devices, wrong totals, read/parse errors or unmet floors cannot launch the child.
Resource failure follows the existing finally restoration path. The gate records
bounded samples and exact thresholds; no claim is made about future outside
allocations racing after the final passive sample. Existing command cleanup
grace can extend wall time beyond a nominal polling/command deadline.
