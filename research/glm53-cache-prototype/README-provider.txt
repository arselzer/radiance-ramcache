Bounded immutable-file provider and CPU route replay
2026-10-25; standalone source prototype, no production integration/deployment.

file_provider.h/.cpp adapt the agreed Provider/RecordRef API to real local POSIX
pread. Identity is a TRUSTED OUT-OF-BAND manifest, not a new self-describing model
container or a checkpoint packer. The exact source/model/layout/schema, TP world/
rank and per-class record lengths must match every manifest key. Every requested
RecordRef must equal the registered immutable entry, including offset/length/hash.
All offset+length arithmetic is checked against a regular file's fstat extent and
off_t. Record extents cannot overlap; duplicate keys/classes with wrong lengths
are refused. Files open read-only with O_CLOEXEC/O_NOFOLLOW/O_NONBLOCK (the last
avoids blocking on a FIFO before rejecting non-regular files). Replacing the path
after open does not replace the retained descriptor/inode.

Reads retry pread EINTR, advance across short positive reads, fail on early EOF or
other syscall error, return exact bytes_read, and compare FNV1a64 over ALL record
bytes. FNV1a64 is NONCRYPTOGRAPHIC corruption detection, not authentication.
Before/after fstat compares device/inode/size/mtime/ctime. This catches ordinary
file mutation/truncation; it does not replace the caller's immutable/no-concurrent-
writer contract or defend against an adversarial writer restoring stamps. A
partial/error read may alter the caller-owned unpublished fill buffer; the cache
must not publish it unless read returns ok with exact length/checksum. No extra
record-sized provider buffer is allocated, nor any asynchronous I/O/pinning/DMA.
Stop requests are checked before I/O, between reads and after completion. A stop
does not forcibly cancel a currently blocked filesystem syscall.

Configured hard ceilings: regular file64MiB, record16MiB, manifest65536 records,
identity string1024B, total manifest identity bytes16MiB. Defaults are smaller:
file4MiB, record16KiB,16384 records,256B strings,total identity bytes4MiB. Metadata
counts/string lengths are bounded before copies. File/manifest publication is
transactional: a failed open does not replace an existing output provider.

Bounded trace/catalog loaders
----------------------------
load_rank0_catalog reads only the explicitly named local metadata file. It checks
schema/source pin/rank0, exactly43 distinct layers3..45 with288 experts, main
stride3342336B and MTP stride6299648B and their distinct class names. These costs
come from the parent's actual Builder/Planner selected-plane allocation probe:
 GLM53_CACHE_RECORDS_EXECUTED_RANK0.json
 source revision b5688f25491202978c19c4d036eef579f61bbe07.
This is not payload, quantization, serving-row or full-checkpoint verification.

load_route_trace accepts these explicit formats, preserving ALL duplicate route
IDs (the cache deduplicates records, not route contributions):
 * CSV header step,phase,layer,experts; e.g 0,decode,3,1;2;1. Strict unquoted CSV;
   semicolons separate IDs, no spaces/trailing separator, CRLF accepted.
 * JSON {"schema":"glm53-route-unions-v1","synthetic":true,"events":[...]},
   JSON array or JSONL objects. Each event has nonnegative integer step, phase
   decode/prefill, catalog layer, experts:[1,2,1] or experts:[[1,2],[1]].
 * Empty per-event unions are allowed; the trace itself must contain an event.
Steps must be nondecreasing. Integers must be exact (no booleans/floats), IDs must
be in catalog range; malformed late events refuse the entire file before output
mutation. Caps: input1MiB,10000events,4096raw IDs/event,64KiB/line,JSON depth16;
catalog input1MiB/depth8. Caller may lower these caps. No recursive unbounded JSON
parse precedes the lexical depth check. A caller-supplied trace is not assumed to
be measured GLM traffic; provenance remains explicitly unverified.

trace_replay
------------
Example command AFTER the parent builds the standalone target:
 ./trace_replay --catalog ../GLM53_CACHE_RECORDS_EXECUTED_RANK0.json \
   --hot-bytes 2048 --cold-bytes 4096 [--trace routes.csv] [--retain-previous]

With no --trace it constructs a labelled synthetic locality/shift/scan demo. Every
run writes deterministic128B(main)/256B(MTP) SYNTHETIC bytes to a unique temporary
file, reads those bytes through the actual provider/cache, independently checks
every returned byte, then removes the file. Max fixture file4MiB; total configured
synthetic hot+cold backing <=8MiB. These are exact record lengths and charges,
NOT virtual/logical GB charges behind tiny buffers. Class identity carries an
@synthetic suffix, and model/source/layout IDs explicitly say synthetic.

JSON output keeps two scopes separate:
  synthetic: actual charged hot/cold peaks and budgets, hit/miss bytes, real
  pread/cache I/O bytes/calls, bypasses and refusals;
  actual_catalog_costs_metadata_only: record-demand/hit/miss/conditional-read
  arithmetic using actual rank0 strides on these SYNTHETIC cache decisions.
Conditional requested-read bytes include all attempted provider reads, even on
a refused fill; these are full-record request sizes, not physical failed-I/O
bytes. Hit/miss weighted bytes cover successful events only. Union counts sum
per event, because a trace may contain multiple events for one step/layer.
The second scope is NOT an actual-stride cache run, actual RAM peak/fit proof,
GLM locality/rate estimate, speed benchmark or model execution. Budgets are never
silently rescaled. Optional --retain-previous holds the preceding batch lease
through the next attempted acquisition (including a refusal), then releases it,
to expose live-cold hits and transient pressure.
Every union is acquired in full or refused; no experts are dropped and no expert
kernels are executed. Refused events explicitly report zero experts executed.
No GPU/static VRAM tier, RDMA, transfer visibility, all-rank rendezvous, routed
kernel/table binding, state rollback or Engine integration is demonstrated.

provider_trace_test.cpp
-----------------------
Nine standalone CPU test groups: independent bytes/FNV published vector, mixed
class/rank/identity/offset/overlap/ref forgery refusal, corrupt/truncated/symlink
files, cancelled/short destination refusal, actual cache late-checksum batch
atomicity, duplicate/live route-union read dedup, strict actual-stride catalog
validation, CSV/JSON/JSONL/cap/late-invalid parse atomicity, and real3342336B
record geometry with ZERO budgets refusing before provider I/O/payload allocation.
The syscall loop implements EINTR/short-positive-read handling; tests do not
claim deterministic forced EINTR or a race-free concurrent-truncate-mid-pread
execution. Truncation is tested at immutable identity/extent gates.

Build dependencies: C++20, existing expert_cache.cpp, POSIX libc and the existing
radiance-src/vendor include directory for nlohmann/json.hpp. No package install,
network, Torch/HIP, model file or server API. Parent owns CMake and all compilation/
execution. Child has written/reviewed source only; no tests/build/GPU were run.
