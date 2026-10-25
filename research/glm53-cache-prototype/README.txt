Standalone GLM expert-record cache prototype (CPU only)
======================================================

This directory is isolated from radiance-src and all validated snapshots. It
implements opaque immutable CPU records and a bounded file-provider/trace seam.
It does not allocate/pin GPU memory, load a model, change serving, add a Radiance
flag, implement RDMA, or integrate with Core/WTAB/collectives. Parent owns builds
and sanitizer runs. Source checks alone are not an execution pass.

Public API is expert_cache.h, namespace glm53::cache:
 - Key: bounded nonempty source/model/layout/class identities and exact TP world,
   rank, layer and expert. Distinct classes never share a record by accident.
 - RecordRef: immutable Key, checked nonnegative64-bit offset, exact byte length,
   and FNV-1a64 checksum. Offsets+length must fit signed64-bit provider offsets.
 - Provider::read writes into a caller-sized span and reports exact bytes read.
   Short reads, errors, exceptions, cancellation and checksum mismatch refuse
   the acquisition. FNV is noncryptographic integrity, NOT authentication.
 - Cache::acquire deduplicates selected keys in first-occurrence order and
   returns one ready Lease per unique demand. Caller maps duplicate route jobs
   to those leases; no selected expert or route contribution is dropped.
 - Batch output is unchanged on every failure. No new batch becomes visible
   until every missing record has validated. Provider-owned files/source must
   remain immutable; the FileProvider separately seals its bounded manifest.

Two independently bounded byte classes
--------------------------------------
Every miss in the current demand union is cold-reserved BEFORE the first read.
The reservation includes all misses, including those that might later enter HOT.
If the entire union does not fit the remaining COLD budget, the call immediately
refuses before I/O. This first implementation does not stream a partial union or
wait forever for its own pinned records. Concurrent outstanding batches add to
the cold demand. A selected HOT hit is retained while the remaining fills run.

After all fills succeed, eligible records can move COLD -> HOT without copying
payload. HOT uses exact record-length charges and an intrusive LRU. A victim is
removed only when the cache owns its sole strong reference. Selected records and
all outstanding user leases are protected. If hot admission cannot fit, the
record still serves as a ready COLD lease. Admission changes retention, never
which experts execute. COLD weak-index entries deduplicate any live ready lease;
after the final cold lease dies the bytes are freed, and a later miss rereads.

The index and live-record caps are separate from byte budgets. max_records must
allow the hot cache PLUS reserved/live cold records; exhausting metadata refuses
before I/O, even if an unleased HOT victim might otherwise be replaceable. This
is an explicit conservative metadata-headroom requirement, not an optimal policy.
max_catalog_records retains the immutable offset/length/checksum of successfully
validated keys after payload eviction; changing an old key's ref refuses even
after its bytes were freed. Failed batches roll back newly staged catalog keys.
Limits also bound batch count and every identity string; absolute implementation
caps are65536 live/catalog records,4096 batch keys and1024 bytes/string.

Ownership and concurrency
-------------------------
Payload is immutable once ready. Lease::bytes returns only a const span. It is
valid as long as that lease (or a copied lease) remains alive; retaining a bare
span after releasing all leases is invalid. A shared ledger outlives Cache while
leases survive. Record destruction frees the vector payload before refunding its
byte/record charge; no record/ledger ownership cycle exists. Accounting observers
can verify outstanding charges after the cache object is destroyed.

One transaction mutex serializes acquisitions and provider fills. The metadata
mutex is released during I/O; providers may inspect stats without deadlocking.
Same-thread recursive cache mutation is explicitly refused, including calls into
another Cache. Provider callbacks MUST NOT initiate or synchronously wait on a
cache mutation in another thread: such callback-induced cross-thread cycles are
outside the contract. This is a simple serialized loader baseline, not a scalable
parallel I/O engine or a throughput promise. Concurrent overlapping requests
share a ready record while an owner/hot cache keeps it alive; this is not a
promise to retain an otherwise unowned cold record indefinitely for late waiters.

Cancellation is checked before and after waiting for the transaction mutex, at
each fill boundary and before commit. It does not forcibly interrupt a blocking
provider call or mutex wait; Provider receives the token for cooperative I/O.
A cancelled call returns no partial batch and releases its unpublished records.
Cache destruction must not race calls on that C++ object. Leases may outlive it.

Tests and scope
---------------
expert_cache_test.cpp has14 bounded CPU cases covering duplicate demand unions,
mixed128/256-byte classes, whole-union budget refusal before I/O, late fill failure
and retry, exact-length/checksum/provider-exception cleanup, pinned HOT admission
bypass, cold dedup, LRU victim choice, leases/ledger after Cache destruction,
immutable identity after eviction, metadata/offset caps, cancellation, callback
reentry, and concurrent same-key fill sharing. Thread construction failure has
an explicit release guard before join; no fixed-party barrier is used.

The actual rank0 catalog is GLM53_CACHE_RECORDS_EXECUTED_RANK0.json in the parent
directory, SHA2564d7a9ecf1855585a406a035422749e0f65e7ab7fbe46d579155555cdeed724e5.
Its main stride3342336 B and MTP stride6299648 B are used only as metadata in a
pre-allocation refusal test. No real model record bytes/checksums were generated
or authenticated. Executed trace fixtures remain tiny synthetic128/256-byte
records; any accompanying actual-GLM traffic arithmetic is separately labelled.
See README-provider.txt for bounded manifest/file-provider/replay contracts.

Budgets charge logical record bytes and record counts until buffers are freed,
including outstanding leases. They do NOT claim allocator RSS, resident/locked
pages, vector/string/map-node/control-block capacity, thread stacks or GPU-pinned
accounting. Auxiliary metadata is bounded by explicit caps but is not part of
the HOT/COLD byte totals. A real mapped/pinned provider will need its own physical
allocation/accounting and visibility/fence proof before GPU integration.

Suggested parent-only build/run (not executed by source authors):
 cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
 cmake --build build -j2
 ctest --test-dir build --output-on-failure
Then ASan/UBSan and a separate TSan build, especially the concurrent/lifetime
cases. No toolchain flags, compiler or device invocation ran while writing this.

Next Radiance gates remain unimplemented: current-step route handoff, selected
sparse WTAB readiness, exact runtime layout packing, producer/last-reader fences,
GPU visibility, all-rank I/O/error readiness BEFORE dependent collectives, and
failure/drain/state ownership. Successful standalone cache tests cannot establish
safe multi-rank serving, model capacity, true routing hit rate or inference speed.
