# Growing physical KV: pre-admission engine seam

Source only, default serving path unchanged. The earlier transaction and page-source commits are
prerequisites. This branch adds an actual Engine/Scheduler quiescence hook and checked drains;
it does **not** expose an enable flag, defer startup backing, or claim an active growing cache.
The remaining publication and admission changes below are required before a safe public mode.

`Scheduler::step_quiescent` holds the existing ownership mutex across a checked callback and all
of `step_locked`: prefix lookup/adoption, state/window seeding, COW/page allocation and the final
BatchBuilder upload. A failed callback returns its status and no batch before any of those
actions. Ordinary `step()` acquires the same mutex and runs the same extracted body. The callback
must never call a public Scheduler method because that would recursively acquire its mutex.

`Engine::kv_quiescent_plan` is a private, currently unused caller of that seam. Its initial scope
is native DS41/TP4, one admitted sequence, fixed deterministic expert placement, no pipelining,
speculation/drafter passes, generator/embedding/media encoder, host-prefix tiers, KV streaming,
general KV loan, P1 or arena lending, or generic PrefillStager. Retained VRAM prefix refs and the
raw state/window snapshot path are compatible with this seam; their physical holder publication
still needs the controller below. The regular recommended configuration remains available with
the growing mode off.

The paired `RankBarrier` generation is checked before and under scheduler ownership. An even
non-aborted generation is before the scheduler releases ranks or after every rank CPU issuer
joined; because the sole step thread does not arrive at the next release during the hook,
workers cannot issue another pass. This check proves CPU parking, not GPU completion. Every rank
then checks that its FileStager CPU/miss pass ended, drains remaining RAM insert helpers, drains
both Ctx compute lanes and its host row-prefetch helper, retires mapped-source copy leases,
drains the mover transfer queue, and uses TierExec's new checked drain. Device selection returns
to rank0 on both success and failure. The backing callback runs only after all ranks succeed.

No drain helper needs the scheduler mutex to finish: FileStager `end_pass`/`hx_drain` joined CPU
expert/miss work before the CPU barrier; RAM insert workers use mover/RAM-cache locks; Ctx host
read-ahead workers only advise weight pages; transfer events execute from already submitted GPU
packets. Background immutable expert/RAM read-ahead never owns KV, and fixed expert residency plus
the KV-loan refusal prevents it from aliasing a KV granule. New work is not released while the
ownership lock is held. `prepare_release`, snapshot restore, `prefill_hash_stage`, media work and
`tier_tick` stay outside this callback, preventing recursive scheduler calls.

Prepared source cases verify paired CPU parking versus active/aborted phases, both Ctx lanes
draining while the context remains reusable, and a pre-plan device-failure refusal preserving
prefix refcounts/free lists with no lookup/adoption/allocation; the following successful plan
then adopts that same prefix. No test was executed.

## Why an enable flag is still absent

The existing manager publishes every carved page in its free heap at startup. Its `live_` mark
is a contiguous expert-loan boundary, not a physical backing mask. A correct growing controller
must first stage distinct per-rank backed-ID masks/free heaps and make **all** block allocation,
specific reservation, prefix adoption, COW, tier/follower restoration and free-count paths honor
them. No unmap can run while an old heap still offers an affected ID. State/window floors remain
fully backed, and every bound layer fragment of a candidate page must be mapped before that ID
is eligible. Prefix refs and queued/issued copy lifetimes must keep their pages mandatory.

Growth must run synchronously before allocation inside admission. Source audit confirms all
`ensure_from` calls in `Scheduler::step_locked` precede its final `BatchBuilder::build`, which is
where copies/uploads are enqueued. This gives a usable pre-upload interval under the new lock
seam. Its availability must be explicit in the controller so callbacks cannot be invoked later
from a path already queueing GPU work. `can_grow`/admission accounting must consider fundable
future backing, not just currently mapped free IDs, without handing out those IDs early.

Current scheduler retry loops treat every negative `ensure_from` as a capacity shortfall and may
evict prefixes or preempt. A failed physical mapping transaction needs a separate fatal/refusal
status latch that is checked before publishing the built batch; an unrepaired mapping must call
the engine error path after releasing scheduler ownership. It cannot be retried as ordinary
memory pressure, and a pending poisoned heap must never be used during those retry loops.

Startup must reserve the maximum fixed VA/strides while deferring full KV commit, publish no
unbacked IDs, and map the configured physical minimum plus window/raw-state floors before any
state clearing or scheduler request starts. The final physical maximum/headroom proof belongs
after other eager allocations and includes later step/queue allowances. It must charge rounded
tail granules honestly and retain the old funding until transaction commit. A smaller physical
maximum than the logical capacity also needs a correct `can_ever_fit` refusal, not an admission
livelock. Unsupported VMem, streaming and loan combinations require explicit refusal/fallback.

Only after those pieces are connected may `step_loop` disable ahead/pipelining and select this
hook. The smallest initial mode can keep expert residents fixed; physical KV release alone does
not expand the existing weights pool. A future joint expert/KV physical ledger remains separate.
