GLM TP4 prepare coordinator — NEW bounded truthful-lookahead derivative

This is a separate source-only derivative of the tested frozen zero-lookahead
coordinator. It still links NO Core and performs no Engine/model/GPU/collective
work. Parent owns compilation and runtime. Original source and all prior test
receipts remain unchanged. No public capability or production profile is enabled.

Only admission arithmetic changes: prefill may retain truthful staged n_ahead
in [0,128], while decode still requires n_tok1 and n_ahead0. Pass::operator==
already compares every field, including n_ahead, so rank mismatch remains fatal
before any owner callback. Existing model/source/epoch/step/phase/draft/spec,
row/count/sequence and exact canonical-ID checks remain intact. No extra input,
cache union, GU grant, model row or mutable token history is derived from these
staged tokens. This is not real issue-ahead/prefetch execution support.

Why this is narrow and truthful: actual BatchBuilder stages following prompt
tokens after n_tok for a middle prefill chunk (sched/batch.cpp1048-1059) and keeps
the count in RadBatch.n_ahead. The only architecture reader of n_ahead in this
pinned tree is PleLayer::ahead_rows; actual GLM main has no PLE and uses n_tok
for embedding/MediaRows/attention/MoE. A FUTURE actual integration must seal that
no-PLE main graph, validate/copy the actual integer count without truncation,
match all four batches, and EXPLICITLY disable Engine issue_ahead (which remains
enabled even n_spec0 in the old Engine). It must not rewrite BatchBuilder fields
to satisfy this specification. No Core or batch semantics are changed here.

Existing process-fatal scope is preserved. Actual experimental negative gates
must notify then invoke fixed std::_Exit(88), before diagnostics, waits or C++
destructors. Tests inspect statuses only in this HOST tool. This neither proves
GPU quiescence nor graceful collective cancellation, safe device frees, driver
recovery or bounded termination of a hardware hang. Coordinator users must join
threads before destruction. Live cache/source/landing leases remain unrelated
to the abort decision and must never be treated as safely retired by it.

Original nine cases remain, except their old n_ahead1 refusal now uses129, the
new actual upper-bound violation. Two further initial refusals cover -1 and a
positive decode lookahead. New case10 executes full42-layer prefill passes with
lookahead1 and128 followed by true decode0; current executed row counts remain3,
3 and1, with resident-only rank2 voting and504 HOST GU counters. New case11 uses
rank3 lookahead127 vs admitted128 after three peers enter: every waiter must
return the same error with zero owner callbacks and zero GU grants. Authored
summary is11 cases/27 refusal groups, original two-pass336 GU counters plus504
new lookahead counters; actual parent runtime must establish these counts.
Bounds remain fixed4*128*8 IDs=16384B, sizeof(Coordinator)<24576; no new state or
hot dynamic allocations. HOST test stacks/stdlib synchronization/RSS remain
outside this logical object quote. It is a CPU scheduling specification only.

Manual parent build/run:
  cmake -S HERE -B ABS_NEW_BUILD -DCMAKE_BUILD_TYPE=Release
  cmake --build ABS_NEW_BUILD --target tp4_cache_prepare_lookahead_check
  ABS_NEW_BUILD/tp4_cache_prepare_lookahead_check
No CTest, automatic execution, Core build, deployment, service or GPU target.
verify_source_pins.py performs stdlib hash and exact arithmetic-diff checks only.
