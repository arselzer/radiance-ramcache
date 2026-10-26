Runner9 — Core8/Fixture10, bounded confirmation of a known late stop

SOURCE ONLY. Parent owns every test/compiler/tokenizer/Docker/model/GPU action.
Runner6, Runner8 and the failed Core8 window/evidence remain unchanged. The
source author has not imported/run this runner or its tests. No auto-start,
service configuration, image replacement, driver action or new model profile.

The ONLY operational change is a helper replacing the original stop/inspect
pair in run_window. Normal Docker stop acknowledgement still requires the same
single fresh stopped() inspection. Stop grace30s/client deadline90s are unchanged.

Only this exact RuntimeError, for the captured FULL container ID, admits the
additional confirmation path (trailing LF allowed):
 Docker stop failed: Error response from daemon: cannot stop container: FULL_ID: tried to kill container, but did not receive an exit event
The original error is retained in the receipt. The helper sends NO second stop
or other signal. It invokes docker wait FULL_ID with timeout120s and accepts
only one decimal exit code0..255. It then requires TWO fresh stopped inspections,
separated by1s, each with the same full ID/image/name, original StartedAt, new
nonzero bounded Docker-UTC-shaped FinishedAt, and matching wait ExitCode. The
second terminal timestamp must equal the first. Malformed/empty/zero timestamp,
wrong generation/identity, running state, code mismatch, timeout, unrelated
error or cancellation refuses. No latency/cause explanation is inferred.

A successful late branch records stop_acknowledgement with stop_request_count1,
initial_ACKfalse, terminal_ACKtrue, wait code, BOTH selected safe-state samples,
kind late_terminal_ACK_after_known_missed_exit_event and the original error.
The count is the one stop API invocation attempt, not proof of signal delivery.
Normal acknowledgement records initial_ACKtrue and one fresh stopped sample.
A cancelled-before-stop helper records count0. Late wait/inspect/owned-child
cleanup deadlines are separately bounded;120s is the wait-client deadline only.

On ANY refusal the exception reaches the unchanged outer finally/restore path.
That preexisting recovery routine may repeat a stop to establish a safe exited
boundary when the original stop remains ambiguous. It is NOT called inside the
new confirmation helper and has not been weakened. Once a normal or explicit
late terminal ACK returns, old stop_confirmed=true behavior follows. Same-ID
restoration, new-start generation and40s healthy recovery remain mandatory.
External SIGKILL/power loss/daemon or driver failure guarantees are unchanged.

ALL existing Docker capture/inspect/identity/stopped/restore, resource, health,
signal, owned-process, log, lock and prepare leaves remain BYTE-EXACT Runner6.
The composite run_window changes ONLY its two-line stop/inspect pair to the
helper call. Reversing that single call yields exact Runner6 bytes. The sole
other old-function change is the explicit preparation-source bridge below.
Core8/Fixture10/Components14, all3full HOST proofs, exact focused Release3 proof,
complete source/data/layout prewalk, fresh idle and physical gates are unchanged.
No DeepSeek, R4 prefill or new cache/precision behavior is admitted here.

Existing preparation may be reused ONLY if bound either to this exact current
Runner9 source receipt/path OR the preserved absolute Runner6 receipt:
 glm53-manual-maintenance-runner-v6-20261026/SOURCE_RECEIPT.json
 SHA de397a3497b2ba3e381e1c3d71275da186992bb8e978a61394b57cc7c979b1b0
Its actual file and complete runner.py/README source closure are rehashed and
pinned in this receipt. There is no general ancestor allowance. The preparation
contains CPU tokenizer and exclusive HOME evidence; all matching Core8/HOME/
model/source/prewalk checks still run unchanged. No old receipt is rewritten.

Parent-only source/fake controls (NO actual Docker subprocess is used):
 python3 -B test_stop_confirmation.py --run-fake-docker-tests
The fake callable replaces docker_capture for the entire helper call, rejects
all unexpected commands, and uses a nonexistent executable prefix. Tests cover
normal ACK, late ACK, wait0/255/137, timeout, malformed/out-of-range output,
unrelated errors, first/second state/identity/generation/code drift, prior-nonzero
FinishedAt versus zero/malformed, cancellation and exact old-source recovery.
The author performed AST/hash checks only; these tests are authored, not claimed
executed. Parent must run/review them before using Runner9.

Parent run-window CLI and bounded choices remain exactly Runner6, including:
 --outputs2|16 --hot-mib64|2048 --host-payload-gib8|16
 --read-chunk-kib64|1024; baseline2/64/8/64
 --vocabulary-receipt validation-glm53-v8-vocabulary-release-20261026T000613Z/receipt.json
Use the SAME existing Core8 preparation, full HOST proof and complete-prewalk
paths after their mandatory hashes/gates pass; only a NEW output evidence dir.
The two choices changing outputs/HOT/chunk together are a combined experiment,
not isolated evidence of RAM-cache speed or output quality. Current baseline
still uses raw one-token a, context512/maxTok1/seq1/spec0/exact/emptyresidents.
The same named host budgets, buffered-record counters, valid-token prefix and
full per-output decode chain checks are untouched. Parent alone authorizes and
executes each maintenance window; freezing source is not permission to run.
