# program-set

## Batch-backed Strands models

`program_set.batch_transport.BatchTransport` persists opaque provider-specific
JSON payloads, envelope IDs, batch status, and normalized terminal results in
SQLite. It validates payload serializability and generic envelope IDs, not
provider request fields.
Normalized messages retain supported content and a Strands-compatible stop
reason; provider message metadata such as IDs, model names, and usage is
discarded. Anthropic `stop_sequence` remains distinct, while provider-only
reasons are mapped to the closest supported Strands reason.
`DeferredBatchModel` shares stable request identity, enqueue/reuse, pending
interrupts, terminal checks, and result-event replay; it never submits or polls
by itself.

`BatchBackedModel` maps the tested Strands text/function-tool flow to Anthropic
Messages and supports Anthropic threshold compaction. Enable it by configuring
`AnthropicBatchClient(client, betas=["compact-2026-01-12"])` and passing
`context_management={"edits": [{"type": "compact_20260112"}]}` in model config
or request parameters. Beta mode is explicit, not inferred from queued request
contents. `ProviderProfile` durably stores this non-secret adapter config in
SQLite, so restarts can rebuild the same adapter without inspecting request
payloads. A typical setup saves a profile once, then on each start opens the
transport without a client, loads the profile, builds the adapter, and binds it:

```python
profile = ProviderProfile(
    "agent-provider", "anthropic", {"betas": ["compact-2026-01-12"]}
)
transport = BatchTransport(None, "batches.sqlite")
transport.save_provider_profile(profile)  # initial setup; repeat only to update
transport.set_client(
    build_batch_client(profile, anthropic_client=application_anthropic_client)
)

# On restart, load the saved config before binding the injected SDK client:
transport = BatchTransport(None, "batches.sqlite")
profile = transport.get_provider_profile("agent-provider")
transport.set_client(
    build_batch_client(profile, anthropic_client=application_anthropic_client)
)
```

Profiles accept adapter configuration only (Anthropic `betas`; no OpenAI
options currently). Never put API keys or other credentials in a profile;
credentials belong to application-owned SDK client construction and are not
stored by the profile table or factory. Compaction blocks are carried opaquely
through Agent history and restored to their original Anthropic
content/signature fields. On-demand compaction is not supported.
`OpenAIBatchBackedModel` maps the supported text/function-tool subset to
Responses and carries reasoning items opaquely across tool calls. OpenAI's
separate JSONL adapter validates its own Responses request shape. There is no
universal request schema or cross-provider conversion layer. SDK clients are
injected; constructing adapters makes no network calls.

Callers own Strands invocation/advancement and the batch lifecycle: invoke an
Agent, Graph, or Swarm; persist its Strands snapshot as needed; flush queued
requests; poll their batches; then pass completed responses back on a later
invocation. `ready_batch_interrupt_responses(result.interrupts, transport)`
builds ordinary `interruptResponse` input blocks only for this package's
durably succeeded requests. It ignores foreign interrupts and ordinary pending
requests, and surfaces unknown, mismatched, and failed requests. A
`submit_unknown` request raises `SubmissionUncertainError` with its durable
submission/request IDs; the caller must establish non-acceptance before
explicitly requeueing it. The helper never invokes Strands, flushes, polls, or
retries. Only the deferred model's explicit source marker identifies its
interrupts; a matching name alone is not enough.

```python
from program_set.batch_interrupts import ready_batch_interrupt_responses

result = agent("do the work")
# Persist the Strands snapshot here if the application needs restart recovery.
batch = transport.flush()  # one caller-owned submission point
if batch is not None:
    if transport.poll(batch.batch_id).processing_status == "ended":
        responses = ready_batch_interrupt_responses(result.interrupts or [], transport)
        if responses:
            resumed = agent(responses)  # caller-owned advancement
# If the batch is not ended, the caller polls again later.
```

Strands' `SnapshotSessionManager` can persist Agent, Graph, and Swarm runtime
state. Rebuild the same definitions with stable model `agent_id`s after a
restart. Nested `Agent.as_tool()` children without their own session manager
are part of their parent's snapshot; a `preserve_context=True` child keeps its
own history and therefore needs its own session manager and stable session ID.
The batch transport persists provider requests/results, not Strands snapshots.

Raising `InterruptException` directly from `Model.stream` is an undocumented
Strands provider behavior. Mocked production snapshot/restore/tool-cycle tests
pin this path to Strands 1.61.1; upgrades require revalidation. TODO: determine
whether to formalize the deferred-inference contract through an upstream
Strands contribution.

Scope is limited to Anthropic text and function tools, and OpenAI Responses text,
function tools, and opaque reasoning replay. Images/audio and structured output
are unsupported. There is no worker, scheduler, splitting, or automatic retry;
ambiguous submissions stay `submit_unknown` until non-acceptance is confirmed.
OpenAI live use requires Batch Write (`api.batch.write`): [RBAC
permissions](https://developers.openai.com/api/docs/guides/rbac).
