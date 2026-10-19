# program-set

## Batch-backed Strands models

`program_set.batch_transport.BatchTransport` persists opaque provider-specific
JSON payloads, envelope IDs, batch status, and normalized terminal results in
SQLite. It validates payload serializability and generic envelope IDs, not
provider request fields.
`DeferredBatchModel` shares stable request identity, enqueue/reuse, pending
interrupts, terminal checks, and result-event replay; it never submits or polls
by itself.

`BatchBackedModel` maps the tested Strands text/function-tool flow to Anthropic
Messages. `OpenAIBatchBackedModel` maps the supported text/function-tool subset
to Responses and carries reasoning items opaquely across tool calls. OpenAI's
separate JSONL adapter validates its own Responses request shape. There is no
universal request schema or cross-provider conversion layer. SDK clients are
injected; constructing adapters makes no network calls.

Callers explicitly flush and poll batches, then re-enter the Agent. Raising
`InterruptException` directly from `Model.stream` is an undocumented Strands
provider behavior. The mocked production snapshot/restore/tool-cycle test pins
this path to Strands 1.61.1; upgrades require revalidation.

Scope is limited to Anthropic text and function tools, and OpenAI Responses text,
function tools, and opaque reasoning replay. Images/audio and structured output
are unsupported. There is no worker, scheduler, splitting, or automatic retry;
ambiguous submissions stay `submit_unknown` until non-acceptance is confirmed.
OpenAI live use requires Batch Write (`api.batch.write`): [RBAC
permissions](https://developers.openai.com/api/docs/guides/rbac).
