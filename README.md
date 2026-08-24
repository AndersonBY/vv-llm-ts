# vv-llm-ts

`vv-llm-ts` is a small Node 20+ ESM client for the common OpenAI-compatible HTTP
surface shared by the Python and Rust `vv-llm` implementations. It uses the
native `fetch` API and has no runtime dependencies.

## Supported MVP surface

- Promise-based non-streaming chat completions and `AsyncIterable` SSE chat
  completions (`client.chat.completions.create(...)`).
- Provider-neutral canonical `ChatRequest`/`ChatRequestOptions` with a pure
  adapter and typed `client.create(...)`/`client.createChatRequest(...)` entry
  points.
- OpenAI-compatible tool definitions and streamed/non-streamed tool calls.
- Text and image URL content parts shared with the Python/Rust core; the
  `input_audio` part is accepted as an OpenAI-compatible passthrough extension.
- Embeddings at `POST /embeddings` and normalized rerank responses at
  `POST /rerank`.
- `base_url`/`baseURL`, bearer API key, default and per-request headers/query,
  arbitrary JSON options, `extra_body`, provider extensions, timeout, and
  caller `AbortSignal`.
- Typed `ModelConfig`/`ModelCatalog` and a settings resolver covering
  `endpoints`, `backends`, `embedding_backends`, `rerank_backends`, string or
  object endpoint bindings, disabled entries, binding `model_id` overrides,
  endpoint headers, and transport metadata.
- Stable `VvLlmError` classification with status, request ID, retry hints, and
  `retry-after-ms` precedence matching the Python/Rust protocol fixture.
- The default `ModelCatalog` is generated from the pinned
  `vv-llm-contract` v1.0.1 catalog and exposes contract version, schema,
  fixture, catalog-revision, and artifact-hash metadata.

## Canonical request API

The canonical request keeps provider-neutral options nested. The client adapts
it to the flat OpenAI-compatible wire shape without mutating the request:

```ts
import { decodeChatRequest, VvLlmClient } from "vv-llm-ts";

const request = decodeChatRequest({
  model: "my-chat-model",
  messages: [{ role: "user", content: "Hello" }],
  options: { temperature: 0.2, stream: false },
  tools: [{
    name: "lookup_weather",
    description: "Look up current weather",
    parameters: { type: "object", properties: { city: { type: "string" } } },
  }],
});

const client = new VvLlmClient({
  baseURL: "http://127.0.0.1:8000/v1",
  apiKey: process.env.OPENAI_API_KEY,
});
const completionResult = await client.createChatRequest(request);
if (Symbol.asyncIterator in completionResult) throw new Error("expected a completion");
const completion = completionResult;
console.log(completion.choices[0]?.message.content);

const streamRequest = decodeChatRequest({
  model: "my-chat-model",
  messages: [{ role: "user", content: "Stream this answer" }],
  options: { stream: true },
});
const streamResult = await client.createChatRequest(streamRequest);
if (!(Symbol.asyncIterator in streamResult)) throw new Error("expected a stream");
const stream = streamResult;
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}
```

`client.createChatRequest(...)` is the explicitly named equivalent canonical
entry point. An `AsyncIterable` is single-consumption; use
`collectChatStream(stream)` instead when a normalized final completion is
preferred. Runtime-only `signal`/`timeout_ms` controls can be passed as the
second argument to either canonical entry point. `decodeChatRequest` performs
closed-object, content, tool, `tool_choice`, and `x_*` extension validation and
returns a deep-cloned request.

## Runnable examples

See [`examples/`](examples/README.md) for typed chat, streaming, tools,
multimodal, contract JSON, middleware, metadata, and fallback examples.

## OpenAI-compatible low-level API

The lower-level namespace keeps the familiar flat OpenAI-compatible shape:

```ts
import { VvLlmClient } from "vv-llm-ts";

const client = new VvLlmClient({
  baseURL: "http://127.0.0.1:8000/v1",
  apiKey: process.env.OPENAI_API_KEY,
});

const completion = await client.chat.completions.create({
  model: "my-chat-model",
  messages: [{
    role: "user",
    content: [
      { type: "text", text: "What is in this image?" },
      { type: "image_url", image_url: { url: "data:image/png;base64,..." } },
    ],
  }],
  tools: [{
    type: "function",
    function: {
      name: "lookup_weather",
      description: "Look up current weather",
      parameters: { type: "object", properties: { city: { type: "string" } } },
    },
  }],
});

const stream = await client.chat.completions.create({
  model: "my-chat-model",
  messages: [{ role: "user", content: "Stream this answer" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}
```

The methods are Promise based because Node's safe native HTTP API is
asynchronous. “Synchronous completion” here means a non-streaming completion;
the stream variant is an async iterator.

## Settings

```ts
import { createChatClientFromSettings } from "vv-llm-ts";

const { client, resolved } = createChatClientFromSettings(settings, {
  backend: "openai",
  model: "my-model",
});
console.log(resolved.endpoint.endpoint_type, resolved.model_id);
const answer = await client.completeChat({
  model: resolved.model_id,
  messages: [{ role: "user", content: "Hello" }],
});
```

## Deliberate boundaries

This MVP sends the common OpenAI-compatible wire protocol only. It does not
claim native Anthropic Messages, Bedrock Converse, Vertex authentication,
Azure deployment URL rewriting, provider-specific image resizing/tokenizers,
or provider SDK behavior. The base `VvLlmClient` does not implicitly retry or
fallback; the explicit execution layer provides `RetryPolicy`,
`MiddlewareChatClient`, `ProviderRegistry`, and capability-aware ordered
`FallbackChatClient`. Gateways that need provider-specific wire options can use
`extra_body`, headers, and query options, but the endpoint must accept the
resulting OpenAI-compatible shape. Rerank is supported only for
the normalized JSON shape (`results[].index` and `results[].relevance_score`)
at `/rerank`; custom response mapping is intentionally deferred.

## Capability matrix

| Surface | Common OpenAI-compatible | Provider-native claim |
| --- | --- | --- |
| `VvLlmClient` chat/stream/tools/multimodal | Supported | Not claimed |
| Embeddings and normalized `/rerank` | Supported | Not claimed |
| Timeout, `AbortSignal`, classified errors | Supported | Not provider-specific |
| `RetryPolicy` and middleware | Explicit opt-in | Not implicit |
| Ordered capability-aware fallback | Explicit `FallbackChatClient` | Not implicit |
| Anthropic Messages, Bedrock Converse, Vertex auth, Azure URL rewriting | No | Deferred |

The language-neutral contract is maintained in the independent
`vv-llm-contract` repository. Runtime types live in `src/types.ts`.

## Contract consumption

The checked-in `contract/v1.0.1/` tree is a byte-for-byte vendor snapshot of
the canonical release: `consumer-lock.v1.json`, `manifest.json`,
`checksums.sha256`, the catalog, the v2 OpenAI-compatible fixture, the
independent retry fixture, and ten schemas.
`scripts/contract-sync.mjs` uses only Node standard-library APIs. Sync requires
an explicit source (`--source` or `VV_LLM_CONTRACT_SOURCE`), for example
`npm run contract:sync -- --source <contract-source>`; it refreshes the
snapshot and regenerates `src/generated/contract-catalog.ts`. `npm run contract:check`
validates the lock, manifest/checksum hashes, every artifact SHA-256, and the
generated catalog. The consumer lock itself is also pinned to the exported
`CONTRACT_CONSUMER_LOCK_SHA256`, so a lock-plus-artifacts drift cannot silently
bootstrap a new contract. Without an explicit `--source` (or
`VV_LLM_CONTRACT_SOURCE`), `--check` uses only the vendored lock and does not
need a source tree or network access; when a source is supplied, it validates
and compares both trees.

The npm package includes the generated runtime catalog plus the lock, manifest,
and checksum metadata. Raw schemas and conformance fixtures remain in the
source tree for tests and are intentionally excluded from the npm tarball.

## Opt-in live smoke

Unit tests and CI never call a provider. A separate script performs one real
non-streaming completion and one streaming completion only when explicitly
enabled:

```text
npm run build
$env:VV_LLM_RUN_LIVE_TESTS = "1"
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1" # optional
$env:VV_LLM_MODEL = "your-model"
npm run live:test
```

Alternatively set `VV_LLM_SETTINGS_JSON` to a JSON file path;
for settings also set `VV_LLM_BACKEND` and `VV_LLM_MODEL`. The output only
contains model name, content/reasoning/tool-call presence flags, and usage
counts. API keys and prompts are never printed. Without
`VV_LLM_RUN_LIVE_TESTS=1`, the command exits without making a request.

## Quality gates

```text
npm install
npm run contract:check
npm run lint            # contract check + strict TypeScript gate
npm test                # native node:test with mocked HTTP and SSE
npm run build
npm run examples:check  # canonical examples typecheck
npm run pack:check      # package contents smoke check
```

Offline tests never call a real API. They inject fetch or scripted clients and
exercise JSON, SSE, tool calls, multimodal request bodies, embeddings, rerank,
settings resolution, retry headers, middleware, metadata, capability skips,
fallback boundaries, errors, timeout, cancellation, and contract-lock tamper
rejection.
