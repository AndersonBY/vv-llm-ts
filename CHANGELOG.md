# Changelog

## 0.2.3 - 2026-10-03

- Adopt contract 1.2.1, catalog revision 16 with the single public ID `qwen3.8-flash-next`;
  DashScope uses endpoint `model_id` to select `qwen3.8-flash`.

## 0.2.2 - 2026-09-30

- Adopt catalog revision 14 with OpenAI `gpt-6.1-sol`.

## 0.2.1 - 2026-09-29

- Adopt catalog revision 13 with Claude Opus 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, GPT-6 Sol, and GPT-6 Luna.

## 0.2.0 - 2026-09-29

- Adopt vv-llm-contract 1.2.0 and catalog revision 10 with per-model effort choices, compatibility aliases, and binding overrides.
- Add configurable effort validation, capability-aware fallback, and preservation of caller-supplied model catalogs.
- Preserve alias values on the wire, omit null top-level effort, and reject conflicting reasoning controls and model overrides.
- Reject Responses endpoints in the Chat Completions settings factory.

### Compatibility

Validation defaults to warn; strict is opt-in. Conflicting controls are rejected even with passthrough. Applications that relied on extra-body overrides of model or reasoning parameters must send one consistent value. This release is distributed as a GitHub Release tarball by the repository workflow.

## 0.1.4 - 2026-09-10

- Add `deepseek-v4.1-flash` and `deepseek-flash` with image input, tool calling, structured output, and configurable thinking from catalog revision 4.

## 0.1.3 - 2026-09-08

- Add validated endpoint priorities and stable `orderEndpoints` ordering for chat, embeddings, and rerank.
- Adopt shared contract 1.1.0.

## 0.1.2 - 2026-09-08

- Add `gpt-6-astra` from shared contract catalog revision 3.

## 0.1.1 - 2026-08-27

- Add the contract revision 2 catalog, including ZhiPuAI `glm-5.3-flash`.

## 0.1.0 - 2026-08-24

- Node 20+ ESM client for chat, SSE streaming, tools, images, embeddings, and
  normalized rerank APIs.
- Typed canonical `ChatRequest` API with validation and OpenAI-compatible
  transport mapping.
- Settings resolution, generated model catalog, stable errors, retries, middleware,
  metadata, provider registry, and capability-aware fallback.
- Pinned `vv-llm-contract` v1.0.1 artifacts with deterministic checks.
- Offline tests, opt-in live smoke tests, and runnable TypeScript examples.
