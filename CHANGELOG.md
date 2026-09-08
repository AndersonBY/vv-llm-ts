# Changelog

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
