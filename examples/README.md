# Examples

All network examples read `VV_LLM_MODEL`, `OPENAI_BASE_URL` (optional), and
`OPENAI_API_KEY` (optional for unauthenticated local gateways). They decode the
provider-neutral `ChatRequest` shape before calling `VvLlmClient`.

```sh
npm run examples:check
npm run examples:build
node dist-examples/basic_chat.js
node dist-examples/streaming.js
node dist-examples/tools.js
node dist-examples/multimodal.js
node dist-examples/contract_json.js ./request.json
node dist-examples/middleware_fallback.js
```

`contract_json.js` is offline when no path is supplied. `streaming.js` consumes
its `AsyncIterable` once; `tools.js` demonstrates object `tool_choice`, and
`multimodal.js` demonstrates both flat and nested image URL forms.
`middleware_fallback.js` is an offline middleware, metadata, and fallback example.
