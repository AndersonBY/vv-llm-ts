import assert from "node:assert/strict";
import { test } from "node:test";
import {
  collectChatStream,
  createChatClientFromSettings,
  createEmbeddingClientFromSettings,
  createRerankClientFromSettings,
  decodeChatRequest,
  ChatRequestDecodeError,
  orderEndpoints,
  CONTRACT_CATALOG_REVISION,
  CONTRACT_CONSUMER_LOCK_SHA256,
  CONTRACT_VERSION,
  DEFAULT_MODEL_CATALOG,
  EMPTY_MODEL_CATALOG,
  ModelCatalog,
  parseRetryAfterHeaders,
  resolveSettingsModel,
  toChatCompletionCreateParams,
  type FetchLike,
  type CanonicalImageUrlContentPart,
  type CanonicalTextContentPart,
  type EndpointBindingInput,
  type JsonObject,
  type Settings,
  VvLlmClient,
  VvLlmError,
} from "../src/index.js";
import { readContractFixture } from "./contract-fixtures.js";
import { validateReasoningEffort } from "../src/reasoning.js";

test("shared reasoning-effort cases distinguish defaults, unknown and unsupported", () => {
  const fixture = readContractFixture<{ capability_cases: Array<{ model: string; reasoning_efforts: string[] | null; reasoning_effort_aliases?: Record<string, string>; reasoning_effort: string | null; valid: boolean }> }>("fixtures/reasoning-effort.v1.json");
  for (const item of fixture.capability_cases) {
    const run = () => validateReasoningEffort(item.model, item.reasoning_effort, { reasoning_efforts: item.reasoning_efforts, reasoning_effort_aliases: item.reasoning_effort_aliases }, "strict");
    if (item.valid) assert.doesNotThrow(run);
    else assert.throws(run, /reasoning_effort/);
    validateReasoningEffort(item.model, item.reasoning_effort, { reasoning_efforts: item.reasoning_efforts }, "passthrough");
  }
});

test("all TypeScript chat entries validate the request model and preserve explicit none", async () => {
  const bodies: Record<string, unknown>[] = [];
  const catalog = new ModelCatalog([
    { id: "first", capabilities: { reasoning_efforts: ["low"] } },
    { id: "second", capabilities: { reasoning_efforts: ["none", "high"] } },
  ]);
  const client = new VvLlmClient({ modelCatalog: catalog, capabilityPolicy: "strict", fetch: async (_input, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return new Response(JSON.stringify({ choices: [] }), { headers: { "content-type": "application/json" } });
  } });
  await client.create({ model: "second", messages: [], options: { reasoning_effort: "none" } });
  await client.chat.completions.create({ model: "first", messages: [] });
  assert.equal(bodies[0]?.reasoning_effort, "none");
  assert.equal(Object.hasOwn(bodies[1] ?? {}, "reasoning_effort"), false);
  await assert.rejects(client.createChatRequest({ model: "first", messages: [], options: { reasoning_effort: "high" } }), /Supported values: low/);
  await assert.rejects(client.streamChatCompletion({ model: "first", messages: [], reasoning_effort: "high" }), /reasoning_effort/);
  await assert.rejects(client.completeChat({ model: "unknown", messages: [], reasoning_effort: "high" }), /support is unknown/);
  await client.completeChat({ model: "unknown", messages: [], reasoning_effort: "ultra", capability_policy: "passthrough" });
  assert.equal(bodies[2]?.reasoning_effort, "ultra");
  assert.equal(bodies.length, 3);
});

test("reasoning conflicts fail before sending, including provider options and zero budgets", async () => {
  const client = new VvLlmClient({ fetch: async () => { assert.fail("must not send"); } });
  for (const extra_body of [
    { reasoning_effort: "low" },
    { google: { thinking_config: { thinking_budget: 0 } } },
    { extra_body: { google: { thinking_config: { thinking_level: "low" } } } },
  ] as JsonObject[]) {
    await assert.rejects(client.completeChat({ model: "model", messages: [], reasoning_effort: "high", extra_body }), /[Cc]onflict/);
  }
  await assert.rejects(client.completeChat({ model: "model", messages: [], reasoning_effort: "high", provider_options: { reasoning_effort: "low" } }), /[Cc]onflict/);
});

test("settings apply binding capability overrides without changing model defaults", async () => {
  const fixture = readContractFixture<{ settings: Settings }>("fixtures/settings-resolution.v1.json");
  const resolved = resolveSettingsModel(fixture.settings, "chat", "deepseek", "chat-alias");
  assert.deepEqual(resolved.model.capabilities?.reasoning_efforts, ["low", "high"]);
  assert.deepEqual(resolved.model.capabilities?.reasoning_effort_aliases, { ultra: "low" });
  validateReasoningEffort(resolved.model_id, "ultra", resolved.model.capabilities, "strict");
  assert.throws(() => validateReasoningEffort(resolved.model_id, "max", resolved.model.capabilities, "strict"), /reasoning_effort/);
  assert.deepEqual(fixture.settings.backends?.deepseek?.models?.["chat-alias"]?.capabilities?.reasoning_efforts, ["low", "medium", "high"]);
  const { client } = createChatClientFromSettings(fixture.settings, { backend: "deepseek", model: "chat-alias", capabilityPolicy: "strict", fetch: async () => { assert.fail("must not send"); } });
  await assert.rejects(client.completeChat({ model: resolved.model_id, messages: [], reasoning_effort: "medium" }), /Supported values: low, high/);
});

test("settings inherit catalog efforts and preserve alias inputs without local capability copies", async () => {
  const bodies: Array<Record<string, unknown>> = [];
  const settings: Settings = {
    endpoints: [{ id: "test", api_base: "https://example.invalid", api_key: "test-key" }],
    backends: { deepseek: { models: { "deepseek-flash": { id: "deepseek-flash", endpoints: ["test"] } } } },
  };
  const { client, resolved } = createChatClientFromSettings(settings, { backend: "deepseek", model: "deepseek-flash", fetch: async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return new Response(JSON.stringify({ choices: [] }), { headers: { "content-type": "application/json" } });
  } });
  assert.deepEqual(resolved.model.capabilities?.reasoning_efforts, ["none", "low", "high", "max"]);
  await client.create({ model: resolved.model_id, messages: [], options: { reasoning_effort: "xhigh" } }, { capability_policy: "strict" });
  assert.equal(bodies[0]?.reasoning_effort, "xhigh");
  settings.backends!.deepseek!.models!["deepseek-flash"]!.endpoints = [{ endpoint_id: "test", capabilities: { reasoning_efforts: [], reasoning_effort_aliases: {} } }];
  const overridden = resolveSettingsModel(settings, "chat", "deepseek", "deepseek-flash");
  assert.throws(() => validateReasoningEffort(overridden.model_id, "xhigh", overridden.model.capabilities, "strict"), /reasoning_effort/);
});

test("ZhiPuAI catalog distinguishes ordinary API efforts, aliases and thinking", async () => {
  const bodies: Array<Record<string, unknown>> = [];
  const client = new VvLlmClient({ capabilityPolicy: "strict", fetch: async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return new Response(JSON.stringify({ choices: [] }), { headers: { "content-type": "application/json" } });
  } });
  for (const model of ["glm-5.2", "glm-5.3", "glm-5.3-flash"]) {
    const capabilities = client.getModelConfig(model)?.capabilities;
    const aliases = model === "glm-5.2" ? { minimal: "none", low: "high", medium: "high", xhigh: "max" } : {};
    assert.deepEqual(capabilities?.reasoning_efforts, model === "glm-5.2" ? ["none", "high", "max"] : ["low", "high", "max"]);
    assert.deepEqual(capabilities?.reasoning_effort_aliases ?? {}, aliases);
    assert.equal(capabilities?.thinking, model === "glm-5.2" ? "configurable" : "always_enabled");
    const settings: Settings = {
      endpoints: [{ id: "test", api_base: "https://example.invalid", api_key: "test-key" }],
      backends: { zhipuai: { models: { [model]: { id: model, endpoints: ["test"] } } } },
    };
    const resolved = resolveSettingsModel(settings, "chat", "zhipuai", model);
    assert.equal(resolved.model.capabilities?.thinking, capabilities?.thinking);
    assert.deepEqual(resolved.model.capabilities?.reasoning_efforts, capabilities?.reasoning_efforts);
    assert.deepEqual(resolved.model.capabilities?.reasoning_effort_aliases, capabilities?.reasoning_effort_aliases);
    settings.backends!.zhipuai!.models![model]!.capabilities = { thinking: "unknown" };
    assert.equal(resolveSettingsModel(settings, "chat", "zhipuai", model).model.capabilities?.thinking, "unknown");
    settings.backends!.zhipuai!.models![model]!.endpoints = [{ endpoint_id: "test", capabilities: { thinking: "always_enabled" } }];
    assert.equal(resolveSettingsModel(settings, "chat", "zhipuai", model).model.capabilities?.thinking, "always_enabled");
    await client.completeChat({ model, messages: [], thinking: { mode: "enabled" } });
    assert.equal(Object.hasOwn(bodies.at(-1) ?? {}, "reasoning_effort"), false);
    for (const effort of [...capabilities!.reasoning_efforts!, ...Object.keys(aliases)]) {
      await client.completeChat({ model, messages: [], reasoning_effort: effort, thinking: { mode: "enabled" } });
      assert.equal(bodies.at(-1)?.reasoning_effort, effort);
      assert.deepEqual(bodies.at(-1)?.thinking, { type: "enabled" });
    }
    for (const effort of model === "glm-5.2" ? ["ultra"] : ["none", "minimal", "medium", "xhigh", "ultra"]) {
      await assert.rejects(client.completeChat({ model, messages: [], reasoning_effort: effort }), /reasoning_effort/);
    }
  }
});

test("default client consumes the pinned contract catalog and metadata", () => {
  const client = new VvLlmClient({ fetch: jsonFetch({}) });
  const vision = client.getModelConfig("deepseek-v4-flash-vision-exp");
  for (const id of ["deepseek-v4.1-flash", "deepseek-flash"]) {
    const expected = { ...vision, id, capabilities: { ...vision?.capabilities } };
    expected.capabilities.reasoning_efforts = ["none", "low", "high", "max"];
    expected.capabilities.reasoning_effort_aliases = { minimal: "low", medium: "high", xhigh: "high", ultra: "max" };
    assert.deepEqual(client.getModelConfig(id), expected);
  }
  const glmFlash = client.getModelConfig("glm-5.3-flash");
  assert.equal(CONTRACT_VERSION, "1.2.0");
  assert.equal(CONTRACT_CATALOG_REVISION, 10);
  assert.equal(CONTRACT_CONSUMER_LOCK_SHA256, "f6a1c18c71555686abf3797e132c4f53cfc0c08ddd60c6ce4e7e89d30544793a");
  assert.equal(client.modelCatalog, DEFAULT_MODEL_CATALOG);
  assert.equal(vision?.max_image_dimension, 8192);
  assert.equal(vision?.capabilities?.thinking, "configurable");
  assert.equal(glmFlash?.context_length, 1_000_000);
  assert.equal(glmFlash?.max_output_tokens, 128_000);
  assert.equal(glmFlash?.function_call_available, true);
  assert.equal(glmFlash?.response_format_available, true);
  assert.equal(glmFlash?.native_multimodal, true);
  assert.deepEqual(glmFlash?.capabilities, {
    tools: true,
    structured_output: "json_schema",
    input_modalities: ["text", "image", "video"],
    thinking: "always_enabled", reasoning_efforts: ["low", "high", "max"],
  });
});

function jsonFetch(value: unknown, status = 200, responseHeaders?: HeadersInit): FetchLike {
  return async () => new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json", ...responseHeaders },
  });
}

test("OpenAI-compatible contract fixture drives canonical request and SSE response consumption", async () => {
  const fixture = readContractFixture<{
    request_case: {
      canonical_request: unknown;
      expected_wire_request: JsonObject;
    };
    completion_case: { raw_response: JsonObject; expected_response: { content: string; reasoning_content: string; usage: JsonObject } };
    stream_case: { raw_chunks: JsonObject[]; expected_deltas: Array<{ content: string; reasoning_content: string; tool_calls: JsonObject[]; usage?: JsonObject }> };
  }>("fixtures/openai-compatible.v2.json");
  const canonical = decodeChatRequest(fixture.request_case.canonical_request);
  const completionParams = toChatCompletionCreateParams(canonical);
  const calls: RequestInit[] = [];
  const fetchImpl: FetchLike = async (_input, init) => {
    calls.push(init ?? {});
    if (calls.length === 1) {
      return new Response(JSON.stringify(fixture.completion_case.raw_response), { headers: { "content-type": "application/json" } });
    }
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of fixture.stream_case.raw_chunks) {
          controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
        }
        controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  };
  const client = new VvLlmClient({ fetch: fetchImpl });
  if (completionParams.stream !== false) throw new Error("fixture completion unexpectedly enabled streaming");
  const completion = await client.completeChat(completionParams);
  assert.deepEqual(JSON.parse(String(calls[0]?.body)), fixture.request_case.expected_wire_request);
  assert.equal(completion.id, fixture.completion_case.raw_response.id);
  assert.equal(completion.choices[0]?.message.content, fixture.completion_case.expected_response.content);
  assert.equal(completion.choices[0]?.message.reasoning_content, fixture.completion_case.expected_response.reasoning_content);

  const streamRequest = decodeChatRequest({
    ...canonical,
    options: { ...(canonical.options ?? {}), stream: true },
  });
  const streamParams = toChatCompletionCreateParams(streamRequest);
  if (streamParams.stream !== true) throw new Error("fixture stream unexpectedly disabled streaming");
  const stream = await client.createChatCompletion(streamParams);
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.equal(chunks.length, fixture.stream_case.expected_deltas.length);
  assert.equal(chunks[0]?.choices[0]?.delta.content, fixture.stream_case.expected_deltas[0]?.content);
  assert.equal(chunks[0]?.choices[0]?.delta.reasoning_content, fixture.stream_case.expected_deltas[0]?.reasoning_content);
  assert.equal(chunks[2]?.usage?.total_tokens, fixture.stream_case.expected_deltas[2]?.usage?.total_tokens);
});

test("canonical tools and tool calls are adapted when options are omitted", async () => {
  let body: unknown;
  const client = new VvLlmClient({
    fetch: async (_input, init) => {
      body = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        id: "canonical-no-options",
        object: "chat.completion",
        created: 1,
        model: "m",
        choices: [{
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: "ok" },
        }],
      }), { headers: { "content-type": "application/json" } });
    },
  });
  await client.create({
    model: "m",
    messages: [{
      role: "assistant",
      content: "",
      tool_calls: [{ id: "call-1", name: "lookup", arguments: "{}" }],
    }],
    tools: [{ name: "lookup", parameters: { type: "object" } }],
    tool_choice: "auto",
  });
  assert.deepEqual(body, {
    model: "m",
    messages: [{
      role: "assistant",
      content: "",
      tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: "{}" } }],
    }],
    stream: false,
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
    tool_choice: "auto",
  });
});

test("canonical decoder validates closed fields and maps both image URL forms", () => {
  const typedText: CanonicalTextContentPart = { type: "text", text: "typed", x_trace: "text" };
  const typedNestedImage: CanonicalImageUrlContentPart = {
    type: "image_url",
    image_url: { url: "https://example.test/typed.png", x_media: "nested" },
    x_trace: "image",
  };
  assert.equal(typedText["x_trace"], "text");
  assert.equal(typedNestedImage["x_trace"], "image");

  const input = {
    model: "vision-model",
    messages: [{
      role: "user",
      content: [
        { type: "image_url", url: "https://example.test/flat.png", detail: "high", cache_control: { type: "ephemeral" } },
        { type: "image_url", image_url: { url: "https://example.test/nested.png", detail: "low", "x_media": "preserve" }, cache_control: "ephemeral" },
        { type: "text", text: "describe", cache_control: { type: "ephemeral" } },
      ],
    }],
    options: { max_tokens_details: { cached_tokens: 3 } },
    "x_trace": { id: "trace-1" },
  };
  const decoded = decodeChatRequest(input);
  assert.notEqual(decoded, input);
  assert.notEqual(decoded.messages, input.messages);
  const params = toChatCompletionCreateParams(decoded);
  assert.deepEqual(params.messages[0]?.content, [
    { type: "image_url", image_url: { url: "https://example.test/flat.png", detail: "high" }, cache_control: { type: "ephemeral" } },
    { type: "image_url", image_url: { url: "https://example.test/nested.png", detail: "low", "x_media": "preserve" }, cache_control: "ephemeral" },
    { type: "text", text: "describe", cache_control: { type: "ephemeral" } },
  ]);
  assert.deepEqual(params.max_tokens_details, { cached_tokens: 3 });
  assert.deepEqual(params.extra_body, { x_trace: { id: "trace-1" } });

  assert.throws(
    () => decodeChatRequest({ model: "m", messages: [{ role: "user", content: "x", unknown: true }] }),
    ChatRequestDecodeError,
  );
  assert.throws(
    () => decodeChatRequest({ model: "m", messages: [{ role: "user", content: "x", "x_Bad": true }] }),
    /invalid extension key/,
  );
  assert.throws(
    () => decodeChatRequest({ model: "m", messages: [{ role: "user", content: "x" }], options: { unknown: true } }),
    /is not allowed/,
  );
  assert.throws(
    () => decodeChatRequest({ model: "m", messages: [{ role: "user", content: "x" }], tool_choice: 42 }),
    /must be an object/,
  );
  assert.throws(
    () => decodeChatRequest({ model: "   ", messages: [] }),
    /must be a non-empty string/,
  );
});

test("legacy flat create keeps top-level options when using shorthand tools", async () => {
  let body: unknown;
  const client = new VvLlmClient({
    fetch: async (_input, init) => {
      body = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({
        id: "legacy-flat",
        object: "chat.completion",
        created: 1,
        model: "m",
        choices: [{
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: "ok" },
        }],
      }), { headers: { "content-type": "application/json" } });
    },
  });

  await client.create({
    model: "m",
    messages: [{ role: "user", content: "hello" }],
    temperature: 0.2,
    max_tokens: 16,
    tools: [{ name: "lookup", parameters: { type: "object" } }],
  });

  assert.deepEqual(body, {
    model: "m",
    messages: [{ role: "user", content: "hello" }],
    temperature: 0.2,
    max_tokens: 16,
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
  });
});

test("non-streaming chat sends multimodal tools, options, thinking, headers, and query", async () => {
  let requestUrl = "";
  let requestInit: RequestInit | undefined;
  const fetchImpl: FetchLike = async (input, init) => {
    requestUrl = input;
    requestInit = init;
    return new Response(JSON.stringify({
      id: "chat-1",
      object: "chat.completion",
      created: 1,
      model: "vision-model",
      choices: [{
        index: 0,
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call-1", type: "function", function: { name: "lookup", arguments: "{\"city\":\"Shanghai\"}" } }],
        },
      }],
      usage: { prompt_tokens: 4, completion_tokens: 7, total_tokens: 11 },
    }), { headers: { "content-type": "application/json" } });
  };
  const client = new VvLlmClient({
    baseURL: "https://gateway.example/v1/",
    apiKey: "secret",
    fetch: fetchImpl,
  });
  const result = await client.chat.completions.create({
    model: "vision-model",
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "Read this" },
        { type: "image_url", image_url: { url: "data:image/png;base64,abc", detail: "high" } },
        { type: "input_audio", input_audio: { data: "abc", format: "wav" } },
      ],
    }],
    temperature: 0.2,
    max_completion_tokens: 64,
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
    tool_choice: "auto",
    thinking: { mode: "enabled", budget_tokens: 128 },
    extra_body: { custom_flag: true },
    extra_headers: { "x-request-tag": "test" },
    extra_query: { trace: "yes", attempt: 1 },
  });

  assert.equal(result.id, "chat-1");
  assert.equal(result.choices[0]?.message.tool_calls?.[0]?.function.name, "lookup");
  assert.equal(requestUrl, "https://gateway.example/v1/chat/completions?trace=yes&attempt=1");
  assert.equal(new Headers(requestInit?.headers).get("authorization"), "Bearer secret");
  assert.equal(new Headers(requestInit?.headers).get("x-request-tag"), "test");
  const body = JSON.parse(String(requestInit?.body)) as Record<string, unknown>;
  assert.equal(body.model, "vision-model");
  assert.equal(body.temperature, 0.2);
  assert.equal((body.extra_flag as unknown), undefined);
  assert.deepEqual(body.thinking, { type: "enabled", budget_tokens: 128 });
  assert.equal(body.custom_flag, true);
  assert.deepEqual((body.messages as Array<{ content: unknown }>)[0]?.content, [
    { type: "text", text: "Read this" },
    { type: "image_url", image_url: { url: "data:image/png;base64,abc", detail: "high" } },
    { type: "input_audio", input_audio: { data: "abc", format: "wav" } },
  ]);
});

test("streaming chat parses split SSE events and aggregates content and tool calls", async () => {
  const sse = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
  const payload = [
    sse({ id: "chat-stream", object: "chat.completion.chunk", created: 2, model: "m", choices: [{ index: 0, delta: { role: "assistant", content: "Hel" }, finish_reason: null }] }),
    sse({ id: "chat-stream", object: "chat.completion.chunk", created: 2, model: "m", choices: [{ index: 0, delta: { content: "lo", tool_calls: [{ index: 0, id: "call", type: "function", function: { name: "lookup", arguments: "{\"city\":\"" } }] }, finish_reason: null }] }),
    sse({ id: "chat-stream", object: "chat.completion.chunk", created: 2, model: "m", choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: "Shanghai\"}" } }] }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 } }),
    "data: [DONE]\n\n",
  ];
  const fetchImpl: FetchLike = async () => {
    let offset = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (offset >= payload.length) {
          controller.close();
          return;
        }
        const text = payload[offset++];
        controller.enqueue(new TextEncoder().encode(text));
      },
    });
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  };
  const client = new VvLlmClient({ fetch: fetchImpl });
  const stream = await client.chat.completions.create({
    model: "m",
    messages: [{ role: "user", content: "stream" }],
    stream: true,
  });
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.equal(chunks.length, 3);
  assert.equal(chunks[0]?.choices[0]?.delta.content, "Hel");
  assert.equal(chunks[1]?.choices[0]?.delta.content, "lo");
  assert.equal(chunks[2]?.usage?.total_tokens, 5);
  const complete = await collectChatStream((async function* () {
    yield* chunks;
  })());
  assert.equal(complete.choices[0]?.message.content, "Hello");
  assert.equal(complete.choices[0]?.message.tool_calls?.[0]?.function.name, "lookup");
  assert.equal(complete.choices[0]?.message.tool_calls?.[0]?.function.arguments, "{\"city\":\"Shanghai\"}");
  assert.equal(complete.usage?.total_tokens, 5);
});

test("embeddings and rerank normalize common responses", async () => {
  const requests: Record<string, unknown>[] = [];
  const fetchImpl: FetchLike = async (_input, init) => {
    requests.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    const body = requests.length === 1
      ? { object: "list", model: "embed", data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }], usage: { prompt_tokens: 2 } }
      : { model: "rank", results: [{ index: 1, relevance_score: 0.9, document: "b" }, { index: 0, relevance_score: 0.1 }] };
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  };
  const client = new VvLlmClient({ baseURL: "http://localhost/v1", fetch: fetchImpl });
  const embedding = await client.embeddings.create({ model: "embed", input: ["a", "b"], dimensions: 2 });
  const rerank = await client.rerank.create({ model: "rank", query: "q", documents: ["a", "b"], top_n: 2 });
  assert.deepEqual(embedding.data[0]?.embedding, [0.1, 0.2]);
  assert.equal(rerank.results[0]?.index, 1);
  assert.deepEqual(requests[0], { model: "embed", input: ["a", "b"], dimensions: 2 });
  assert.equal(requests[1]?.return_documents, true);
});

test("settings resolves endpoint bindings and factories preserve transport metadata", async () => {
  const fixture = readContractFixture<{
    settings: Settings;
    cases: Array<{ kind: "chat" | "embedding" | "rerank"; backend: string; model: string; expected: { model_id: string; endpoint_id: string; endpoint_type: string } }>;
  }>("fixtures/settings-resolution.v1.json");
  const settings = fixture.settings;
  for (const item of fixture.cases) {
    const resolved = resolveSettingsModel(settings, item.kind, item.backend, item.model);
    assert.equal(resolved.model_id, item.expected.model_id, item.kind);
    assert.equal(resolved.endpoint.id, item.expected.endpoint_id, item.kind);
    assert.equal(resolved.endpoint.endpoint_type, item.expected.endpoint_type, item.kind);
  }
  const chatCase = fixture.cases.find((item) => item.kind === "chat");
  assert.ok(chatCase);
  const catalog = new ModelCatalog([{ id: chatCase.expected.model_id, capabilities: { tools: true } }]);
  const fetchImpl = jsonFetch({ id: "ok", object: "chat.completion", created: 0, model: chatCase.expected.model_id, choices: [] });
  const { client, resolved } = createChatClientFromSettings(settings, { backend: chatCase.backend, model: chatCase.model, fetch: fetchImpl, modelCatalog: catalog });
  assert.equal(client.getModelConfig(chatCase.expected.model_id)?.capabilities?.tools, true);
  const embeddingCase = fixture.cases.find((item) => item.kind === "embedding");
  assert.ok(embeddingCase);
  const embedding = createEmbeddingClientFromSettings(settings, { backend: embeddingCase.backend, model: embeddingCase.model, fetch: jsonFetch({ model: embeddingCase.expected.model_id, data: [] }) });
  assert.equal(embedding.resolved.model_id, embeddingCase.expected.model_id);
  const rerankCase = fixture.cases.find((item) => item.kind === "rerank");
  assert.ok(rerankCase);
  const rerank = createRerankClientFromSettings(settings, { backend: rerankCase.backend, model: rerankCase.model, fetch: jsonFetch({ results: [] }) });
  assert.equal(rerank.resolved.endpoint.id, rerankCase.expected.endpoint_id);
  assert.equal(EMPTY_MODEL_CATALOG.list().length, 0);
});

test("settings rejects top-level provider config", () => {
  for (const kind of ["chat", "embedding", "rerank"] as const) {
    assert.throws(
      () => resolveSettingsModel({ openai: { models: {} } }, kind, "openai", "test-model"),
      (error: unknown) => error instanceof VvLlmError
        && error.kind === "configuration"
        && error.code === "UNSUPPORTED_SETTINGS_SHAPE"
        && error.message.includes("backends.openai"),
    );
  }
});

function prioritySettings(section: "backends" | "embedding_backends" | "rerank_backends"): Settings {
  return {
    endpoints: [
      { id: "low", enabled: true },
      { id: "high", enabled: true },
      { id: "peer", enabled: true },
    ],
    [section]: {
      openai: {
        models: {
          test: {
            id: "test",
            endpoints: [
              { endpoint_id: "low", model_id: "low-model", priority: 2 },
              { endpoint_id: "high", model_id: "high-model" },
              "peer",
            ],
          },
        },
      },
    },
  };
}

function priorityBindings(
  settings: Settings,
  section: "backends" | "embedding_backends" | "rerank_backends",
): readonly EndpointBindingInput[] {
  const backendMap = section === "backends"
    ? settings.backends
    : section === "embedding_backends"
      ? settings.embedding_backends
      : settings.rerank_backends;
  return backendMap?.openai?.models?.test?.endpoints ?? [];
}

test("endpoint binding priority validates, round-trips, and orders stably", () => {
  for (const section of ["backends", "embedding_backends", "rerank_backends"] as const) {
    const settings = JSON.parse(JSON.stringify(prioritySettings(section))) as Settings;
    const bindings = priorityBindings(settings, section);
    assert.deepEqual(bindings, priorityBindings(prioritySettings(section), section));
    assert.equal((bindings[0] as { priority?: number }).priority, 2);

    for (const priority of [0, -1, true, false, "2", 1.5, null]) {
      const invalid = prioritySettings(section);
      (priorityBindings(invalid, section) as unknown as Array<Record<string, unknown>>)[0]!.priority = priority;
      const kind = section === "backends" ? "chat" : section === "embedding_backends" ? "embedding" : "rerank";
      assert.throws(() => resolveSettingsModel(invalid, kind, "openai", "test"), /priority/);
    }
  }

  const endpoints = priorityBindings(prioritySettings("backends"), "backends");
  const original = structuredClone(endpoints);
  assert.deepEqual(orderEndpoints(endpoints), [endpoints[1], endpoints[2], endpoints[0]]);
  assert.deepEqual(orderEndpoints(endpoints, "low"), [endpoints[1], endpoints[2], endpoints[0]]);
  assert.deepEqual(orderEndpoints(endpoints, "peer"), [endpoints[2], endpoints[1], endpoints[0]]);
  assert.deepEqual(orderEndpoints(endpoints, "missing"), orderEndpoints(endpoints));
  assert.deepEqual(endpoints, original);
  assert.notEqual(orderEndpoints(endpoints), endpoints);
  assert.deepEqual(orderEndpoints([]), []);
});

test("chat and retrieval auto-selection filters before priority", () => {
  for (const section of ["backends", "embedding_backends", "rerank_backends"] as const) {
    const settings = prioritySettings(section);
    const kind = section === "backends" ? "chat" : section === "embedding_backends" ? "embedding" : "rerank";
    const automatic = resolveSettingsModel(settings, kind, "openai", "test");
    assert.equal(automatic.endpoint.id, "high", section);
    assert.equal(automatic.model_id, "high-model", section);

    settings.endpoints = settings.endpoints!.map((endpoint) => endpoint.id === "high" ? { ...endpoint, enabled: false } : endpoint);
    assert.equal(resolveSettingsModel(settings, kind, "openai", "test").endpoint.id, "peer", section);
  }
});

test("retry headers follow the Python/Rust protocol fixture", () => {
  const fixture = readContractFixture<{
    cases: Array<{ name: string; headers: { "retry-after-ms"?: string; "retry-after"?: string }; now_unix_seconds: number; expected_seconds: number | null }>;
  }>("fixtures/retry-after.v1.json");
  for (const item of fixture.cases) {
    const now = item.now_unix_seconds * 1000;
    const expected = item.expected_seconds === null ? undefined : item.expected_seconds * 1000;
    assert.equal(parseRetryAfterHeaders(item.headers["retry-after-ms"], item.headers["retry-after"], now), expected, item.name);
  }
});

test("HTTP failures expose classified errors and retry metadata", async () => {
  const client = new VvLlmClient({
    fetch: jsonFetch({ error: { message: "slow down", code: "rate_limit" } }, 429, { "x-request-id": "req-1", "retry-after-ms": "1250" }),
  });
  await assert.rejects(
    client.completeChat({ model: "m", messages: [{ role: "user", content: "x" }] }),
    (error: unknown) => {
      assert.ok(error instanceof VvLlmError);
      assert.equal(error.kind, "rate_limited");
      assert.equal(error.status, 429);
      assert.equal(error.request_id, "req-1");
      assert.equal(error.retry_after_ms, 1250);
      return true;
    },
  );
});

test("timeout and caller AbortSignal are distinguishable", async () => {
  const pendingFetch: FetchLike = (_input, init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
  });
  const timeoutClient = new VvLlmClient({ fetch: pendingFetch });
  await assert.rejects(
    timeoutClient.completeChat({ model: "m", messages: [], timeout_ms: 10 }),
    (error: unknown) => error instanceof VvLlmError && error.kind === "timeout",
  );

  const controller = new AbortController();
  const abortClient = new VvLlmClient({ fetch: pendingFetch });
  const request = abortClient.completeChat({ model: "m", messages: [], signal: controller.signal, timeout_ms: 1000 });
  controller.abort();
  await assert.rejects(request, (error: unknown) => error instanceof VvLlmError && error.kind === "cancelled");

  const timeoutController = new AbortController();
  const callerTimeoutClient = new VvLlmClient({ fetch: pendingFetch });
  const callerTimeoutRequest = callerTimeoutClient.completeChat({ model: "m", messages: [], signal: timeoutController.signal });
  timeoutController.abort(new DOMException("Request timed out", "TimeoutError"));
  await assert.rejects(
    callerTimeoutRequest,
    (error: unknown) => error instanceof VvLlmError && error.kind === "timeout" && error.code === "TIMEOUT_ERR",
  );

  const nativeTimeoutRequest = callerTimeoutClient.completeChat({ model: "m", messages: [], signal: AbortSignal.timeout(10) });
  await assert.rejects(nativeTimeoutRequest, (error: unknown) => error instanceof VvLlmError && error.kind === "timeout");
});


test("settings preserve an explicit catalog and binding overrides", async () => {
  const catalog = new ModelCatalog([{ id: "custom", context_length: 1234, capabilities: { tools: true, reasoning_efforts: ["high"] } }]);
  const settings: Settings = { endpoints: [{ id: "test", api_base: "https://example.invalid", api_key: "test-key" }], backends: { openai: { models: { custom: { id: "custom", endpoints: ["test"] } } } } };
  const first = createChatClientFromSettings(settings, { backend: "openai", model: "custom", modelCatalog: catalog, fetch: jsonFetch({ choices: [] }) });
  assert.deepEqual(first.client.getModelConfig("custom")?.capabilities?.reasoning_efforts, ["high"]);
  assert.equal(first.client.getModelConfig("custom")?.context_length, 1234);
  assert.equal(first.client.getModelConfig("custom")?.capabilities?.tools, true);
  await first.client.completeChat({ model: "custom", messages: [], reasoning_effort: "high", capability_policy: "strict" });
  settings.backends!.openai!.models!.custom!.endpoints = [{ endpoint_id: "test", capabilities: { reasoning_efforts: ["low"] } }];
  const overridden = createChatClientFromSettings(settings, { backend: "openai", model: "custom", modelCatalog: catalog, fetch: jsonFetch({ choices: [] }) });
  await assert.rejects(overridden.client.completeChat({ model: "custom", messages: [], reasoning_effort: "high", capability_policy: "strict" }), /reasoning_effort/);
  assert.deepEqual(catalog.get("custom")?.capabilities?.reasoning_efforts, ["high"]);
});

test("extra fields cannot change the validated model and null effort stays omitted", async () => {
  const bodies: Record<string, unknown>[] = [];
  const client = new VvLlmClient({ fetch: async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    return new Response(JSON.stringify({ choices: [] }), { headers: { "content-type": "application/json" } });
  } });
  for (const field of ["extra_body", "provider_options"] as const) {
    await assert.rejects(client.completeChat({ model: "gpt-5", messages: [], [field]: { model: "other" }, capability_policy: "passthrough" }), /model/);
  }
  assert.equal(bodies.length, 0);
  await client.completeChat({ model: "gpt-5", messages: [], reasoning_effort: null });
  assert.equal(Object.hasOwn(bodies[0]!, "reasoning_effort"), false);
  await assert.rejects(client.completeChat({ model: "gpt-5", messages: [], extra_body: { reasoning_effort: null } }), /non-empty string/);
});
