import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ChatMiddlewareV1,
  FallbackChatClient,
  FallbackRoute,
  MiddlewareChatClient,
  ProviderRegistry,
  RetryPolicy,
  ScriptedChatClient,
  ScriptedStream,
  VvLlmError,
  executeWithRetry,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatCompletionCreateParams,
  type ChatExecutionRequest,
  type ChatExecutionResponse,
  type ChatCompletionStreamParams,
  type MiddlewareContext,
} from "../src/index.js";

function completion(content: string, model = "model"): ChatCompletion {
  return {
    id: `completion-${content}`,
    object: "chat.completion",
    created: 0,
    model,
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
  };
}

function chunk(content: string, model = "model"): ChatCompletionChunk {
  return {
    id: `chunk-${content}`,
    object: "chat.completion.chunk",
    created: 0,
    model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

function rolePrelude(model = "model"): ChatCompletionChunk {
  return {
    id: "prelude",
    object: "chat.completion.chunk",
    created: 0,
    model,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  };
}

const request: ChatCompletionCreateParams = {
  model: "model",
  messages: [{ role: "user", content: "hello" }],
};

test("fallback checks each model of one provider without downgrading effort", async () => {
  const scripted = new ScriptedChatClient([completion("ok")]);
  const registry = new ProviderRegistry();
  registry.register("provider", () => scripted, { capabilities: {}, model_capabilities: {
    low: { reasoning_efforts: ["low"] },
    high: { reasoning_efforts: ["high"] },
  } });
  const client = new FallbackChatClient(registry, [new FallbackRoute("provider", "low"), new FallbackRoute("provider", "high")]);
  const result = await client.createWithMetadata({ ...request, reasoning_effort: "high" });
  assert.equal(result.metadata.fallback_index, 1);
  assert.equal(scripted.requests.length, 1);
  assert.equal(scripted.requests[0]?.model, "high");
  assert.equal(scripted.requests[0]?.reasoning_effort, "high");
});

test("RetryPolicy retries classified transient errors and honors Retry-After", async () => {
  let calls = 0;
  const delays: number[] = [];
  const policy = new RetryPolicy({ maxAttempts: 3, baseDelayMs: 10, maxDelayMs: 100, jitterRatio: 0 });
  const value = await executeWithRetry(() => {
    calls += 1;
    if (calls < 3) throw new VvLlmError("busy", { kind: "rate_limited", retry_after_ms: 25 });
    return "ok";
  }, policy, { sleep: (milliseconds) => { delays.push(milliseconds); }, random: () => 0.5 });
  assert.equal(value, "ok");
  assert.equal(calls, 3);
  assert.deepEqual(delays, [25, 25]);
});

test("RetryPolicy does not retry non-retryable failures", async () => {
  let calls = 0;
  const policy = new RetryPolicy({ maxAttempts: 4, baseDelayMs: 0, jitterRatio: 0 });
  await assert.rejects(
    executeWithRetry(() => {
      calls += 1;
      throw new VvLlmError("bad request", { kind: "invalid_request" });
    }, policy, { sleep: () => undefined }),
    (error: unknown) => error instanceof VvLlmError && error.kind === "invalid_request",
  );
  assert.equal(calls, 1);
});

test("MiddlewareChatClient runs hooks and returns CompletionResult metadata", async () => {
  const events: string[] = [];
  class Hooks extends ChatMiddlewareV1 {
    public override onRequest(context: MiddlewareContext, value: ChatExecutionRequest): ChatExecutionRequest {
      events.push(`request:${context.attempt}`);
      return { ...value, extra_body: { ...(value.extra_body ?? {}), hooked: true } };
    }

    public override onResponse(_context: MiddlewareContext, value: ChatExecutionResponse): ChatExecutionResponse {
      events.push("response");
      return value;
    }

    public override onStreamStart() {
      events.push("stream-start");
    }
  }
  const inner = new ScriptedChatClient([completion("answer")], { provider: "scripted-provider" });
  const client = new MiddlewareChatClient(inner, { middleware: [new Hooks()] });
  const result = await client.createWithMetadata(request);
  assert.equal((inner.requests[0]?.extra_body as Record<string, unknown> | undefined)?.hooked, true);
  assert.equal((result.response as ChatCompletion).choices[0]?.message.content, "answer");
  assert.equal(result.metadata.provider, "scripted-provider");
  assert.equal(result.metadata.model, "model");
  assert.equal(result.metadata.attempts, 1);
  assert.deepEqual(events, ["request:0", "response"]);
});

test("MiddlewareChatClient establishes streams before the first chunk", async () => {
  const events: string[] = [];
  class Hooks extends ChatMiddlewareV1 {
    public override onStreamStart() {
      events.push("stream-start");
    }
  }
  const inner = new ScriptedChatClient([
    new ScriptedStream<ChatCompletionChunk>([rolePrelude(), chunk("visible")]),
  ]);
  const client = new MiddlewareChatClient(inner, { middleware: [new Hooks()] });
  const stream = await client.create({ ...request, stream: true });
  assert.deepEqual(events, []);
  const iterator = (stream as AsyncIterable<ChatCompletionChunk>)[Symbol.asyncIterator]();
  const first = await iterator.next();
  assert.deepEqual(events, ["stream-start"]);
  assert.equal(first.value?.choices[0]?.delta.role, "assistant");
  const second = await iterator.next();
  assert.equal(second.value?.choices[0]?.delta.content, "visible");
  assert.deepEqual(events, ["stream-start"]);
});

test("ProviderRegistry skips routes that cannot satisfy capabilities", async () => {
  const registry = new ProviderRegistry();
  let rejectedCalls = 0;
  registry.register("no-tools", () => {
    rejectedCalls += 1;
    return new ScriptedChatClient([completion("wrong")]);
  }, { capabilities: { tools: false, streaming: true, input_modalities: ["text"] } });
  registry.register("tools", () => new ScriptedChatClient([completion("right")]), {
    capabilities: { tools: true, streaming: true, input_modalities: ["text"] },
  });
  const client = new FallbackChatClient(registry, [new FallbackRoute("no-tools", "m1"), new FallbackRoute("tools", "m2")]);
  const response = await client.create({
    ...request,
    tools: [{ type: "function", function: { name: "lookup", parameters: { type: "object" } } }],
  });
  assert.equal((response as ChatCompletion).choices[0]?.message.content, "right");
  assert.equal(rejectedCalls, 0);
});

test("ProviderRegistry skips routes with incompatible thinking capabilities", async () => {
  const unsupported = new ProviderRegistry();
  let unsupportedCalls = 0;
  unsupported.register("unsupported", () => {
    unsupportedCalls += 1;
    return new ScriptedChatClient([completion("wrong")]);
  }, { capabilities: { tools: true, streaming: true, thinking: "unsupported" } });
  unsupported.register("configurable", () => new ScriptedChatClient([completion("right")]), {
    capabilities: { tools: true, streaming: true, thinking: "configurable" },
  });
  const enabledResponse = await new FallbackChatClient(unsupported, [
    new FallbackRoute("unsupported", "m1"),
    new FallbackRoute("configurable", "m2"),
  ]).create({ ...request, thinking: { mode: "enabled" } });
  assert.equal((enabledResponse as ChatCompletion).choices[0]?.message.content, "right");
  assert.equal(unsupportedCalls, 0);

  const unknown = new ProviderRegistry();
  let unknownCalls = 0;
  unknown.register("unknown", () => {
    unknownCalls += 1;
    return new ScriptedChatClient([completion("wrong")]);
  }, { capabilities: { tools: true, streaming: true } });
  unknown.register("configurable", () => new ScriptedChatClient([completion("right")]), {
    capabilities: { tools: true, streaming: true, thinking: "configurable" },
  });
  const unknownResponse = await new FallbackChatClient(unknown, [
    new FallbackRoute("unknown", "m1"),
    new FallbackRoute("configurable", "m2"),
  ]).create({ ...request, thinking: { mode: "enabled" } });
  assert.equal((unknownResponse as ChatCompletion).choices[0]?.message.content, "right");
  assert.equal(unknownCalls, 0);

  const alwaysEnabled = new ProviderRegistry();
  let alwaysEnabledCalls = 0;
  alwaysEnabled.register("always-enabled", () => {
    alwaysEnabledCalls += 1;
    return new ScriptedChatClient([completion("wrong")]);
  }, { capabilities: { tools: true, streaming: true, thinking: "always_enabled" } });
  alwaysEnabled.register("configurable", () => new ScriptedChatClient([completion("right")]), {
    capabilities: { tools: true, streaming: true, thinking: "configurable" },
  });
  const disabledResponse = await new FallbackChatClient(alwaysEnabled, [
    new FallbackRoute("always-enabled", "m1"),
    new FallbackRoute("configurable", "m2"),
  ]).create({ ...request, thinking: { mode: "disabled" } });
  assert.equal((disabledResponse as ChatCompletion).choices[0]?.message.content, "right");
  assert.equal(alwaysEnabledCalls, 0);
});

test("ScriptedChatClient isolates nested request snapshots and preserves AbortSignal", async () => {
  const controller = new AbortController();
  const messages = [{ role: "user" as const, content: [{ type: "text" as const, text: "before" }] }];
  const extraBody = { nested: { value: "before" } };
  const scripted = new ScriptedChatClient([completion("ok")]);
  await scripted.create({ ...request, messages, extra_body: extraBody, signal: controller.signal });

  messages[0]!.content = [{ type: "text", text: "after" }];
  extraBody.nested.value = "after";
  const captured = scripted.requests[0];
  assert.ok(captured);
  assert.equal(captured.signal, controller.signal);
  assert.equal((captured.messages[0]?.content as unknown as readonly [{ text: string }])[0]?.text, "before");
  assert.equal(((captured.extra_body?.nested as Record<string, unknown>).value), "before");

  ((captured.extra_body?.nested as Record<string, unknown>).value) = "mutated-return";
  assert.equal(((scripted.requests[0]?.extra_body?.nested as Record<string, unknown>).value), "before");
});

test("FallbackChatClient falls back on configured errors but not by default on invalid requests", async () => {
  const registry = new ProviderRegistry();
  registry.register("bad", () => new ScriptedChatClient([new VvLlmError("invalid", { kind: "invalid_request" })]), {
    capabilities: { tools: true, streaming: true, input_modalities: ["text"] },
  });
  registry.register("good", () => new ScriptedChatClient([completion("good")]), {
    capabilities: { tools: true, streaming: true, input_modalities: ["text"] },
  });
  const noFallback = new FallbackChatClient(registry, [new FallbackRoute("bad", "bad"), new FallbackRoute("good", "good")]);
  await assert.rejects(noFallback.create(request), (error: unknown) => error instanceof VvLlmError && error.kind === "invalid_request");

  const configured = new FallbackChatClient(registry, [new FallbackRoute("bad", "bad"), new FallbackRoute("good", "good")], { fallbackOn: ["invalid_request"] });
  const response = await configured.create(request);
  assert.equal((response as ChatCompletion).choices[0]?.message.content, "good");
});

test("stream fallback may switch before first visible chunk", async () => {
  const registry = new ProviderRegistry();
  const first = new ScriptedChatClient([new ScriptedStream<ChatCompletionChunk>([new VvLlmError("before visible", { kind: "network" })])]);
  const second = new ScriptedChatClient([new ScriptedStream<ChatCompletionChunk>([chunk("fallback")])]);
  registry.register("first", () => first, { capabilities: { tools: true, streaming: true, input_modalities: ["text"] } });
  registry.register("second", () => second, { capabilities: { tools: true, streaming: true, input_modalities: ["text"] } });
  const client = new FallbackChatClient(registry, [new FallbackRoute("first", "m1"), new FallbackRoute("second", "m2")]);
  const stream = await client.create({ ...request, stream: true }) as AsyncIterable<ChatCompletionChunk>;
  const chunks: ChatCompletionChunk[] = [];
  for await (const value of stream) chunks.push(value);
  assert.equal(chunks[0]?.choices[0]?.delta.content, "fallback");
  assert.equal(first.requests.length, 1);
  assert.equal(second.requests.length, 1);
});

test("stream fallback never replays after the first visible chunk", async () => {
  const registry = new ProviderRegistry();
  const first = new ScriptedChatClient([new ScriptedStream<ChatCompletionChunk>([
    chunk("visible"),
    new VvLlmError("after visible", { kind: "network" }),
  ])]);
  const second = new ScriptedChatClient([new ScriptedStream<ChatCompletionChunk>([chunk("must-not-run")])]);
  registry.register("first", () => first, { capabilities: { tools: true, streaming: true, input_modalities: ["text"] } });
  registry.register("second", () => second, { capabilities: { tools: true, streaming: true, input_modalities: ["text"] } });
  const client = new FallbackChatClient(registry, [new FallbackRoute("first", "m1"), new FallbackRoute("second", "m2")]);
  const stream = await client.create({ ...request, stream: true } as ChatCompletionStreamParams) as AsyncIterable<ChatCompletionChunk>;
  const iterator = stream[Symbol.asyncIterator]();
  const firstChunk = await iterator.next();
  assert.equal(firstChunk.value?.choices[0]?.delta.content, "visible");
  await assert.rejects(iterator.next(), (error: unknown) => error instanceof VvLlmError && error.kind === "network");
  assert.equal(second.requests.length, 0);
});
