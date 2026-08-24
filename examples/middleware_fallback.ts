import {
  ChatMiddlewareV1,
  decodeChatRequest,
  FallbackChatClient,
  FallbackRoute,
  MiddlewareChatClient,
  ProviderRegistry,
  ScriptedChatClient,
  VvLlmError,
  toChatCompletionCreateParams,
  type ChatCompletion,
  type ChatExecutionRequest,
  type MiddlewareContext,
} from "vv-llm-ts";

const completion: ChatCompletion = {
  id: "chatcmpl-example",
  object: "chat.completion",
  created: 0,
  model: "backup-model",
  choices: [{
    index: 0,
    message: { role: "assistant", content: "Fallback succeeded." },
    finish_reason: "stop",
  }],
};

const events: string[] = [];

class RequestTrace extends ChatMiddlewareV1 {
  public override onRequest(
    context: MiddlewareContext,
    request: ChatExecutionRequest,
  ): ChatExecutionRequest {
    events.push(`${context.provider}:${request.model}`);
    return request;
  }
}

const middleware = [new RequestTrace()];
const registry = new ProviderRegistry();
registry.register("primary", () => new MiddlewareChatClient(
  new ScriptedChatClient([
    new VvLlmError("primary unavailable", { kind: "network" }),
  ], { provider: "primary" }),
  { middleware },
), { capabilities: { tools: true, streaming: true, input_modalities: ["text"] } });
registry.register("backup", () => new MiddlewareChatClient(
  new ScriptedChatClient([completion], { provider: "backup" }),
  { middleware },
), {
  capabilities: { tools: true, streaming: true, input_modalities: ["text"] },
});

const fallback = new FallbackChatClient(registry, [
  new FallbackRoute("primary", "primary-model"),
  new FallbackRoute("backup", "backup-model"),
]);
const request = decodeChatRequest({
  model: "primary-model",
  messages: [{ role: "user", content: "Use the available provider." }],
  options: { stream: false },
});

const result = await fallback.createWithMetadata(toChatCompletionCreateParams(request));
console.log((result.response as ChatCompletion).choices[0]?.message.content ?? "");
console.log(JSON.stringify({ events, metadata: result.metadata }, null, 2));
