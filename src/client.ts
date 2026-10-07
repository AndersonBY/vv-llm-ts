import { HttpTransport, type TransportOptions } from "./transport.js";
import { normalizeGeminiBody, mergeProviderBody, mergeReasoningBody, resolveReasoningEffort, validateReasoningEffort } from "./reasoning.js";
import type { CapabilityPolicy } from "./types.js";
import { DEFAULT_MODEL_CATALOG, ModelCatalog } from "./catalog.js";
import {
  type ChatCompletion,
  type ChatCompletionChoice,
  type ChatCompletionCreateParams,
  type ChatCompletionDelta,
  type ChatCompletionChunk,
  type ChatCompletionMessage,
  type ChatCompletionStreamParams,
  type ChatMessage,
  type ChatRequest,
  type EmbeddingCreateParams,
  type EmbeddingData,
  type EmbeddingResponse,
  type JsonObject,
  type JsonValue,
  type NonStreamingChatRequest,
  type RerankCreateParams,
  type RerankResponse,
  type RerankResult,
  type StreamingChatRequest,
  type ChatRequestTransportOptions,
  type ThinkingPreference,
  type ToolCall,
  toChatCompletionCreateParams,
} from "./types.js";
import { isJsonValue, isRecord, VvLlmError } from "./errors.js";
import type { ChatExecutionRequest, ChatExecutionResponse } from "./execution.js";

export type { FetchLike } from "./transport.js";
export interface ClientOptions extends TransportOptions {
  modelCatalog?: ModelCatalog;
  model_catalog?: ModelCatalog;
  capabilityPolicy?: CapabilityPolicy;
  capability_policy?: CapabilityPolicy;
}

export interface ChatCreateMethod {
  (params: ChatCompletionStreamParams): Promise<AsyncIterable<ChatCompletionChunk>>;
  (params: ChatCompletionCreateParams): Promise<ChatCompletion>;
}

export interface VvLlmChatNamespace {
  completions: {
    create: ChatCreateMethod;
  };
}

export interface VvLlmEmbeddingsNamespace {
  create: (params: EmbeddingCreateParams) => Promise<EmbeddingResponse>;
}

export interface VvLlmRerankNamespace {
  create: (params: RerankCreateParams) => Promise<RerankResponse>;
}


const CHAT_OPTION_KEYS = [
  "temperature",
  "max_tokens",
  "max_completion_tokens",
  "top_p",
  "response_format",
  "stream_options",
  "audio",
  "frequency_penalty",
  "logit_bias",
  "logprobs",
  "max_tokens_details",
  "metadata",
  "modalities",
  "n",
  "parallel_tool_calls",
  "prediction",
  "presence_penalty",
  "reasoning_effort",
  "seed",
  "service_tier",
  "stop",
  "store",
  "top_logprobs",
  "user",
] as const;

type ChatOptionKey = (typeof CHAT_OPTION_KEYS)[number];

/**
 * Native-fetch OpenAI-compatible client. Network methods are Promise based;
 * streams are returned as AsyncIterables once the HTTP response is available.
 */
export class VvLlmClient {
  public readonly chat: VvLlmChatNamespace;
  public readonly embeddings: VvLlmEmbeddingsNamespace;
  public readonly rerank: VvLlmRerankNamespace;
  public readonly modelCatalog: ModelCatalog;
  public readonly providerName = "openai-compatible";

  private readonly transport: HttpTransport;
  private readonly capabilityPolicy: CapabilityPolicy;

  public constructor(options: ClientOptions = {}) {
    this.transport = new HttpTransport(options);
    this.modelCatalog = options.modelCatalog ?? options.model_catalog ?? DEFAULT_MODEL_CATALOG;
    this.capabilityPolicy = options.capabilityPolicy ?? options.capability_policy ?? "warn";

    this.chat = {
      completions: {
        create: this.createChatCompletion.bind(this) as ChatCreateMethod,
      },
    };
    this.embeddings = {
      create: this.createEmbedding.bind(this),
    };
    this.rerank = {
      create: this.createRerank.bind(this),
    };
  }

  public getModelConfig(modelId: string) {
    return this.modelCatalog.get(modelId);
  }

  /**
   * Create from the canonical shape or the flat execution shape used by
   * middleware and fallback clients.
   */
  public async create(
    request: StreamingChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  public async create(
    request: NonStreamingChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<ChatCompletion>;
  public async create(
    request: ChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<ChatExecutionResponse>;
  public async create(request: ChatExecutionRequest): Promise<ChatExecutionResponse>;
  public async create(
    request: ChatRequest | ChatExecutionRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<ChatExecutionResponse> {
    const params = isCanonicalChatRequest(request) ? toChatCompletionCreateParams(request, transport) : request;
    return this.createChatCompletionInternal(params);
  }

  /** Explicitly named canonical entry point for callers that prefer clarity. */
  public async createChatRequest(
    request: StreamingChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  public async createChatRequest(
    request: NonStreamingChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<ChatCompletion>;
  public async createChatRequest(
    request: ChatRequest,
    transport?: ChatRequestTransportOptions,
  ): Promise<ChatExecutionResponse> {
    return this.createChatCompletionInternal(toChatCompletionCreateParams(request, transport));
  }

  public async createChatCompletion(params: ChatCompletionCreateParams): Promise<ChatCompletion>;
  public async createChatCompletion(
    params: ChatCompletionStreamParams,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  public async createChatCompletion(
    params: ChatCompletionCreateParams | ChatCompletionStreamParams,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    return this.createChatCompletionInternal(params);
  }

  private async createChatCompletionInternal(
    params: ChatCompletionCreateParams | ChatCompletionStreamParams,
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    const stream = params.stream === true;
    const body = buildChatBody(params);
    validateReasoningEffort(params.model, body.reasoning_effort, this.modelCatalog.capabilities(params.model), params.capability_policy ?? this.capabilityPolicy);
    const response = await this.transport.send("/chat/completions", body, {
      headers: params.extra_headers,
      query: params.extra_query,
      timeoutMs: params.timeout_ms,
      signal: params.signal,
      accept: stream ? "text/event-stream" : "application/json",
    });

    if (!response.ok) {
      throw await this.transport.errorFromResponse(response);
    }
    if (stream) {
      return parseSseStream(response);
    }
    const raw = await this.transport.readJson(response);
    return normalizeChatCompletion(raw, params.model);
  }

  public async streamChatCompletion(
    params: Omit<ChatCompletionStreamParams, "stream">,
  ): Promise<AsyncIterable<ChatCompletionChunk>> {
    return this.createChatCompletion({ ...params, stream: true });
  }

  public async completeChat(
    params: Omit<ChatCompletionCreateParams, "stream">,
  ): Promise<ChatCompletion> {
    return this.createChatCompletion({ ...params, stream: false });
  }

  public async createEmbedding(params: EmbeddingCreateParams): Promise<EmbeddingResponse> {
    const body: Record<string, unknown> = {
      model: params.model,
      input: params.input,
    };
  copyDefined(body, params as unknown as Record<string, unknown>, ["encoding_format", "dimensions", "user"]);
    Object.assign(body, params.extra_body ?? {});
    const response = await this.transport.send("/embeddings", body, {
      headers: params.extra_headers,
      query: params.extra_query,
      timeoutMs: params.timeout_ms,
      signal: params.signal,
    });
    if (!response.ok) {
      throw await this.transport.errorFromResponse(response);
    }
    const raw = await this.transport.readJson(response);
    return normalizeEmbeddingResponse(raw, params.model);
  }

  public async embed(
    input: string,
    options: Omit<EmbeddingCreateParams, "input">,
  ): Promise<number[] | string> {
    const response = await this.createEmbedding({ ...options, input });
    const first = response.data[0];
    if (!first) return [];
    return first.embedding;
  }

  public async createRerank(params: RerankCreateParams): Promise<RerankResponse> {
    const body: Record<string, unknown> = {
      model: params.model,
      query: params.query,
      documents: params.documents,
      return_documents: params.return_documents ?? true,
    };
    if (params.top_n !== undefined) body.top_n = params.top_n;
    Object.assign(body, params.extra_body ?? {});
    const response = await this.transport.send("/rerank", body, {
      headers: params.extra_headers,
      query: params.extra_query,
      timeoutMs: params.timeout_ms,
      signal: params.signal,
    });
    if (!response.ok) {
      throw await this.transport.errorFromResponse(response);
    }
    const raw = await this.transport.readJson(response);
    return normalizeRerankResponse(raw, params.model);
  }

  public async rerankDocuments(params: RerankCreateParams): Promise<RerankResponse> {
    return this.createRerank(params);
  }


}

function isCanonicalChatRequest(request: ChatRequest | ChatExecutionRequest): request is ChatRequest {
  if (Object.keys(request).some((key) => FLAT_CHAT_REQUEST_KEYS.has(key))) return false;
  if (Object.prototype.hasOwnProperty.call(request, "options")) return true;
  if (Object.keys(request).some((key) => key.startsWith("x_"))) return true;
  if (Array.isArray(request.tools) && request.tools.some((tool) => {
    return isRecord(tool) && typeof tool.name === "string" && tool.type === undefined;
  })) return true;
  return request.messages.some((message) => {
    if (!isRecord(message) || !Array.isArray(message.tool_calls)) return false;
    return message.tool_calls.some((toolCall) => {
      return isRecord(toolCall) && typeof toolCall.name === "string" && toolCall.function === undefined;
    });
  });
}

const FLAT_CHAT_REQUEST_KEYS = new Set([
  "temperature",
  "max_tokens",
  "max_completion_tokens",
  "stream",
  "top_p",
  "stop",
  "response_format",
  "stream_options",
  "audio",
  "frequency_penalty",
  "logit_bias",
  "logprobs",
  "max_tokens_details",
  "metadata",
  "modalities",
  "n",
  "parallel_tool_calls",
  "prediction",
  "presence_penalty",
  "reasoning_effort",
  "thinking",
  "seed",
  "service_tier",
  "store",
  "top_logprobs",
  "user",
  "extra_headers",
  "extra_query",
  "provider_options",
  "timeout_ms",
  "signal",
]);


function buildChatBody(params: ChatCompletionCreateParams | ChatCompletionStreamParams): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: params.model,
    messages: params.messages,
  };
  for (const key of CHAT_OPTION_KEYS) {
    const value = params[key as ChatOptionKey];
    if (value !== undefined && value !== null) body[key] = value;
  }
  if (params.stream !== undefined) body.stream = params.stream;
  if (params.tools !== undefined) body.tools = normalizeTools(params.tools);
  if (params.tool_choice !== undefined) body.tool_choice = params.tool_choice;
  if (params.stream === true && params.stream_options === undefined) {
    body.stream_options = { include_usage: true };
  }

  const extraBody = mergeProviderBody(params.extra_body ?? {}, params.provider_options ?? {});
  const thinking = resolveThinking(params.thinking);
  if (thinking !== undefined) Object.assign(extraBody, mergeReasoningBody(extraBody, { thinking }));
  const effort = resolveReasoningEffort(params.reasoning_effort, extraBody, params.model);
  if (effort !== undefined) extraBody.reasoning_effort = effort;
  else delete body.reasoning_effort;
  Object.assign(body, extraBody);
  return normalizeGeminiBody(params.model, body);
}

function normalizeTools(tools: readonly unknown[]): readonly unknown[] {
  return tools.map((tool) => {
    if (!isRecord(tool) || tool.type !== undefined) return tool;
    if (typeof tool.name !== "string") return tool;
    return {
      type: "function",
      function: {
        name: tool.name,
        ...(typeof tool.description === "string" ? { description: tool.description } : {}),
        ...(isRecord(tool.parameters) ? { parameters: tool.parameters } : {}),
      },
    };
  });
}

function resolveThinking(value: JsonObject | ThinkingPreference | null | undefined): JsonObject | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value) || typeof value.mode !== "string") return value as JsonObject;
  const preference = value as unknown as ThinkingPreference;
  if (preference.mode === "default") return undefined;
  if (preference.mode === "disabled") return { type: "disabled" };
  if (preference.mode === "enabled") {
    return preference.budget_tokens === undefined
      ? { type: "enabled" }
      : { type: "enabled", budget_tokens: preference.budget_tokens };
  }
  if (preference.mode === "provider_defined") return preference.value ?? {};
  return value as JsonObject;
}

function copyDefined(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
  keys: readonly string[],
): void {
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined && value !== null) target[key] = value;
  }
}

export async function* parseSseStream(response: Response): AsyncGenerator<ChatCompletionChunk> {
  if (!response.body) {
    throw new VvLlmError("Streaming response has no body", { kind: "serialization" });
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];

  const emit = async function* (): AsyncGenerator<ChatCompletionChunk> {
    if (dataLines.length === 0) return;
    const data = dataLines.join("\n");
    dataLines = [];
    if (data.trim() === "[DONE]") return;
    try {
      const raw = JSON.parse(data) as unknown;
      yield normalizeChatChunk(raw);
    } catch (error) {
      if (error instanceof VvLlmError) throw error;
      throw new VvLlmError("Provider returned invalid SSE JSON", {
        kind: "serialization",
        cause: error,
      });
    }
  };

  while (true) {
    const item = await reader.read();
    buffer += decoder.decode(item.value, { stream: !item.done });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).replace(/\r$/, "");
      buffer = buffer.slice(newline + 1);
      if (line === "") {
        yield* emit();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
      newline = buffer.indexOf("\n");
    }
    if (item.done) break;
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    if (buffer.startsWith("data:")) dataLines.push(buffer.slice(5).trimStart());
    else dataLines.push(buffer.trim());
  }
  yield* emit();
}

export async function collectChatStream(stream: AsyncIterable<ChatCompletionChunk>): Promise<ChatCompletion> {
  let id = "";
  let model = "";
  let created = 0;
  let role = "assistant";
  let content = "";
  let reasoning = "";
  let finishReason: string | null = null;
  let usage;
  const toolCalls = new Map<number, ToolCall>();

  for await (const chunk of stream) {
    id = chunk.id || id;
    model = chunk.model || model;
    created = chunk.created || created;
    usage = chunk.usage ?? usage;
    const choice = chunk.choices[0];
    if (!choice) continue;
    finishReason = choice.finish_reason ?? finishReason;
    const delta = choice.delta;
    role = delta.role ?? role;
    if (typeof delta.content === "string") content += delta.content;
    if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
    for (const [position, call] of (delta.tool_calls ?? []).entries()) {
      const index = call.index ?? position;
      const existing = toolCalls.get(index);
      if (!existing) {
        toolCalls.set(index, {
          id: call.id ?? `tool_call_${index}`,
          type: call.type ?? "function",
          function: {
            name: call.function?.name ?? "",
            arguments: call.function?.arguments ?? "",
          },
          index,
        });
      } else {
        if (call.id) existing.id = call.id;
        if (call.type) existing.type = call.type;
        if (call.function?.name) existing.function.name += call.function.name;
        if (call.function?.arguments) existing.function.arguments += call.function.arguments;
      }
    }
  }

  const message: ChatCompletionMessage = {
    role,
    content: content || null,
  };
  if (reasoning) message.reasoning_content = reasoning;
  if (toolCalls.size) message.tool_calls = [...toolCalls.values()].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  const choice: ChatCompletionChoice = { index: 0, message, finish_reason: finishReason };
  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [choice],
    ...(usage ? { usage } : {}),
  };
}

function normalizeChatCompletion(raw: unknown, fallbackModel: string): ChatCompletion {
  if (!isRecord(raw) || !Array.isArray(raw.choices)) {
    throw new VvLlmError("Provider returned an invalid chat completion", { kind: "serialization" });
  }
  return {
    ...raw,
    id: typeof raw.id === "string" ? raw.id : "",
    object: typeof raw.object === "string" ? raw.object : "chat.completion",
    created: typeof raw.created === "number" ? raw.created : Math.floor(Date.now() / 1000),
    model: typeof raw.model === "string" ? raw.model : fallbackModel,
    choices: raw.choices as ChatCompletionChoice[],
    usage: isRecord(raw.usage) ? raw.usage as unknown as ChatCompletion["usage"] : raw.usage === null ? null : undefined,
  };
}

function normalizeChatChunk(raw: unknown): ChatCompletionChunk {
  if (!isRecord(raw) || !Array.isArray(raw.choices)) {
    throw new VvLlmError("Provider returned an invalid chat completion chunk", { kind: "serialization" });
  }
  return {
    ...raw,
    id: typeof raw.id === "string" ? raw.id : "",
    object: typeof raw.object === "string" ? raw.object : "chat.completion.chunk",
    created: typeof raw.created === "number" ? raw.created : Math.floor(Date.now() / 1000),
    model: typeof raw.model === "string" ? raw.model : "",
    choices: raw.choices as ChatCompletionChunk["choices"],
    usage: isRecord(raw.usage) ? raw.usage as unknown as ChatCompletionChunk["usage"] : raw.usage === null ? null : undefined,
  };
}

function normalizeEmbeddingResponse(raw: unknown, fallbackModel: string): EmbeddingResponse {
  if (!isRecord(raw) || !Array.isArray(raw.data)) {
    throw new VvLlmError("Provider returned an invalid embedding response", { kind: "serialization" });
  }
  const data: EmbeddingData[] = raw.data.map((item, position) => {
    if (!isRecord(item)) {
      throw new VvLlmError(`Embedding item ${position} is not an object`, { kind: "serialization" });
    }
    const embedding = item.embedding;
    if (!(typeof embedding === "string" || (Array.isArray(embedding) && embedding.every((value) => typeof value === "number")))) {
      throw new VvLlmError(`Embedding item ${position} has an invalid vector`, { kind: "serialization" });
    }
    return {
      object: typeof item.object === "string" ? item.object : "embedding",
      index: typeof item.index === "number" ? item.index : position,
      embedding: embedding as number[] | string,
      ...(typeof item.text === "string" ? { text: item.text } : {}),
      ...(isRecord(item.metadata) && isJsonValue(item.metadata) ? { metadata: item.metadata as JsonObject } : {}),
    };
  });
  return {
    ...raw,
    object: typeof raw.object === "string" ? raw.object : "list",
    model: typeof raw.model === "string" ? raw.model : fallbackModel,
    data,
    usage: isRecord(raw.usage) ? raw.usage as JsonObject : raw.usage === null ? null : undefined,
  };
}

function normalizeRerankResponse(raw: unknown, fallbackModel: string): RerankResponse {
  if (!isRecord(raw) || !Array.isArray(raw.results)) {
    throw new VvLlmError("Provider returned an invalid rerank response", { kind: "serialization" });
  }
  const results: RerankResult[] = raw.results.map((item, position) => {
    if (!isRecord(item) || typeof item.relevance_score !== "number") {
      throw new VvLlmError(`Rerank result ${position} is invalid`, { kind: "serialization" });
    }
    return {
      ...item,
      index: typeof item.index === "number" ? item.index : position,
      relevance_score: item.relevance_score,
    } as RerankResult;
  });
  return {
    ...raw,
    model: typeof raw.model === "string" ? raw.model : fallbackModel,
    results,
    usage: isRecord(raw.usage) ? raw.usage as JsonObject : raw.usage === null ? null : undefined,
  };
}
