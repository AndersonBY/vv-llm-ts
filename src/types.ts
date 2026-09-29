/** OpenAI-compatible public types and canonical request adapter inputs. */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JSONValue = JsonValue;
export interface JsonObject {
  [key: string]: JsonValue;
}

export type ChatRole = "system" | "developer" | "user" | "assistant" | "tool";

export interface TextContentPart {
  type: "text";
  text: string;
  cache_control?: JsonValue;
}

export type ImageDetail = "auto" | "low" | "high";

export interface ImageUrlContentPart {
  type: "image_url";
  image_url: {
    url: string;
    detail?: ImageDetail;
    [key: `x_${string}`]: JsonValue | undefined;
  };
  cache_control?: JsonValue;
}

export interface InputAudioContentPart {
  type: "input_audio";
  input_audio: {
    data: string;
    format: string;
  };
}

/** OpenAI-compatible multimodal parts supported by this MVP. */
export type ContentPart = TextContentPart | ImageUrlContentPart | InputAudioContentPart;
export type MessageContent = string | readonly ContentPart[];

export interface ToolCall {
  id: string;
  type: "function" | string;
  function: {
    name: string;
    arguments: string;
  };
  index?: number;
  extra_content?: JsonValue;
}

export interface ChatMessage {
  role: ChatRole;
  content?: MessageContent | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: readonly ToolCall[];
  reasoning_content?: string;
}

export interface FunctionToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: JsonValue;
    strict?: boolean;
  };
}

export type ToolDefinition = FunctionToolDefinition | JsonObject;
export type ToolChoice = "none" | "auto" | "required" | JsonObject;

/** Canonical contract roles (the OpenAI-compatible type additionally accepts developer). */
export type CanonicalChatRole = "system" | "user" | "assistant" | "tool";

export interface CanonicalTextContentPart extends TextContentPart {
  cache_control?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

export interface CanonicalImageUrlContentPart extends ImageUrlContentPart {
  cache_control?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

/** Canonical contract image form with the URL at the content-part root. */
export interface CanonicalFlatImageUrlContentPart {
  type: "image_url";
  url: string;
  detail?: ImageDetail;
  cache_control?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

export type CanonicalContentPart =
  | CanonicalTextContentPart
  | CanonicalImageUrlContentPart
  | CanonicalFlatImageUrlContentPart;
export type CanonicalMessageContent = string | readonly CanonicalContentPart[];

/** Tool-call shape used by the provider-neutral contract. */
export interface CanonicalToolCall {
  id: string;
  name: string;
  arguments: string;
  index?: number;
  extra_content?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

export interface CanonicalChatMessage {
  role: CanonicalChatRole;
  content: CanonicalMessageContent;
  name?: string;
  tool_call_id?: string;
  tool_calls?: readonly CanonicalToolCall[];
  reasoning_content?: string;
  [key: `x_${string}`]: JsonValue | undefined;
}

/** Canonical tool definitions are normalized to OpenAI function tools by the adapter. */
export interface CanonicalChatTool {
  name: string;
  description?: string;
  parameters: JsonValue;
  cache_control?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

export type CanonicalThinkingType =
  | "default"
  | "enabled"
  | "disabled"
  | "adaptive"
  | "provider_defined";

export interface CanonicalThinkingPreference {
  type: CanonicalThinkingType;
  budget_tokens?: number;
  value?: JsonValue;
  [key: `x_${string}`]: JsonValue | undefined;
}

/** Nested options from the language-neutral canonical chat request. */
export interface ChatRequestOptions {
  temperature?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  stream?: boolean;
  top_p?: number;
  stop?: string | readonly string[];
  response_format?: JsonValue;
  stream_options?: JsonValue;
  audio?: JsonValue;
  frequency_penalty?: number;
  logit_bias?: JsonObject;
  logprobs?: boolean;
  max_tokens_details?: JsonValue;
  metadata?: JsonValue;
  modalities?: readonly string[];
  n?: number;
  parallel_tool_calls?: boolean;
  prediction?: JsonValue;
  presence_penalty?: number;
  reasoning_effort?: string;
  thinking?: CanonicalThinkingPreference;
  seed?: number;
  service_tier?: string;
  store?: boolean;
  top_logprobs?: number;
  user?: string;
  [key: `x_${string}`]: JsonValue | undefined;
}

/** Provider-neutral canonical chat request. Use the adapter for OpenAI wire calls. */
export interface ChatRequest {
  model: string;
  messages: readonly CanonicalChatMessage[];
  options?: ChatRequestOptions;
  tools?: readonly CanonicalChatTool[];
  tool_choice?: ToolChoice;
  extra_body?: JsonObject;
  [key: `x_${string}`]: JsonValue | undefined;
}

/** Runtime-only transport controls kept separate from the canonical JSON shape. */
export interface ChatRequestTransportOptions {
  capability_policy?: CapabilityPolicy;
  signal?: AbortSignal;
  timeout_ms?: number;
}

export type NonStreamingChatRequest = ChatRequest & {
  options?: ChatRequestOptions & { stream?: false };
};

export type StreamingChatRequest = ChatRequest & {
  options: ChatRequestOptions & { stream: true };
};

export type StructuredOutputCapability = "none" | "json_object" | "json_schema";
export type CapabilityPolicy = "strict" | "warn" | "passthrough";
export type ThinkingCapability = "unknown" | "unsupported" | "configurable" | "always_enabled";
export type Modality = "text" | "image" | "audio" | "video";

export interface ModelCapabilities {
  tools?: boolean;
  structured_output?: StructuredOutputCapability;
  input_modalities?: readonly Modality[];
  output_modalities?: readonly Modality[];
  streaming?: boolean;
  parallel_tool_calls?: boolean;
  thinking?: ThinkingCapability;
  reasoning_efforts?: readonly string[] | null;
  reasoning_effort_aliases?: Readonly<Record<string, string>> | null;
}

/** Provider-neutral model metadata, kept close to the Python/Rust fields. */
export interface ModelConfig {
  id: string;
  enabled?: boolean;
  context_length?: number;
  max_output_tokens?: number;
  function_call_available?: boolean;
  response_format_available?: boolean;
  native_multimodal?: boolean;
  max_image_dimension?: number;
  protocol?: string;
  dimensions?: number;
  default_top_n?: number;
  capabilities?: ModelCapabilities;
}

export type ThinkingMode = "default" | "enabled" | "disabled" | "provider_defined";

export interface ThinkingPreference {
  mode: ThinkingMode;
  budget_tokens?: number;
  value?: JsonObject;
}

export interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
  prompt_tokens_details?: JsonObject;
  completion_tokens_details?: JsonObject;
  [key: string]: JsonValue | undefined;
}

export interface ChatCompletionMessage {
  role: "assistant" | string;
  content: string | null;
  refusal?: string | null;
  tool_calls?: ToolCall[];
  function_call?: JsonObject;
  reasoning_content?: string | null;
  [key: string]: JsonValue | ToolCall[] | undefined;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatCompletionMessage;
  finish_reason: string | null;
  logprobs?: JsonValue;
  [key: string]: JsonValue | ChatCompletionMessage | undefined;
}

export interface ChatCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage?: ChatUsage | null;
  system_fingerprint?: string | null;
  [key: string]: JsonValue | ChatCompletionChoice[] | ChatUsage | null | undefined;
}

export interface ChatCompletionDelta {
  role?: string;
  content?: string | null;
  refusal?: string | null;
  tool_calls?: ToolCall[];
  function_call?: JsonObject;
  reasoning_content?: string | null;
  [key: string]: JsonValue | ToolCall[] | undefined;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionDelta;
  finish_reason: string | null;
  logprobs?: JsonValue;
  [key: string]: JsonValue | ChatCompletionDelta | undefined;
}

export interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: ChatUsage | null;
  [key: string]: JsonValue | ChatCompletionChunkChoice[] | ChatUsage | null | undefined;
}

export interface ChatCompletionCreateParamsBase {
  model: string;
  messages: readonly ChatMessage[];
  temperature?: number | null;
  max_tokens?: number | null;
  max_completion_tokens?: number | null;
  top_p?: number | null;
  response_format?: JsonValue;
  stream_options?: JsonValue;
  audio?: JsonValue;
  frequency_penalty?: number | null;
  logit_bias?: JsonValue;
  logprobs?: boolean | null;
  max_tokens_details?: JsonValue;
  metadata?: JsonValue;
  modalities?: readonly string[];
  n?: number | null;
  parallel_tool_calls?: boolean | null;
  prediction?: JsonValue;
  presence_penalty?: number | null;
  reasoning_effort?: string | null;
  capability_policy?: CapabilityPolicy;
  thinking?: JsonObject | ThinkingPreference | null;
  seed?: number | null;
  service_tier?: string | null;
  stop?: string | readonly string[] | null;
  store?: boolean | null;
  top_logprobs?: number | null;
  user?: string | null;
  tools?: readonly ToolDefinition[];
  tool_choice?: ToolChoice;
  extra_headers?: HeadersInit;
  extra_query?: Record<string, string | number | boolean | null | undefined>;
  extra_body?: JsonObject;
  /** Additional root-level OpenAI-compatible extensions. */
  provider_options?: JsonObject;
  timeout_ms?: number;
  signal?: AbortSignal;
}

export interface ChatCompletionCreateParams extends ChatCompletionCreateParamsBase {
  stream?: false;
}

export interface ChatCompletionStreamParams extends ChatCompletionCreateParamsBase {
  stream: true;
}

/**
 * Convert a canonical request's nested options to the flat OpenAI-compatible
 * request shape. The function is pure and does not mutate the input.
 */
export function toChatCompletionCreateParams(
  request: StreamingChatRequest,
  transport?: ChatRequestTransportOptions,
): ChatCompletionStreamParams;
export function toChatCompletionCreateParams(
  request: NonStreamingChatRequest,
  transport?: ChatRequestTransportOptions,
): ChatCompletionCreateParams;
export function toChatCompletionCreateParams(
  request: ChatRequest,
  transport?: ChatRequestTransportOptions,
): ChatCompletionCreateParams | ChatCompletionStreamParams;
export function toChatCompletionCreateParams(
  request: ChatRequest,
  transport: ChatRequestTransportOptions = {},
): ChatCompletionCreateParams | ChatCompletionStreamParams {
  const options = request.options ?? {};
  const extraBody = {
    ...canonicalExtensions(options),
    ...canonicalExtensions(request),
    ...(request.extra_body ?? {}),
  };
  const base: ChatCompletionCreateParamsBase = {
    model: request.model,
    messages: request.messages.map(toOpenAIMessage),
    temperature: options.temperature,
    max_tokens: options.max_tokens,
    max_completion_tokens: options.max_completion_tokens,
    top_p: options.top_p,
    response_format: options.response_format,
    stream_options: options.stream_options,
    audio: options.audio,
    frequency_penalty: options.frequency_penalty,
    logit_bias: options.logit_bias,
    logprobs: options.logprobs,
    max_tokens_details: options.max_tokens_details,
    metadata: options.metadata,
    modalities: options.modalities,
    n: options.n,
    parallel_tool_calls: options.parallel_tool_calls,
    prediction: options.prediction,
    presence_penalty: options.presence_penalty,
    reasoning_effort: options.reasoning_effort,
    thinking: toOpenAIThinking(options.thinking),
    seed: options.seed,
    service_tier: options.service_tier,
    stop: options.stop,
    store: options.store,
    top_logprobs: options.top_logprobs,
    user: options.user,
    tools: request.tools?.map(toOpenAITool),
    tool_choice: request.tool_choice,
    extra_body: Object.keys(extraBody).length > 0 ? extraBody : undefined,
    signal: transport.signal,
    capability_policy: transport.capability_policy,
    timeout_ms: transport.timeout_ms,
  };

  if (options.stream === true) {
    return { ...base, stream: true };
  }
  return { ...base, stream: false };
}

function toOpenAIMessage(message: CanonicalChatMessage): ChatMessage {
  const converted: ChatMessage = {
    role: message.role,
    content: toOpenAIContent(message.content),
    name: message.name,
    tool_call_id: message.tool_call_id,
    reasoning_content: message.reasoning_content,
    tool_calls: message.tool_calls?.map((toolCall) => ({
      id: toolCall.id,
      type: "function",
      function: {
        name: toolCall.name,
        arguments: toolCall.arguments,
      },
      index: toolCall.index,
      extra_content: toolCall.extra_content,
    })),
  };
  copyCanonicalExtensions(message, converted as unknown as Record<string, unknown>);
  return converted;
}

function toOpenAIContent(content: CanonicalMessageContent): MessageContent {
  if (typeof content === "string") return content;
  return content.map((part) => {
    if (part.type !== "image_url" || !("url" in part)) return part;
    const { url, detail, cache_control: cacheControl, ...extensions } = part;
    const converted: ImageUrlContentPart & Record<string, unknown> = {
      ...extensions,
      type: "image_url",
      image_url: {
        url,
        ...(detail === undefined ? {} : { detail }),
      },
      ...(cacheControl === undefined ? {} : { cache_control: cacheControl }),
    };
    return converted;
  });
}

function toOpenAITool(tool: CanonicalChatTool): FunctionToolDefinition {
  const converted: FunctionToolDefinition = {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };
  copyCanonicalExtensions(tool, converted as unknown as Record<string, unknown>);
  return converted;
}

function toOpenAIThinking(thinking: CanonicalThinkingPreference | undefined): JsonObject | undefined {
  if (!thinking) return undefined;
  const converted: JsonObject = { type: thinking.type };
  if (thinking.budget_tokens !== undefined) converted.budget_tokens = thinking.budget_tokens;
  if (thinking.value !== undefined) converted.value = thinking.value;
  copyCanonicalExtensions(thinking, converted);
  return converted;
}

function canonicalExtensions(value: object): JsonObject {
  const extensions: JsonObject = {};
  for (const [key, item] of Object.entries(value)) {
    if (key.startsWith("x_") && item !== undefined) extensions[key] = item as JsonValue;
  }
  return extensions;
}

function copyCanonicalExtensions(source: object, target: Record<string, unknown>): void {
  for (const [key, item] of Object.entries(source)) {
    if (key.startsWith("x_") && item !== undefined) target[key] = item;
  }
}

export interface EmbeddingCreateParams {
  model: string;
  input: string | readonly string[];
  encoding_format?: "float" | "base64" | string;
  dimensions?: number;
  user?: string;
  extra_headers?: HeadersInit;
  extra_query?: Record<string, string | number | boolean | null | undefined>;
  extra_body?: JsonObject;
  timeout_ms?: number;
  signal?: AbortSignal;
}

export interface EmbeddingData {
  object: "embedding" | string;
  index: number;
  embedding: number[] | string;
  text?: string;
  metadata?: JsonObject;
}

export interface EmbeddingResponse {
  object?: string;
  model: string;
  data: EmbeddingData[];
  usage?: JsonObject | null;
  [key: string]: JsonValue | EmbeddingData[] | null | undefined;
}

export interface RerankCreateParams {
  model: string;
  query: string;
  documents: readonly (string | JsonObject)[];
  top_n?: number;
  return_documents?: boolean;
  extra_headers?: HeadersInit;
  extra_query?: Record<string, string | number | boolean | null | undefined>;
  extra_body?: JsonObject;
  timeout_ms?: number;
  signal?: AbortSignal;
}

export interface RerankResult {
  index: number;
  relevance_score: number;
  document?: string | JsonObject | null;
  id?: string | null;
  metadata?: JsonObject | null;
  [key: string]: JsonValue | string | number | JsonObject | null | undefined;
}

export interface RerankResponse {
  model: string;
  results: RerankResult[];
  usage?: JsonObject | null;
  [key: string]: JsonValue | RerankResult[] | null | undefined;
}
