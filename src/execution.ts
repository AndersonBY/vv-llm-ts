import { ModelCatalog } from "./catalog.js";
import {
  classifyError,
  type ErrorKind,
  VvLlmError,
} from "./errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ChatCompletionStreamParams,
  ChatMessage,
  JsonObject,
  ModelCapabilities,
} from "./types.js";

export type ChatExecutionRequest = ChatCompletionCreateParams | ChatCompletionStreamParams;
export type ChatExecutionResponse = ChatCompletion | AsyncIterable<ChatCompletionChunk>;
export type MaybePromise<T> = T | PromiseLike<T>;

export interface ChatClientLike {
  create(request: ChatExecutionRequest): Promise<ChatExecutionResponse>;
  providerName?: string;
  model?: string;
  modelCatalog?: ModelCatalog;
}

export interface ResponseMetadata {
  provider?: string;
  model?: string;
  response_id?: string;
  request_id?: string;
  finish_reason?: string;
  attempts: number;
  latency_ms?: number;
  fallback_index: number;
  attributes: JsonObject;
}

export interface CompletionResult<Response> {
  response: Response;
  metadata: ResponseMetadata;
}

export interface RetryPolicyOptions {
  maxAttempts?: number;
  max_attempts?: number;
  baseDelayMs?: number;
  base_delay_ms?: number;
  maxDelayMs?: number;
  max_delay_ms?: number;
  jitterRatio?: number;
  jitter_ratio?: number;
  totalTimeoutMs?: number;
  total_timeout_ms?: number;
  retryableKinds?: ReadonlySet<ErrorKind> | readonly ErrorKind[];
  retryable_kinds?: ReadonlySet<ErrorKind> | readonly ErrorKind[];
}

const DEFAULT_RETRYABLE_KINDS: ReadonlySet<ErrorKind> = new Set([
  "rate_limited",
  "network",
  "timeout",
  "provider_internal",
]);

export class RetryPolicy {
  public readonly maxAttempts: number;
  public readonly baseDelayMs: number;
  public readonly maxDelayMs: number;
  public readonly jitterRatio: number;
  public readonly totalTimeoutMs?: number;
  public readonly retryableKinds: ReadonlySet<ErrorKind>;

  public constructor(options: RetryPolicyOptions = {}) {
    this.maxAttempts = options.maxAttempts ?? options.max_attempts ?? 3;
    this.baseDelayMs = options.baseDelayMs ?? options.base_delay_ms ?? 500;
    this.maxDelayMs = options.maxDelayMs ?? options.max_delay_ms ?? 8_000;
    this.jitterRatio = options.jitterRatio ?? options.jitter_ratio ?? 0.2;
    this.totalTimeoutMs = options.totalTimeoutMs ?? options.total_timeout_ms;
    const kinds = options.retryableKinds ?? options.retryable_kinds ?? DEFAULT_RETRYABLE_KINDS;
    this.retryableKinds = kinds instanceof Set ? new Set(kinds) : new Set(kinds);
    if (!Number.isInteger(this.maxAttempts) || this.maxAttempts < 1) throw new RangeError("maxAttempts must be at least 1");
    if (this.baseDelayMs < 0 || this.maxDelayMs < 0) throw new RangeError("retry delays cannot be negative");
    if (this.jitterRatio < 0 || this.jitterRatio > 1) throw new RangeError("jitterRatio must be between 0 and 1");
    if (this.totalTimeoutMs !== undefined && this.totalTimeoutMs <= 0) throw new RangeError("totalTimeoutMs must be positive");
  }

  public shouldRetry(error: VvLlmError, attempt: number): boolean {
    return attempt < this.maxAttempts && this.retryableKinds.has(error.kind);
  }

  public should_retry(error: VvLlmError, attempt: number): boolean {
    return this.shouldRetry(error, attempt);
  }

  public delayFor(error: VvLlmError, attempt: number, randomValue = Math.random()): number {
    if (error.retry_after_ms !== undefined) return Math.min(this.maxDelayMs, Math.max(0, error.retry_after_ms));
    const delay = Math.min(this.maxDelayMs, this.baseDelayMs * (2 ** Math.max(0, attempt - 1)));
    const sample = Math.min(1, Math.max(0, randomValue));
    const jitter = delay * this.jitterRatio * ((sample * 2) - 1);
    return Math.max(0, delay + jitter);
  }

  public delay_for(error: VvLlmError, attempt: number, randomValue = Math.random()): number {
    return this.delayFor(error, attempt, randomValue);
  }
}

export interface ExecuteWithRetryOptions {
  provider?: string;
  model?: string;
  sleep?: (milliseconds: number) => MaybePromise<void>;
  now?: () => number;
  random?: () => number;
}

export async function executeWithRetry<T>(
  operation: () => MaybePromise<T>,
  policy = new RetryPolicy(),
  options: ExecuteWithRetryOptions = {},
): Promise<T> {
  const sleep = options.sleep ?? ((milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds)));
  const now = options.now ?? (() => Date.now());
  const random = options.random ?? (() => Math.random());
  const started = now();
  let attempt = 1;
  while (true) {
    try {
      return await operation();
    } catch (cause) {
      const error = classifyError(cause, { provider: options.provider, model: options.model });
      if (!policy.shouldRetry(error, attempt)) throw error;
      const delay = policy.delayFor(error, attempt, random());
      if (policy.totalTimeoutMs !== undefined && now() - started + delay >= policy.totalTimeoutMs) throw error;
      await sleep(delay);
      attempt += 1;
    }
  }
}

export const execute_with_retry = executeWithRetry;

export interface MiddlewareContext {
  provider?: string;
  model?: string;
  attempt: number;
  attributes: JsonObject;
}

export class ChatMiddlewareV1 {
  public readonly apiVersion = "v1" as const;
  public readonly api_version = "v1" as const;

  public onRequest(_context: MiddlewareContext, request: ChatExecutionRequest): MaybePromise<ChatExecutionRequest> {
    return request;
  }

  public onResponse(_context: MiddlewareContext, response: ChatExecutionResponse): MaybePromise<ChatExecutionResponse> {
    return response;
  }

  public onError(_context: MiddlewareContext, _error: VvLlmError): MaybePromise<void> {
    return undefined;
  }

  public onStreamStart(_context: MiddlewareContext): MaybePromise<void> {
    return undefined;
  }
}

export interface MiddlewareLike {
  apiVersion?: string;
  api_version?: string;
  onRequest?: (context: MiddlewareContext, request: ChatExecutionRequest) => MaybePromise<ChatExecutionRequest>;
  onResponse?: (context: MiddlewareContext, response: ChatExecutionResponse) => MaybePromise<ChatExecutionResponse>;
  onError?: (context: MiddlewareContext, error: VvLlmError) => MaybePromise<void>;
  onStreamStart?: (context: MiddlewareContext) => MaybePromise<void>;
}

export interface MiddlewareChatClientOptions {
  middleware?: readonly MiddlewareLike[];
  retryPolicy?: RetryPolicy;
  retry_policy?: RetryPolicy;
}

export class MiddlewareChatClient implements ChatClientLike {
  public readonly inner: ChatClientLike;
  public readonly middleware: readonly MiddlewareLike[];
  public readonly retryPolicy: RetryPolicy;
  public readonly providerName: string;

  public constructor(inner: ChatClientLike, options: MiddlewareChatClientOptions = {}) {
    this.inner = inner;
    this.middleware = options.middleware ?? [];
    this.retryPolicy = options.retryPolicy ?? options.retry_policy ?? new RetryPolicy({ maxAttempts: 1 });
    this.providerName = inner.providerName ?? "middleware";
    for (const item of this.middleware) {
      const version = item.apiVersion ?? item.api_version ?? "v1";
      if (version !== "v1") throw new TypeError(`unsupported middleware API version: ${version}`);
    }
  }

  public async create(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    const result = await this.execute(request);
    return result.response;
  }

  public async createWithMetadata(request: ChatExecutionRequest): Promise<CompletionResult<ChatExecutionResponse>> {
    const started = Date.now();
    const result = await this.execute(request);
    return {
      response: result.response,
      metadata: metadataFromResponse(result.response, result.context, {
        latency_ms: Math.max(0, Date.now() - started),
      }),
    };
  }

  public async create_with_metadata(request: ChatExecutionRequest): Promise<CompletionResult<ChatExecutionResponse>> {
    return this.createWithMetadata(request);
  }

  public async createCompletion(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    return this.create(request);
  }

  public async create_completion(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    return this.create(request);
  }

  public async createStream(request: ChatCompletionStreamParams): Promise<AsyncIterable<ChatCompletionChunk>> {
    const response = await this.create({ ...request, stream: true });
    if (!isAsyncIterable(response)) throw new VvLlmError("stream request returned a non-stream response", { kind: "serialization" });
    return response;
  }

  private async execute(request: ChatExecutionRequest): Promise<{ response: ChatExecutionResponse; context: MiddlewareContext }> {
    const context: MiddlewareContext = {
      provider: this.providerName,
      model: request.model ?? this.inner.model,
      attempt: 0,
      attributes: {},
    };
    let prepared = request;
    for (const item of this.middleware) {
      if (item.onRequest) prepared = await item.onRequest(context, prepared);
    }
    context.model = prepared.model ?? context.model;
    const response = await executeWithRetry(async () => {
      context.attempt += 1;
      try {
        let value = await this.inner.create(prepared);
        for (const item of [...this.middleware].reverse()) {
          if (item.onResponse) value = await item.onResponse(context, value);
        }
        return this.wrapStream(value, context);
      } catch (cause) {
        const error = classifyError(cause, { provider: context.provider, model: context.model });
        await this.notifyError(context, error);
        throw error;
      }
    }, this.retryPolicy, { provider: context.provider, model: context.model });
    return { response, context };
  }

  private async notifyError(context: MiddlewareContext, error: VvLlmError): Promise<void> {
    for (const item of [...this.middleware].reverse()) {
      if (item.onError) await item.onError(context, error);
    }
  }

  private wrapStream(response: ChatExecutionResponse, context: MiddlewareContext): ChatExecutionResponse {
    if (!isAsyncIterable(response)) return response;
    const middlewares = this.middleware;
    const self = this;
    return (async function* () {
      try {
        // Middleware observes stream setup before role-only or usage-only chunks.
        for (const item of middlewares) {
          if (item.onStreamStart) await item.onStreamStart(context);
        }
        for await (const chunk of response) {
          yield chunk;
        }
      } catch (cause) {
        const error = classifyError(cause, { provider: context.provider, model: context.model });
        await self.notifyError(context, error);
        throw error;
      }
    })();
  }
}

export interface ProviderRegistration {
  name: string;
  factory: () => ChatClientLike;
  capabilities: ModelCapabilities;
}

export class ProviderRegistry {
  private readonly providers = new Map<string, ProviderRegistration>();

  public register(
    name: string,
    factory: () => ChatClientLike,
    options: { capabilities: ModelCapabilities; replace?: boolean },
  ): this {
    if (this.providers.has(name) && !options.replace) throw new VvLlmError(`provider is already registered: ${name}`, { kind: "configuration", provider: name });
    this.providers.set(name, { name, factory, capabilities: options.capabilities });
    return this;
  }

  public get(name: string): ProviderRegistration {
    const registration = this.providers.get(name);
    if (!registration) throw new VvLlmError(`provider is not registered: ${name}`, { kind: "configuration", provider: name });
    return registration;
  }

  public create(name: string): ChatClientLike {
    return this.get(name).factory();
  }

  public get names(): readonly string[] {
    return [...this.providers.keys()];
  }
}

export class FallbackRoute {
  public constructor(public readonly provider: string, public readonly model: string) {
    if (!provider || !model) throw new TypeError("fallback route provider and model are required");
  }
}

export interface FallbackChatClientOptions {
  fallbackOn?: ReadonlySet<ErrorKind> | readonly ErrorKind[];
  fallback_on?: ReadonlySet<ErrorKind> | readonly ErrorKind[];
}

const DEFAULT_FALLBACK_KINDS: ReadonlySet<ErrorKind> = new Set([
  "rate_limited",
  "network",
  "timeout",
  "provider_internal",
  "model_not_found",
]);

export class FallbackChatClient implements ChatClientLike {
  public readonly registry: ProviderRegistry;
  public readonly routes: readonly FallbackRoute[];
  public readonly fallbackOn: ReadonlySet<ErrorKind>;
  public readonly providerName = "fallback";

  public constructor(registry: ProviderRegistry, routes: readonly FallbackRoute[], options: FallbackChatClientOptions = {}) {
    if (routes.length === 0) throw new TypeError("fallback routes cannot be empty");
    this.registry = registry;
    this.routes = routes;
    const kinds = options.fallbackOn ?? options.fallback_on ?? DEFAULT_FALLBACK_KINDS;
    this.fallbackOn = kinds instanceof Set ? new Set(kinds) : new Set(kinds);
  }

  public async create(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    if (request.stream === true) return (await this.prepareStream(request)).stream;
    return (await this.complete(request)).response;
  }

  public async createWithMetadata(request: ChatExecutionRequest): Promise<CompletionResult<ChatExecutionResponse>> {
    const started = Date.now();
    if (request.stream === true) {
      const prepared = await this.prepareStream(request);
      return {
        response: prepared.stream,
        metadata: metadataFromResponse(prepared.stream, {
          provider: prepared.route.provider,
          model: prepared.route.model,
          attempt: 1,
          attributes: {},
        }, {
          fallback_index: prepared.index,
          latency_ms: Math.max(0, Date.now() - started),
        }),
      };
    }
    const completed = await this.complete(request);
    return {
      response: completed.response,
      metadata: metadataFromResponse(completed.response, {
        provider: completed.route.provider,
        model: completed.route.model,
        attempt: 1,
        attributes: {},
      }, {
        fallback_index: completed.index,
        latency_ms: Math.max(0, Date.now() - started),
      }),
    };
  }

  public async create_with_metadata(request: ChatExecutionRequest): Promise<CompletionResult<ChatExecutionResponse>> {
    return this.createWithMetadata(request);
  }

  private async complete(request: ChatExecutionRequest): Promise<{ response: ChatCompletion; index: number; route: FallbackRoute }> {
    let lastError: VvLlmError | undefined;
    for (const [index, route] of this.routes.entries()) {
      const registration = this.registry.get(route.provider);
      const routed = { ...request, model: route.model, stream: false } as ChatCompletionCreateParams;
      const capabilityError = capabilityErrorFor(registration, routed);
      if (capabilityError) {
        lastError = capabilityError;
        continue;
      }
      try {
        const response = await registration.factory().create(routed);
        if (isAsyncIterable(response)) throw new VvLlmError("non-stream route returned a stream", { kind: "serialization", provider: route.provider, model: route.model });
        return { response, index, route };
      } catch (cause) {
        const error = classifyError(cause, { provider: route.provider, model: route.model });
        if (!this.fallbackOn.has(error.kind) || index === this.routes.length - 1) throw error;
        lastError = error;
      }
    }
    throw lastError ?? new VvLlmError("no fallback route was eligible", { kind: "configuration" });
  }

  private async prepareStream(request: ChatCompletionStreamParams): Promise<{ stream: AsyncIterable<ChatCompletionChunk>; index: number; route: FallbackRoute }> {
    let lastError: VvLlmError | undefined;
    for (const [index, route] of this.routes.entries()) {
      const registration = this.registry.get(route.provider);
      const routed = { ...request, model: route.model, stream: true } as ChatCompletionStreamParams;
      const capabilityError = capabilityErrorFor(registration, routed);
      if (capabilityError) {
        lastError = capabilityError;
        continue;
      }
      try {
        const response = await registration.factory().create(routed);
        if (!isAsyncIterable(response)) throw new VvLlmError("stream route returned a non-stream response", { kind: "serialization", provider: route.provider, model: route.model });
        const iterator = response[Symbol.asyncIterator]();
        const prelude: ChatCompletionChunk[] = [];
        while (true) {
          const next = await iterator.next();
          if (next.done) {
            return { stream: asyncIterableFrom(prelude), index, route };
          }
          prelude.push(next.value);
          if (isVisibleChunk(next.value)) {
            return { stream: prependAsync(prelude, iterator), index, route };
          }
        }
      } catch (cause) {
        const error = classifyError(cause, { provider: route.provider, model: route.model });
        if (!this.fallbackOn.has(error.kind) || index === this.routes.length - 1) throw error;
        lastError = error;
      }
    }
    throw lastError ?? new VvLlmError("no fallback route was eligible", { kind: "configuration" });
  }
}

export class ScriptedStream<T> implements AsyncIterable<T> {
  public readonly chunks: readonly (T | Error)[];

  public constructor(chunks: Iterable<T | Error>) {
    this.chunks = [...chunks];
  }

  public async *[Symbol.asyncIterator](): AsyncGenerator<T> {
    for (const chunk of this.chunks) {
      if (chunk instanceof Error) throw chunk;
      yield chunk;
    }
  }
}

export type ScriptedStep<T> = T | Error | ScriptedStream<T> | (() => MaybePromise<T | Error | ScriptedStream<T>>);

export interface ScriptedChatClientOptions {
  provider?: string;
  model?: string;
  capabilities?: ModelCapabilities;
  modelCatalog?: ModelCatalog;
}

export type ScriptedChatStep =
  | ChatExecutionResponse
  | Error
  | ScriptedStream<ChatCompletionChunk>
  | (() => MaybePromise<ChatExecutionResponse | Error | ScriptedStream<ChatCompletionChunk>>);

export class ScriptedChatClient implements ChatClientLike {
  public readonly providerName: string;
  public readonly model: string;
  public readonly capabilities: ModelCapabilities;
  public readonly modelCatalog?: ModelCatalog;
  private readonly steps: ScriptedChatStep[];
  private readonly seenRequests: ChatExecutionRequest[] = [];

  public constructor(steps: Iterable<ScriptedChatStep>, options: ScriptedChatClientOptions = {}) {
    this.steps = [...steps];
    this.providerName = options.provider ?? "scripted";
    this.model = options.model ?? "scripted-model";
    this.capabilities = options.capabilities ?? { tools: true, streaming: true, input_modalities: ["text", "image"], output_modalities: ["text"] };
    this.modelCatalog = options.modelCatalog;
  }

  public get requests(): readonly ChatExecutionRequest[] {
    return this.seenRequests.map((request) => cloneScriptedValue(request));
  }

  public async create(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    this.seenRequests.push(cloneScriptedValue(request));
    const step = this.steps.shift();
    if (step === undefined) throw new VvLlmError("scripted client has no remaining steps", { kind: "configuration", provider: this.providerName, model: request.model });
    const resolved = typeof step === "function" ? await step() : step;
    if (resolved instanceof Error) throw resolved;
    return resolved;
  }

  public async createCompletion(request: ChatExecutionRequest): Promise<ChatExecutionResponse> {
    return this.create(request);
  }
}

function capabilityErrorFor(registration: ProviderRegistration, request: ChatExecutionRequest): VvLlmError | undefined {
  const capabilities = registration.capabilities;
  const conflicts: string[] = [];
  if (request.tools && request.tools.length > 0 && capabilities.tools === false) conflicts.push("the model does not support tools");
  if (request.response_format !== undefined && capabilities.structured_output === "none") conflicts.push("the model does not support structured output");
  if (request.stream === true && capabilities.streaming === false) conflicts.push("the model does not support streaming");
  if (request.messages.some(messageHasImage) && !capabilities.input_modalities?.includes("image")) conflicts.push("the model does not support image input");
  const thinking = thinkingConstraint(request.thinking);
  if (thinking.configured) {
    const capability = capabilities.thinking ?? "unknown";
    if (capability === "unsupported") conflicts.push("the model does not support thinking controls");
    if (capability === "unknown") conflicts.push("the model's thinking capability is unknown");
    if (capability === "always_enabled" && thinking.disabled) conflicts.push("thinking is always enabled for this model");
  }
  if (conflicts.length === 0) return undefined;
  return new VvLlmError(conflicts.join("; "), { kind: "configuration", provider: registration.name, model: request.model });
}

function thinkingConstraint(value: ChatExecutionRequest["thinking"]): { configured: boolean; disabled: boolean } {
  if (value === undefined || value === null) return { configured: false, disabled: false };
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    if (record.mode === "default") return { configured: false, disabled: false };
    if (record.mode === "disabled" || record.type === "disabled") return { configured: true, disabled: true };
  }
  return { configured: true, disabled: false };
}

function messageHasImage(message: ChatMessage): boolean {
  return Array.isArray(message.content) && message.content.some((part) => part.type === "image_url");
}

function isAsyncIterable(value: unknown): value is AsyncIterable<ChatCompletionChunk> {
  return typeof value === "object" && value !== null && Symbol.asyncIterator in value;
}

function isVisibleChunk(value: unknown): value is ChatCompletionChunk {
  if (!isAsyncChunk(value)) return false;
  const choice = value.choices[0];
  if (!choice) return false;
  const delta = choice.delta;
  return Boolean(
    (typeof delta.content === "string" && delta.content.length > 0)
    || (typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0)
    || (delta.tool_calls && delta.tool_calls.length > 0),
  );
}

function isAsyncChunk(value: unknown): value is ChatCompletionChunk {
  return typeof value === "object" && value !== null && Array.isArray((value as { choices?: unknown }).choices);
}

/** Clone scripted requests without JSON-stringifying runtime values such as AbortSignal. */
function cloneScriptedValue<T>(value: T, seen = new WeakMap<object, unknown>()): T {
  if (value === null || typeof value !== "object") return value;
  if (typeof AbortSignal !== "undefined" && value instanceof AbortSignal) return value;
  if (value instanceof Headers) return new Headers(value) as T;
  if (value instanceof Date) return new Date(value.getTime()) as T;

  const existing = seen.get(value);
  if (existing !== undefined) return existing as T;
  if (Array.isArray(value)) {
    const result: unknown[] = [];
    seen.set(value, result);
    for (const item of value) result.push(cloneScriptedValue(item, seen));
    return result as T;
  }
  if (value instanceof Map) {
    const result = new Map<unknown, unknown>();
    seen.set(value, result);
    for (const [key, item] of value) result.set(cloneScriptedValue(key, seen), cloneScriptedValue(item, seen));
    return result as T;
  }
  if (value instanceof Set) {
    const result = new Set<unknown>();
    seen.set(value, result);
    for (const item of value) result.add(cloneScriptedValue(item, seen));
    return result as T;
  }

  const result = Object.create(Object.getPrototypeOf(value)) as Record<PropertyKey, unknown>;
  seen.set(value, result);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !("value" in descriptor)) continue;
    result[key] = cloneScriptedValue(descriptor.value, seen);
  }
  return result as T;
}

async function* prependAsync<T>(prefix: readonly T[], iterator: AsyncIterator<T>): AsyncGenerator<T> {
  yield* prefix;
  while (true) {
    const next = await iterator.next();
    if (next.done) return;
    yield next.value;
  }
}

async function* asyncIterableFrom<T>(values: readonly T[]): AsyncGenerator<T> {
  yield* values;
}

function metadataFromResponse(
  response: unknown,
  context: MiddlewareContext,
  overrides: Partial<ResponseMetadata> = {},
): ResponseMetadata {
  const object = response && typeof response === "object" ? response as Record<string, unknown> : {};
  const choices = Array.isArray(object.choices) ? object.choices : [];
  const finishReason = choices[0] && typeof choices[0] === "object" ? (choices[0] as Record<string, unknown>).finish_reason : undefined;
  return {
    provider: context.provider,
    model: context.model,
    response_id: typeof object.id === "string" ? object.id : undefined,
    request_id: typeof object.request_id === "string" ? object.request_id : undefined,
    finish_reason: typeof finishReason === "string" ? finishReason : undefined,
    attempts: Math.max(1, context.attempt),
    latency_ms: overrides.latency_ms,
    fallback_index: overrides.fallback_index ?? 0,
    attributes: { ...context.attributes, ...(overrides.attributes ?? {}) },
  };
}
