import type { JsonValue } from "./types.js";

export type ErrorKind =
  | "authentication"
  | "rate_limited"
  | "network"
  | "timeout"
  | "invalid_request"
  | "context_length"
  | "content_policy"
  | "model_not_found"
  | "provider_internal"
  | "serialization"
  | "configuration"
  | "cancelled"
  | "unknown";

export interface VvLlmErrorOptions {
  kind?: ErrorKind;
  status?: number;
  code?: string;
  provider?: string;
  model?: string;
  request_id?: string;
  retry_after_ms?: number;
  details?: JsonValue;
  cause?: unknown;
}

/** Stable error surface for transport, provider, and request failures. */
export class VvLlmError extends Error {
  public readonly kind: ErrorKind;
  public readonly status?: number;
  public readonly code?: string;
  public readonly provider?: string;
  public readonly model?: string;
  public readonly request_id?: string;
  public readonly retry_after_ms?: number;
  public readonly details?: JsonValue;

  public constructor(message: string, options: VvLlmErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = "VvLlmError";
    this.kind = options.kind ?? "unknown";
    this.status = options.status;
    this.code = options.code;
    this.provider = options.provider;
    this.model = options.model;
    this.request_id = options.request_id;
    this.retry_after_ms = options.retry_after_ms;
    this.details = options.details;
    Object.setPrototypeOf(this, new.target.prototype);
  }

  public static fromResponse(
    status: number,
    body: unknown,
    headers?: Headers,
  ): VvLlmError {
    const object = isRecord(body) ? body : undefined;
    const nested = object && isRecord(object.error) ? object.error : object;
    const message = typeof nested?.message === "string" ? nested.message : `HTTP ${status}`;
    const code = typeof nested?.code === "string" ? nested.code : undefined;
    const requestId = headers?.get("x-request-id") ?? headers?.get("request-id") ?? undefined;
    return new VvLlmError(message, {
      kind: classifyHttpStatus(status, code),
      status,
      code,
      request_id: requestId,
      retry_after_ms: parseRetryAfterHeaders(
        headers?.get("retry-after-ms"),
        headers?.get("retry-after"),
      ),
      details: isJsonValue(body) ? body : undefined,
    });
  }
}

export function classifyHttpStatus(status: number, code?: string): ErrorKind {
  const normalized = code?.toLowerCase() ?? "";
  if (status === 401 || status === 403) return "authentication";
  if (status === 404 || normalized.includes("model_not_found")) return "model_not_found";
  if (status === 408 || status === 504) return "timeout";
  if (status === 429) return "rate_limited";
  if (status === 409 || status === 422 || (status >= 400 && status < 500)) {
    if (normalized.includes("context") || normalized.includes("length")) return "context_length";
    if (normalized.includes("content") || normalized.includes("policy")) return "content_policy";
    return "invalid_request";
  }
  if (status >= 500) return "provider_internal";
  return "unknown";
}

export interface ErrorContext {
  provider?: string;
  model?: string;
}

/** Normalize arbitrary provider/client failures into the stable VvLlmError surface. */
export function classifyError(error: unknown, context: ErrorContext = {}): VvLlmError {
  if (error instanceof VvLlmError) {
    if (error.provider === context.provider && error.model === context.model) return error;
    return new VvLlmError(error.message, {
      kind: error.kind,
      status: error.status,
      code: error.code,
      provider: context.provider ?? error.provider,
      model: context.model ?? error.model,
      request_id: error.request_id,
      retry_after_ms: error.retry_after_ms,
      details: error.details,
      cause: error,
    });
  }
  if (error instanceof DOMException && error.name === "AbortError") {
    return new VvLlmError("Request was cancelled", {
      kind: "cancelled",
      code: "ABORT_ERR",
      provider: context.provider,
      model: context.model,
      cause: error,
    });
  }
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return new VvLlmError("Request timed out", {
      kind: "timeout",
      provider: context.provider,
      model: context.model,
      cause: error,
    });
  }
  const message = error instanceof Error ? error.message : String(error);
  return new VvLlmError(message || "Unknown LLM failure", {
    kind: error instanceof TypeError ? "network" : "unknown",
    provider: context.provider,
    model: context.model,
    cause: error,
  });
}

/** Parse the standard `Retry-After` seconds or HTTP-date value into milliseconds. */
export function parseRetryAfter(value: string | null | undefined, now = Date.now()): number | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const numeric = Number(trimmed);
  if (Number.isFinite(numeric)) {
    return Math.max(0, numeric * 1000);
  }
  const date = Date.parse(trimmed);
  if (!Number.isFinite(date)) return undefined;
  return Math.max(0, date - now);
}

/**
 * Apply the vv-llm retry-header contract: valid `retry-after-ms` wins and is
 * interpreted as milliseconds; an invalid millisecond value falls back to
 * numeric seconds or an HTTP date in `Retry-After`.
 */
export function parseRetryAfterHeaders(
  retryAfterMs: string | null | undefined,
  retryAfter: string | null | undefined,
  now = Date.now(),
): number | undefined {
  if (retryAfterMs !== null && retryAfterMs !== undefined && retryAfterMs.trim() !== "") {
    const milliseconds = Number(retryAfterMs.trim());
    if (Number.isFinite(milliseconds)) return Math.max(0, milliseconds);
  }
  return parseRetryAfter(retryAfter, now);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return true;
  }
  if (Array.isArray(value)) return value.every(isJsonValue);
  if (isRecord(value)) return Object.values(value).every(isJsonValue);
  return false;
}
