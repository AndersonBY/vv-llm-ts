import { VvLlmError } from "./errors.js";

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;
export interface TransportOptions {
  apiKey?: string;
  api_key?: string;
  baseURL?: string;
  base_url?: string;
  headers?: HeadersInit;
  timeoutMs?: number;
  timeout_ms?: number;
  fetch?: FetchLike;
}

/** Shared JSON/SSE HTTP transport; each client owns its request protocol. */
export class HttpTransport {
  private readonly baseURL: string;
  private readonly apiKey?: string;
  private readonly defaultHeaders: Headers;
  private readonly timeoutMs: number;
  private readonly fetchImpl: FetchLike;

  constructor(options: TransportOptions = {}) {
    this.baseURL = (options.baseURL ?? options.base_url ?? "https://api.openai.com/v1").replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? options.api_key;
    this.defaultHeaders = new Headers(options.headers);
    this.timeoutMs = options.timeoutMs ?? options.timeout_ms ?? 60_000;
    this.fetchImpl = options.fetch ?? ((input, init) => fetch(input, init));
  }

  async send(
    path: string,
    body: Record<string, unknown>,
    request: {
      headers?: HeadersInit;
      query?: Record<string, string | number | boolean | null | undefined>;
      timeoutMs?: number;
      signal?: AbortSignal;
      accept?: string;
    },
  ): Promise<Response> {
    const url = new URL(`${this.baseURL}${path}`);
    for (const [key, value] of Object.entries(request.query ?? {})) {
      if (value !== undefined && value !== null) url.searchParams.set(key, String(value));
    }

    const headers = new Headers(this.defaultHeaders);
    if (!headers.has("accept")) headers.set("accept", request.accept ?? "application/json");
    if (!headers.has("content-type")) headers.set("content-type", "application/json");
    if (this.apiKey && !headers.has("authorization")) {
      headers.set("authorization", `Bearer ${this.apiKey}`);
    }
    new Headers(request.headers).forEach((value, key) => headers.set(key, value));

    const controller = new AbortController();
    let timeoutTriggered = false;
    const timeoutMs = request.timeoutMs ?? this.timeoutMs;
    const callerSignal = request.signal;
    const abortFromCaller = () => controller.abort(callerSignal?.reason);
    if (callerSignal) {
      if (callerSignal.aborted) {
        abortFromCaller();
      } else {
        callerSignal.addEventListener("abort", abortFromCaller, { once: true });
      }
    }
    const timeoutId = timeoutMs > 0 ? setTimeout(() => {
      timeoutTriggered = true;
      controller.abort(new DOMException("Request timed out", "TimeoutError"));
    }, timeoutMs) : undefined;

    try {
      return await this.fetchImpl(url.toString(), {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      if (timeoutTriggered) {
        throw new VvLlmError(`Request timed out after ${timeoutMs} ms`, {
          kind: "timeout",
          cause: error,
        });
      }
      if (callerSignal?.aborted && isTimeoutAbortReason(callerSignal.reason)) {
        throw new VvLlmError("Request timed out by the caller", {
          kind: "timeout",
          code: "TIMEOUT_ERR",
          cause: error,
        });
      }
      if (callerSignal?.aborted) {
        throw new VvLlmError("Request was cancelled by the caller", {
          kind: "cancelled",
          code: "ABORT_ERR",
          cause: error,
        });
      }
      throw new VvLlmError(error instanceof Error ? error.message : "Network request failed", {
        kind: "network",
        cause: error,
      });
    } finally {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
      callerSignal?.removeEventListener("abort", abortFromCaller);
    }
  }

  async readJson(response: Response): Promise<unknown> {
    const text = await response.text();
    if (!text) return {};
    try {
      return JSON.parse(text) as unknown;
    } catch (error) {
      throw new VvLlmError("Provider returned invalid JSON", {
        kind: "serialization",
        status: response.status,
        cause: error,
      });
    }
  }

  async errorFromResponse(response: Response): Promise<VvLlmError> {
    const body = await this.readJson(response);
    return VvLlmError.fromResponse(response.status, body, response.headers);
  }
}

function isTimeoutAbortReason(reason: unknown): boolean {
  return typeof reason === "object"
    && reason !== null
    && "name" in reason
    && (reason as { name?: unknown }).name === "TimeoutError";
}
