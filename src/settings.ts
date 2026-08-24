import { ModelCatalog, type ModelConfig } from "./catalog.js";
import type { JsonObject } from "./types.js";
import { VvLlmError } from "./errors.js";
import { type ClientOptions, VvLlmClient } from "./client.js";

export interface EndpointConfig {
  id: string;
  enabled?: boolean;
  api_base?: string | null;
  api_key?: string;
  organization?: string;
  response_api?: boolean;
  endpoint_type?: string;
  region?: string;
  is_azure?: boolean;
  is_bedrock?: boolean;
  is_vertex?: boolean;
  credentials?: JsonObject;
  rpm?: number;
  tpm?: number;
  concurrent_requests?: number;
  proxy?: string;
  headers?: Record<string, string>;
  [key: string]: unknown;
}

export interface EndpointBinding {
  endpoint_id: string;
  model_id?: string;
  enabled?: boolean;
  rpm?: number;
  tpm?: number;
  concurrent_requests?: number;
  [key: string]: unknown;
}

export type EndpointBindingInput = string | EndpointBinding;

export interface SettingsModelConfig extends ModelConfig {
  endpoints?: readonly EndpointBindingInput[];
  request_mapping?: JsonObject;
  response_mapping?: JsonObject;
  [key: string]: unknown;
}

export interface BackendConfig {
  models?: Record<string, SettingsModelConfig>;
  default_endpoint?: string | null;
  [key: string]: unknown;
}

export interface SettingsV2 {
  VERSION?: string;
  endpoints?: readonly EndpointConfig[];
  backends?: Record<string, BackendConfig>;
  embedding_backends?: Record<string, BackendConfig>;
  rerank_backends?: Record<string, BackendConfig>;
  /** Preserve unrelated settings fields when loading a shared config. */
  [key: string]: unknown;
}

export type SettingsKind = "chat" | "embedding" | "rerank";

export interface ResolvedModelConfig {
  kind: SettingsKind;
  backend: string;
  model: SettingsModelConfig;
  model_id: string;
  endpoint: EndpointConfig;
  binding: EndpointBinding;
}

export interface SettingsFactoryOptions extends Omit<ClientOptions, "baseURL" | "base_url" | "apiKey" | "api_key" | "headers" | "modelCatalog" | "model_catalog"> {
  backend: string;
  model: string;
  baseURL?: string;
  base_url?: string;
  apiKey?: string;
  api_key?: string;
  headers?: HeadersInit;
  modelCatalog?: ModelCatalog;
  model_catalog?: ModelCatalog;
}

export interface SettingsClientResult {
  client: VvLlmClient;
  resolved: ResolvedModelConfig;
}

/** Resolve a V2 model and its first enabled endpoint without making a request. */
export function resolveSettingsModel(
  settings: SettingsV2,
  kind: SettingsKind,
  backend: string,
  modelId: string,
): ResolvedModelConfig {
  const backendMap = getBackendMap(settings, kind);
  const backendConfig = backendMap[backend];
  if (!backendConfig) {
    throw new VvLlmError(`Backend '${backend}' is not configured for ${kind}`, {
      kind: "model_not_found",
      code: "BACKEND_NOT_FOUND",
    });
  }
  const models = backendConfig.models ?? {};
  const entry = models[modelId] ?? Object.values(models).find((candidate) => candidate.id === modelId);
  if (!entry || entry.enabled === false) {
    throw new VvLlmError(`Model '${modelId}' is not configured for backend '${backend}'`, {
      kind: "model_not_found",
      code: "MODEL_NOT_FOUND",
    });
  }

  const configuredBindings = entry.endpoints ?? [];
  const fallbackBindings = configuredBindings.length > 0
    ? configuredBindings
    : backendConfig.default_endpoint
      ? [backendConfig.default_endpoint]
      : [];
  const bindingInput = fallbackBindings.find((candidate) => bindingEnabled(candidate));
  if (!bindingInput) {
    throw new VvLlmError(`Model '${modelId}' has no enabled endpoint binding`, {
      kind: "configuration",
      code: "NO_ENABLED_ENDPOINT_BINDING",
    });
  }
  const binding = normalizeBinding(bindingInput);
  const endpoint = (settings.endpoints ?? []).find(
    (candidate) => candidate.id === binding.endpoint_id && candidate.enabled !== false,
  );
  if (!endpoint) {
    throw new VvLlmError(`Endpoint '${binding.endpoint_id}' is not configured or disabled`, {
      kind: "configuration",
      code: "ENDPOINT_NOT_FOUND",
    });
  }
  return {
    kind,
    backend,
    model: { ...entry },
    model_id: binding.model_id ?? entry.id,
    endpoint: { ...endpoint, headers: endpoint.headers ? { ...endpoint.headers } : undefined },
    binding,
  };
}

export function createChatClientFromSettings(
  settings: SettingsV2,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "chat", options);
}

export function createEmbeddingClientFromSettings(
  settings: SettingsV2,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "embedding", options);
}

export function createRerankClientFromSettings(
  settings: SettingsV2,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "rerank", options);
}

function createClientFromSettings(
  settings: SettingsV2,
  kind: SettingsKind,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  const resolved = resolveSettingsModel(settings, kind, options.backend, options.model);
  const endpointHeaders = resolved.endpoint.headers ?? {};
  const clientHeaders = options.headers ? new Headers(options.headers) : new Headers();
  for (const [key, value] of Object.entries(endpointHeaders)) {
    if (!clientHeaders.has(key)) clientHeaders.set(key, value);
  }
  const clientOptions: ClientOptions = {
    ...options,
    baseURL: options.baseURL ?? options.base_url ?? resolved.endpoint.api_base ?? undefined,
    apiKey: options.apiKey ?? options.api_key ?? resolved.endpoint.api_key,
    headers: clientHeaders,
    modelCatalog: options.modelCatalog ?? options.model_catalog,
  };
  delete (clientOptions as Record<string, unknown>).backend;
  delete (clientOptions as Record<string, unknown>).model;
  return { client: new VvLlmClient(clientOptions), resolved };
}

function getBackendMap(settings: SettingsV2, kind: SettingsKind): Record<string, BackendConfig> {
  if (kind === "chat") {
    if (settings.backends) return settings.backends;
    // Preserve V1 top-level backend blocks exactly as provided.
    const known = [
      "anthropic", "deepseek", "gemini", "groq", "local", "minimax", "mistral",
      "moonshot", "openai", "qwen", "yi", "zhipuai", "baichuan", "stepfun",
      "xai", "xiaomi", "ernie",
    ];
    return Object.fromEntries(
      known.flatMap((name) => {
        const value = settings[name];
        return isBackendConfig(value) ? [[name, value]] : [];
      }),
    );
  }
  return kind === "embedding"
    ? settings.embedding_backends ?? {}
    : settings.rerank_backends ?? {};
}

function isBackendConfig(value: unknown): value is BackendConfig {
  return typeof value === "object" && value !== null;
}

function bindingEnabled(value: EndpointBindingInput): boolean {
  return typeof value === "string" || value.enabled !== false;
}

function normalizeBinding(value: EndpointBindingInput): EndpointBinding {
  return typeof value === "string" ? { endpoint_id: value } : { ...value };
}
