import { DEFAULT_MODEL_CATALOG, DEFAULT_MODEL_CONFIGS, ModelCatalog, type ModelConfig } from "./catalog.js";
import type { JsonObject, ModelCapabilities } from "./types.js";
import { VvLlmError } from "./errors.js";
import { type ClientOptions, VvLlmClient } from "./client.js";

const CHAT_BACKENDS = [
  "anthropic", "deepseek", "gemini", "groq", "local", "minimax", "mistral",
  "moonshot", "openai", "qwen", "yi", "zhipuai", "baichuan", "stepfun",
  "xai", "xiaomi", "ernie",
] as const;

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
  capabilities?: ModelCapabilities;
  endpoint_id: string;
  model_id?: string;
  priority?: number;
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

export interface Settings {
  /** Optional wire-format metadata preserved from shared settings JSON. */
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

/** Return endpoint bindings in stable ascending-priority order. */
export function orderEndpoints(
  endpoints: readonly EndpointBindingInput[],
  preferredEndpointId?: string,
): EndpointBindingInput[] {
  return endpoints
    .map((endpoint) => ({
      endpoint,
      priority: typeof endpoint === "string" ? 1 : endpoint.priority ?? 1,
      preferred: endpointIdOf(endpoint) === preferredEndpointId,
    }))
    .sort((left, right) => left.priority - right.priority || Number(right.preferred) - Number(left.preferred))
    .map(({ endpoint }) => endpoint);
}

/** Resolve a settings model and its first enabled endpoint without making a request. */
export function resolveSettingsModel(
  settings: Settings,
  kind: SettingsKind,
  backend: string,
  modelId: string,
  modelCatalog?: ModelCatalog,
): ResolvedModelConfig {
  validateSettings(settings);
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

  const catalogModel = modelCatalog?.get(modelId) ?? modelCatalog?.get(entry.id);
  const defaults = kind === "chat" && modelCatalog === undefined ? DEFAULT_MODEL_CONFIGS.find((model) => model.backend === backend && (model.id === modelId || model.id === entry.id))?.capabilities : undefined;
  const reasoningDefaults = defaults ? { thinking: defaults.thinking, reasoning_efforts: defaults.reasoning_efforts, reasoning_effort_aliases: defaults.reasoning_effort_aliases } : {};
  const configuredBindings = entry.endpoints ?? [];
  const fallbackBindings = configuredBindings.length > 0
    ? configuredBindings
    : backendConfig.default_endpoint
      ? [backendConfig.default_endpoint]
      : [];
  const bindingInput = orderEndpoints(
    fallbackBindings.filter((candidate) => bindingEnabled(candidate)).filter((candidate) => {
      const candidateEndpointId = endpointIdOf(candidate);
      return (settings.endpoints ?? []).some(
        (endpoint) => endpoint.id === candidateEndpointId && endpoint.enabled !== false,
      );
    }),
  )[0];
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
  const model = { ...catalogModel, ...entry, capabilities: { ...reasoningDefaults, ...catalogModel?.capabilities, ...entry.capabilities, ...binding.capabilities } };
  return {
    kind,
    backend,
    model: { ...model, capabilities: new ModelCatalog([model]).capabilities(model.id) },
    model_id: binding.model_id ?? entry.id,
    endpoint: { ...endpoint, headers: endpoint.headers ? { ...endpoint.headers } : undefined },
    binding,
  };
}

export function createChatClientFromSettings(
  settings: Settings,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "chat", options);
}

export function createEmbeddingClientFromSettings(
  settings: Settings,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "embedding", options);
}

export function createRerankClientFromSettings(
  settings: Settings,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  return createClientFromSettings(settings, "rerank", options);
}

function createClientFromSettings(
  settings: Settings,
  kind: SettingsKind,
  options: SettingsFactoryOptions,
): SettingsClientResult {
  const resolved = resolveSettingsModel(settings, kind, options.backend, options.model, options.modelCatalog ?? options.model_catalog);
  if (kind === "chat" && resolved.endpoint.response_api) {
    throw new VvLlmError("Responses endpoints are not supported by the TypeScript OpenAI-compatible client", { kind: "configuration" });
  }
  const catalog = new ModelCatalog((options.modelCatalog ?? options.model_catalog ?? DEFAULT_MODEL_CATALOG).list());
  catalog.add({ ...resolved.model, id: resolved.model_id });
  catalog.add({ ...resolved.model, id: options.model });
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
    modelCatalog: catalog,
  };
  delete (clientOptions as Record<string, unknown>).backend;
  delete (clientOptions as Record<string, unknown>).model;
  return { client: new VvLlmClient(clientOptions), resolved };
}

function getBackendMap(settings: Settings, kind: SettingsKind): Record<string, BackendConfig> {
  const legacyBackend = CHAT_BACKENDS.find((backend) => Object.hasOwn(settings, backend));
  if (legacyBackend) {
    throw new VvLlmError(
      `Top-level provider setting '${legacyBackend}' is unsupported; use 'backends.${legacyBackend}'`,
      { kind: "configuration", code: "UNSUPPORTED_SETTINGS_SHAPE" },
    );
  }
  if (kind === "chat") {
    return settings.backends ?? {};
  }
  return kind === "embedding"
    ? settings.embedding_backends ?? {}
    : settings.rerank_backends ?? {};
}

function bindingEnabled(value: EndpointBindingInput): boolean {
  return typeof value === "string" || value.enabled !== false;
}

function normalizeBinding(value: EndpointBindingInput): EndpointBinding {
  return typeof value === "string" ? { endpoint_id: value } : { ...value };
}

function endpointIdOf(value: EndpointBindingInput): string {
  return typeof value === "string" ? value : value.endpoint_id;
}

function validateSettings(settings: Settings): void {
  if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
    throw new TypeError("settings must be an object");
  }
  for (const backendMap of [settings.backends, settings.embedding_backends, settings.rerank_backends]) {
    for (const backend of Object.values(backendMap ?? {})) {
      for (const model of Object.values(backend.models ?? {})) {
        for (const binding of model.endpoints ?? []) {
          if (typeof binding === "string" || binding.priority === undefined) continue;
          if (!Number.isInteger(binding.priority) || binding.priority < 1) {
            throw new TypeError("endpoint binding priority must be a strict integer >= 1");
          }
        }
      }
    }
  }
}
