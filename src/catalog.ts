import type { ModelCapabilities, Modality, ModelConfig } from "./types.js";
import {
  DEFAULT_MODEL_CONFIGS,
  CONTRACT_ARTIFACTS,
  CONTRACT_CATALOG_REVISION,
  CONTRACT_CHECKSUMS_SHA256,
  CONTRACT_CONSUMER_LOCK_SHA256,
  CONTRACT_DEFAULT_MODELS,
  CONTRACT_FIXTURE_VERSION,
  CONTRACT_MANIFEST_SHA256,
  CONTRACT_SCHEMA_VERSION,
  CONTRACT_VERSION,
} from "./generated/contract-catalog.js";

export {
  DEFAULT_MODEL_CONFIGS,
  CONTRACT_ARTIFACTS,
  CONTRACT_CATALOG_REVISION,
  CONTRACT_CHECKSUMS_SHA256,
  CONTRACT_CONSUMER_LOCK_SHA256,
  CONTRACT_DEFAULT_MODELS,
  CONTRACT_FIXTURE_VERSION,
  CONTRACT_MANIFEST_SHA256,
  CONTRACT_SCHEMA_VERSION,
  CONTRACT_VERSION,
} from "./generated/contract-catalog.js";

// Preserve the public catalog import path.
export type { ModelConfig } from "./types.js";

/**
 * Model metadata is deliberately advisory. Unknown model IDs remain valid so
 * custom OpenAI-compatible gateways do not require a package release.
 */
/** A user-owned catalog that can be shared by chat, embedding, and rerank calls. */
export class ModelCatalog {
  private readonly entries: Map<string, ModelConfig>;

  public constructor(configs: Iterable<ModelConfig> | Record<string, ModelConfig> = []) {
    this.entries = new Map<string, ModelConfig>();
    if (Symbol.iterator in Object(configs)) {
      for (const config of configs as Iterable<ModelConfig>) {
        this.add(config);
      }
    } else {
      for (const [key, value] of Object.entries(configs)) {
        this.add({ ...value, id: value.id || key });
      }
    }
  }

  public add(config: ModelConfig): this {
    if (!config.id.trim()) {
      throw new TypeError("ModelConfig.id must not be empty");
    }
    this.entries.set(config.id, { enabled: true, ...config });
    return this;
  }

  public get(modelId: string): ModelConfig | undefined {
    return this.entries.get(modelId);
  }

  public has(modelId: string): boolean {
    return this.entries.has(modelId);
  }

  public list(): ModelConfig[] {
    return [...this.entries.values()].map((config) => ({ ...config }));
  }

  public enabled(modelId: string): boolean {
    return this.entries.get(modelId)?.enabled !== false;
  }

  public capabilities(modelId: string): ModelCapabilities | undefined {
    const config = this.entries.get(modelId);
    if (!config) {
      return undefined;
    }
    if (config.capabilities) {
      return config.capabilities;
    }
    const modalities: Modality[] = ["text"];
    if (config.native_multimodal) {
      modalities.push("image");
    }
    return {
      tools: config.function_call_available ?? false,
      structured_output: config.response_format_available ? "json_schema" : "none",
      input_modalities: modalities,
      output_modalities: ["text"],
      streaming: true,
    };
  }
}

/** Catalog generated from the pinned vv-llm-contract release. */
export const DEFAULT_MODEL_CATALOG = new ModelCatalog(DEFAULT_MODEL_CONFIGS);

/** Empty catalog for callers that explicitly opt out of defaults. */
export const EMPTY_MODEL_CATALOG = new ModelCatalog();

export function defineModel(config: ModelConfig): ModelConfig {
  return { enabled: true, ...config };
}
