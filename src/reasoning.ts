import { isRecord, VvLlmError } from "./errors.js";
import type { CapabilityPolicy, ModelCapabilities } from "./types.js";

export function validateReasoningEffort(model: string, effort: unknown, capabilities: ModelCapabilities | undefined, policy: CapabilityPolicy = "warn"): void {
  if (effort === undefined || effort === null || policy === "passthrough") return;
  const levels = capabilities?.reasoning_efforts;
  const message = levels === undefined || levels === null
    ? "Model '" + model + "' reasoning_effort support is unknown"
    : typeof effort === "string" && (levels.includes(effort) || (Object.hasOwn(capabilities?.reasoning_effort_aliases ?? {}, effort) && levels.includes(capabilities!.reasoning_effort_aliases![effort]!)))
      ? undefined
      : "Model '" + model + "' does not support reasoning_effort='" + String(effort) + "'. Supported values: " + (levels.join(", ") || "(none)");
  if (!message) return;
  if (policy === "strict") throw new VvLlmError(message, { kind: "configuration", model });
  console.warn(message);
}

export function mergeReasoningBody(body: Record<string, unknown>, additions: Record<string, unknown>): Record<string, unknown> {
  const result = { ...body };
  for (const [key, value] of Object.entries(additions)) {
    if (Object.hasOwn(result, key) && JSON.stringify(result[key]) !== JSON.stringify(value)) {
      if (isRecord(result[key]) && isRecord(value)) {
        result[key] = mergeReasoningBody(result[key], value);
        continue;
      }
      throw new VvLlmError("Conflicting reasoning control: " + key, { kind: "configuration" });
    }
    result[key] = value;
  }
  return result;
}

export function mergeProviderBody(body: Record<string, unknown>, additions: Record<string, unknown>): Record<string, unknown> {
  let result = { ...body };
  for (const [key, value] of Object.entries(additions)) {
    if (["reasoning_effort", "reasoning", "output_config", "thinking", "google"].includes(key)) result = mergeReasoningBody(result, { [key]: value });
    else result[key] = value;
  }
  return result;
}

export function resolveReasoningEffort(effort: string | null | undefined, body: Record<string, unknown>, model?: string): string | undefined {
  if (model !== undefined && Object.hasOwn(body, "model") && body.model !== model) throw new VvLlmError("Conflicting model values", { kind: "configuration" });
  const values: unknown[] = effort == null ? [] : [effort];
  if (Object.hasOwn(body, "reasoning_effort") && body.reasoning_effort !== undefined) values.push(body.reasoning_effort);
  if (values.some((value) => typeof value !== "string" || !value.trim())) throw new VvLlmError("reasoning_effort must be a non-empty string", { kind: "configuration" });
  if (new Set(values).size > 1) throw new VvLlmError("Conflicting reasoning_effort values", { kind: "configuration" });
  const resolved = values[0] as string | undefined;
  if (resolved !== undefined) {
    for (const container of [body, body.extra_body]) {
      if (!isRecord(container) || !isRecord(container.google)) continue;
      const thinking = container.google.thinking_config;
      if (isRecord(thinking) && ["thinking_level", "thinking_budget", "thinkingLevel", "thinkingBudget"].some((key) => Object.hasOwn(thinking, key))) {
        throw new VvLlmError("reasoning_effort conflicts with Gemini thinking_level/thinking_budget", { kind: "configuration" });
      }
    }
  }
  return resolved;
}

export function normalizeGeminiBody(model: string, body: Record<string, unknown>): Record<string, unknown> {
  const match = /^gemini-(\d+)(?:[.-]|$)/i.exec(model.split("/").at(-1) ?? "");
  if (!match || Number(match[1]) < 3) return body;
  const result = { ...body };
  for (const key of ["temperature", "top_p", "top_k", "topP", "topK", "thinking_budget", "thinkingBudget"]) delete result[key];
  if (Object.hasOwn(result, "thinkingLevel")) {
    if (Object.hasOwn(result, "thinking_level") && result.thinking_level !== result.thinkingLevel) {
      throw new VvLlmError("Conflicting Gemini thinking_level values", { kind: "configuration" });
    }
    result.thinking_level = result.thinkingLevel;
    delete result.thinkingLevel;
  }
  for (const key of ["extra_body", "google", "thinking_config"]) {
    if (isRecord(result[key])) result[key] = normalizeGeminiBody(model, result[key]);
  }
  return result;
}
