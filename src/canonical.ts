import type {
  CanonicalChatMessage,
  CanonicalChatRole,
  CanonicalChatTool,
  CanonicalContentPart,
  CanonicalFlatImageUrlContentPart,
  CanonicalImageUrlContentPart,
  CanonicalThinkingPreference,
  CanonicalThinkingType,
  CanonicalToolCall,
  ChatRequest,
  ChatRequestOptions,
  ImageUrlContentPart,
  JsonObject,
  JsonValue,
} from "./types.js";

const EXTENSION_KEY = /^x_[a-z][a-z0-9_]*$/;

const REQUEST_KEYS = new Set(["model", "messages", "options", "tools", "tool_choice", "extra_body"]);
const MESSAGE_KEYS = new Set(["role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"]);
const TEXT_PART_KEYS = new Set(["type", "text", "cache_control"]);
const FLAT_IMAGE_PART_KEYS = new Set(["type", "url", "detail", "cache_control"]);
const NESTED_IMAGE_PART_KEYS = new Set(["type", "image_url", "cache_control"]);
const NESTED_IMAGE_URL_KEYS = new Set(["url", "detail"]);
const TOOL_CALL_KEYS = new Set(["id", "name", "arguments", "index", "extra_content"]);
const TOOL_KEYS = new Set(["name", "description", "parameters", "cache_control"]);
const THINKING_KEYS = new Set(["type", "budget_tokens", "value"]);
const OPTION_KEYS = new Set([
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
]);

const ROLES = new Set<CanonicalChatRole>(["system", "user", "assistant", "tool"]);
const IMAGE_DETAILS = new Set(["auto", "low", "high"]);
const THINKING_TYPES = new Set<CanonicalThinkingType>([
  "default",
  "enabled",
  "disabled",
  "adaptive",
  "provider_defined",
]);

/** A stable, path-aware error raised when untrusted canonical JSON is invalid. */
export class ChatRequestDecodeError extends TypeError {
  public readonly path: string;

  public constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "ChatRequestDecodeError";
    this.path = path;
  }
}

/**
 * Validate and deep-clone an unknown canonical request.
 *
 * The contract objects are closed except for lower-case `x_*` extensions. The
 * returned object never shares mutable arrays/objects with the input.
 */
export function decodeChatRequest(value: unknown): ChatRequest {
  const input = objectValue(value, "request");
  validateKeys(input, REQUEST_KEYS, "request");

  const result: ChatRequest = {
    model: requiredString(input, "model", "request", true, true),
    messages: requiredArray(input, "messages", "request").map((item, index) => decodeMessage(item, `request.messages[${index}]`)),
  };
  assignOptional(result, "options", input, "options", decodeOptions, "request");
  if (hasValue(input, "tools")) {
    result.tools = arrayValue(input.tools, "request.tools").map((item, index) => decodeTool(item, `request.tools[${index}]`));
  }
  if (hasValue(input, "tool_choice")) result.tool_choice = decodeToolChoice(input.tool_choice, "request.tool_choice");
  if (hasValue(input, "extra_body")) result.extra_body = jsonObjectValue(input.extra_body, "request.extra_body");
  copyExtensions(input, result, "request");
  return result;
}

function decodeMessage(value: unknown, path: string): CanonicalChatMessage {
  const input = objectValue(value, path);
  validateKeys(input, MESSAGE_KEYS, path);
  const role = requiredString(input, "role", path, false);
  if (!ROLES.has(role as CanonicalChatRole)) fail(`${path}.role`, "must be system, user, assistant, or tool");
  if (!hasValue(input, "content")) fail(`${path}.content`, "is required");

  const result: CanonicalChatMessage = {
    role: role as CanonicalChatRole,
    content: decodeContent(input.content, `${path}.content`),
  };
  assignOptional(result, "name", input, "name", (item, itemPath) => stringValue(item, itemPath, false), path);
  assignOptional(result, "tool_call_id", input, "tool_call_id", (item, itemPath) => stringValue(item, itemPath, false), path);
  if (hasValue(input, "tool_calls")) {
    result.tool_calls = arrayValue(input.tool_calls, `${path}.tool_calls`)
      .map((item, index) => decodeToolCall(item, `${path}.tool_calls[${index}]`));
  }
  assignOptional(
    result,
    "reasoning_content",
    input,
    "reasoning_content",
    (item, itemPath) => stringValue(item, itemPath, false),
    path,
  );
  copyExtensions(input, result, path);
  return result;
}

function decodeContent(value: unknown, path: string): string | CanonicalContentPart[] {
  if (typeof value === "string") return value;
  return arrayValue(value, path).map((item, index) => decodeContentPart(item, `${path}[${index}]`));
}

function decodeContentPart(value: unknown, path: string): CanonicalContentPart {
  const input = objectValue(value, path);
  const type = requiredString(input, "type", path, false);
  if (type === "text") {
    validateKeys(input, TEXT_PART_KEYS, path);
    const result: CanonicalContentPart = {
      type: "text",
      text: requiredString(input, "text", path, false),
    };
    if (hasValue(input, "cache_control")) result.cache_control = cloneJson(input.cache_control, `${path}.cache_control`);
    copyExtensions(input, result, path);
    return result;
  }
  if (type !== "image_url") fail(`${path}.type`, "must be text or image_url");

  const hasFlatUrl = hasValue(input, "url");
  const hasNestedUrl = hasValue(input, "image_url");
  if (hasFlatUrl === hasNestedUrl) fail(path, "image_url content must use exactly one of url or image_url");
  if (hasFlatUrl) {
    validateKeys(input, FLAT_IMAGE_PART_KEYS, path);
    const result: CanonicalFlatImageUrlContentPart = {
      type: "image_url",
      url: requiredString(input, "url", path, true),
    };
    assignImageDetail(result, input.detail, `${path}.detail`);
    if (hasValue(input, "cache_control")) result.cache_control = cloneJson(input.cache_control, `${path}.cache_control`);
    copyExtensions(input, result, path);
    return result;
  }

  validateKeys(input, NESTED_IMAGE_PART_KEYS, path);
  const image = objectValue(input.image_url, `${path}.image_url`);
  validateKeys(image, NESTED_IMAGE_URL_KEYS, `${path}.image_url`);
  const nestedImage: ImageUrlContentPart["image_url"] & Record<string, JsonValue> = {
    url: requiredString(image, "url", `${path}.image_url`, true),
  };
  assignImageDetail(nestedImage, image.detail, `${path}.image_url.detail`);
  copyExtensions(image, nestedImage, `${path}.image_url`);
  const nested: CanonicalImageUrlContentPart = {
    type: "image_url",
    image_url: nestedImage,
  };
  if (hasValue(input, "cache_control")) nested.cache_control = cloneJson(input.cache_control, `${path}.cache_control`);
  copyExtensions(input, nested, path);
  return nested;
}

function decodeToolCall(value: unknown, path: string): CanonicalToolCall {
  const input = objectValue(value, path);
  validateKeys(input, TOOL_CALL_KEYS, path);
  const result: CanonicalToolCall = {
    id: requiredString(input, "id", path, true),
    name: requiredString(input, "name", path, true),
    arguments: requiredString(input, "arguments", path, false),
  };
  if (hasValue(input, "index")) result.index = integerValue(input.index, `${path}.index`, 0);
  if (hasValue(input, "extra_content")) result.extra_content = cloneJson(input.extra_content, `${path}.extra_content`);
  copyExtensions(input, result, path);
  return result;
}

function decodeTool(value: unknown, path: string): CanonicalChatTool {
  const input = objectValue(value, path);
  validateKeys(input, TOOL_KEYS, path);
  const result: CanonicalChatTool = {
    name: requiredString(input, "name", path, true),
    parameters: requiredJson(input, "parameters", path),
  };
  assignOptional(result, "description", input, "description", (item, itemPath) => stringValue(item, itemPath, false), path);
  if (hasValue(input, "cache_control")) result.cache_control = cloneJson(input.cache_control, `${path}.cache_control`);
  copyExtensions(input, result, path);
  return result;
}

function decodeToolChoice(value: unknown, path: string): "none" | "auto" | "required" | JsonObject {
  if (typeof value === "string") {
    if (value === "none" || value === "auto" || value === "required") return value;
    fail(path, "must be none, auto, or required when represented as a string");
  }
  return jsonObjectValue(value, path);
}

function decodeOptions(value: unknown, path: string): ChatRequestOptions {
  const input = objectValue(value, path);
  validateKeys(input, OPTION_KEYS, path);
  const result: ChatRequestOptions = {};
  assignNumber(result, "temperature", input, path, 0);
  assignInteger(result, "max_tokens", input, path, 0);
  assignInteger(result, "max_completion_tokens", input, path, 0);
  assignBoolean(result, "stream", input, path);
  assignNumber(result, "top_p", input, path, 0);
  if (hasValue(input, "stop")) {
    result.stop = typeof input.stop === "string"
      ? input.stop
      : arrayValue(input.stop, `${path}.stop`).map((item, index) => stringValue(item, `${path}.stop[${index}]`, false));
  }
  assignJson(result, "response_format", input, path);
  assignJson(result, "stream_options", input, path);
  assignJson(result, "audio", input, path);
  assignNumber(result, "frequency_penalty", input, path);
  if (hasValue(input, "logit_bias")) result.logit_bias = integerMap(input.logit_bias, `${path}.logit_bias`);
  assignBoolean(result, "logprobs", input, path);
  assignJson(result, "max_tokens_details", input, path);
  assignJson(result, "metadata", input, path);
  if (hasValue(input, "modalities")) {
    result.modalities = arrayValue(input.modalities, `${path}.modalities`)
      .map((item, index) => stringValue(item, `${path}.modalities[${index}]`, false));
  }
  assignInteger(result, "n", input, path, 1);
  assignBoolean(result, "parallel_tool_calls", input, path);
  assignJson(result, "prediction", input, path);
  assignNumber(result, "presence_penalty", input, path);
  assignOptional(result, "reasoning_effort", input, "reasoning_effort", (item, itemPath) => stringValue(item, itemPath, false), path);
  if (hasValue(input, "thinking")) result.thinking = decodeThinking(input.thinking, `${path}.thinking`);
  assignInteger(result, "seed", input, path);
  assignOptional(result, "service_tier", input, "service_tier", (item, itemPath) => stringValue(item, itemPath, false), path);
  assignBoolean(result, "store", input, path);
  assignInteger(result, "top_logprobs", input, path, 0);
  assignOptional(result, "user", input, "user", (item, itemPath) => stringValue(item, itemPath, false), path);
  copyExtensions(input, result, path);
  return result;
}

function decodeThinking(value: unknown, path: string): CanonicalThinkingPreference {
  const input = objectValue(value, path);
  validateKeys(input, THINKING_KEYS, path);
  const type = requiredString(input, "type", path, false);
  if (!THINKING_TYPES.has(type as CanonicalThinkingType)) fail(`${path}.type`, "has an unsupported thinking mode");
  const result: CanonicalThinkingPreference = { type: type as CanonicalThinkingType };
  if (hasValue(input, "budget_tokens")) result.budget_tokens = integerValue(input.budget_tokens, `${path}.budget_tokens`, 1);
  if (hasValue(input, "value")) result.value = cloneJson(input.value, `${path}.value`);
  copyExtensions(input, result, path);
  return result;
}

function assignImageDetail(target: { detail?: "auto" | "low" | "high" }, value: unknown, path: string): void {
  if (value === undefined) return;
  const detail = stringValue(value, path, false);
  if (!IMAGE_DETAILS.has(detail)) fail(path, "must be auto, low, or high");
  target.detail = detail as "auto" | "low" | "high";
}

function assignNumber(target: object, key: string, source: Record<string, unknown>, path: string, minimum?: number): void {
  if (hasValue(source, key)) (target as Record<string, unknown>)[key] = numberValue(source[key], `${path}.${key}`, minimum);
}

function assignInteger(target: object, key: string, source: Record<string, unknown>, path: string, minimum?: number): void {
  if (hasValue(source, key)) (target as Record<string, unknown>)[key] = integerValue(source[key], `${path}.${key}`, minimum);
}

function assignBoolean(target: object, key: string, source: Record<string, unknown>, path: string): void {
  if (hasValue(source, key)) {
    if (typeof source[key] !== "boolean") fail(`${path}.${key}`, "must be a boolean");
    (target as Record<string, unknown>)[key] = source[key];
  }
}

function assignJson(target: object, key: string, source: Record<string, unknown>, path: string): void {
  if (hasValue(source, key)) (target as Record<string, unknown>)[key] = cloneJson(source[key], `${path}.${key}`);
}

function assignOptional<T extends object, K extends keyof T>(
  target: T,
  targetKey: K,
  source: Record<string, unknown>,
  sourceKey: string,
  decode: (value: unknown, path: string) => T[K],
  path: string,
): void {
  if (hasValue(source, sourceKey)) target[targetKey] = decode(source[sourceKey], `${path}.${sourceKey}`);
}

function integerMap(value: unknown, path: string): JsonObject {
  const input = objectValue(value, path);
  const result: JsonObject = {};
  for (const [key, item] of Object.entries(input)) result[key] = integerValue(item, `${path}.${key}`);
  return result;
}

function requiredJson(source: Record<string, unknown>, key: string, path: string): JsonValue {
  if (!hasValue(source, key)) fail(`${path}.${key}`, "is required");
  return cloneJson(source[key], `${path}.${key}`);
}

function requiredArray(source: Record<string, unknown>, key: string, path: string): unknown[] {
  if (!hasValue(source, key)) fail(`${path}.${key}`, "is required");
  return arrayValue(source[key], `${path}.${key}`);
}

function requiredString(
  source: Record<string, unknown>,
  key: string,
  path: string,
  nonEmpty: boolean,
  nonBlank = false,
): string {
  if (!hasValue(source, key)) fail(`${path}.${key}`, "is required");
  return stringValue(source[key], `${path}.${key}`, nonEmpty, nonBlank);
}

function stringValue(value: unknown, path: string, nonEmpty: boolean, nonBlank = false): string {
  if (typeof value !== "string" || (nonEmpty && (nonBlank ? value.trim().length === 0 : value.length === 0))) {
    fail(path, nonEmpty ? "must be a non-empty string" : "must be a string");
  }
  return value;
}

function numberValue(value: unknown, path: string, minimum?: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || (minimum !== undefined && value < minimum)) {
    fail(path, minimum === undefined ? "must be a finite number" : `must be a finite number >= ${minimum}`);
  }
  return value;
}

function integerValue(value: unknown, path: string, minimum?: number): number {
  if (!Number.isInteger(value) || (minimum !== undefined && (value as number) < minimum)) {
    fail(path, minimum === undefined ? "must be an integer" : `must be an integer >= ${minimum}`);
  }
  return value as number;
}

function arrayValue(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(path, "must be an array");
  return value;
}

function objectValue(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(path, "must be an object");
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) fail(path, "must be a plain object");
  return value as Record<string, unknown>;
}

function jsonObjectValue(value: unknown, path: string): JsonObject {
  return cloneJson(objectValue(value, path), path) as JsonObject;
}

function cloneJson(value: unknown, path: string): JsonValue {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return numberValue(value, path);
  if (Array.isArray(value)) return value.map((item, index) => cloneJson(item, `${path}[${index}]`));
  const input = objectValue(value, path);
  const result: JsonObject = {};
  for (const [key, item] of Object.entries(input)) result[key] = cloneJson(item, `${path}.${key}`);
  return result;
}

function hasValue(source: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(source, key) && source[key] !== undefined;
}

function validateKeys(source: Record<string, unknown>, allowed: ReadonlySet<string>, path: string): void {
  for (const key of Object.keys(source)) {
    if (allowed.has(key)) continue;
    if (key.startsWith("x_") && !EXTENSION_KEY.test(key)) fail(`${path}.${key}`, "has an invalid extension key");
    if (!key.startsWith("x_")) fail(`${path}.${key}`, "is not allowed");
  }
}

function copyExtensions(source: Record<string, unknown>, target: object, path: string): void {
  const output = target as Record<string, JsonValue>;
  for (const key of Object.keys(source)) {
    if (!key.startsWith("x_")) continue;
    if (!EXTENSION_KEY.test(key)) fail(`${path}.${key}`, "has an invalid extension key");
    output[key] = cloneJson(source[key], `${path}.${key}`);
  }
}

function fail(path: string, message: string): never {
  throw new ChatRequestDecodeError(path, message);
}
