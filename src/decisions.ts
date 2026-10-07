import { DEFAULT_MODEL_CATALOG, ModelCatalog } from "./catalog.js";
import { isRecord, VvLlmError } from "./errors.js";
import { HttpTransport, type TransportOptions } from "./transport.js";

export type DecisionType = "predicate" | "choice" | "score";
export type ChoiceValue = string | boolean;
export interface DecisionChoice { id: ChoiceValue; description: string; }
export interface DecisionLevel { label: string; description?: string; }
export interface PredicateQuestion { type: "predicate"; instructions: string; name?: string | null; }
export interface ChoiceQuestion { type: "choice"; instructions: string; name?: string | null; choices: readonly (ChoiceValue | DecisionChoice)[]; }
export interface ScoreQuestion { type: "score"; instructions: string; name?: string | null; rubric: string | readonly DecisionLevel[]; }
export type DecisionQuestion = PredicateQuestion | ChoiceQuestion | ScoreQuestion;
export type DecisionContent = { type: "text"; text: string } | { type: "image_url"; image_url: string; detail?: "auto" | "low" | "high" | "original" };
export interface DecisionMessage { type?: "message"; role: "user"; content: string | readonly DecisionContent[]; }
export interface DecisionRequest {
  model?: string;
  input: string | readonly DecisionMessage[];
  questions: readonly DecisionQuestion[];
  safety_identifier?: string | null;
}
export type DecisionAnswer =
  | { type: "predicate"; name?: string | null; probability: number }
  | { type: "choice"; name?: string | null; choice: ChoiceValue; options: readonly { choice: ChoiceValue; probability: number }[] }
  | { type: "score"; name?: string | null; score: number; confidence: number; probabilities: readonly { value: number; label: string; probability: number }[] }
  | { type: "refusal"; name?: string | null };
export interface DecisionUsage {
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
  input_tokens_details?: { cached_tokens?: number | null; cache_write_tokens?: number | null; [key: string]: unknown } | null;
  output_tokens_details?: { reasoning_tokens?: number | null; [key: string]: unknown } | null;
  [key: string]: unknown;
}
export interface DecisionResponse { model: string; answers: readonly DecisionAnswer[]; usage?: DecisionUsage | null; }
export interface DecisionClientOptions extends TransportOptions {
  model?: string;
  /** Provider model ID for the configured default model binding. */
  modelId?: string;
  modelCatalog?: ModelCatalog;
}
export interface DecisionTransportOptions {
  signal?: AbortSignal;
  timeout_ms?: number;
  extra_headers?: HeadersInit;
}

function check(condition: unknown, message: string): asserts condition {
  if (!condition) throw new TypeError(message);
}
function object(value: unknown): asserts value is Record<string, unknown> { check(isRecord(value), "decision value must be an object"); }
function keys(value: Record<string, unknown>, allowed: readonly string[]): void {
  check(Object.keys(value).every((key) => allowed.includes(key)), "unknown decision field");
}
function text(value: unknown): asserts value is string { check(typeof value === "string", "decision field must be a string"); }
function choiceValue(value: unknown): asserts value is ChoiceValue { check(typeof value === "string" || typeof value === "boolean", "decision choice must be a string or boolean"); }
function named(value: Record<string, unknown>): void { if (value.name != null) text(value.name); }
function list(value: unknown): asserts value is unknown[] { check(Array.isArray(value) && value.length > 0, "decision list must be non-empty"); }
function finite(value: unknown): asserts value is number { check(typeof value === "number" && Number.isFinite(value), "decision number must be finite"); }
function probability(value: unknown): void { finite(value); check(value >= 0 && value <= 1, "decision probability must be between zero and one"); }
function level(value: unknown): void {
  object(value); keys(value, ["label", "description"]); text(value.label);
  if (value.description !== undefined) text(value.description);
}

/** Decode the closed canonical request and return an isolated copy. */
export function decodeDecisionRequest(value: unknown): DecisionRequest & { model: string } {
  object(value); keys(value, ["model", "input", "questions", "safety_identifier"]);
  text(value.model); check(value.model.trim(), "decision model is required");
  if (value.safety_identifier != null) { text(value.safety_identifier); check(Array.from(value.safety_identifier).length <= 128, "safety identifier is too long"); }
  let images = 0;
  if (typeof value.input !== "string") {
    list(value.input);
    for (const message of value.input) {
      object(message); keys(message, ["type", "role", "content"]); check(message.role === "user", "decision messages must have user role");
      if (message.type !== undefined) check(message.type === "message", "unsupported decision message type");
      if (typeof message.content === "string") continue;
      list(message.content);
      for (const part of message.content) {
        object(part);
        if (part.type === "text") { keys(part, ["type", "text"]); text(part.text); }
        else {
          check(part.type === "image_url", "unsupported decision input content"); keys(part, ["type", "image_url", "detail"]); text(part.image_url);
          check(/^data:image\/[^;,]+;base64,[A-Za-z0-9+/]+={0,2}$/.test(part.image_url), "decision images must use inline base64 data URLs");
          if (part.detail !== undefined) check(["auto", "low", "high", "original"].includes(String(part.detail)), "invalid image detail");
          images++;
        }
      }
    }
  }
  check(images <= 128, "decision input supports at most 128 images");
  list(value.questions);
  const names = new Set<string>();
  for (const q of value.questions) {
    object(q); text(q.instructions); check(q.instructions.trim(), "question instructions are required"); named(q);
    if (typeof q.name === "string") { check(!names.has(q.name), "question names must be unique"); names.add(q.name); }
    if (q.type === "predicate") keys(q, ["type", "name", "instructions"]);
    else if (q.type === "choice") {
      keys(q, ["type", "name", "instructions", "choices"]); list(q.choices);
      const ids = q.choices.map((item) => {
        if (isRecord(item)) { keys(item, ["id", "description"]); choiceValue(item.id); text(item.description); return item.id; }
        choiceValue(item); return item;
      });
      check(new Set(ids).size === ids.length, "choice IDs must be unique");
    } else {
      check(q.type === "score", "unsupported decision question type"); keys(q, ["type", "name", "instructions", "rubric"]);
      if (typeof q.rubric !== "string") { list(q.rubric); q.rubric.forEach(level); }
    }
  }
  return structuredClone(value) as unknown as DecisionRequest & { model: string };
}

export function decodeDecisionResponse(value: unknown): DecisionResponse {
  object(value); keys(value, ["model", "answers", "usage"]); text(value.model); check(value.model.length > 0, "decision response model is required");
  list(value.answers);
  for (const answer of value.answers) {
    object(answer); named(answer);
    if (answer.type === "predicate") { keys(answer, ["type", "name", "probability"]); probability(answer.probability); }
    else if (answer.type === "choice") {
      keys(answer, ["type", "name", "choice", "options"]); choiceValue(answer.choice); list(answer.options);
      for (const option of answer.options) { object(option); keys(option, ["choice", "probability"]); choiceValue(option.choice); probability(option.probability); }
    } else if (answer.type === "score") {
      keys(answer, ["type", "name", "score", "confidence", "probabilities"]); finite(answer.score); probability(answer.confidence); list(answer.probabilities);
      for (const item of answer.probabilities) { object(item); keys(item, ["value", "label", "probability"]); check(Number.isSafeInteger(item.value) && (item.value as number) >= 0, "score level value must be a non-negative integer"); text(item.label); probability(item.probability); }
    } else { check(answer.type === "refusal", "unsupported decision answer type"); keys(answer, ["type", "name"]); }
  }
  if (value.usage != null) {
    object(value.usage);
    const counts = (record: Record<string, unknown>, fields: readonly string[]) => {
      for (const field of fields) if (record[field] != null) check(typeof record[field] === "number" && Number.isSafeInteger(record[field]) && (record[field] as number) >= 0, "invalid decision token count");
    };
    counts(value.usage, ["input_tokens", "output_tokens", "total_tokens"]);
    if (value.usage.input_tokens_details != null) { object(value.usage.input_tokens_details); counts(value.usage.input_tokens_details, ["cached_tokens", "cache_write_tokens"]); }
    if (value.usage.output_tokens_details != null) { object(value.usage.output_tokens_details); counts(value.usage.output_tokens_details, ["reasoning_tokens"]); }
  }
  return structuredClone(value) as unknown as DecisionResponse;
}

function validateAnswers(response: DecisionResponse, request: DecisionRequest): void {
  check(response.answers.length === request.questions.length, "decision answer count does not match questions");
  response.answers.forEach((answer, index) => {
    const question = request.questions[index]!;
    check((answer.name ?? null) === (question.name ?? null) && (answer.type === "refusal" || answer.type === question.type), "decision answer type/name does not match question");
    if (question.type === "choice" && answer.type === "choice") {
      const ids = question.choices.map((item) => typeof item === "object" ? item.id : item);
      const options = answer.options.map((option) => option.choice);
      check(options.length === ids.length && new Set(options).size === options.length && options.every((id) => ids.includes(id)) && ids.includes(answer.choice), "decision choice does not match supplied choices");
    }
    if (question.type === "score" && answer.type === "score" && typeof question.rubric !== "string") {
      const rubric = question.rubric;
      const values = answer.probabilities.map((item) => item.value);
      check(values.length === rubric.length && new Set(values).size === values.length && answer.probabilities.every((item) => rubric[item.value]?.label === item.label) && answer.score >= 0 && answer.score <= rubric.length - 1, "decision score does not match supplied rubric");
    }
  });
}

function normalizeResponse(raw: unknown): DecisionResponse {
  object(raw); list(raw.answers);
  const select = (value: unknown, fields: string[]) => { object(value); return Object.fromEntries(Object.entries(value).filter(([key]) => fields.includes(key))); };
  const answers = raw.answers.map((answer) => {
    object(answer);
    const fields = answer.type === "predicate" ? ["probability"] : answer.type === "choice" ? ["choice", "options"] : answer.type === "score" ? ["score", "confidence", "probabilities"] : [];
    const result = select(answer, ["type", "name", ...fields]);
    if (Array.isArray(result.options)) result.options = result.options.map((item) => select(item, ["choice", "probability"]));
    if (Array.isArray(result.probabilities)) result.probabilities = result.probabilities.map((item) => select(item, ["value", "label", "probability"]));
    return result;
  });
  return decodeDecisionResponse({ ...select(raw, ["model", "usage"]), answers });
}

/** A standalone decision client; thresholds and routing belong to the caller. */
export class DecisionClient {
  private readonly transport: HttpTransport;
  private readonly catalog: ModelCatalog;
  private readonly model?: string;
  private readonly modelId?: string;
  constructor(options: DecisionClientOptions = {}) {
    this.transport = new HttpTransport(options);
    this.modelId = options.modelId;
    this.catalog = options.modelCatalog ?? DEFAULT_MODEL_CATALOG;
    this.model = options.model ?? this.catalog.list().find((model) => model.capabilities?.decision_types?.length)?.id;
  }
  async create(request: DecisionRequest, options: DecisionTransportOptions = {}): Promise<DecisionResponse> {
    const resolved = decodeDecisionRequest({ ...request, model: request.model ?? this.model });
    const supported = this.catalog.get(resolved.model)?.capabilities?.decision_types;
    check(supported && resolved.questions.every((q) => supported.includes(q.type)), "selected model does not declare support for the requested decision types");
    const body = { ...resolved, model: resolved.model === this.model ? this.modelId ?? resolved.model : resolved.model };
    const response = await this.transport.send("/decisions", body as unknown as Record<string, unknown>, { signal: options.signal, timeoutMs: options.timeout_ms, headers: options.extra_headers });
    if (!response.ok) throw await this.transport.errorFromResponse(response);
    try {
      const result = normalizeResponse(await this.transport.readJson(response));
      validateAnswers(result, resolved);
      return result;
    } catch (error) {
      if (error instanceof VvLlmError) throw error;
      throw new VvLlmError("Provider returned an invalid decision response", { kind: "serialization", cause: error });
    }
  }
}

export function createDecisionClient(options: DecisionClientOptions = {}): DecisionClient { return new DecisionClient(options); }
