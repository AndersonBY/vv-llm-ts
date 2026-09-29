#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const enabled = process.env.VV_LLM_RUN_LIVE_TESTS === "1" || process.argv.includes("--run");
if (!enabled) {
  console.log("live smoke skipped; set VV_LLM_RUN_LIVE_TESTS=1 (and a key/settings source) to opt in");
  process.exit(0);
}

const { VvLlmClient, createChatClientFromSettings } = await import("../dist/index.js");

function parseSettings() {
  const rawPath = process.env.VV_LLM_SETTINGS_JSON;
  if (!rawPath) return undefined;
  const candidatePath = resolve(process.cwd(), rawPath);
  if (!existsSync(candidatePath)) {
    throw new Error("VV_LLM_SETTINGS_JSON must name a readable JSON file");
  }
  try {
    return JSON.parse(readFileSync(candidatePath, "utf8"));
  } catch {
    throw new Error("VV_LLM_SETTINGS_JSON file must contain valid JSON");
  }
}

function flagUsage(usage) {
  if (!usage || typeof usage !== "object") return undefined;
  const result = {};
  for (const key of ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"]) {
    if (typeof usage[key] === "number") result[key] = usage[key];
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function completionFlags(response) {
  const message = response?.choices?.[0]?.message;
  return {
    content_present: typeof message?.content === "string" && message.content.length > 0,
    reasoning_present: typeof message?.reasoning_content === "string" && message.reasoning_content.length > 0,
    tool_calls_present: Array.isArray(message?.tool_calls) && message.tool_calls.length > 0,
    usage: flagUsage(response?.usage),
  };
}

function streamFlags(chunks) {
  let content = false;
  let reasoning = false;
  let toolCalls = false;
  let usage;
  for (const chunk of chunks) {
    const delta = chunk?.choices?.[0]?.delta;
    content ||= typeof delta?.content === "string" && delta.content.length > 0;
    reasoning ||= typeof delta?.reasoning_content === "string" && delta.reasoning_content.length > 0;
    toolCalls ||= Array.isArray(delta?.tool_calls) && delta.tool_calls.length > 0;
    usage = flagUsage(chunk?.usage) ?? usage;
  }
  return { content_present: content, reasoning_present: reasoning, tool_calls_present: toolCalls, usage };
}

let parsedApiKey;
try {
  const settings = parseSettings();
  const backend = process.env.VV_LLM_BACKEND ?? settings?.backend;
  const requestedModel = process.env.VV_LLM_MODEL ?? process.env.OPENAI_MODEL ?? settings?.model ?? settings?.model_id;
  const prompt = process.env.VV_LLM_PROMPT ?? "Reply with one short sentence confirming this live smoke test.";
  parsedApiKey = process.env.VV_LLM_API_KEY ?? process.env.OPENAI_API_KEY ?? settings?.api_key ?? settings?.apiKey;
  const baseURL = process.env.VV_LLM_BASE_URL ?? process.env.OPENAI_BASE_URL ?? settings?.base_url ?? settings?.baseURL;
  if (!requestedModel) throw new Error("set VV_LLM_MODEL or OPENAI_MODEL (or provide model in VV_LLM_SETTINGS_JSON)");

  let client;
  let model = requestedModel;
  if (settings?.endpoints && settings?.backends && backend) {
    const resolved = createChatClientFromSettings(settings, { backend, model: requestedModel });
    client = resolved.client;
    model = resolved.resolved.model_id;
    parsedApiKey ??= resolved.resolved.endpoint.api_key;
  } else {
    if (!parsedApiKey) throw new Error("set OPENAI_API_KEY/VV_LLM_API_KEY or provide api_key in VV_LLM_SETTINGS_JSON");
    client = new VvLlmClient({ baseURL, apiKey: parsedApiKey });
  }

  const messages = [{ role: "user", content: prompt }];
  const reasoningEffort = process.env.VV_LLM_REASONING_EFFORT;
  const maxTokens = reasoningEffort ? 256 : 64;
  const liveTimeoutMs = 60_000;
  const completion = await client.create({
    model,
    messages,
    options: { max_tokens: maxTokens, stream: false, reasoning_effort: reasoningEffort },
  }, {
    signal: AbortSignal.timeout(liveTimeoutMs),
    capability_policy: reasoningEffort ? "strict" : undefined,
  });
  const stream = await client.create({
    model,
    messages,
    options: { max_tokens: maxTokens, stream: true, reasoning_effort: reasoningEffort },
  }, {
    signal: AbortSignal.timeout(liveTimeoutMs),
    capability_policy: reasoningEffort ? "strict" : undefined,
  });
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  console.log(JSON.stringify({
    model,
    reasoning_effort: reasoningEffort,
    completion: completionFlags(completion),
    stream: streamFlags(chunks),
  }, null, 2));
} catch (error) {
  const safe = error && typeof error === "object" ? {
    error_type: error.constructor?.name,
    kind: error.kind,
    status: error.status,
    code: error.code,
  } : { error_type: typeof error };
  console.error(JSON.stringify({ live_smoke_error: safe }));
  process.exitCode = 1;
}
