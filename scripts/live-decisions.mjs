#!/usr/bin/env node
import { readFileSync } from "node:fs";

if (process.env.VV_LLM_RUN_LIVE_TESTS !== "1") {
  console.log("decision live smoke skipped; set VV_LLM_RUN_LIVE_TESTS=1");
  process.exit(0);
}
try {
  const { createDecisionClientFromSettings } = await import("../dist/index.js");
  if (!process.env.VV_LLM_SETTINGS_JSON) throw new Error("Set VV_LLM_SETTINGS_JSON to an explicit local settings file");
  const settings = JSON.parse(readFileSync(process.env.VV_LLM_SETTINGS_JSON, "utf8"));
  if (!settings.decision_backends) {
    const backend = structuredClone(settings.backends?.openai ?? {});
    if (!backend.default_endpoint) {
      for (const model of Object.values(backend.models ?? {})) {
        for (const binding of model.endpoints ?? []) {
          const id = typeof binding === "string" ? binding : binding.endpoint_id;
          if (typeof binding === "object" && binding.enabled === false) continue;
          const endpoint = settings.endpoints?.find((entry) => entry.id === id);
          if (endpoint?.enabled !== false && endpoint?.api_key && [undefined, null, "default", "openai"].includes(endpoint.endpoint_type) && !endpoint.is_azure && !endpoint.is_vertex && !endpoint.is_bedrock) { backend.default_endpoint = id; break; }
        }
        if (backend.default_endpoint) break;
      }
    }
    settings.decision_backends = { openai: backend };
  }
  const model = process.env.VV_LLM_MODEL ?? "gpt-6-luna";
  const backend = settings.decision_backends.openai;
  const selected = backend.models?.[model];
  const ids = new Set([backend.default_endpoint, ...(selected?.endpoints ?? []).map((binding) => typeof binding === "string" ? binding : binding.endpoint_id)]);
  const decisionSettings = { endpoints: settings.endpoints?.filter((endpoint) => ids.has(endpoint.id)), decision_backends: { openai: { default_endpoint: backend.default_endpoint, models: selected ? { [model]: selected } : {} } } };
  const fixture = JSON.parse(readFileSync(new URL("../contract/v1.3.0/fixtures/decisions.v1.json", import.meta.url), "utf8"));
  const { client } = createDecisionClientFromSettings(decisionSettings, { backend: "openai", model });
  const request = structuredClone(fixture.request);
  delete request.model;
  const response = await client.create(request, { signal: AbortSignal.timeout(60_000) });
  const [damaged, intent, urgency] = response.answers;
  if (response.answers.length !== 3 || damaged?.type !== "predicate" || damaged.probability <= 0.5 || intent?.type !== "choice" || intent.choice !== "replacement" || urgency?.type !== "score" || urgency.score < 0 || urgency.score > 2) throw new Error("Decision semantic check failed");
  console.log(JSON.stringify({ decision_live_smoke: "passed", answer_types: response.answers.map((answer) => answer.type), semantic_checks_passed: true, usage: response.usage ? Object.fromEntries(["input_tokens", "output_tokens", "total_tokens"].filter((key) => typeof response.usage[key] === "number").map((key) => [key, response.usage[key]])) : undefined }));
} catch (error) {
  console.error(JSON.stringify({ decision_live_smoke: "failed", error_type: error?.constructor?.name, kind: error?.kind, status: error?.status }));
  process.exitCode = 1;
}
