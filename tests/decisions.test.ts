import assert from "node:assert/strict";
import { test } from "node:test";
import { DecisionClient, createDecisionClientFromSettings, decodeDecisionRequest, decodeDecisionResponse, type DecisionRequest, type DecisionResponse, type Settings, VvLlmError } from "../src/index.js";
import { readContractFixture } from "./contract-fixtures.js";

const fixture = readContractFixture<{ request: DecisionRequest; response: DecisionResponse; refusal_response: DecisionResponse; valid_requests: unknown[]; valid_responses: unknown[]; invalid_requests: unknown[]; invalid_responses: unknown[] }>("fixtures/decisions.v1.json");

test("decisions share canonical fixtures and preserve observed zero versus missing usage", () => {
  assert.deepEqual(decodeDecisionRequest(fixture.request), fixture.request);
  for (const valid of fixture.valid_requests) assert.deepEqual(decodeDecisionRequest(valid), valid);
  assert.deepEqual(decodeDecisionResponse(fixture.response), fixture.response);
  for (const valid of fixture.valid_responses) assert.deepEqual(decodeDecisionResponse(valid), valid);
  assert.deepEqual(decodeDecisionResponse(fixture.refusal_response), fixture.refusal_response);
  for (const raw of fixture.invalid_requests) assert.throws(() => decodeDecisionRequest(raw));
  for (const raw of fixture.invalid_responses) assert.throws(() => decodeDecisionResponse(raw));
  const response = decodeDecisionResponse({ model: "gpt-6-luna", answers: [{ type: "predicate", probability: 0 }], usage: { output_tokens: 0 } });
  assert.equal(response.usage?.output_tokens, 0);
  assert.equal(Object.hasOwn(response.usage!, "input_tokens"), false);
});

test("standalone decision settings select enabled bindings and provider model IDs", async () => {
  const bodies: unknown[] = [];
  const settings: Settings = { endpoints: [{ id: "disabled", enabled: false }, { id: "active", api_base: "https://example.invalid/v1", api_key: "test-key", headers: { "x-test": "preserved" } }], decision_backends: { openai: { models: { "gpt-6-luna": { id: "gpt-6-luna", endpoints: [{ endpoint_id: "disabled" }, { endpoint_id: "active", priority: 2, model_id: "luna-deployment" }] } } } } };
  const { client } = createDecisionClientFromSettings(settings, { backend: "openai", model: "gpt-6-luna", fetch: async (url, init) => {
    assert.equal(url, "https://example.invalid/v1/decisions");
    assert.equal(new Headers(init?.headers).get("x-test"), "preserved");
    const body = JSON.parse(String(init?.body));
    assert.deepEqual(body.questions[1].choices, ["replacement", "refund"]);
    assert.deepEqual(body.questions[2].rubric, [{ label: "low" }, { label: "medium" }, { label: "high" }]);
    bodies.push(body);
    return new Response(JSON.stringify({ ...fixture.response, future_provider_field: true }));
  } });
  const request = structuredClone(fixture.request);
  delete request.model;
  assert.deepEqual(await client.create(request), fixture.response);
  assert.deepEqual(await client.create(fixture.request), fixture.response);
  assert.equal((bodies[0] as { model: string }).model, "luna-deployment");
  assert.deepEqual(fixture.request.model, "gpt-6-luna");
  const binding = settings.decision_backends!.openai!.models!["gpt-6-luna"]!.endpoints![1]!;
  assert.equal(typeof binding, "object");
  if (typeof binding === "object") binding.capabilities = { decision_types: [] };
  const disabled = createDecisionClientFromSettings(settings, { backend: "openai", model: "gpt-6-luna", fetch: async () => { assert.fail("must not send"); } });
  await assert.rejects(disabled.client.create(request), /declare support/);
});

test("decision HTTP errors and malformed successful responses remain errors", async () => {
  const client = new DecisionClient({ fetch: async () => new Response(JSON.stringify({ error: { message: "slow down" } }), { status: 429, headers: { "retry-after": "2" } }) });
  await assert.rejects(client.create(fixture.request), (error: unknown) => error instanceof VvLlmError && error.kind === "rate_limited" && error.retry_after_ms === 2000);
  const malformed = new DecisionClient({ fetch: async () => new Response(JSON.stringify({ ...fixture.response, answers: [] })) });
  await assert.rejects(malformed.create(fixture.request), (error: unknown) => error instanceof VvLlmError && error.kind === "serialization");
  const refusal = new DecisionClient({ fetch: async () => new Response(JSON.stringify({ model: "gpt-6-luna", answers: [{ type: "refusal", name: "greeting" }] })) });
  const result = await refusal.create({ input: "hello", questions: [{ type: "predicate", name: "greeting", instructions: "Is this a greeting?" }] });
  assert.equal(result.answers[0]?.type, "refusal");
});
