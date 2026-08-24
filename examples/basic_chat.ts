import { decodeChatRequest, type ChatRequest } from "vv-llm-ts";
import { clientFromEnv, expectCompletion, modelFromEnv } from "./common.js";

const request: ChatRequest = decodeChatRequest({
  model: modelFromEnv(),
  messages: [{ role: "user", content: "Give me one concise greeting." }],
  options: { stream: false, temperature: 0.2 },
});

const response = expectCompletion(await clientFromEnv().create(request));
console.log(response.choices[0]?.message.content ?? "");
