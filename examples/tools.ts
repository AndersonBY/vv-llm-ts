import { decodeChatRequest, type ChatRequest } from "vv-llm-ts";
import { clientFromEnv, expectCompletion, modelFromEnv } from "./common.js";

const request: ChatRequest = decodeChatRequest({
  model: modelFromEnv(),
  messages: [{ role: "user", content: "What is the weather in Shanghai?" }],
  tools: [{
    name: "lookup_weather",
    description: "Look up current weather for a city.",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  }],
  tool_choice: {
    type: "function",
    function: { name: "lookup_weather" },
  },
  options: { stream: false },
});

const response = expectCompletion(await clientFromEnv().create(request));
const toolCalls = response.choices.flatMap((choice) => choice.message.tool_calls ?? []);
console.log(JSON.stringify(toolCalls, null, 2));
