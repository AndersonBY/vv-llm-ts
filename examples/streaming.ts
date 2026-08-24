import { decodeChatRequest, type ChatRequest } from "vv-llm-ts";
import { clientFromEnv, expectStream, modelFromEnv } from "./common.js";

const request: ChatRequest = decodeChatRequest({
  model: modelFromEnv(),
  messages: [{ role: "user", content: "Explain streaming in one sentence." }],
  options: { stream: true },
});

const stream = expectStream(await clientFromEnv().create(request));
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta.content ?? "");
}
process.stdout.write("\n");
