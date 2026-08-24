import { readFile } from "node:fs/promises";
import { decodeChatRequest, type ChatRequest } from "vv-llm-ts";

const inputPath = process.argv[2];
const source = inputPath
  ? await readFile(inputPath, "utf8")
  : JSON.stringify({
      model: "example-model",
      messages: [{ role: "user", content: "Validate this canonical request." }],
      options: { stream: false },
    });

const request: ChatRequest = decodeChatRequest(JSON.parse(source) as unknown);
console.log(JSON.stringify(request, null, 2));
