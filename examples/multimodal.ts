import { decodeChatRequest, type ChatRequest } from "vv-llm-ts";
import { clientFromEnv, expectCompletion, modelFromEnv } from "./common.js";

const imageDataUrl =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

const request: ChatRequest = decodeChatRequest({
  model: modelFromEnv(),
  messages: [{
    role: "user",
    content: [
      { type: "text", text: "Compare these two image parts." },
      {
        type: "image_url",
        url: imageDataUrl,
        detail: "low",
        cache_control: { type: "ephemeral" },
      },
      {
        type: "image_url",
        image_url: { url: imageDataUrl, detail: "high", x_source: "example" },
        cache_control: { type: "ephemeral" },
      },
    ],
  }],
  options: { stream: false },
});

const response = expectCompletion(await clientFromEnv().create(request));
console.log(response.choices[0]?.message.content ?? "");
