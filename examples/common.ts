import {
  VvLlmClient,
  type ChatCompletion,
  type ChatCompletionChunk,
  type ChatExecutionResponse,
  type ClientOptions,
} from "vv-llm-ts";

export function clientFromEnv(): VvLlmClient {
  const options: ClientOptions = {
    baseURL: process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1",
  };
  if (process.env.OPENAI_API_KEY !== undefined) options.apiKey = process.env.OPENAI_API_KEY;
  return new VvLlmClient(options);
}

export function modelFromEnv(): string {
  const model = process.env.VV_LLM_MODEL;
  if (!model) throw new Error("Set VV_LLM_MODEL before running a network example");
  return model;
}

export function isAsyncIterable<T>(value: unknown): value is AsyncIterable<T> {
  return typeof value === "object" && value !== null && Symbol.asyncIterator in value;
}

export function expectCompletion(response: ChatExecutionResponse): ChatCompletion {
  if (isAsyncIterable<ChatCompletionChunk>(response)) {
    throw new Error("Expected a non-streaming completion; set options.stream to false");
  }
  return response as ChatCompletion;
}

export function expectStream(response: ChatExecutionResponse): AsyncIterable<ChatCompletionChunk> {
  if (!isAsyncIterable<ChatCompletionChunk>(response)) {
    throw new Error("Expected a stream; set options.stream to true");
  }
  return response;
}
