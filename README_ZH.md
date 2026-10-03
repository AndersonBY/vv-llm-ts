# vv-llm-ts

`vv-llm-ts` 是面向 Node 20+、ESM 的轻量 TypeScript 客户端，使用原生
`fetch`，没有运行时依赖，覆盖 OpenAI-compatible 的 chat、SSE 流式输出、
tool calls、多模态内容、embeddings 和标准化 rerank。

## 能力边界

| 能力 | 状态 |
| --- | --- |
| OpenAI-compatible chat/stream/tools/image | 支持 |
| Embeddings 与 `/rerank` 标准结果 | 支持 |
| Timeout、AbortSignal、分类错误 | 支持 |
| RetryPolicy、Middleware、ProviderRegistry | 显式启用 |
| 能力感知有序 fallback | `FallbackChatClient` 显式启用 |
| Anthropic Messages、Bedrock Converse、Vertex 鉴权、Azure URL 改写 | 未声称支持 |

基础 `VvLlmClient` 不会隐式 retry 或 fallback。需要这些行为时，组合
`RetryPolicy`、`MiddlewareChatClient` 和 `FallbackChatClient`；流式 fallback
只允许在首个可见 chunk 之前切换，已产生可见输出后不会 replay 到其它 provider。

Settings 的 endpoint binding 支持 `priority`（严格整数且不小于 1，缺省为 1）。
自动选择先过滤禁用或不可用端点，再按 priority 稳定排序。

## Canonical ChatRequest

跨语言公共入口使用嵌套 `options`；客户端会通过纯 adapter 映射到
OpenAI-compatible 的扁平请求，不修改原对象：

```ts
import { decodeChatRequest, VvLlmClient } from "vv-llm-ts";

const request = decodeChatRequest({
  model: "my-chat-model",
  messages: [{ role: "user", content: "Hello" }],
  options: { stream: false, temperature: 0.2 },
});

const client = new VvLlmClient({ apiKey: process.env.OPENAI_API_KEY });
const result = await client.createChatRequest(request);
if (Symbol.asyncIterator in result) throw new Error("expected completion");
const completion = result;
```

`client.createChatRequest(...)` 是同等的显式命名入口。原有
`client.chat.completions.create(...)` 仍保留，作为扁平 OpenAI-compatible
低层 API；流式返回的 `AsyncIterable` 只能消费一次。运行时的
`signal`/`timeout_ms` 可作为 canonical 入口的第二个参数传入。
`decodeChatRequest` 会校验闭合对象、内容、tools、`tool_choice` 和 `x_*`
扩展，并返回深拷贝后的请求。

## 可运行示例

[`examples/`](examples/README.md) 提供类型化 chat、streaming、tools、
multimodal、contract JSON、middleware、metadata 和 fallback 示例。

## 推理强度

`capabilities.reasoning_efforts` 缺省或 null 表示未知，空列表表示不支持。客户端可设置 `capabilityPolicy: "strict"`，也可通过请求的 `capability_policy` 设置策略，默认 warn，另支持 passthrough；冲突参数始终拒绝发送。

切换模型请使用请求的 `model` 或端点绑定的 `model_id`。`extra_body.model`
与选中的实际模型冲突时，即使使用 passthrough 也会报错。

settings 在本地未声明档位时继承固定目录的推理能力，再应用选中绑定的局部覆盖；显式传入 `modelCatalog` 时保留其元数据，本地能力字段和绑定覆盖仍优先；registry 的 `model_capabilities` 按模型配置 fallback，保持请求强度。Gemini 原生 level/budget 与 effort 不能同时设置。本运行时使用 Chat Completions，chat settings 工厂会拒绝 Responses endpoint。

`reasoning_effort_aliases` 单独记录兼容输入及其实际目标。只有目标仍在 `reasoning_efforts` 中时才接受别名，请求原值由服务商映射。绑定中的档位列表和别名映射均整体替换。DeepSeek 实际为 low/high/max 三档，none 表示关闭；minimal → low，medium/xhigh → high，ultra → max。兼容别名不作为独立档位展示。


## Contract 与模型目录

canonical language-neutral contract 位于独立的 `vv-llm-contract` 仓库。
本仓库 vendor 了锁定的 `contract/v1.2.1/` artifact tree，
并生成 `src/generated/contract-catalog.ts`。默认 `ModelCatalog` 使用该目录，
同时导出 contract version、revision 和 SHA-256 metadata。该 release 使用
OpenAI fixture v2，并将 retry fixture/schema 独立锁定。

```text
npm run contract:check
npm run contract:sync -- --source <contract-source>
```

同步必须显式指定 `--source` 或 `VV_LLM_CONTRACT_SOURCE`；未提供 source 时，check
只依赖已 vendor 的 lock 和 artifacts；提供 source 时会比较 source 与 vendor。

## 真实 API smoke（显式 opt-in）

默认不访问网络。要运行一次真实非流式和一次流式 completion：

```powershell
npm run build
$env:VV_LLM_RUN_LIVE_TESTS = "1"
$env:OPENAI_API_KEY = "..."
$env:VV_LLM_MODEL = "your-model"
npm run live:test
```

也可以把 `VV_LLM_SETTINGS_JSON` 设置为 Settings JSON 文件路径；
使用 settings 解析时另需 `VV_LLM_BACKEND` 和 `VV_LLM_MODEL`。输出只包含 model、内容/推理/
tool-call 是否存在和 usage 数量，不输出 key 或 prompt。没有
`VV_LLM_RUN_LIVE_TESTS=1` 时不会发请求。

运行 `npm run live:test` 时可设置 `VV_LLM_REASONING_EFFORT=xhigh`，用严格策略验证非流式和流式请求的推理参数。
