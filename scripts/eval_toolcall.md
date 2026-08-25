# `eval_toolcall.py` 介绍

对应源码：[`eval_toolcall.py`](./eval_toolcall.py)

## 文件定位

用于检查 MiniMind Tool Call 生成与多轮工具交互流程的命令行评测脚本。

## 主要能力

- 内置数学、时间、随机数、文本长度、单位、天气、汇率和翻译八类工具定义。
- 工具返回值为本地模拟数据，不会访问真实天气、汇率或翻译服务。
- `local` 后端直接加载本地模型；`api` 后端调用 OpenAI 兼容接口，并支持流式响应。
- 可解析原生 `<tool_call>...</tool_call>` 文本，也可处理 OpenAI `tool_calls` 字段。
- `run_case` 会执行模型提出的工具调用，将结果追加到上下文，再让模型继续回答，直到不再产生调用。
- 支持预设测试用例和手动输入模式。

## 运行示例

```bash
cd scripts
python eval_toolcall.py --backend local --weight full_sft
python eval_toolcall.py --backend api --api_base_url http://localhost:11434/v1
```

该文件验证的是 Tool Call 协议与交互链路，不代表模拟工具返回内容的真实准确性。

## 项目脉络

普通语言模型只生成文本；Tool Call 模型需要先生成结构化调用，再接收工具结果，最后完成回答。这个脚本把整条链路放进终端中：

```text
用户问题 + 工具定义
  └─> 模型
        ├─> 普通回答 ─> 结束
        └─> tool_call
              └─> 本地模拟执行
                    └─> tool result 写回消息
                          └─> 再次调用模型
```

它位于训练之后、正式集成之前，适合快速观察 `full_sft`、`agent` 等不同权重是否能生成合法调用、选择正确工具，并利用返回结果继续回答。

## 工具与测试数据

`TOOLS` 使用 OpenAI function tool schema，包含函数名、描述和 JSON Schema 参数。`TOOL_MAP` 便于按名字选取工具，`get_tools(names)` 为每个测试用例组装可用工具列表。

`MOCK_RESULTS` 为八种工具提供本地函数。天气、汇率和翻译返回固定示例；随机数和时间在本机生成；数学表达式通过 `eval` 计算。它们的目的只是验证协议闭环，而不是提供生产能力。

`TEST_CASES` 为自动模式提供“问题 + 可用工具名”的组合，可检查模型是否在受限工具集合中选择合适调用。

## 两种推理后端

### Local

`init_model` 与根目录 `eval_llm.py` 类似：根据 `load_from` 构造原生 MiniMind 或加载 Transformers 模型。`generate` 完成以下步骤：

1. 把 `messages` 和 `tools` 交给聊天模板。
2. Tokenize 后发送到本地设备。
3. 使用 `TextStreamer` 流式显示。
4. 从生成 ids 中切出新 token，解码为完整响应。

本地模型的工具调用以 `<tool_call>JSON</tool_call>` 文本表示，由 `parse_tool_calls` 提取。

### API

`chat_api` 调用 OpenAI 兼容 Chat Completions。非流式响应可直接读取 `message.tool_calls`；流式响应需要按 `index` 把多个 chunk 中零散的 `id`、函数名和参数字符串拼回完整调用。

部分服务会把工具调用放在普通文本标签中而不是标准字段，`parse_tool_call_from_text` 为这种情况提供回退解析。

## 多轮工具循环：`run_case`

每个 case 从单条 user 消息开始。只要模型返回工具调用，函数就会：

1. 规范化本地/API 两种调用结构。
2. 把 assistant 的调用写入消息历史。
3. 调用 `execute_tool` 解析参数并执行模拟函数。
4. 把结果序列化为 tool 消息。
5. 再次调用模型，让模型读取观察结果。

当模型不再产生工具调用时循环结束。脚本没有预设最大循环次数，因此异常模型若持续调用工具，会一直执行；实际评测时应观察输出并按需中断。

## 技术点解释

### 为什么要把工具定义放进聊天模板

模型无法直接发现 Python 函数。工具 schema 会告诉模型“有哪些函数、各自用途、参数类型和必填字段”。Tokenizer 的聊天模板把这些 JSON 定义包装进 system prompt，并规定 `<tool_call>` 的输出协议。

### 工具调用为什么是两阶段生成

模型只负责决定“调用什么”和“传什么参数”，真正结果由外部环境产生。结果写回后模型才能基于观察继续推理。这种“Action → Observation → Next Action/Answer”是 Agent 多轮交互的基本结构。

### 本地格式与 OpenAI 格式

本地训练语料使用 XML 标签包裹 JSON，便于纯文本语言模型学习；API 层通常将其抽象为结构化 `tool_calls` 数组。脚本同时支持两者，正好用于检查格式转换是否正确。

## 输入输出与评测边界

| 模式 | 输入 | 输出 |
| --- | --- | --- |
| 自动 | 预置 prompt 与受限工具集合 | 调用过程、模拟结果、最终回答 |
| 手动 | 终端问题与全部工具 | 同上 |
| API | 兼容接口地址、模型名和 Key | 服务返回的流式/非流式结果 |

脚本没有自动计算准确率或与标准答案对比，评测主要依靠人工观察调用格式、工具选择和最终回答。

## 安全与使用注意事项

- `calculate_math` 使用 Python `eval`，且没有像 `train_agent.py` 那样限制 builtins；只应在可信、本地测试输入上使用。
- 模拟单位换算只固定乘一个系数，并不根据所有单位组合真实换算。
- 天气、汇率、翻译均为固定数据，不能用于真实查询。
- API 默认 Key 是占位值，是否需要真实凭据由目标服务决定。
- 本地原生权重的尺寸、层数和 MoE 参数必须与命令行一致。
