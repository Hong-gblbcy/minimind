# `serve_openai_api.py` 介绍

对应源码：[`serve_openai_api.py`](./serve_openai_api.py)

## 文件定位

基于 FastAPI 和 Uvicorn 的轻量 OpenAI Chat Completions 兼容服务。

## 核心对象

- `ChatRequest`：描述 `/v1/chat/completions` 请求，包含消息、采样参数、工具、流式开关和 thinking 配置。
- `init_model`：加载原生 MiniMind/LoRA 权重或 Transformers 模型。
- `CustomStreamer`：把生成文本写入线程安全队列。
- `parse_response`：将 `<think>` 和 `<tool_call>` 标签拆分为 `reasoning_content`、正文和 OpenAI 风格工具调用。
- `generate_stream_response`：在线程中执行生成，以 SSE 增量发送 reasoning、正文、工具调用和停止原因。
- `chat_completions`：同时实现流式和非流式响应。

## 接口与运行

```text
POST /v1/chat/completions
```

```bash
cd scripts
python serve_openai_api.py
```

默认监听 `0.0.0.0:8998`。原生权重默认从 `../out/` 读取，也可通过 `--load_from` 指定 Transformers 模型目录。

## 项目脉络

这个文件把“Python 内存中的生成函数”包装成“其他应用可调用的 HTTP 接口”：

```text
OpenAI SDK / WebUI / 第三方应用
  └─> POST /v1/chat/completions
        └─> FastAPI 参数校验
              └─> Tokenizer 聊天模板
                    └─> MiniMind.generate
                          └─> OpenAI 风格 JSON 或 SSE
```

它不执行工具，只负责把工具 schema 交给模型，并把模型生成的 `<tool_call>` 解析成结构化响应。真正的工具执行应由客户端或 Agent 框架完成。

## 启动与模型初始化

直接运行文件时，入口按以下顺序执行：

1. 解析模型结构、权重、LoRA、设备等参数。
2. 设置模块级 `device`。
3. `init_model` 加载模型和 Tokenizer，并写入模块级 `model`、`tokenizer`。
4. 启动 Uvicorn。

原生模型判断仍使用“`load_from` 路径是否包含 `model`”。原生权重路径为：

```text
../<save_dir>/<weight>_<hidden_size>[_moe].pth
```

LoRA 路径为：

```text
../<save_dir>/lora/<lora_weight>_<hidden_size>.pth
```

这个 LoRA 路径约定与其他脚本可能不同，部署前应确认实际文件位置。

## 请求模型：`ChatRequest`

Pydantic 会把 JSON 请求解析为带类型和默认值的对象。主要字段包括：

- `messages`：OpenAI 风格消息数组。
- `temperature`、`top_p`、`max_tokens`：生成参数。
- `stream`：选择 SSE 或普通 JSON。
- `tools`：工具 schema 列表。
- `open_thinking`、`chat_template_kwargs`：thinking 配置。

`get_open_thinking` 兼容三种写法：顶层 `open_thinking`、`chat_template_kwargs.open_thinking` 和 `chat_template_kwargs.enable_thinking`。请求中的 `model` 字段是兼容接口所需，但当前服务只有一个全局模型，不会根据该字段切换模型。

## 非流式请求脉络

非流式分支执行：

1. `apply_chat_template` 生成完整 prompt。
2. Tokenize 并移动到设备。
3. `torch.no_grad()` 下生成输入长度加 `max_tokens` 的完整序列。
4. 切掉输入 ids，只解码新增 token。
5. `parse_response` 分离 reasoning、正文和工具调用。
6. 返回 `chat.completion` JSON。

结束原因根据是否存在工具调用设为 `tool_calls` 或 `stop`。

## 流式请求脉络

### `CustomStreamer`

Transformers 在生成过程中调用 `on_finalized_text`。自定义 Streamer 把文本片段放入 `Queue`，结束时放入 `None` 作为哨兵。

### 后台生成线程

`generate_stream_response` 创建线程调用 `model.generate`。HTTP 生成器在主线程不断读取队列，这样 FastAPI 可以一边生成一边向客户端发送数据，不必等整段回答完成。

### Thinking 分流

当开启 thinking 时，在遇到 `</think>` 之前的文本被放进 `reasoning_content`；闭合标签后的文本进入 `content`。代码维护 `full_text` 和 `emitted` 下标，确保每段字符只发送一次。

生成结束后再对完整文本执行 `parse_response`。若发现工具调用，会额外发送一个带 `tool_calls` 的 delta，最后发送含 `finish_reason` 的 chunk。

FastAPI 最外层把每个 JSON 字符串包装成：

```text
data: {...}\n\n
```

这就是 Server-Sent Events 的消息边界。

## 响应解析：`parse_response`

函数处理三类模型输出：

- 完整 `<think>...</think>`：标签内部作为 reasoning。
- 只有 `</think>`：把闭合标签前内容视为 reasoning，兼容 Streamer 可能移除起始特殊标记的情况。
- `<tool_call>...</tool_call>`：解析 JSON，生成带 id、类型、函数名和 arguments 的 OpenAI 工具调用对象。

工具调用存在时会从正文中移除原始 XML，避免客户端同时收到文本和结构化调用两份重复内容。

## 技术特点

- FastAPI/Pydantic 负责路由和请求解析。
- Queue + Thread 把同步模型生成桥接为流式 HTTP 响应。
- 同一个解析逻辑兼容 thinking 和 Tool Call。
- 原生 MiniMind 与 Transformers 模型共享同一服务接口。

## 边界与注意事项

- 入口依赖模块级 `model`、`tokenizer`、`device`。直接用 `uvicorn scripts.serve_openai_api:app` 导入模块不会执行 `__main__` 初始化，默认会缺少这些全局对象；应按文档直接运行脚本，或自行增加应用启动钩子。
- 服务没有认证、限流或租户隔离，却默认绑定 `0.0.0.0`。只应在可信网络使用，公开部署前必须增加安全层。
- 客户端传入的 `tools` 只进入 prompt，服务端不会执行函数。
- 当前 SSE 末尾发送 finish chunk，但没有显式发送常见的 `data: [DONE]`；严格依赖该标记的客户端可能需要适配。
- 多个并发请求共享同一个全局模型。线程安全、显存峰值和吞吐需要在目标部署环境中实际验证。
- Streamer 线程异常会以 `{"error": ...}` 放入流，而不是改成 HTTP 5xx，因为响应流可能已经开始发送。
