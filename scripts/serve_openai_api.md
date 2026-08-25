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

