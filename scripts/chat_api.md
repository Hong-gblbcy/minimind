# `chat_api.py` 介绍

对应源码：[`chat_api.py`](./chat_api.py)

## 文件定位

使用 OpenAI Python SDK 调用本地兼容接口的最小聊天客户端示例。

## 工作方式

- 默认连接 `http://localhost:11434/v1`，模型名为 `minimind-local:latest`。
- 在终端循环读取问题，将用户消息和模型回复写入对话历史。
- 支持流式与非流式响应；流式模式会分别显示 `reasoning_content` 和最终正文。
- `history_messages_num` 控制请求携带的历史消息数，值应为偶数，`0` 表示不携带历史。
- `extra_body` 示例展示了 thinking 开关和 reasoning effort 的传递方式。

## 运行方式

先启动 OpenAI 兼容服务，再执行：

```bash
cd scripts
python chat_api.py
```

连接地址、API Key、模型名和采样参数目前直接写在脚本中，属于本地调用示例。

