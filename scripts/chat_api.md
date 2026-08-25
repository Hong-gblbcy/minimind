# `chat_api.py` 介绍

对应源码：[`chat_api.py`](./chat_api.py)

## 文件定位

使用 OpenAI Python SDK 调用本地兼容接口的最小聊天客户端示例。

## 工作方式

- 默认连接 `http://localhost:11434/v1`，模型名为 `minimind-local:latest`。
- 在终端循环读取问题，将用户消息和模型回复写入对话历史。
- 支持流式与非流式响应；流式模式会分别显示 `reasoning_content` 和最终正文。
- `history_messages_num` 参与控制请求切片范围，`0` 时只发送当前问题；当前实现的切片会把当前 user 消息也计入该数值。
- `extra_body` 示例展示了 thinking 开关和 reasoning effort 的传递方式。

## 运行方式

先启动 OpenAI 兼容服务，再执行：

```bash
cd scripts
python chat_api.py
```

连接地址、API Key、模型名和采样参数目前直接写在脚本中，属于本地调用示例。

## 项目脉络

这个脚本是“客户端”，不是模型服务本身：

```text
终端输入
  └─> OpenAI Python SDK
        └─> OpenAI 兼容 HTTP 服务
              └─> MiniMind 或其他本地模型
                    └─> 流式 / 非流式响应
```

它展示的是第三方程序如何用通用 OpenAI SDK 接入本地模型。服务端可以是仓库中的 `serve_openai_api.py`，也可以是 Ollama 或其他兼容实现；默认端口 `11434` 更接近本地模型服务的示例配置，并不等于仓库服务端默认的 `8998`。

## 执行脉络

脚本启动时创建一个全局 `OpenAI` 客户端，然后进入无限循环：

1. 从终端读取用户问题。
2. 把问题以 `role='user'` 追加到 `conversation_history`。
3. 根据 `history_messages_num` 截取要发送的消息。
4. 调用 `client.chat.completions.create`。
5. 读取完整响应或遍历流式 chunk。
6. 把最终正文以 assistant 消息写回历史，供下一轮使用。

当 `history_messages_num=0` 时，表达式 `-(history_messages_num or 1)` 会只截取当前最后一条用户消息。大于 0 时，它截取的是“包括当前 user 消息在内的最后 N 条”，这与源码注释中“偶数表示 Q+A 历史”的描述并不完全一致。例如已经有一轮历史时，值为 2 实际发送的是“上一条 assistant + 当前 user”，而不是完整上一轮问答。若希望携带一轮完整历史和当前问题，当前切片需要取 3 条，或把代码改为截取 `history_messages_num + 1` 条。

## 流式响应处理

OpenAI 流式接口会把一次回复拆成多个 chunk，每个 chunk 可能没有 choice、没有 delta，或只包含部分字段。脚本逐层判空后分别处理：

- `delta.reasoning_content`：用灰色终端样式即时显示。
- `delta.content`：作为最终正文显示，并累加到 `assistant_res`。

历史记录只保存最终正文，不保存 reasoning。这样后续对话不会把内部思考再次作为普通 assistant 内容发送。

非流式模式则直接读取 `response.choices[0].message.content`，并在响应为空时抛出异常，避免把无效内容写进历史。

## 技术点解释

### OpenAI 兼容接口

“兼容”表示请求路径和主要 JSON 字段遵循 OpenAI Chat Completions 形式，因此可以复用官方 SDK。后端不一定由 OpenAI 提供，模型名也由本地服务决定。`api_key='sk-123'` 只是示例占位符，是否校验取决于服务端。

### `extra_body`

SDK 会把 `extra_body` 合并到请求 JSON，用于传递标准接口之外的扩展字段。这里通过 `chat_template_kwargs.open_thinking` 和 `reasoning_effort` 演示思考模式配置。目标服务若不认识这些字段，可能忽略或拒绝请求。

### 历史窗口

只截取最近若干条消息可以限制上下文长度和推理成本，但也会丢失更早信息。由于当前切片按“消息条数”而不是 token 数控制，它不能保证请求一定落在模型最大上下文内。

## 输入输出

| 类型 | 内容 |
| --- | --- |
| 输入 | 终端文本、脚本内连接和采样配置 |
| 网络请求 | `model`、`messages`、stream、temperature、max_tokens、top_p 等 |
| 输出 | 终端中的 reasoning 和正文 |
| 状态 | 当前进程内的消息历史 |

## 注意事项

- 脚本没有命令行参数，切换地址或模型需要编辑源码。
- 默认无限循环，没有专门退出指令，可用 `Ctrl+C` 结束。
- `max_tokens` 控制新生成 token 数，不代表整个上下文长度。
- reasoning 字段不是所有兼容服务都会返回，代码已用 `getattr` 做兼容。
- 用于真实服务时不要把有效凭据硬编码进版本库，应从安全配置或环境变量读取。
