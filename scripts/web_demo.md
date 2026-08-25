# `web_demo.py` 介绍

对应源码：[`web_demo.py`](./web_demo.py)

## 文件定位

基于 Streamlit 的 MiniMind 多语言聊天页面，支持流式生成、thinking 展示和模拟 Tool Call。

## 页面能力

- 自动扫描 `scripts/` 下含 `.bin`、`.safetensors`、`.pt` 或分片索引的模型子目录。
- 通过 Transformers 接口缓存加载所选模型。
- 提供中英文界面、历史轮数、最大生成长度、温度和 thinking 开关。
- 最多选择四个内置模拟工具；模型调用后，页面执行工具、写回结果并继续生成，最多循环 16 次。
- `process_assistant_content` 将思考块和工具调用渲染为折叠或高亮 HTML。
- `TextIteratorStreamer` 配合后台线程逐步更新回答。

## 运行方式

先把 Transformers 格式模型目录放到 `scripts/` 下，然后执行：

```bash
cd scripts
streamlit run web_demo.py
```

内置天气、汇率、翻译等工具均返回示例数据，仅用于展示交互流程。

## 项目脉络

`web_demo.py` 是直接加载模型的前端应用，不经过 `serve_openai_api.py`：

```text
浏览器
  └─> Streamlit 脚本
        ├─> 侧边栏参数 / Session State
        ├─> Transformers 模型与 Tokenizer
        ├─> 本地流式生成
        └─> 可选模拟 Tool Call 循环
```

它适合个人演示和本地体验。相比 API 架构，它部署简单，但 UI、模型推理和工具执行都在同一个 Python 进程中，不适合直接视为高并发生产服务。

## Streamlit 的执行模型

Streamlit 在用户修改控件或提交输入时会重新执行整个脚本。代码因此把跨次执行需要保留的信息存入 `st.session_state`，把昂贵的模型加载放进 `@st.cache_resource`。

文件顶部的页面配置、CSS、模型目录扫描和侧边栏控件都在模块执行阶段运行；`main()` 主要处理聊天记录、用户输入和生成。

## 模型发现与加载

脚本扫描自身所在目录的一级子目录。满足以下任一条件时视为模型目录：

- 含 `.bin`、`.safetensors` 或 `.pt` 文件。
- 含 `model.safetensors.index.json`。

选择模型后，`load_model_tokenizer` 使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 加载，并启用 `trust_remote_code=True`。模型转为 FP16、eval 模式并移动到 CUDA 或 CPU。

缓存以 `model_path` 为参数键，切换回已经加载过的模型时可以复用资源，避免每次页面重跑都重新读取权重。

## 两套消息状态

脚本维护两个列表：

- `messages`：用于页面展示，包含用户可见的完整答案以及渲染用工具结果 HTML。
- `chat_messages`：用于下一次模型输入，遵循聊天模板需要的 role/content 结构。

分开保存可以让页面展示丰富内容，同时避免把 HTML 装饰全部送回模型。历史轮数滑块控制的是 `chat_messages` 截取范围。

## 单轮生成脉络

1. `st.chat_input` 接收问题。
2. 问题写入两套消息状态。
3. 随机生成种子并调用 `setup_seed`。
4. 根据工具选择决定 system prompt 和 `tools` 参数。
5. `apply_chat_template` 生成模型 prompt。
6. Tokenize 后建立 `TextIteratorStreamer`。
7. 后台线程调用 `model.generate`。
8. 主线程迭代 Streamer，不断更新占位区域。
9. 完整答案写回 session state。

当没有工具时，脚本加入一个普通 MiniMind system prompt；有工具时，聊天模板本身会生成工具说明，因此 `sys_prompt` 为空，避免重复定义。

## Thinking 的展示

`process_assistant_content` 处理四种情况：

- 完整 `<think>...</think>`：渲染为“已思考”折叠块。
- 只有起始标签：显示“思考中”。
- 只有结束标签：把此前文本视为思考内容。
- 流式阶段尚无显式标签：根据 thinking 开关暂时把内容放入思考块，并尝试识别答案开始位置。

这部分只改变页面显示，不改变保存在模型上下文中的原始生成文本。

## Tool Call 脉络

用户最多选择四个工具。工具 schema 通过聊天模板传给模型；生成结束后脚本用正则查找 `<tool_call>`：

1. 解析工具名和 JSON 参数。
2. 调用本地 `execute_tool`。
3. 以 `role='tool'` 把结果写回 `chat_messages`。
4. 重新套用聊天模板并继续生成。
5. 最多循环 16 次，防止模型无限调用。

页面会分别渲染 ToolCalling 和 ToolCalled 卡片，让用户看到动作与观察结果。

## 技术点解释

### 为什么用生成线程

Transformers 的 `generate` 是同步阻塞调用。把它放到后台线程后，主线程可以迭代 `TextIteratorStreamer`，每得到一小段文本就刷新页面，实现视觉上的实时输出。

### Session State 与模型上下文

网页每次重跑时普通局部变量都会丢失，Session State 相当于每个用户会话的内存存储。它不是数据库，服务重启后历史不会保留，也不适合长期持久化。

### 模型目录为什么必须放在 `scripts/`

扫描基准是 `os.path.dirname(__file__)`，而不是仓库根目录或命令行参数。模型不在该目录的一级子目录时，选择框不会发现它。

## 边界与安全注意事项

- `calculate_math` 使用 `eval` 执行模型生成的表达式，只适合可信的本地演示，不应直接暴露给不受信用户。
- 用户文本和模型文本通过 `unsafe_allow_html=True` 渲染，且没有完整 HTML 转义；公开部署可能存在内容注入风险。
- 天气、汇率、翻译、单位换算均为示例实现，不是真实工具。
- `trust_remote_code=True` 会执行模型目录中的代码，只能加载可信模型。
- 模型统一 `.half()`；CPU 对 FP16 的支持和性能需要单独验证。
- 页面没有账号、鉴权、队列、显存调度或请求隔离，不应直接当作生产推理服务。
- `clear_chat_messages`、`init_chat_messages`、`regenerate_answer` 已定义，但当前主流程没有完整接入独立按钮；阅读时应区分已定义辅助函数和实际页面路径。
