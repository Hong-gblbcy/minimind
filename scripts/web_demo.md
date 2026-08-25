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

