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

