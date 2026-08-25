# `eval_llm.py` 介绍

对应源码：[`eval_llm.py`](./eval_llm.py)

## 文件定位

MiniMind 的命令行推理与交互式对话入口。它既能读取项目原生 `.pth` 权重，也能加载 Transformers 格式模型。

## 主要流程

1. `init_model(args)` 加载 Tokenizer 和模型。
2. 原生模型路径会构造 `MiniMindForCausalLM`，再读取 `out/<weight>_<hidden_size>.pth`；也可叠加 LoRA 权重。
3. Transformers 路径通过 `AutoModelForCausalLM.from_pretrained` 加载。
4. `main()` 提供自动问题集和手动输入两种模式，以流式方式输出回答并可统计生成速度。

## 关键能力

- Dense/MoE 模型选择。
- 多轮历史对话。
- thinking 模式和 YaRN RoPE 推理外推。
- `temperature`、`top_p`、最大生成长度等采样参数。

## 运行示例

```bash
python eval_llm.py --load_from ./model --weight full_sft
python eval_llm.py --load_from ./minimind-3
```

