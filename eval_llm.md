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

## 项目脉络

这个文件位于完整模型流程的最后一段：

```text
训练数据
  └─> trainer/train_*.py
        └─> out/*.pth 或转换后的 Transformers 模型
              └─> eval_llm.py
                    ├─> 自动问题集
                    └─> 人工多轮对话
```

它不负责训练，也不启动网络服务。它的价值是用最短链路验证“Tokenizer、模型结构、权重文件、聊天模板、生成参数”是否能够协同工作。如果需要 HTTP 接口，应使用 `scripts/serve_openai_api.py`；如果需要浏览器界面，应使用 `scripts/web_demo.py`。

## 执行脉络

### 1. 判断模型格式

`init_model` 先从 `args.load_from` 加载 Tokenizer，然后用下面的代码分支选择加载方式：

- 路径字符串中包含 `model`：按项目原生模型处理，手动构造 `MiniMindConfig` 和 `MiniMindForCausalLM`。
- 其他路径：交给 `AutoModelForCausalLM.from_pretrained`，按 Transformers 格式处理。

原生权重文件名由以下信息拼出：

```text
./<save_dir>/<weight>_<hidden_size>[_moe].pth
```

因此 `hidden_size`、`num_hidden_layers` 和 `use_moe` 必须与训练权重一致。结构参数不匹配时，`load_state_dict(..., strict=True)` 会直接失败，这种严格校验可以避免把不兼容权重静默装入模型。

### 2. 可选加载 LoRA

当 `lora_weight != 'None'` 时，脚本先调用 `apply_lora` 给线性层挂载低秩分支，再调用 `load_lora` 读取增量参数。这里的 `weight` 仍指定基础模型，`lora_weight` 只指定适配器名称；两者必须来自相同模型尺寸。

### 3. 构造对话输入

每次提问都会写入 `conversation`：

- `historys=0`：只保留当前用户消息。
- `historys>0`：截取末尾指定数量的历史消息，再追加当前问题。一个完整问答包含 user 和 assistant 两条消息，因此该值通常应为偶数。
- 权重名含 `pretrain`：直接使用 `BOS + 用户文本`，因为预训练模型尚未学习聊天模板。
- 其他权重：调用 `tokenizer.apply_chat_template`，加入角色标记、生成提示和可选 thinking 模板。

### 4. 流式生成与结果回填

`TextStreamer` 在模型逐 token 生成时立即打印解码文本。生成结束后，脚本仍会从完整 `generated_ids` 中切掉输入部分，再解码出纯回复，并把它追加到历史对话。这样“终端显示”与“下一轮上下文”使用的是同一次生成结果。

## 技术点解释

### 原生权重与 Transformers 权重

原生 `.pth` 通常只是 PyTorch `state_dict`，体积直接、结构透明，但加载方必须知道模型配置。Transformers 目录则同时保存 `config.json`、Tokenizer 和权重，通用工具可以通过 AutoClass 自动恢复结构。脚本同时支持两者，方便开发阶段快速验证，也方便验证导出结果。

### Temperature 与 Top-P

模型先用 `temperature` 缩放 logits。温度越低，概率分布越尖锐，输出更稳定；越高则更随机。Top-P 随后从高概率 token 开始累加，只在累计概率不超过阈值的候选集中采样。两者共同控制生成的确定性和多样性。

### RoPE 外推不等于长文本能力

`--inference_rope_scaling` 会启用配置中的 YaRN 缩放，缓解推理位置超过训练长度时的位置编码问题。但脚本参数说明也明确指出：位置编码可用并不意味着模型已经通过长文本训练获得稳定的长上下文理解能力。

### 随机种子

每个问题前会随机生成一个新种子，再交给 `setup_seed`。这意味着同一问题默认不保证跨次运行得到相同回答；若要复现实验，需要改为固定种子。

## 主要输入与输出

| 类型 | 内容 |
| --- | --- |
| 模型输入 | 原生 `.pth` 或 Transformers 模型目录 |
| 用户输入 | 自动问题列表或终端输入 |
| 模型输出 | 终端中的流式回答和可选 tokens/s |
| 状态 | 仅保存在当前进程内的对话历史，不写入文件 |

## 使用注意事项

- 默认相对路径按仓库根目录解析，应从根目录运行。
- `args.load_from` 的格式判断是“路径是否包含字符串 `model`”，不是读取文件结构后自动识别；自定义目录名时要留意这一规则。
- 模型最终会调用 `.half()`。CUDA 推理通常适合 FP16；某些 CPU 算子对 FP16 支持有限，CPU 环境出现算子错误时应改用合适精度。
- `open_thinking` 只影响聊天模板，不会凭空赋予未经过相应训练的权重稳定推理能力。

## 运行示例

```bash
python eval_llm.py --load_from ./model --weight full_sft
python eval_llm.py --load_from ./minimind-3
```
