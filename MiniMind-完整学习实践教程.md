# MiniMind 完整学习实践教程

> 基于仓库 [`jingyaogong/minimind`](https://github.com/jingyaogong/minimind) 的 `master` 分支提交  
> [`89d674b8a517010f5561b6d8ab2dcbb58e2fb91b`](https://github.com/jingyaogong/minimind/tree/89d674b8a517010f5561b6d8ab2dcbb58e2fb91b)（2026-07-23，`[refactor] PPO cleanup`）逐文件整理。  
> 教程目标：从读懂 tokenizer、数据、模型开始，完成 Pretrain → SFT 主线，并能继续实践 LoRA、蒸馏、DPO、PPO、GRPO/CISPO、Agentic RL、评测、转换和部署。

---

## 导航

- [范围、地图、环境](#0-先明确教程边界)：第 0～2 节；
- [Tokenizer、模型、数据](#3-第一课读懂-tokenizer-和聊天协议)：第 3～5 节；
- [训练公共机制与全部训练路线](#6-第四课训练公共机制)：第 6～15 节；
- [推理、工具、API、WebUI、转换、部署和评测](#16-第十四课cli-推理与-yarn)：第 16～22 节；
- [排障、完整参数、实践路线、源码注记与文件覆盖](#23-当前版本的已知不一致与排障表)：第 23～30 节。

---

## 0. 先明确教程边界

这份教程覆盖当前 **MiniMind 主仓库**里的全部功能和入口：

- 6400 词表 ByteLevel BPE tokenizer 与聊天模板；
- 64M Dense 和 198M-A64M MoE 两种语言模型；
- Pretrain、全参数 SFT、LoRA、白盒蒸馏；
- DPO、PPO、GRPO、CISPO、Agentic RL；
- 本地 CLI、Tool Calling、WebUI、OpenAI 兼容 API；
- Torch / Transformers 格式转换，SGLang、vLLM、llama.cpp、Ollama、MNN 接入；
- 主观评测、Tool Calling 演示、`lm-evaluation-harness` 客观评测和 YaRN 长度外推；
- 断点恢复、DDP、SwanLab、检查点与所有公开命令参数；
- 当前代码中的限制、路径陷阱、安全风险和 README 与代码不一致之处。

这里的“完整”指**覆盖全部公开模块、训练阶段和执行入口**，而不是把每一行源码换一种说法复述。仓库提到的 MiniMind-V、MiniMind-O、MiniMind-dLM、MiniMind-Linear 属于主线之外的扩展方向：V/O 有各自仓库，dLM/Linear 当前由 README 链接到 Discussions 中的实验方案；它们都不属于本提交 `master` 的受跟踪源码文件：

- [MiniMind-V](https://github.com/jingyaogong/minimind-v)：在语言模型旁接视觉编码器，让模型理解图片；
- [MiniMind-O](https://github.com/jingyaogong/minimind-o)：进一步处理图像、音频等多模态输入/输出；
- [MiniMind-dLM](https://github.com/jingyaogong/minimind/discussions/618)：用扩散式、非传统自回归方式建模语言；
- [MiniMind-Linear](https://github.com/jingyaogong/minimind/discussions/704)：探索线性复杂度的序列建模结构。

README 说明 dLM/Linear 的实验可从主线自回归权重继续训练；具体做法应以对应 Discussion 为准，本教程不虚构当前目录中不存在的脚本。

主仓库 tokenizer 虽然预留了视觉、音频和 TTS token，但 `model/model_minimind.py` 仍是**纯文本、自回归、decoder-only Causal LM**，不能因为存在这些 token 就把它当成多模态模型。

---

## 1. 建立全局地图

### 1.1 你最终要走通的路线

```text
仓库自带 tokenizer（主线直接使用）
        │
        ▼
Pretrain：从随机参数学习语言规律
train_pretrain.py
        │  out/pretrain_768.pth
        ▼
Full SFT：学习聊天、指令、思考和工具协议
train_full_sft.py
        │  out/full_sft_768.pth
        ├─ LoRA：train_lora.py
        ├─ 蒸馏：train_distillation.py
        ├─ DPO：train_dpo.py
        ├─ PPO：train_ppo.py
        ├─ GRPO/CISPO：train_grpo.py
        └─ 多轮 Agentic RL：train_agent.py
```

最小可复现路线只需要：

```text
pretrain_t2t_mini.jsonl
  → train_pretrain.py
  → sft_t2t_mini.jsonl
  → train_full_sft.py
  → eval_llm.py
```

其余训练都是可选分支，不是必须依次执行。尤其不要把 DPO、PPO、GRPO 和 Agent 理解成必须全部串联的四个连续阶段；通常从同一个高质量 `full_sft` 基线分别实验，再根据评测选择路线。

### 1.2 仓库目录职责

```text
minimind/
├── model/
│   ├── model_minimind.py       # Dense/MoE 模型、loss、KV Cache、generate
│   ├── model_lora.py           # 手写 LoRA 注入、保存、加载、合并
│   ├── tokenizer.json          # 6400 词 BPE
│   └── tokenizer_config.json   # special token 与 chat template
├── dataset/
│   ├── lm_dataset.py           # 5 类 Dataset 与聊天增强
│   └── dataset.md              # 提醒把数据放在本目录
├── trainer/
│   ├── trainer_utils.py        # DDP、学习率、checkpoint、加载模型
│   ├── train_tokenizer.py      # tokenizer 教学实验
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   ├── train_lora.py
│   ├── train_distillation.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   ├── train_grpo.py           # 同时实现 GRPO / CISPO
│   ├── train_agent.py          # 多轮 Tool-Use RL
│   └── rollout_engine.py       # Torch / SGLang rollout 抽象
├── scripts/
│   ├── convert_model.py        # Torch、Transformers、Qwen3、LoRA 转换
│   ├── serve_openai_api.py     # OpenAI 风格 API
│   ├── chat_api.py             # API 客户端示例
│   ├── eval_toolcall.py        # Tool Calling 演示
│   └── web_demo.py             # Streamlit WebUI
├── eval_llm.py                 # CLI 对话推理
├── requirements.txt
├── README.md / README_en.md
├── LICENSE                     # 仓库 Work 使用 Apache-2.0
├── CODE_OF_CONDUCT.md
└── images/                     # README 使用的结构、曲线、界面和评测图片
```

`model/__init__.py` 与 `dataset/__init__.py` 是空的包标记文件。仓库没有正式测试目录，也没有验证集训练循环，因此本教程会为每一阶段给出人工验收点。

---

## 2. 环境、下载与路径约定

### 2.1 参考环境与现实要求

README 给出的参考环境是 Ubuntu 20.04、Python 3.10.16、CUDA 12.2、8×RTX 3090 24GB、128GB RAM。它是作者环境，不是硬性最低配置。

重要事实：

- `requirements.txt` 中 `torch`、`torchvision` 被注释，必须按你的 CUDA/CPU/MPS 平台另行安装；
- 普通 Dense Pretrain/SFT 可以尝试 CPU 或 MPS，但速度和兼容性会差很多；
- 多卡代码把 DDP 后端固定为 NCCL，实际要求 NVIDIA CUDA；
- PPO/GRPO/Agent 同时驻留多个模型，显存要求远高于普通 SFT；
- `fastapi`、`uvicorn`、SGLang、vLLM、llama.cpp、MNN 不在基础依赖里，需要按使用场景另装；
- requirements 中 `jsonlines==4.0.0` 重复一次，无功能影响；
- 当前固定 `transformers==4.57.6`，不要在未验证时随意升级大版本。

建议先建立隔离环境：

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell 使用 .venv\Scripts\Activate.ps1

# 先按 PyTorch 官方页面为自己的平台安装 torch/torchvision，再执行：
python -m pip install -r requirements.txt
```

验证：

```bash
python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA count:", torch.cuda.device_count())
PY
```

若只运行 API，再补：

```bash
python -m pip install fastapi uvicorn
```

### 2.2 三种工作目录，不能混用

这是仓库最容易踩的坑之一。脚本大量使用相对路径：

| 任务 | 应在何处执行 | 原因 |
|---|---|---|
| `eval_llm.py` | 项目根目录 | 默认读取 `model/`、`out/` |
| 所有 `train_*.py` | `trainer/` | 默认数据 `../dataset`、权重 `../out`、tokenizer `../model` |
| `serve_openai_api.py`、`eval_toolcall.py`、`web_demo.py`、`convert_model.py`、`chat_api.py` | `scripts/` | 默认模型、权重路径按 `scripts/` 计算 |

后文命令默认先定义：

```text
ROOT=/path/to/minimind
```

但命令会直接写出 `cd`，避免你依赖未定义的环境变量。

后文每个 shell 代码块都视为**独立执行，并从项目根目录开始**；不要把上一代码块留下的当前目录带入下一块。若要在同一个 shell 连续粘贴，可把子目录命令写成 `(cd trainer && python ...)` / `(cd scripts && python ...)`，避免改变父 shell 的目录。

### 2.3 下载模型

只体验推理时，在项目根目录下载 Transformers 格式模型：

```bash
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
```

或：

```bash
git clone https://huggingface.co/jingyaogong/minimind-3
```

Hugging Face 的 Git 方式依赖 Git LFS/Xet；若 clone 后只有指针文件，应先安装对应扩展，或改用 Hugging Face CLI 下载。

仓库列出的发布家族包括：

| 模型 | 参数量 | Release |
|---|---:|---|
| MiniMind-3 | 64M | 2026-04-01 |
| MiniMind-3-MoE | 198M，总体；约 A64M/token | 2026-04-01 |
| MiniMind2-small | 26M | 2025-04-26 |
| MiniMind2-MoE | 145M | 2025-04-26 |
| MiniMind2 | 104M | 2025-04-26 |
| MiniMind-v1-small | 26M | 2024-08-28 |
| MiniMind-v1-MoE | 4×26M | 2024-09-17 |
| MiniMind-v1 | 108M | 2024-09-01 |

历史兼容性不能只看 hidden size：2025-04-26 曾统一重命名模型参数，并把 `<s></s>` 改为 `<|im_start|><|im_end|>`，当前代码不能直接加载此前旧权重；v1 系列也已停止维护。需要历史实现时应使用 README 指向的旧提交，不能把 v1/v2/v3 权重混入当前训练链。

当前发布的 Transformers 模型通常由 `full_sft` 转换而来；DPO、PPO、GRPO、CISPO、Agent、LoRA 等实验权重并不保证全部公开或持续同步维护，应以实际 release 文件为准。

原生 PyTorch 阶段权重可从项目 README 指向的 ModelScope/Hugging Face `minimind-3-pytorch` 仓库下载，放入 `out/` 并保持命名规则。

### 2.4 下载数据

数据下载页：

- [ModelScope：gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [Hugging Face：jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

无需整库下载，只取要练习的文件并放入 `dataset/`：

```text
dataset/
├── pretrain_t2t_mini.jsonl   1.2GB   最小主线必需
├── sft_t2t_mini.jsonl        1.6GB   最小主线必需
├── pretrain_t2t.jsonl         10GB   完整预训练
├── sft_t2t.jsonl              14GB   完整 SFT
├── dpo.jsonl                  53MB   DPO
├── rlaif.jsonl                24MB   PPO/GRPO/CISPO
├── agent_rl.jsonl             86MB   Agentic RL
└── agent_rl_math.jsonl        18MB   数学 Agent/RLVR 补充
```

LoRA 示例默认还需要自备 `dataset/lora_medical.jsonl`。README 的最小主线是前两个 mini 文件；完整主线推荐 `pretrain_t2t + sft_t2t + rlaif/agent_rl`。

仓库 Work 与外部数据的许可证是两件事。仓库采用 Apache-2.0；训练数据混合了 Apache、CC-BY-NC 等来源约束。若要商用或再分发，必须逐项核对数据来源，不能用仓库的 Apache-2.0 覆盖数据本身的许可。

---

## 3. 第一课：读懂 tokenizer 和聊天协议

### 3.1 为什么主线不要重训 tokenizer

`model/tokenizer.json` 是一个 6400 词 ByteLevel BPE：

| 组成 | 数量 |
|---|---:|
| 预留/协议 token | 36 |
| ByteLevel 基础字节 token | 256 |
| BPE merge 结果 | 6108 |
| 合计 | 6400 |

ID 分区为：

```text
0—35      协议与预留 token
36—291    完整 256 字节 alphabet
292—6399  BPE merge token
```

词表很小能减少 embedding 和输出层占比，但中英文压缩率不如超大中文词表。仓库经验值：

- 中文约 1 token / 1.5～1.7 字符；
- 英文约 1 token / 4～5 字符。

重新训练 tokenizer 会同时破坏：

- 已有模型权重中 token ID 与 embedding 的对应关系；
- 当前数据和聊天模板的边界；
- 社区发布权重、Transformers 配置、部署生态；
- PPL 等按 token 统计指标的可比性。

因此 `trainer/train_tokenizer.py` 只适合作为 BPE 教学实验，**不属于正式训练主线**。

### 3.2 36 个预留 token

| ID | token/分组 | 是否被 tokenizer 标成 special |
|---:|---|---|
| 0 | `<\|endoftext\|>`，同时 PAD/UNK | 是 |
| 1 | `<\|im_start\|>`，BOS | 是 |
| 2 | `<\|im_end\|>`，EOS | 是 |
| 3–8 | object ref、box、quad 起止标记 | 是 |
| 9–13 | vision/image/video 标记 | 是 |
| 14–16 | audio 标记 | 是 |
| 17–20 | TTS 标记 | 是 |
| 21–24 | `<tool_call>`、`</tool_call>`、`<tool_response>`、`</tool_response>` | 否 |
| 25–26 | `<think>`、`</think>` | 否 |
| 27–35 | `<\|buffer1\|>`～`<\|buffer9\|>` | 否 |

Tool/Think 标签虽然是不可再分的 added token，却故意不是 special token。这意味着：

- `decode(..., skip_special_tokens=True)` 后仍然可见；
- 模型能对这些标签本身计算 loss；
- 推理代码可从文本中解析它们。

视觉、音频、TTS 标记只是为系列生态预留，主模型没有对应 encoder。

### 3.3 tokenizer 配置的三个长度概念

`tokenizer_config.json` 声明：

```text
model_max_length = 131072
BOS = 1, EOS = 2, PAD = UNK = 0
add_bos_token = false
add_eos_token = false
```

不要混淆：

1. `model_max_length=131072`：tokenizer 元数据；
2. 模型默认 `max_position_embeddings=32768`：RoPE 缓冲长度；
3. 各训练脚本的 `max_seq_len`：本次数据截断/训练长度。

第 1 项不是模型具备 128K 能力的保证。启用 `inference_rope_scaling` 后，内置 YaRN 配置按 `2048 × 16 ≈ 32768` 外推，同样不是 131K。

因为 tokenizer 不自动加 BOS/EOS：

- PretrainDataset 手动加；
- 对话数据由 chat template 加；
- 直接 `tokenizer("文本")` 不会自动出现 ID 1/2。

### 3.4 Chat Template 做了什么

普通对话大致渲染为：

```text
<|im_start|>system
系统提示<|im_end|>
<|im_start|>user
用户文本<|im_end|>
<|im_start|>assistant
<think>
reasoning_content
</think>

assistant content<|im_end|>
```

核心规则完整列举如下：

1. 有 `tools` 时，模板会建立 system 块，追加固定英文 Tool Calling 说明，并把工具 JSON 放入 `<tools>...</tools>`。
2. 无 tools 时，仅首条 system 消息按 system 块处理。
3. assistant 总会包含 `<think>...</think>`，之后才是正文和可能的 tool calls。
4. 优先读取独立的 `reasoning_content`；否则尝试从已含 `</think>` 的 content 中拆分思考和正文。
5. `tool_calls` 同时接受 `{name, arguments}` 和 OpenAI 风格 `{function:{name, arguments}}`。
6. arguments 若是字符串则原样写出，否则转 JSON。
7. 连续多个 `tool` 消息被合并到一个 user turn，每项包在 `<tool_response>` 中。
8. 未知 role 没有渲染分支，会被忽略。
9. `add_generation_prompt=True, open_thinking=True` 时，prompt 结尾是：

   ```text
   <|im_start|>assistant
   <think>
   ```

10. `open_thinking=False` 时，prompt 先放一个完整空思考块，模型直接续写正文：

    ```text
    <|im_start|>assistant
    <think>

    </think>

    ```

当前不再有独立 `train_reason.py`，也不再把“思考模型”当单独权重。思考能力由模板、SFT 数据的 `reasoning_content`、空 think 混合，以及 RL 的 `thinking_ratio` 一起控制。Tool Calling 与显式 thinking 同时开启时可能不稳定，因为联合样本仍不足。

### 3.5 tokenizer 实践

在项目根目录运行：

```bash
python - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./model")
print("vocab:", len(tok))
for i in [0, 1, 2, 21, 22, 23, 24, 25, 26]:
    print(i, repr(tok.convert_ids_to_tokens(i)))

messages = [
    {"role": "system", "content": "你是一个严谨的助手。"},
    {"role": "user", "content": "解释 GQA。"},
]
for enabled in [False, True]:
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=enabled,
    )
    print("\nopen_thinking =", enabled)
    print(text)

raw = "<think>先分析</think><tool_call>{\"name\":\"demo\",\"arguments\":{}}</tool_call>"
ids = tok.encode(raw, add_special_tokens=False)
print(tok.decode(ids, skip_special_tokens=True))
PY
```

验收：

- 词表长度为 6400；
- 0/1/2 分别是 PAD/UNK、BOS、EOS 约定；
- `skip_special_tokens=True` 后 Tool/Think 标签仍保留；
- thinking 开关只改变 generation prompt，而不是替换模型。

### 3.6 重训 tokenizer 的教学实验

`trainer/train_tokenizer.py`：

- 只读取 `sft_t2t_mini.jsonl` 前 10000 行；
- 从 `conversations[*].content` 提取文本；
- 训练 6400 词 ByteLevel BPE；
- 输出到 `model_learn_tokenizer/`；
- 写出 tokenizer、vocab、merges 和 config；
- 自动测试模板、编解码、压缩率与多字节流式解码。

运行：

```bash
cd trainer
python train_tokenizer.py
```

它没有 CLI 参数。正式训练的 `init_model()` 仍固定读取 `../model`，不会自动改用新目录。若你真要建立独立模型家族，必须同时重训所有权重并同步模板、数据、转换和部署配置。

---

## 4. 第二课：从张量到 MiniMind 模型

### 4.1 默认配置

| 参数 | Dense/MoE 默认值 | 含义 |
|---|---:|---|
| `vocab_size` | 6400 | 与 tokenizer 严格一致 |
| `hidden_size` | 768 | 隐藏维度 |
| `num_hidden_layers` | 8 | Transformer Block 数 |
| `num_attention_heads` | 8 | Q heads |
| `num_key_value_heads` | 4 | KV heads，GQA 2:1 |
| `head_dim` | 96 | `768 / 8` |
| `intermediate_size` | 2432 | `ceil(768π/64)×64` |
| `hidden_act` | SiLU | SwiGLU 的激活 |
| `dropout` | 0 | embedding、attention、residual |
| `rms_norm_eps` | `1e-6` | RMSNorm |
| `rope_theta` | `1e6` | RoPE base |
| `max_position_embeddings` | 32768 | RoPE buffer |
| `flash_attn` | true | 使用 PyTorch SDPA，不是外部 flash-attn 包 |
| `tie_word_embeddings` | true | embedding 与 LM head 绑权重 |
| `num_experts` | 4 | MoE 专家数 |
| `num_experts_per_tok` | 1 | Top-1 |
| `router_aux_loss_coef` | `5e-4` | 负载均衡系数 |

按代码计算：

- Dense：63,912,192 参数，约 63.91M；
- MoE：198,416,640 总参数，约 198.42M；
- MoE 每 token 激活路径约 63.94M，所以写作 `198M-A64M`；
- 若不绑定 embedding 与 LM head，会额外增加 `6400×768=4,915,200` 参数。

约束：

- attention heads 必须能被 KV heads 整除；
- `head_dim` 必须是偶数，否则当前 RoPE 实现会发生维度不匹配；
- 配置没有显式设 `pad_token_id=0`，原生流程能工作，外部 HF 流程可能警告。

### 4.2 一个 Block 的数据流

```text
x
├─ RMSNorm → GQA Attention → residual add
└─ RMSNorm → SwiGLU FFN / MoE → residual add
```

这是 Pre-Norm decoder-only Transformer。Dense FFN：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

所有投影均无 bias。

RMSNorm 先把输入转成 FP32 计算：

```text
x / sqrt(mean(x²) + eps) × weight
```

再转回输入 dtype，以提高混合精度稳定性。

### 4.3 GQA、RoPE 与 KV Cache

默认张量形状：

```text
x          [B, T, 768]
Q          [B, T, 8, 96]
K/V        [B, T, 4, 96]
缓存 K/V   [B, total_T, 4, 96]
repeat KV  [B, total_T, 8, 96]
scores     [B, 8, T, total_T]
output     [B, T, 768]
```

Q/K 会先做每头 RMSNorm，再应用 RoPE。缓存保存的是已归一化、已旋转的 K，以及 `v_proj(x)` 后但不做旋转的 V。4 个 KV head 各重复两次，服务 8 个 Q head，从而减少缓存占用。

SDPA 快路径只有在这些条件同时满足时启用：

- 当前 `seq_len > 1`；
- 没有历史 cache；
- attention mask 为空或全 1；
- PyTorch 提供 `scaled_dot_product_attention`；
- `flash_attn=True`。

否则走手写 attention，显式构造 causal/padding mask。训练 Dataset 都不传 attention mask；因为使用右 padding，有效 token 不会看到未来 pad，但 padding 仍浪费算力。

RoPE 的 cos/sin buffer：

- 形状 `[32768, 96]`，各一份；
- FP32 总计约 24MiB；
- `persistent=False`，不进入 checkpoint；
- 超出 32768 没有友好错误，常表现为张量形状异常。

YaRN 固定配置：

```text
factor = 16
original_max_position_embeddings = 2048
beta_fast = 32
beta_slow = 1
attention_factor = 1
```

它只缓解位置编码外推，不会凭空赋予模型训练过的长文理解能力。

### 4.4 MoE 的真实含义

每个 token 的 router：

```text
scores = softmax(gate(x))
topk_weight, topk_idx = topk(scores)
```

每个专家都是完整 SwiGLU FFN，无 shared expert。默认 Top-1 且归一化被选概率时，唯一权重约等于 1，因此 router 的主要可学习信号来自辅助负载均衡损失：

```text
load = one_hot(topk_idx).mean(tokens)
aux  = sum(load × mean_router_probability) × num_experts × 5e-4
```

要点：

- `output.loss` 只含语言模型 CE，不自动加 MoE aux；
- 仓库训练脚本显式使用 `loss + aux_loss`；
- 若改用 HF Trainer，必须自行把 aux 加入目标；
- 空闲专家通过一个零值参数连接保留在计算图中，避免 DDP 判为未使用参数；
- 198M 是存储总参数，不等于每个 token 都激活 198M。

### 4.5 Causal LM loss

模型内部完成 next-token shift：

```text
x = logits[..., :-1, :]
y = labels[..., 1:]
CE(x, y, ignore_index=-100)
```

所以 Dataset 通常应返回等长的 `input_ids` 与 `labels`，不要再额外 shift；只有 DPO 不走模型 CE，因此 Dataset 自己构造 `x/y`。

`logits_to_keep` 可只计算最后 K 个位置，RL 用它减少 LM head 计算。训练时不要同时给正数 K 和完整 labels，否则长度可能不匹配。

若一个 SFT 样本因截断导致 labels 全是 `-100`，mean CE 可能变 NaN。后续数据课会专门验证这个风险。

### 4.6 自定义 generate

默认：

```text
max_new_tokens=8192
temperature=0.85
top_p=0.85
top_k=50
do_sample=True
repetition_penalty=1.0
eos_token_id=2
use_cache=True
```

它实现 KV cache、temperature、top-k、top-p、重复惩罚、streamer 和 greedy/sample，但不是完整 HF Generation API：

- 无 beam search、stop string、min tokens、多 EOS 等；
- `temperature=0` 会除零；要 greedy 应设 `do_sample=False`；
- `top_k` 不能超过 vocab size；
- EOS 预期为单个整数；
- 不会自动 `eval()`；
- 默认返回 prompt+completion；
- 输入+输出不能超过 RoPE buffer。

### 4.7 模型最小实践

在项目根目录：

```bash
python - <<'PY'
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

for use_moe in [False, True]:
    cfg = MiniMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=6400,
        use_moe=use_moe,
        max_position_embeddings=128,
    )
    model = MiniMindForCausalLM(cfg)
    ids = torch.randint(3, 6400, (2, 16))
    labels = ids.clone()
    out = model(input_ids=ids, labels=labels)
    n = sum(p.numel() for p in model.parameters())
    print("moe=", use_moe, "params=", n,
          "logits=", tuple(out.logits.shape),
          "ce=", float(out.loss),
          "aux=", float(out.aux_loss))
PY
```

验收：

- logits 为 `[2,16,6400]`；
- Dense aux 为 0；
- MoE aux 非负；
- 总训练目标应是 `out.loss + out.aux_loss`。

---

## 5. 第三课：五类 Dataset 与数据契约

`dataset/lm_dataset.py` 在 import 时设置 `TOKENIZERS_PARALLELISM=false`，以避免多进程 tokenizer 警告。

本节为便于阅读会把一条 JSON 展开成多行；真正保存为 `.jsonl` 时，**每条完整样本必须占一个物理行**，不能直接把展示中的跨行格式逐行写入文件。尤其 `AgentRLDataset` 会对每一物理行单独执行 `json.loads`。

### 5.1 聊天数据增强

`pre_processing_chat`：

- 只要任意消息带 truthy `tools`，整条样本原样返回；
- 非工具样本且首条不是 system 时，有 20% 概率加一条随机中/英文 system；
- 已有 system 不替换；
- 空 conversations 会访问 `[0]` 报错。

`post_processing_chat`：

- chat template 会为无 reasoning 的 assistant 生成精确的空 `<think>\n\n</think>\n\n`；
- 80% 概率删除这个空块，20% 保留；
- 只匹配精确换行格式；
- 每次 `__getitem__` 都可能重新随机。

### 5.2 PretrainDataset

输入 JSONL：

```jsonl
{"text":"一段用于 next-token prediction 的文本。"}
```

处理：

1. tokenizer 截至 `max_length-2`，不自动加 special token；
2. 手动拼 `[BOS] + text + [EOS]`；
3. 右 padding 到固定长度；
4. labels 克隆 input IDs；
5. ID 0 的 label 全改成 `-100`。

模型内部 shift 后实际学习：

```text
BOS → 第一个正文 token
正文 → 下一个正文 token
最后正文 token → EOS
```

限制：

- 无 document packing，短样本 padding 浪费明显；
- 无 attention mask；
- 正文若真实含 ID 0，也会被当 pad 屏蔽；
- `max_length<2` 无校验。

### 5.3 SFTDataset

输入支持普通对话、reasoning 和 Tool Calling：

```json
{"conversations":[
  {"role":"system","content":"你是助手","tools":"[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"},
  {"role":"user","content":"计算 256*37"},
  {"role":"assistant","content":"","reasoning_content":"需要调用计算器","tool_calls":"[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256*37\"}}]"},
  {"role":"tool","content":"{\"result\":\"9472\"}"},
  {"role":"assistant","content":"结果是 9472。"}
]}
```

数据 Features 固定包含：

```text
role, content, reasoning_content, tools, tool_calls
```

`tools` 和 `tool_calls` 的主线文件形式是 JSON 字符串，Dataset 在套模板前 `json.loads`。工具定义必须挂在 system 消息上。

label mask：

- system、user、tool response、padding：`-100`；
- 每一轮 assistant marker 后的内容直到并包含 `<|im_end|>\n` 整个结束边界：参与训练；
- assistant 的 `<think>`、正文、`<tool_call>`、EOS 及 EOS 后的换行 token 都训练；
- 多轮 assistant 全部训练。

风险：

- assistant 边界依赖模板中精确的 `<|im_start|>assistant\n` 与 `<|im_end|>\n`；
- 自定义模板后若边界不匹配，可能全为 `-100`；
- 右截断可能把 assistant 全切掉，导致 CE NaN；
- 非法 tools/tool_calls JSON 直接报错；
- 用户正文若恰好伪造完整 assistant marker，可能被误判。

### 5.4 DPODataset

输入：

```json
{"chosen":[
  {"role":"user","content":"问题"},
  {"role":"assistant","content":"更好的回答"}
],"rejected":[
  {"role":"user","content":"问题"},
  {"role":"assistant","content":"较差的回答"}
]}
```

chosen/rejected 分别套模板、截断、右 padding，再生成 assistant-only mask。因为 DPO 直接比较 token log probability，Dataset 已提前 shift，返回：

```text
x_chosen, y_chosen, mask_chosen
x_rejected, y_rejected, mask_rejected
```

每个形状是 `[max_length-1]`。

mask 的 assistant 结束边界同样是完整的 `<|im_end|>\n`。类中的 `self.padding` 成员被设置但后续未使用，不影响当前结果。

限制：

- chosen 与 rejected 分数是 assistant token logprob **求和**，不做长度归一化；
- 不验证两边 prompt 是否相同；
- 两边独立随机删除空 think；
- mask 全 0 时几乎无偏好信号；
- 这里没有 SFTDataset 那套 tools JSON 解析，更适合普通文本偏好对。

### 5.5 RLAIFDataset

输入与 SFT 类似，但最后必须留一个 assistant 占位：

```json
{"conversations":[
  {"role":"user","content":"解释光合作用"},
  {"role":"assistant","content":""}
]}
```

处理：

1. 可随机加 system；
2. 无条件丢掉最后一条 `conversations[:-1]`；
3. 按 `thinking_ratio` 决定 generation prompt 是否打开 `<think>`；
4. 返回 `{"prompt": prompt_string, "answer": ""}`。

`max_length`、`bos_id`、`eos_id` 在 Dataset 内都没有使用，真正的 tokenization/truncation 在 PPO/GRPO 训练脚本中。若末条不是 assistant 占位，真实消息也会被静默丢弃。

### 5.6 AgentRLDataset

输入：

```json
{"conversations":[
  {"role":"system","content":"可使用工具","tools":"[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"description\":\"计算数学表达式\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"},
  {"role":"user","content":"计算并给出结果"},
  {"role":"assistant","content":""}
],"gt":["9472"]}
```

它逐行手动读 JSONL：

- 从 system 解析 tools；
- 返回 `messages[:-1]`；
- 原样返回 `gt`，实践中必须是 list；
- `max_length` 也不在 Dataset 内使用。

空行、坏 JSON、缺 `gt` 会直接失败；多个 system 都有 tools 时，后一个覆盖前一个。

它不像 SFTDataset 那样主动解析字符串形式的历史 `tool_calls`；若数据含已有工具轨迹，应确保传给模板的类型本来就正确。

### 5.7 数据到 loss 的总接线

| 文件类型 | Dataset | 输出/目标 | 训练脚本 |
|---|---|---|---|
| `{"text":...}` | PretrainDataset | 等长 IDs/labels，模型内 shift CE | Pretrain |
| `conversations` | SFTDataset | assistant-only labels | SFT、LoRA、蒸馏 |
| chosen/rejected | DPODataset | 已 shift 的 x/y/mask | DPO |
| prompt+空 assistant | RLAIFDataset | 字符串 prompt，在线采样 | PPO、GRPO/CISPO |
| tools+gt+空 assistant | AgentRLDataset | messages/tools/gt | Agent |

### 5.8 强烈建议先做 mask 验收

在正式 SFT 前写一个一次性检查：

```bash
python - <<'PY'
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

tok = AutoTokenizer.from_pretrained("./model")
ds = SFTDataset("./dataset/sft_t2t_mini.jsonl", tok, max_length=256)
ids, labels = ds[0]
active = labels.ne(-100)
print("shape:", ids.shape)
print("train tokens:", int(active.sum()), "/", len(ids))
print("full:")
print(tok.decode(ids[ids.ne(tok.pad_token_id)].tolist()))
print("targets:")
print(tok.decode(labels[active].tolist()))
assert active.any(), "样本被截断成全 -100，训练会有 NaN 风险"
PY
```

要同时抽查：

- 普通单轮；
- 多轮；
- 带 reasoning；
- tool call + tool response；
- 很长 user prompt。

验收结论必须是：只有 assistant 的思考、工具调用、正文和 `<|im_end|>\n` 结束边界被训练，用户、系统、工具返回不被训练。

---

## 6. 第四课：训练公共机制

### 6.1 所有正式训练从 `trainer/` 启动

单进程：

```bash
cd trainer
python train_xxx.py
```

单机多卡：

```bash
cd trainer
torchrun --nproc_per_node 2 train_xxx.py
```

对 Pretrain、SFT、LoRA、蒸馏、DPO，`batch_size` 是**每个进程**的数据 microbatch，近似有效全局样本 batch：

```text
batch_size × world_size × accumulation_steps
```

例如 Pretrain 默认单卡有效 batch 是 `32×1×8=256` 条；双卡是 512 条。增加 GPU 而不调 batch/lr 会改变优化条件。

GRPO/Agent 的 `batch_size` 是 prompt 数，实际 policy 轨迹数还要乘 `num_generations`。PPO 则在每批 rollout 内按 `mini_batch_size` 和 `ppo_update_iters` 多次 backward，并在 rollout batch 末尾 flush，不能直接套用上述样本 batch 公式。

### 6.2 初始化和权重命名

`trainer_utils.init_model()` 固定：

- tokenizer 从 `../model` 加载；
- 前置权重从 `../out/<from_weight>_<hidden_size>[_moe].pth` 加载；
- `from_weight=none` 时随机初始化；
- `load_state_dict(strict=False)`。

命名：

```text
Dense: ../out/full_sft_768.pth
MoE:   ../out/full_sft_768_moe.pth
LoRA:  ../out/lora_medical_768.pth
```

同一个模型的连续加载链必须保持 hidden/layers 和 Dense/MoE 架构一致；蒸馏中的 student/teacher 是两套独立配置，允许 Dense 学生配 MoE 教师。`strict=False` 便于附加 value head 等结构，但也可能把错误的缺失/多余 key 隐藏在日志里，加载后应检查输出。

一个重要路径陷阱：`--save_dir` 只改变输出权重目录，下一阶段的 `init_model()` 仍固定从 `../out` 加载。若你把 Pretrain 保存到别处，SFT 不会自动找到它；应复制回符合命名规则的位置，或明确修改加载代码。

### 6.3 优化器、学习率和混合精度

所有脚本使用 AdamW，只显式传学习率，因此其余是 PyTorch 默认值：

```text
betas=(0.9,0.999), eps=1e-8, weight_decay=0.01
```

Pretrain、SFT、LoRA、蒸馏、DPO 的学习率：

```text
lr(step) = base_lr × [0.1 + 0.45 × (1 + cos(π × step / total_steps))]
```

即从约 100% 降至 10%，没有 warmup。PPO、GRPO、Agent 使用 `CosineAnnealingLR`，最低同样为初始值的 1/10。

公共机制：

- CUDA 默认使用 BF16 autocast；CPU/MPS 走 `nullcontext`，`--dtype bfloat16` 不会自动把训练变成 BF16；
- 传 `--dtype float16` 时，前五类监督/偏好脚本使用 GradScaler；
- PPO/GRPO/Agent 不使用 GradScaler，FP16 稳定性需自行验证；
- dtype 没有 choices 校验：在 CUDA 上，非 `bfloat16` 字符串会进入 FP16 autocast，但只有精确的 `float16` 才开 scaler；
- 默认 grad clip 为 1.0；
- 可选 `--use_compile 1`，但 LoRA 会自动关闭；
- 默认 DataLoader workers 为 8，低内存或 Windows 环境可调成 0～2。

训练入口按 `42+rank` 设置 Python、NumPy、Torch 和 CUDA seed，同时关闭 cuDNN benchmark 并启用 deterministic；这提高可复现性，但不等于前述 resume 能逐 bit 复现。

### 6.4 DDP 的适用边界

普通 Pretrain/SFT/LoRA/蒸馏/DPO 会通过 DDP wrapper 做 forward。

需要谨慎的是：

- GRPO、Agent，以及 PPO 的 actor/critic 更新显式取 `.module` 后 forward，绕过标准 DDP forward 生命周期；
- 参数上仍可能安装了 DDP hooks，但多卡同步是否严格等价需要单独验证；
- teacher、reference、reward model 会在每张卡各复制一份，不做参数分片；
- 仓库 README 曾提及 DeepSpeed，但当前提交没有 DeepSpeed 配置、launcher 或训练接线；仓库内直接可复现的是原生 PyTorch DDP。

因此普通 SFT 可放心先从 DDP 入门；PPO/GRPO/Agent 应先做单卡与双卡小数据权重变化对比，再投入长跑。

### 6.5 日志

`--use_wandb` 这个参数名沿用历史习惯，当前代码实际：

```python
import swanlab as wandb
```

所以对应的是 SwanLab。`wandb_project` 决定项目名，resume checkpoint 还会保存 run id。没有 `--use_wandb` 时只输出终端日志。

### 6.6 三类保存文件

```text
../out/<name>_<hidden>[_moe].pth
../checkpoints/<name>_<hidden>[_moe].pth
../checkpoints/<name>_<hidden>[_moe]_resume.pth
```

- `out`：CPU FP16 `state_dict`，供推理或下一阶段加载；
- `checkpoints` 不带 `_resume`：也是模型权重；
- `_resume`：模型、optimizer、epoch、step、world size、SwanLab run id，以及 scaler/scheduler/critic 等附加状态；
- 同名文件每次覆盖，不保留每个 step 的历史；
- checkpoint 通过临时文件 + `os.replace` 原子替换；
- `out` 的直接保存不是原子的；
- LoRA 的 `out` 只含 adapter，而 resume 包含“基模+LoRA”，所以大很多；
- `--save_dir` 不改变 resume 目录，resume 仍固定在 `../checkpoints`。

### 6.7 断点恢复

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
```

恢复后 `SkipBatchSampler` 跳过当前 epoch 已完成的 batch。GPU 数变化时：

```text
new_step = old_step × old_world_size // new_world_size
```

这不是 bitwise 精确恢复：

- 不保存 Python/NumPy/Torch RNG 完整状态；
- Dataset 的随机 system 和 think 删除不能精确复现；
- GPU 数改变后 step 是整数近似；
- RL scheduler 从旧 checkpoint 恢复，可能不匹配新 world size；
- 梯度累积中途的未提交梯度不会保存；
- 最后一个不完整 accumulation 可能在最终保存后才 optimizer step，更新没有再落盘。

实践建议：

- `save_interval` 尽量与 `accumulation_steps` 对齐；
- 恢复后先看前几个 step 的 loss 与学习率；
- 对重要实验额外复制有意义的里程碑权重，而不是依赖被覆盖的单文件；
- 训练脚本的 `checkpoints/` 目前没有被 `.gitignore` 忽略，避免误提交大文件。

### 6.8 一个统一的训练前检查

```bash
cd trainer
python - <<'PY'
from pathlib import Path
import torch
from transformers import AutoTokenizer

required = [
    Path("../model/tokenizer.json"),
    Path("../model/tokenizer_config.json"),
]
for p in required:
    assert p.exists(), p

tok = AutoTokenizer.from_pretrained("../model")
assert len(tok) == 6400
assert tok.bos_token_id == 1
assert tok.eos_token_id == 2
assert tok.pad_token_id == 0
print("device CUDA:", torch.cuda.is_available())
print("tokenizer contract: OK")
PY
```

---

## 7. 第五课：从零完成 Pretrain

### 7.1 目标

预训练让随机模型学习语言统计与基础知识，本质目标是 next-token prediction。它不是聊天模型，测试时要把输入当成“续写前缀”。

### 7.2 默认配置

| 参数 | 默认值 |
|---|---:|
| `save_weight` | `pretrain` |
| `epochs` | 2 |
| `batch_size` | 32 |
| `learning_rate` | `5e-4` |
| `accumulation_steps` | 8 |
| `max_seq_len` | 340 |
| hidden/layers | 768 / 8 |
| `use_moe` | 0 |
| 数据 | `../dataset/pretrain_t2t_mini.jsonl` |
| log/save interval | 100 / 1000 |
| dtype | BF16 |
| `from_weight` | `none` |

README 对 mini 数据建议 `max_seq_len≈768`，但当前代码默认是 **340**；完整 `pretrain_t2t` 的 README 建议约 380。两者不是同一个概念：

- “代码默认”保证你不传参数时实际发生什么；
- “README 推荐”是对当前数据长度分布的经验实验值。

长序列 attention 显存/计算近似按长度平方增长，先用默认 340 跑通，再按资源试 768。

### 7.3 第一次建议先做短 smoke run

仓库没有 `--max_steps`，可先从原 JSONL 复制少量行到一个临时教学文件。不要修改公开数据本体。例如在项目根目录：

```bash
python - <<'PY'
from itertools import islice
from pathlib import Path

src = Path("dataset/pretrain_t2t_mini.jsonl")
dst = Path("dataset/pretrain_smoke.jsonl")
with src.open("r", encoding="utf-8") as r, dst.open("w", encoding="utf-8") as w:
    for line in islice(r, 256):
        w.write(line)
print(dst)
PY
```

然后：

```bash
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain_smoke.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --num_workers 0 \
  --max_seq_len 128 \
  --log_interval 1 \
  --save_interval 20 \
  --save_weight pretrain_smoke
```

接线测试结束后可验证产物能被加载：

```bash
python eval_llm.py --weight pretrain_smoke --max_new_tokens 128
```

它不会得到好模型。若要实际验证 resume，把 smoke 设为 `--epochs 2`、较短 `save_interval`，看到 checkpoint 后中断，再用完全相同参数追加 `--from_resume 1`，检查是否从保存 step 继续。正式 Dense：

```bash
cd trainer
python train_pretrain.py
```

正式 MoE：

```bash
cd trainer
python train_pretrain.py --use_moe 1
```

多卡示例：

```bash
cd trainer
torchrun --nproc_per_node 2 train_pretrain.py
```

### 7.4 产物与验收

```text
Dense: ../out/pretrain_768.pth
MoE:   ../out/pretrain_768_moe.pth
```

回到项目根目录：

```bash
python eval_llm.py --load_from ./model --weight pretrain --max_new_tokens 128
```

MoE：

```bash
python eval_llm.py --load_from ./model --weight pretrain --use_moe 1 --max_new_tokens 128
```

验收：

- loss 是有限值并总体下降；
- MoE 同时监控 CE 与 aux；当前脚本没有逐专家负载日志，若要判断专家坍塌，还需在 router 处记录每个 expert 的 token count；
- 文件能被加载；
- 生成至少形成连贯局部文本；
- 不用聊天指令服从能力要求 Pretrain 权重。

### 7.5 训练开销如何理解

README 在单张 RTX 3090、**1 epoch** 上估计：

| 模型 | mini Pretrain | mini SFT | 两阶段合计 |
|---|---:|---:|---:|
| Dense 64M | 约 1.21h / ¥1.57 | 约 1.10h / ¥1.43 | 约 2.31h / ¥3.0 |
| MoE 198M-A64M | 约 1.69h / ¥2.20 | 约 1.54h / ¥2.00 | 约 3.23h / ¥4.2 |

README 的“约 2 小时”指 Dense 的 mini Pretrain+SFT 一轮估算；当前两个脚本默认都是 2 epochs，所以默认命令不能机械等同于这张一轮表。硬件、PyTorch、序列长度、compile、I/O 和 batch 都会改变结果。

---

## 8. 第六课：Full SFT 得到聊天模型

### 8.1 前置条件

默认寻找：

```text
../out/pretrain_768.pth
```

MoE 则找 `_moe` 文件。数据是 `sft_t2t_mini.jsonl`；其中已经混入约 10 万条 Qwen3-4B 合成 Tool Call 数据以及 reasoning 数据，所以当前版本通常不需要独立 Tool Calling SFT。

### 8.2 默认配置

| 参数 | 默认值 |
|---|---:|
| `save_weight` | `full_sft` |
| `epochs` | 2 |
| `batch_size` | 16 |
| `learning_rate` | `1e-5` |
| `accumulation_steps` | 1 |
| `max_seq_len` | 768 |
| hidden/layers | 768 / 8 |
| `use_moe` | 0 |
| 数据 | `../dataset/sft_t2t_mini.jsonl` |
| `from_weight` | `pretrain` |
| log/save interval | 100 / 1000 |

### 8.3 运行

Dense：

```bash
cd trainer
python train_full_sft.py
```

MoE：

```bash
cd trainer
python train_full_sft.py --use_moe 1
```

多卡：

```bash
cd trainer
torchrun --nproc_per_node 2 train_full_sft.py
```

输出：

```text
../out/full_sft_768.pth
../out/full_sft_768_moe.pth
```

### 8.4 CLI 验收

回到根目录：

```bash
python eval_llm.py --weight full_sft --max_new_tokens 512
```

显式思考：

```bash
python eval_llm.py --weight full_sft --open_thinking 1 --max_new_tokens 512
```

携带最近两轮 user/assistant 历史：

```bash
python eval_llm.py --weight full_sft --historys 4 --max_new_tokens 512
```

验收至少覆盖：

- 普通问答是否遵循 assistant 格式；
- 多轮指代；
- 中英文；
- 不打开 thinking 时能否直接回答；
- 打开 thinking 时是否闭合 `</think>`；
- 简单工具 schema 下能否生成合法 `<tool_call>`；
- 长输入截断边界；
- 事实幻觉、重复和格式失败率，而不仅仅看 loss。

这一级别模型的容量非常小。README 自己展示的主观样例包含明显知识错误与重复退化，因此“能生成流畅文字”绝不等于“答案可靠”。

---

## 9. 第七课：手写 LoRA 垂域适配

### 9.1 这不是 PEFT LoRA

仓库的 LoRA 数学：

```text
Linear(x) + B(A(x))
A: in_features → rank
B: rank → out_features
```

默认 rank 16：

- A 用标准差 0.02 的正态初始化；
- B 全零，所以注入瞬间输出不变；
- 无 alpha/rank 缩放；
- 无 LoRA dropout；
- 只给 `in_features == out_features` 的方形 Linear 注入。

在默认 8Q/4KV、vocab=6400 的 8 层模型中，实际命中每层：

- `q_proj: 768→768`；
- `o_proj: 768→768`。

不命中 K/V、FFN、MoE 专家、router、LM head。总 LoRA 参数：

```text
16 个 adapter × (768×16 + 16×768) = 393,216
```

### 9.2 准备数据

格式与 SFT 完全一致，例如 `dataset/lora_medical.jsonl`：

```jsonl
{"conversations":[{"role":"user","content":"颈椎病患者怎样选择枕头？"},{"role":"assistant","content":"应结合颈椎曲度、睡姿和医生建议选择合适高度……"}]}
```

LoRA 不会自动保护通用能力；小而重复的数据仍会过拟合。应准备独立验证问题，并确保基模与训练时完全一致。

### 9.3 默认配置与运行

| 参数 | 默认值 |
|---|---:|
| `lora_name` | `lora_medical` |
| rank | 训练 CLI 未暴露；当前调用使用 `apply_lora` 默认值 16 |
| epochs/batch | 10 / 32 |
| lr | `1e-4` |
| max length | 340 |
| base | `full_sft` |
| log/save | 10 / 1000 |

```bash
cd trainer
python train_lora.py
```

自定义：

```bash
cd trainer
python train_lora.py \
  --lora_name lora_identity \
  --data_path ../dataset/lora_identity.jsonl \
  --epochs 5
```

`torch.compile` 会被自动关闭。输出只包含 adapter：

```text
../out/lora_medical_768.pth
```

### 9.4 推理与验收

```bash
python eval_llm.py \
  --weight full_sft \
  --lora_weight lora_medical \
  --max_new_tokens 512
```

验收：

1. 加载的基础权重必须与训练时的 `from_weight`、hidden/layers、Dense/MoE 一致；
2. 输出文件 key 应只有 `.lora.A.weight` / `.lora.B.weight`；
3. 注入前后、尚未训练时输出应相同，因为 B=0；
4. 比较“基模 vs 基模+LoRA”的垂域集和通用集；
5. adapter 加载前必须先 `apply_lora`，否则 `load_lora` 可能什么也没加载却不主动报错；
6. 不要在同一模型上重复 `apply_lora`。

查看权重：

```bash
python - <<'PY'
import torch
d = torch.load("out/lora_medical_768.pth", map_location="cpu")
print("tensors:", len(d))
print(*list(d)[:8], sep="\n")
assert d and all(".lora." in k for k in d)
PY
```

### 9.5 合并 LoRA

`model_lora.merge_lora` 计算：

```text
W_merged = W_base + B @ A
```

`scripts/convert_model.py` 提供 `convert_merge_base_lora`，但该脚本没有 argparse，而且底部默认执行的是 `convert_torch2transformers(...)`，**直接运行不会合并 LoRA**。应先注释默认转换调用，取消 LoRA merge 段的注释，并核对名称与 Dense/MoE 后缀。底部应等价于：

```python
if __name__ == "__main__":
    lm_config = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=8192,
        use_moe=False,
    )
    base_torch_path = "../out/full_sft_768.pth"
    lora_path = "../out/lora_identity_768.pth"
    merged_torch_path = "../out/merge_identity_768.pth"
    convert_merge_base_lora(
        base_torch_path,
        lora_path,
        merged_torch_path,
    )
```

修改并复核路径后才运行：

```bash
cd scripts
python convert_model.py
```

若是 MoE，三个路径都应按实际产物使用 `_moe` 后缀，且 `use_moe=True`。`B@A` 在当前模型 device/dtype 上计算，随后 base 与 delta 转成 CPU FP16 相加并保存，因此会有 FP16 舍入误差。合并结果删除 `.lora.` key，成为普通完整权重。只加载可信 `.pth`，因为 `torch.load` 涉及 pickle 反序列化。

---

## 10. 第八课：白盒知识蒸馏

### 10.1 当前实现是什么

黑盒蒸馏是拿教师生成文本继续做 SFT；主线 SFT 数据已经包含这类合成信号。`train_distillation.py` 实现的是白盒 token-distribution 蒸馏：

```text
L = α × CE(y, student)
  + (1-α) × T² × KL(teacher/T || student/T)
```

KL 只在 assistant 非 `-100` token 上计算；教师冻结、eval、`no_grad`。若学生是 MoE，router aux 加进 CE 分支，所以也会被 α 缩放。

### 10.2 默认前置权重是一个隐含门槛

默认：

```text
学生：../out/full_sft_768.pth       Dense
教师：../out/full_sft_768_moe.pth   MoE
```

如果你只完成了 Dense 主线，直接运行一定会缺教师权重。你要么先完整训练/下载同规格 MoE SFT，要么显式改变 teacher 配置和命名。

教师 logits 会裁到学生 vocab size，但二者仍应使用同一 tokenizer。教师模型不做 DDP 分片，每个 rank 都复制一份。

### 10.3 默认配置

| 参数 | 默认值 |
|---|---:|
| 输出 | `full_dist` |
| epochs/batch | 6 / 32 |
| lr | `5e-6` |
| max length | 340 |
| 数据 | `sft_t2t_mini.jsonl` |
| student | 768×8 Dense，`full_sft` |
| teacher | 768×8 MoE，`full_sft` |
| α / T | 0.5 / 1.5 |
| save interval | 100 |

运行：

```bash
cd trainer
python train_distillation.py
```

输出：

```text
../out/full_dist_768.pth
```

测试：

```bash
python eval_llm.py --weight full_dist --max_new_tokens 512
```

使用 Dense 教师做接线实验：

```bash
cd trainer
python train_distillation.py \
  --teacher_use_moe 0 \
  --from_teacher_weight full_sft
```

此时学生/教师完全同初始化，只适合验证实现，不代表有价值的知识转移。真正实验应选择更强教师，并同步指定：

```text
teacher_hidden_size
teacher_num_layers
teacher_use_moe
from_teacher_weight
```

### 10.4 验收

- 开始前确认学生和教师两个确切文件都存在；
- CE、KL、total loss 都是有限值；
- 教师无梯度；
- tokenizer/vocab 一致；
- 比较蒸馏前后固定验证集，而不是只看训练 KL；
- 更大教师会显著增加显存，不要按单模型显存估算。

---

## 11. 第九课：DPO 偏好对齐

### 11.1 模型与目标

DPO 同时加载：

```text
可训练 policy = full_sft
冻结 reference = 同一个 full_sft
```

对 chosen/rejected 的 assistant token logprob 求和：

```text
Δπ   = logπ(chosen|x) - logπ(rejected|x)
Δref = logref(chosen|x) - logref(rejected|x)

L = -log sigmoid[β(Δπ - Δref)] + MoE aux
```

不需要 reward model 或 critic。由于 logprob 按 token 求和、没有长度归一化，回答长度会影响序列分数。

### 11.2 默认配置与运行

| 参数 | 默认值 |
|---|---:|
| 输出 | `dpo` |
| epochs/batch | 1 / 4 个偏好对 |
| 拼接后序列 batch | 8 |
| lr | `4e-8` |
| beta | 0.15 |
| max length | 1024 |
| 数据 | `dpo.jsonl` |
| base/ref | `full_sft` |
| save interval | 100 |

```bash
cd trainer
python train_dpo.py
```

输出：

```text
../out/dpo_768.pth
```

测试：

```bash
python eval_llm.py --weight dpo --max_new_tokens 512
```

脚本帮助明确建议学习率不高于 `5e-8`，以降低灾难性遗忘。DPO 是离线静态偏好优化，不会在线探索，也不能仅凭“chosen 胜出”推断知识推理能力提升。

### 11.3 验收

- 抽样人工确认 chosen/rejected 没有写反且 prompt 相同；
- 两边 assistant mask 非空；
- loss 非 NaN；
- 监控 chosen 与 rejected 相对 margin；
- 用固定通用问答集检查低学习率下能力是否坍塌；
- 偏好/安全改善与可验证任务正确率分开评估。

---

## 12. 第十课：先理解在线 RL 的共用部件

### 12.1 RLAIF 在这里的含义

仓库把 reward model、规则函数、Ground Truth 校验和环境反馈都广义归入自动反馈。对 0.1B 级模型，纯 0/1 数学奖励很容易全部为 0，组内没有差异，策略梯度消失。因此普通 PPO/GRPO 示例使用连续 reward model 分数并叠加格式规则；Agent 则更多依赖工具和 GT 的可验证奖励。

默认 reward model：

```text
InternLM2-1.8B-Reward
```

下载来源：[ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward) / [Hugging Face](https://huggingface.co/internlm/internlm2-1_8b-reward)。

建议放在项目同级：

```text
root/
├── minimind/
└── internlm2-1_8b-reward/
```

因为训练在 `minimind/trainer` 下启动，默认路径才是：

```text
../../internlm2-1_8b-reward
```

`LMForRewardModel` 使用：

```python
AutoTokenizer.from_pretrained(..., trust_remote_code=True)
AutoModel.from_pretrained(..., trust_remote_code=True)
reward_model.get_score(tokenizer, messages)
```

它要求模型实现自定义 `get_score`，不是任意 `AutoModelForSequenceClassification` 都能替换。单个 RM 分数裁剪到 `[-3,3]`。`trust_remote_code=True` 会执行模型仓库代码，只使用可信来源和固定 revision。

### 12.2 PPO/GRPO 共用奖励

每个回答：

```text
长度 20～800 字符：+0.5，否则 -0.5
若出现 </think>：
  think 前内容长度 20～300：+1.0，否则 -0.5
  恰好一个 </think>：+0.25，否则 -0.25
减去 3-gram 重复惩罚，最高 0.5
加 reward model 分数 [-3,3]
```

注意：

- thinking 规则只查 `</think>`，不严格验证开标签；
- 重复惩罚用 `\w+|[^\w\s]` 分词，连续中文可能被视作一个长 token，对无标点中文重复不敏感；
- PPO/GRPO 的总奖励没有再次整体 clip；
- reward 上升不等于通用能力上升，可能出现刻意凑长度、think 标签和 RM 偏好的 reward hacking。

### 12.3 RolloutResult

`rollout_engine.py` 统一返回：

```text
output_ids
completion_ids
per_token_logps
completions
prompt_lens
completion_mask
```

Torch 后端：

- prompt 按 generation 数重复；
- 调原生 `model.generate`；
- rollout temperature 固定传 0.8；
- generate 仍使用默认 top-p=0.85、top-k=50；
- 再用 policy 计算 completion 每 token 的 old logprob；
- 始终引用当前训练模型。

SGLang 后端：

- 向 `/generate` 发 HTTP 请求并索取 token logprob；
- 当前请求只显式发送 temperature、最大 token 和 stop IDs；Torch 还使用 top-p=0.85、top-k=50，因此两后端并非严格同采样分布；
- rank 0 把当前 policy 以 FP16 Transformers 格式写到共享目录；
- 调 `/update_weights_from_disk` 让 server 重载；
- trainer 和 server 必须看到同一个绝对共享路径；
- tokenizer 必须完全一致；
- 没有重试或自动回退；
- logprob 缺失时左侧补 0，可能静默污染 importance ratio；
- 只在初始化和每个 `save_interval` 同步，期间 rollout policy 滞后；
- 这是同步训推分离，不是异步 rollout buffer。

启动 SGLang：

```bash
python -m sglang.launch_server \
  --model-path /absolute/path/to/transformers-model \
  --attention-backend triton \
  --host 0.0.0.0 \
  --port 8998
```

SGLang 不在 requirements 中，且通常需要 CUDA。

---

## 13. 第十一课：PPO

### 13.1 驻留模型

```text
Actor MiniMind（训练）
Reference MiniMind（冻结）
Critic MiniMind + value_head（训练）
InternLM2-1.8B Reward Model（冻结）
Rollout engine
```

Critic 保留完整 CausalLM backbone 和未使用的 LM head，再加 `Linear(hidden,1)` value head，因此存在额外显存占用。基础模型权重用 `strict=False` 加载，value head 随机初始化。当前 `CriticModel.forward()` 还会对 backbone 已完成 final RMSNorm 的 hidden states 再调用一次 `model.norm`，形成双重 final RMSNorm；这是本仓库实现细节，不是 PPO 算法要求。

### 13.2 算法流程

1. 每个 prompt 生成 1 个回答；
2. 外部序列级奖励只放到回答最后一个有效 token；
3. critic 给出每 token value；
4. GAE 默认 `γ=1.0, λ=0.95`；
5. 有效回答 token 的 advantage 在全 batch 归一化；
6. 同一 rollout 做 2 轮 PPO update；
7. policy ratio：

   ```text
   ratio = exp(logp_new - logp_old)
   ```

8. actor 采用 ε=0.2 clip；
9. reference KL 使用非负近似：

   ```text
   d = logp_ref - logp_policy
   KL = exp(d) - d - 1
   ```

10. KL 系数 0.02；
11. value loss clip 0.2、系数 0.5；
12. approximate KL 超过 0.25 时提前停止本批更新；多卡会 all-reduce 后共同决定。

### 13.3 默认参数

| 参数 | 默认值 |
|---|---:|
| 输出 | `ppo_actor` |
| epochs/batch | 1 / 2 |
| actor lr | `3e-7` |
| critic lr | `5e-7` |
| prompt/gen length | 768 / 1024 |
| accumulation | 1 |
| minibatch / update iters | 2 / 2 |
| clip / value clip | 0.2 / 0.2 |
| value coef / KL coef | 0.5 / 0.02 |
| gamma / lambda | 1.0 / 0.95 |
| early-stop KL | 0.25 |
| 数据 | `rlaif.jsonl` |
| base | `full_sft` |
| reward model | `../../internlm2-1_8b-reward` |
| thinking ratio | 0.9 |
| rollout | Torch |
| save interval | 10 |

### 13.4 运行

```bash
cd trainer
python train_ppo.py --debug_mode
```

输出：

```text
../out/ppo_actor_768.pth
```

测试：

```bash
python eval_llm.py --weight ppo_actor --max_new_tokens 512
```

SGLang：

```bash
cd trainer
python train_ppo.py \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_model_path /absolute/path/to/model-tokenizer \
  --sglang_shared_path /absolute/shared/path/ppo
```

resume 除普通状态外还保存 critic、critic optimizer、actor/critic scheduler。

### 13.5 PPO 验收

同时看：

- reward 均值和长度；当前代码未记录 reward 方差，如需监控应补 `rewards.var(unbiased=False)`；
- actor loss、critic/value loss；
- approximate KL、reference KL；
- clip fraction；
- 生成样本和 think 格式；
- 固定通用集与事实集。

若只有 reward 上升，回答却变长、重复或通用能力下降，应判为失败。`log_interval` 虽存在，当前训练日志实际每 step 输出。默认 RM 强制 FP16；主训练 BF16 不改变它。

---

## 14. 第十二课：GRPO 与 CISPO

### 14.1 脚本名不等于默认算法

`train_grpo.py` 同时实现两种 loss，但默认：

```text
loss_type=cispo
```

要运行经典 GRPO 必须显式：

```bash
python train_grpo.py --loss_type grpo
```

### 14.2 组内优势

每个 prompt 默认生成 6 个回答：

```text
A_i = (R_i - group_mean) / (group_std + 1e-4)
```

同一回答的所有 token 共用一个序列级 advantage。

GRPO：

```text
ratio = exp(logp_new - logp_old)
objective = min(ratio×A, clip(ratio,1-ε,1+ε)×A)
```

CISPO：

```text
weight = min(ratio, epsilon_high).detach()
objective = weight × A × logp_new
```

二者都加：

```text
beta × [exp(logp_ref-logp_policy)
        -(logp_ref-logp_policy)-1]
```

先按每条回答有效 token 数做长度归一化，再对样本平均。

### 14.3 默认参数与运行

| 参数 | 默认值 |
|---|---:|
| 输出 | `grpo` |
| epochs/batch | 1 / 2 prompts |
| lr | `3e-7` |
| prompt/gen | 768 / 1024 |
| generations/prompt | 6 |
| beta | 0.1 |
| loss | `cispo` |
| epsilon / high | 0.2 / 5.0 |
| 数据 | `rlaif.jsonl` |
| RM | `../../internlm2-1_8b-reward` |
| thinking | 0.9 |
| rollout | Torch |
| save interval | 10 |

CISPO（显式改名，避免覆盖 GRPO 实验）：

```bash
cd trainer
python train_grpo.py \
  --loss_type cispo \
  --save_weight cispo \
  --debug_mode
```

GRPO：

```bash
cd trainer
python train_grpo.py \
  --loss_type grpo \
  --save_weight grpo \
  --debug_mode
```

输出：

```text
../out/cispo_768.pth
../out/grpo_768.pth
```

分别测试：

```bash
python eval_llm.py --weight cispo --max_new_tokens 512
python eval_llm.py --weight grpo --max_new_tokens 512
```

若两种 loss 都使用默认 `save_weight=grpo`，out 权重、普通 checkpoint 和 resume checkpoint 会互相覆盖；恢复时也必须继续使用与原实验相同的 `save_weight`。SGLang 参数与 PPO 相同，默认共享目录是 `./sglang_ckpt_grpo`。

### 14.4 退化条件与验收

- `num_generations=1` 时组内 advantage 近似恒 0，无法学习；
- 同组奖励完全相同也会退化；
- Torch 每批只更新一次，old/new 初始通常接近，ratio≈1，clip 作用有限；
- SGLang 每 10 step 默认才同步，ratio 偏离可能明显；
- RM 默认每 step 要串行打分 `batch 2 × generation 6 = 12` 次，是吞吐瓶颈；
- 日志中的 policy loss 在 MoE 时可能包含 aux；
- 多卡 forward 绕过 DDP wrapper，需验证。

最关键的监控是**组内 reward std**。若持续接近 0，立即检查题目难度、reward model 分辨率和规则，不要靠延长训练等待奇迹。

当前 `train_grpo.py` 只记录归一化后的 `advantages_std`，没有记录原始组内 reward std；二者不能互相替代。执行上述验收前应增加：

```python
group_reward_std = grouped_rewards.std(
    dim=1,
    unbiased=False,
).mean()
```

并把它写入终端/SwanLab。`train_agent.py` 已经记录 `group_reward_std`。GRPO/Agent 的 debug 当前只显示总 reward；若要诊断 reward composition，还需在 `calculate_rewards()` 中分别记录 RM、长度、thinking、重复等分项。

---

## 15. 第十三课：多轮 Agentic RL

### 15.1 它和普通 GRPO 的区别

普通 GRPO 是 prompt→单次回答→打分；Agent 是：

```text
问题
→ 模型生成 tool call
→ 解析和执行工具
→ tool response 回填上下文
→ 模型继续生成
→ 最多 3 轮
→ 整条轨迹结束后结算延迟奖励
```

策略 loss 只覆盖模型生成 token；工具 observation 虽进入后续上下文，mask 为 0。

### 15.2 数据要求

下例为阅读而换行，写入 `agent_rl*.jsonl` 时必须压成单个物理行：

```json
{"conversations":[
  {"role":"system","content":"你可以调用工具","tools":"[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"parameters\":{\"type\":\"object\",\"properties\":{\"expression\":{\"type\":\"string\"}},\"required\":[\"expression\"]}}}]"},
  {"role":"user","content":"256乘37是多少？"},
  {"role":"assistant","content":""}
],"gt":["9472"]}
```

`gt` 应为 list。训练实际使用每条数据携带的工具 schema，但执行器只认识六个固定名称：

- `calculate_math`
- `unit_converter`
- `get_current_weather`
- `get_current_time`
- `get_exchange_rate`
- `translate_text`

README/`eval_toolcall.py` 的演示还包含 random number、text length 等工具，但当前 `train_agent.py` 执行器没有它们。这是当前版本的真实差异；训练数据 schema 与执行器必须对齐。

天气、时间、汇率、翻译、单位换算均为硬编码模拟数据，不是实时服务。温度转换只有倍率，没有摄氏/华氏偏移，不能当真实换算器。

### 15.3 Rollout 成本

默认每个 prompt 生成 4 条轨迹，最多 3 轮，batch=2。内部逐样本、逐 generation、逐轮串行，最坏每 step：

```text
2 × 4 × 3 = 24 次 generate
```

模型生成的 EOS 被过滤；工具 observation 被重新套模板、tokenize 后拼入，并设 policy mask 0。当前每轮 rollout 的 prompt 没有长度截断；完整轨迹生成结束、打包成训练样本后，若超过 `max_total_len` 才会从**左侧**截断，可能删掉原问题和工具定义。

### 15.4 奖励

先对每轮 `<tool_call>` 开闭标签数量差，每个差额扣 0.5。

无可解析工具调用：

```text
回答长度 5～800：+0.5，否则 -0.5
若有 </think>：
  thinking 20～300：+1，否则 -0.5
  恰好一个闭合：+0.25，否则 -0.25
加 RM 分数
减重复惩罚
总分 clip 到 [-3,3]
```

存在工具调用：

```text
检查工具名在样本 schema 中
检查固定执行器要求的必填字段存在
有效调用数与 len(gt) 完全对齐：+0.5
否则按差距每项 -0.5
最终回答每命中一个 GT：+2.5/len(gt)
3 轮后仍未完成：-0.5
减重复惩罚
总分 clip 到 [-3,3]
```

有工具调用的分支不使用 reward model。GT 先做普通子串匹配，再做数值匹配，因此 `"1"` 可能误命中 `"10"`，有 reward hacking 空间。参数只检查存在性，不验证语义。一个工具返回多个 GT 时，“调用数等于 GT 数”的假设也可能不合理。

### 15.5 默认参数

| 参数 | 默认值 |
|---|---:|
| 代码默认输出 | `agent` |
| epochs/batch | 1 / 2 |
| lr | `3e-7` |
| `max_seq_len` | 1024；当前实现未用它截断 prompt |
| 单轮生成 max | 768 |
| 最终训练 pack max | 2500；完成 rollout 后超出才左截断 |
| generations | 4 |
| max turns | 3，代码硬编码 |
| beta | 0.1 |
| loss | CISPO |
| epsilon/high | 0.2 / 5.0 |
| 数据 | `agent_rl.jsonl` |
| base | `full_sft` |
| reward model | `../../internlm2-1_8b-reward` |
| thinking | 0.1 |
| rollout | Torch |
| save interval | 10 |

当前 `--max_seq_len` 只与 `max_gen_len` 相加，作为额外 config 属性传入 Dataset；模型和 Dataset 都没有用它实施 prompt 截断，模型实际 RoPE buffer 仍由 `max_position_embeddings=32768` 控制。真正生效的是每轮 `max_gen_len`，以及轨迹完成后的 `max_total_len`。

把 `--max_total_len` 降到 1792 可以作为降低训练显存的保守选择，但不能限制 rollout 的中间上下文。若需要严格 token budget，应在 `rollout_single()` 每轮 tokenize 后显式截断，同时保护 system/tool schema 和最近消息。

### 15.6 运行

Torch/CISPO（显式改名以隔离实验）：

```bash
cd trainer
python train_agent.py \
  --loss_type cispo \
  --save_weight agent_cispo \
  --debug_mode
```

真正使用 GRPO loss：

```bash
cd trainer
python train_agent.py \
  --loss_type grpo \
  --save_weight agent_grpo \
  --debug_mode
```

SGLang：

```bash
cd trainer
python train_agent.py \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_model_path /absolute/path/to/model-tokenizer \
  --sglang_shared_path /absolute/shared/path/agent \
  --data_path ../dataset/agent_rl_math.jsonl \
  --save_weight agent_cispo \
  --use_wandb
```

输出：

```text
../out/agent_cispo_768.pth
../out/agent_grpo_768.pth
```

若两种 loss 都保留默认 `save_weight=agent`，它们会覆盖同一组 out/checkpoint/resume 文件；resume 时必须使用原实验相同的 `save_weight`。

测试：

```bash
cd scripts
python eval_toolcall.py --weight agent_cispo
```

### 15.7 Agent 验收与安全

每次调试打印完整轨迹，核对：

- tool call JSON 是否可解析；
- name 与 schema/执行器是否一致；
- observation 是否正确回填且 mask=0；
- GT 是否只在最终回答中被命中；
- unfinished 惩罚是否发生；
- 组内 reward std 是否非零；
- 是否出现故意复述 GT、滥调工具等 reward hacking。

debug mode 会打印 context、completion、长度和总 reward，但不会打印实际 mask tensor；要严格核对 observation mask，应额外输出 `full_response_masks` / `completion_mask`。

安全限制：

- `calculate_math` 使用清空 builtins 的 Python `eval` 和 1 秒 `SIGALRM`，仍只适合受控教学；
- `SIGALRM` 依赖 Unix 主线程，Windows 不兼容；
- JSON 解析失败会被静默忽略；
- 模拟工具不是实际网络/API 能力；
- 生产环境应使用白名单解析器、独立沙箱、权限边界和真实工具返回校验。

---

## 16. 第十四课：CLI 推理与 YaRN

### 16.1 Transformers 格式

在项目根目录：

```bash
python eval_llm.py --load_from ./minimind-3
```

### 16.2 原生 PyTorch 格式

```bash
python eval_llm.py \
  --load_from ./model \
  --save_dir out \
  --weight full_sft
```

MoE：

```bash
python eval_llm.py \
  --load_from ./model \
  --weight full_sft \
  --use_moe 1
```

LoRA：

```bash
python eval_llm.py \
  --load_from ./model \
  --weight full_sft \
  --lora_weight lora_medical
```

启动后：

```text
[0] 自动测试   依次跑 8 个内置问题
[1] 手动输入   输入空行结束
```

Pretrain 处理通过 `if 'pretrain' in args.weight` 判断：命中时直接拼 BOS+prompt，不套聊天模板。

参数生效边界：

| 加载方式 | 生效参数 |
|---|---|
| 原生 `./model` + `.pth` | `save_dir`、`weight`、`lora_weight`、hidden/layers、`use_moe`、`inference_rope_scaling` |
| Transformers 目录 | 架构、Dense/MoE、RoPE 都读取目录内 config；上述本地构造参数和 LoRA 参数被忽略 |
| 两者通用 | max new tokens、temperature、top-p、thinking、history、speed、device |

API 服务也遵循同一边界：只有原生分支会叠加仓库 LoRA；Transformers 分支若需要 LoRA，应事先合并或使用外部 adapter 方案。Transformers 格式启用 YaRN 要改自身 `config.json`，不能依赖 CLI 的 `--inference_rope_scaling`。

### 16.3 完整默认行为

| 参数 | 默认值 |
|---|---:|
| `load_from` | `model` |
| `save_dir` | `out` |
| `weight` | `full_sft` |
| `lora_weight` | 字符串 `None` |
| hidden/layers | 768 / 8 |
| MoE | 0 |
| YaRN | 关闭 |
| max new tokens | 8192 |
| temperature/top-p | 0.85 / 0.95 |
| open thinking | 0 |
| history messages | 0 |
| show speed | 1 |
| device | CUDA 可用则 CUDA，否则 CPU |

每轮会随机重设一次 seed；`historys` 是保留的**消息条数**，应为偶数，但代码不校验。8192 只是生成上限，不是模型一定能输出 8192 个高质量 token。

### 16.4 两个实现陷阱

原生/Transformers 的判断不是检查配置，而是：

```python
if "model" in args.load_from:
```

所以任何 Transformers 路径只要包含小写 `model`，也可能被误判成原生格式。最稳妥的是：

- 原生始终传明确的 `./model`；
- Transformers 目录避免名字或上级路径包含小写 `model`；
- 更长期的修复应改成检测 `config.json`/模型类型。

无论设备为何，代码最后都会 `model.half()`。CPU 的 FP16 支持和性能不理想，MPS 也可能遇到算子问题。仓库默认最适合 CUDA 推理；若要稳定 CPU 推理，应在本地把 dtype 改为 FP32/BF16 并逐算子验证。

### 16.5 YaRN 外推

原生：

```bash
python eval_llm.py \
  --weight full_sft \
  --inference_rope_scaling
```

Transformers 的 `config.json`：

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 16.0,
  "original_max_position_embeddings": 2048,
  "beta_fast": 32.0,
  "beta_slow": 1.0,
  "attention_factor": 1.0
}
```

README 的 PPL 图显示外推后长文本 PPL 相对改善，但不要把位置编码可计算、tokenizer 声明 131K、训练过长文本和可靠长文任务能力混为一谈。

---

## 17. 第十五课：Tool Calling 演示

`scripts/eval_toolcall.py` 支持：

- `local`：原生/Transformers 本地模型；
- `api`：OpenAI 兼容接口；
- 8 个内置测试问题；
- 8 个 mock 工具；
- 多轮“模型调用→执行→回填→继续生成”。

本地：

```bash
cd scripts
python eval_toolcall.py \
  --backend local \
  --load_from ../model \
  --weight full_sft
```

项目自带 API：

```bash
cd scripts
python eval_toolcall.py \
  --backend api \
  --api_base_url http://localhost:8998/v1 \
  --api_key sk-123 \
  --api_model minimind \
  --stream 0
```

默认 API 地址其实是 Ollama 的 `http://localhost:11434/v1`，不是本仓库的 8998；使用仓库 server 时必须显式改。

在 API backend 中，请求的 `max_tokens` 又被脚本固定为 8192，因此 `--max_new_tokens` 只控制 local backend。若要限制 API 演示长度，需要修改 `chat_api()` 里的固定值。

这里特意使用 `--stream 0`：当前本地服务的流式路径会先把原始 `<tool_call>...</tool_call>` 当 `delta.content` 发出，结束时又追加结构化 `delta.tool_calls`，客户端可能把同一次调用重复回填。修复服务端前，结构化 Tool Calling 建议使用非流式；正式修复应缓冲 tool-call 段，不能先把它作为普通 content 发出。

它的工具是：

- math、time、random number、text length；
- unit、weather、exchange、translate。

但多数结果是假数据/硬编码，math 直接使用裸 `eval`。工具 schema 在 `eval_toolcall.py`、`web_demo.py`、`train_agent.py` 之间也有字段差异，例如 weather 用 `city` 或 `location`、translate 用 `target_lang` 或 `target_language`。把真实工具接入前，必须建立一份唯一 schema 并让训练、评测、UI、执行器共享。

该脚本没有准确率统计，只是 8 例交互 demo；循环也没有最大工具轮数，模型持续发 call 时可能不停止。生产/正式评测应增加：

- 最大轮数；
- JSON/schema 验证；
- 工具成功率、参数准确率、最终答案准确率；
- 超时、权限和审计；
- 禁用 `eval`。

README 展示的一次轻 Agent 小测试是 `full_sft 12/20=60%`、`agent 17/20=85%`。这只能视作仓库样例，不是通用 benchmark。

---

## 18. 第十六课：OpenAI 风格 API

### 18.1 启动

先补依赖：

```bash
python -m pip install fastapi uvicorn
```

Transformers 模型：

```bash
cd scripts
python serve_openai_api.py --load_from ../minimind-3
```

原生权重：

```bash
cd scripts
python serve_openai_api.py \
  --load_from ../model \
  --save_dir out \
  --weight full_sft
```

服务固定：

```text
host = 0.0.0.0
port = 8998
POST /v1/chat/completions
```

host/port 没有 CLI 参数。

### 18.2 请求

```bash
curl http://localhost:8998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimind",
    "messages": [{"role":"user","content":"解释 GQA"}],
    "temperature": 0.7,
    "top_p": 0.92,
    "max_tokens": 1024,
    "stream": false,
    "open_thinking": true
  }'
```

请求字段：

| 字段 | 默认/要求 |
|---|---|
| `model` | 必填，但服务端实际忽略 |
| `messages` | 必填 list |
| `temperature` | 0.7 |
| `top_p` | 0.92 |
| `max_tokens` | 8192 |
| `stream` | true |
| `tools` | 空 list |
| `open_thinking` | false |
| `chat_template_kwargs` | null |

thinking 还兼容：

```json
{"chat_template_kwargs":{"open_thinking":true}}
```

以及 `enable_thinking`。非流式响应把 `<think>` 拆到 `reasoning_content`，把 `<tool_call>` 转成 OpenAI 风格 `tool_calls`。

### 18.3 Python SDK

```python
from openai import OpenAI

client = OpenAI(api_key="sk-123", base_url="http://localhost:8998/v1")
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    stream=False,
    extra_body={"chat_template_kwargs": {"open_thinking": True}},
)
print(response.choices[0].message)
```

`scripts/chat_api.py` 默认仍指向：

```text
http://localhost:11434/v1
model=minimind-local:latest
```

要测试本仓库 server，需改为 8998 和 `minimind`。这个客户端只处理文本/reasoning，不完成 Tool Calling 循环。它还发送 `reasoning_effort="medium"`，但本地 `ChatRequest` 没有该字段，Pydantic 默认会忽略；真正控制显式思考的是 `chat_template_kwargs.open_thinking`。

### 18.4 当前兼容性边界

这是“轻量 OpenAI 风格”而非完整 OpenAI server：

- 没有鉴权、限流、并发队列、模型列表、usage 统计；
- request 的 model 名不选择实际模型；
- SSE 结尾没有标准 `[DONE]`；
- 流式 chunk 字段不完全覆盖标准协议；
- 流式 Tool Calling 可能同时发送原始标签 content 和结构化 tool_calls；
- 生成线程每请求启动；
- 对外绑定 `0.0.0.0`；
- `trust_remote_code=True`；
- 无输入长度和工具权限治理。

不要直接暴露公网。至少应在反向代理后增加认证、TLS、限流、超时、日志脱敏和请求体上限。

原生模型的非流式分支还有一个参数接线问题：它把请求的 `max_tokens` 作为 `max_length` 传入，但 MiniMind 自定义 `generate()` 读取的是 `max_new_tokens`，会忽略 `max_length`，于是回落到默认 8192；流式分支则正确传了 `max_new_tokens`。此外服务的 `--max_seq_len` 只作为额外 config 字段传入，没有给 tokenizer 设置实际 `max_length`，也没有改变模型的 `max_position_embeddings`。正式使用前应修正这两处并添加长度测试。

API 的 LoRA 路径是：

```text
../out/lora/<lora_name>_768.pth
```

而 `train_lora.py` 默认保存：

```text
../out/<lora_name>_768.pth
```

这是当前代码不一致。使用 API 前要么把 adapter 放入 `out/lora/`，要么修正 `serve_openai_api.py` 路径。

---

## 19. 第十七课：Streamlit WebUI

WebUI 只扫描 `scripts/` 的直接子目录，且目录内需存在 `.bin`、`.safetensors`、`.pt` 或 safetensors index。

准备：

```bash
cp -R minimind-3 scripts/minimind-3
cd scripts
streamlit run web_demo.py
```

功能：

- 中英文 UI，默认英文；
- 选择扫描到的 Transformers 模型；
- history 0～8，按偶数消息携带；
- max generation 256～8192，默认 8192；
- temperature 0.6～1.2，默认 0.9；
- top-p 固定 0.85；
- thinking 开关；
- 最多选择 4 个工具；
- 最多 16 个工具交互轮次。

限制：

- 只直接加载 Transformers 格式，不读取 `out/*.pth`；
- 未扫描到模型时仍可能尝试从空路径加载并直接报错，启动前必须确认下拉框有真实模型；
- 同样强制 `model.half()`，CPU/MPS 存在与 CLI 相同的 FP16 风险；
- 工具同样是 demo/mock；
- math 使用 `eval`；
- 多处 `unsafe_allow_html=True`，且会显示用户/模型文本；
- 不适合处理不可信内容或直接公网部署；
- 工具与思考同时开启仍可能格式不稳定。

若要做真实产品，先对 HTML 转义、移除 eval、隔离工具、增加认证和资源限制。

---

## 20. 第十八课：模型格式转换

### 20.1 脚本提供的函数

`scripts/convert_model.py` 包含：

| 函数 | 方向/用途 |
|---|---|
| `convert_torch2transformers_minimind` | 原生 `.pth` → 带 MiniMind remote code 的 Transformers |
| `convert_torch2transformers` | 原生 `.pth` → Qwen3/Qwen3MoE 兼容结构 |
| `convert_transformers2torch` | Transformers → CPU FP16 `.pth` |
| `convert_merge_base_lora` | 基模+原生 LoRA → 普通 `.pth` |
| `convert_jinja_to_json` | Jinja 模板转 JSON 字符串 |
| `convert_json_to_jinja` | tokenizer config 中模板导出为 Jinja |

还有 Transformers 5.x 的 config/tokenizer 兼容修补。

### 20.2 当前默认执行

脚本没有 argparse。底部默认：

```text
config: hidden=768, layers=8, Dense
input:  ../out/full_sft_768.pth
output: ../minimind-3
action: convert_torch2transformers（Qwen3-compatible）
```

如果第 2 节已经把发布模型下载到 `minimind-3/`，不要按这个默认值原地转换：`save_pretrained()` 会覆盖同名权重/config，还可能把旧文件残留在目录中。先把脚本底部改为：

```python
transformers_path = "../minimind-3-converted"
```

再执行：

```bash
cd scripts
python convert_model.py
```

若要 MoE、LoRA merge、反向转换或原生 MiniMind Transformers 格式，必须先编辑底部 `lm_config`、路径和要调用的函数。不要只改文件名而忘记 `use_moe`、hidden/layers。

### 20.3 转换后验证

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

p = "minimind-3-converted"
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True)
print(type(model).__name__, len(tok), model.config.hidden_size)
PY
```

`eval_llm.py` 固定 `do_sample=True`，不能承担可复现的 greedy 对照。可在项目根目录直接比较 token ID：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "用一句话解释 GQA。"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

native = MiniMindForCausalLM(
    MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
)
native.load_state_dict(
    torch.load("out/full_sft_768.pth", map_location="cpu"),
    strict=True,
)
converted = AutoModelForCausalLM.from_pretrained(
    "minimind-3-converted",
    trust_remote_code=True,
)

for name, model in [("native", native), ("converted", converted)]:
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_new_tokens=32,
        )
    print(name, generated[0, inputs["input_ids"].shape[1]:].tolist())
```

再比较前 32 个生成 token ID。格式兼容不等于输出一定逐 bit 相同，尤其发生 FP16 转换和不同 generate 实现时；若不一致，应先定位第一处分叉。

---

## 21. 第十九课：第三方推理与端侧部署

以下都先要求 Transformers 或 GGUF 等对应格式，且版本变化快，应以各项目官方文档为准。

### 21.1 SGLang

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --attention-backend triton \
  --host 0.0.0.0 \
  --port 8998
```

适合 CUDA 高吞吐服务，也可作为 RL rollout server。

### 21.2 vLLM

```bash
vllm serve /path/to/model \
  --model-impl transformers \
  --served-model-name minimind \
  --port 8998
```

### 21.3 llama.cpp

仓库 README 当前给出一个临时兼容做法：在 llama.cpp 的 `convert_hf_to_gguf.py`、`get_vocab_base_pre` 末尾为未识别 tokenizer 复用 `qwen2`：

```python
if res is None:
    res = "qwen2"
```

然后：

```bash
cd /path/to/llama.cpp
python convert_hf_to_gguf.py /path/to/minimind-model

./build/bin/llama-quantize \
  /path/to/model/model.gguf \
  /path/to/model/model.q8.gguf \
  Q8_0

./build/bin/llama-cli -m /path/to/model/model.q8.gguf
```

这是兼容补丁，不是稳定上游保证；转换后必须验证 token IDs、chat template 和中文编解码。

### 21.4 Ollama

直接体验：

```bash
ollama run jingyaogong/minimind-3
```

自定义 GGUF 不能只写 `FROM` 和采样参数，否则角色、thinking、tool-call 协议会漂移。按当前 README 创建 `minimind.modelfile`：

```text
FROM /path/to/model/model.q8.gguf

SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。"

TEMPLATE """{{- if .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{{ else if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>
{{ .Thinking }}
</think>

{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}
{{- if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet $.Think -}}
<think>
{{ else -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}"""

PARAMETER repeat_penalty 1
PARAMETER temperature 0.9
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

构建和运行：

```bash
ollama create -f minimind.modelfile minimind-local
ollama run minimind-local
```

推送：

```bash
ollama cp minimind-local:latest your_username/minimind:latest
ollama push your_username/minimind:latest
```

### 21.5 MNN

先按 MNN 官方文档克隆、安装依赖并完成 LLM 工具构建；它是 MiniMind 仓库之外的独立工程。导出时：

```bash
cd /path/to/MNN/transformers/llm/export
python llmexport.py \
  --path /path/to/model \
  --export mnn \
  --hqq \
  --dst_path /path/to/model-mnn
```

然后使用你构建得到的 `llm_demo` **实际绝对路径**：

```bash
/absolute/path/to/llm_demo /path/to/model-mnn/config.json /path/to/prompt.txt
```

`llm_demo` 的构建产物目录随平台/build 配置变化，不能假定它位于 export 目录。这是 4-bit HQQ 量化端侧路线；量化后应单独测 PPL、关键任务准确率、速度和内存，不能只看能否启动。

### 21.6 其他兼容目标

README 声明 OpenAI 风格 API 可接 FastGPT、OpenWebUI、Dify，并把 Transformers、TRL、PEFT、Llama-Factory 列为生态兼容目标。当前仓库没有这些 UI/Llama-Factory 的现成配置或端到端测试，原生 LoRA 格式也不等同于 PEFT adapter；接入时应把它们视为待验证的兼容目标，而不是仓库已交付的一键方案。

---

## 22. 第二十课：评测体系

### 22.1 建立四层评测

1. **接线层**：能加载、生成、resume，loss 非 NaN；
2. **协议层**：chat template、thinking、tool JSON、mask 是否正确；
3. **能力层**：知识、推理、代码、多轮、目标领域；
4. **系统层**：tokens/s、显存、延迟、并发、长上下文和安全。

只看训练 loss 或 reward 都不够。

### 22.2 固定生成条件

同一 MiniMind 家族的不同 checkpoint 比较时至少固定：

- tokenizer/template；
- prompt 和 system；
- open_thinking；
- max tokens、temperature、top-p、top-k；
- seed 或多次采样数；
- 基模/LoRA/MoE 配置；
- 截断长度。

跨模型家族比较不能强行共用 tokenizer/template，否则会破坏对方模型协议；应固定任务语义、prompt 结构和解码策略，各用原生 tokenizer/template，并报告 token 数差异。跨 tokenizer 的语言建模效率优先比较 BPB（Bits Per Byte），不要直接把 PPL 当成同尺度指标。

小模型随机性很大，单个漂亮样例没有统计意义。

### 22.3 lm-evaluation-harness

安装：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
python -m pip install -e .
```

指令模型：

```bash
HF_ENDPOINT=https://hf-mirror.com lm_eval \
  --model hf \
  --model_args pretrained="/path/to/model",dtype=auto \
  --tasks "ceval-valid" \
  --batch_size 16 \
  --device cpu \
  --trust_remote_code \
  --apply_chat_template
```

纯 Pretrain 基座不加 `--apply_chat_template`。可用：

```bash
lm_eval ls tasks
```

仓库当前记录：

| 模型 | C-Eval | CMMLU | ARC | PIQA | OBQA | HellaSwag | SIQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMind-3 64M | 24.89 | 25.38 | 28.49 | 50.65 | 23.60 | 28.28 | 34.19 |
| MiniMind-3-MoE | 25.48 | 24.32 | 27.74 | 50.71 | 26.20 | 27.43 | 34.03 |
| MiniMind-3-exam | 30.98 | 26.12 | 35.61 | 56.26 | 24.20 | 28.40 | 34.19 |

`exam` 不是更大基模，只是用 `lora_exam.jsonl` 对齐选择题格式后合并；7 个评测集与该对齐数据无交集，平均约提升 2.9 个百分点。这说明评测格式本身会显著影响小模型。仓库也做过有污染子集约 97% 的实验，并明确指出那种数字没有参考意义。

### 22.4 RL 评测

RL 权重针对特定 reward 优化，必须同时画：

```text
reward / reward std
KL / clip fraction
回答长度 / 重复率
格式合法率
目标任务准确率
通用能力回归
人工盲评
```

仓库主观比较显示 RL 可能提升工具任务，也可能损害开放问答、事实知识或形成 reward hacking。结论应基于你自己的 holdout，而不是照搬 README 排名。

上表是应建立的评测，不代表当前脚本已经全部记录：PPO 缺 reward 方差；GRPO 缺原始 group reward std；GRPO/Agent 缺 reward 各分项；Agent debug 缺实际 mask tensor。需要按第 13～15 节说明补日志。

### 22.5 推荐的最小回归集

至少准备 50～200 条不参与训练的固定样本：

```text
10% 模板/多轮
15% 中英文基础知识
15% 可验证数学
10% 代码与结构化输出
20% 目标领域
20% 工具调用：name、args、result、final
10% 对抗/超长/非法 schema
```

每个阶段保留同一份原始输出，比较错误类型，而不是只比较一个平均分。

---

## 23. 当前版本的已知不一致与排障表

先查这一表，能省下大量无效调试。

| 现象 | 当前代码事实 | 处理 |
|---|---|---|
| README 推荐 mini Pretrain 约 768，但实际只跑 340 | `train_pretrain.py` 默认 340 | 区分代码默认和数据推荐；按显存显式传参 |
| CLI 写“RoPE 外推 4 倍” | 模型固定 YaRN factor=16、original=2048 | 以实际 config 为准；仍要实测长文 |
| tokenizer 显示 131072 | 模型 RoPE buffer 默认 32768 | 不把 tokenizer 元数据当能力 |
| Transformers 路径加载成原生模型 | 代码用路径中是否含 `"model"` 判断 | 避免该子串或修复检测逻辑 |
| CPU/MPS 推理失败或极慢 | `eval_llm.py` 强制 `.half()` | 改为平台支持 dtype，再验证 |
| 自定义 `--save_dir` 后下一阶段找不到权重 | 加载固定读 `../out` | 把文件放回 `out` 或同步改加载代码 |
| API 加载不到 LoRA | API 查 `out/lora/`，trainer 存 `out/` | 移动/复制 adapter 或改路径 |
| MoE LoRA 在 CLI/API 加载不到 | trainer 文件带 `_moe`，eval/API 的 LoRA 路径不加 `_moe` | 显式统一文件名/路径代码 |
| `chat_api.py` 连不上仓库 server | 默认连接 Ollama 11434 | 改为 `localhost:8998/v1` |
| `eval_toolcall --backend api` 连错服务 | 同样默认 Ollama 11434 | 显式传 `--api_base_url` |
| ToolCall API 模式忽略 `--max_new_tokens` | 客户端把 `max_tokens` 固定为 8192 | 改为传 `args.max_new_tokens` |
| 流式 ToolCall 被重复回填 | server 先发原始标签 content，结束时又发结构化 tool_calls | 修复前使用 `--stream 0`；之后缓冲并只发送结构化调用 |
| README/eval/WebUI 有 random/text_length，但 Agent 训练执行失败 | 当前 Agent trainer 执行器只有另外 6 个工具；数据若声明前两者也无法执行 | 数据、执行器和 UI 共用一份 schema |
| Agent rollout 过长或训练时丢初始问题 | 每轮 rollout 无 prompt 截断；训练 pack 超过 `max_total_len=2500` 后左截断 | 实现逐轮 token budget 并保护 system/tool schema；按显存和任务选择 pack 上限 |
| API 启动缺包 | requirements 无 FastAPI/uvicorn | 单独安装 |
| 原生 API 非流式不遵守 `max_tokens` | 错传为自定义 generate 不读取的 `max_length` | 改传 `max_new_tokens=request.max_tokens` |
| API 的 `--max_seq_len` 没限制 prompt | 只是额外 config 字段，tokenizer 未传 max length | 显式 token 截断并校验 prompt+generation |
| SGLang/vLLM 命令不可用 | 不在 requirements | 按各项目 CUDA/版本要求另装 |
| “开启 FlashAttention”但没安装 flash-attn | 实际调用 PyTorch SDPA | 无需外部 flash-attn |
| 想用 DeepSpeed | 当前仓库无配置/接线 | 使用 DDP，或自行做独立集成 |
| RL 多卡结果异常 | 更新 forward 绕过 DDP wrapper | 先单卡，做单双卡梯度/权重对照 |
| resume 后样本不完全一致 | 不保存完整 RNG 状态 | 把 resume 当近似续跑 |
| SFT loss NaN | 截断后 labels 可能全 `-100` | 训练前统计 active labels |
| GRPO 不学习 | 组内 reward std≈0 | 增加 generations、换数据/奖励 |
| SGLang ratio 异常 | 权重只按 save interval 同步，logp 可补 0 | 缩短同步间隔并验证 logprob |
| 工具调用一直循环 | `eval_toolcall.py` 没最大轮数 | 在正式应用加硬上限 |
| 推理 help 还列出 `reason`/`spo` 等权重 | 当前仓库已移除独立 `train_reason.py`，也无 `train_spo.py` | 视为历史提示，使用当前实际产物名 |
| checkpoints 被 git 发现 | `.gitignore` 只忽略 `out`，未忽略 checkpoints | 本地 ignore/不要提交大权重 |
| 看不到验证指标 | 仓库无 validation loop/test suite | 自建固定 holdout 与回归脚本 |

其他代码卫生事实：

- `requirements.txt` 重复列出 `jsonlines`；
- `CODE_OF_CONDUCT.md` 是 Contributor Covenant 2.0，但 enforcement 联系地址写成了 `.`，项目维护者若正式治理应补有效地址；
- `trust_remote_code=True` 和 `.pth` pickle 都要求可信来源；
- API/WebUI/工具代码是教学实现，不具备生产安全边界。

---

## 24. 所有命令行参数索引

这一节用于“查全”；概念、风险和推荐值仍以前文为准。`device` 的动态默认是“有 CUDA 则 `cuda:0`/`cuda`，否则 CPU”，但 argparse 没有给 device 设置 choices，表中的 `<cuda:0|cpu>` 只是动态默认的简写，并非只允许这两个字符串。布尔 flag 如 `--use_wandb` 不传即 false。

### 24.1 Pretrain

```text
train_pretrain.py
--save_dir=../out
--save_weight=pretrain
--epochs=2
--batch_size=32
--learning_rate=5e-4
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=8
--grad_clip=1.0
--log_interval=100
--save_interval=1000
--hidden_size=768
--num_hidden_layers=8
--max_seq_len=340
--use_moe=0 {0,1}
--data_path=../dataset/pretrain_t2t_mini.jsonl
--from_weight=none
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-Pretrain
--use_compile=0 {0,1}
```

### 24.2 Full SFT

```text
train_full_sft.py
--save_dir=../out
--save_weight=full_sft
--epochs=2
--batch_size=16
--learning_rate=1e-5
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=100
--save_interval=1000
--hidden_size=768
--num_hidden_layers=8
--max_seq_len=768
--use_moe=0 {0,1}
--data_path=../dataset/sft_t2t_mini.jsonl
--from_weight=pretrain
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-Full-SFT
--use_compile=0 {0,1}
```

### 24.3 LoRA

```text
train_lora.py
--save_dir=../out
--lora_name=lora_medical
--epochs=10
--batch_size=32
--learning_rate=1e-4
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=10
--save_interval=1000
--hidden_size=768
--num_hidden_layers=8
--max_seq_len=340
--use_moe=0 {0,1}
--data_path=../dataset/lora_medical.jsonl
--from_weight=full_sft
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-LoRA
--use_compile=0 {0,1}（实际自动关闭）
```

rank=16 没有 CLI 参数。

### 24.4 蒸馏

```text
train_distillation.py
--save_dir=../out
--save_weight=full_dist
--epochs=6
--batch_size=32
--learning_rate=5e-6
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=100
--save_interval=100
--max_seq_len=340
--data_path=../dataset/sft_t2t_mini.jsonl
--student_hidden_size=768
--student_num_layers=8
--teacher_hidden_size=768
--teacher_num_layers=8
--student_use_moe=0 {0,1}
--teacher_use_moe=1 {0,1}
--from_student_weight=full_sft
--from_teacher_weight=full_sft
--from_resume=0 {0,1}
--alpha=0.5
--temperature=1.5
--use_wandb
--wandb_project=MiniMind-Distillation
--use_compile=0 {0,1}
```

### 24.5 DPO

```text
train_dpo.py
--save_dir=../out
--save_weight=dpo
--epochs=1
--batch_size=4
--learning_rate=4e-8
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=100
--save_interval=100
--hidden_size=768
--num_hidden_layers=8
--max_seq_len=1024
--use_moe=0 {0,1}
--data_path=../dataset/dpo.jsonl
--from_weight=full_sft
--from_resume=0 {0,1}
--beta=0.15
--use_wandb
--wandb_project=MiniMind-DPO
--use_compile=0 {0,1}
```

### 24.6 PPO

```text
train_ppo.py
--save_dir=../out
--save_weight=ppo_actor
--epochs=1
--batch_size=2
--learning_rate=3e-7
--critic_learning_rate=5e-7
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=1
--save_interval=10
--hidden_size=768
--num_hidden_layers=8
--use_moe=0 {0,1}
--max_seq_len=768
--max_gen_len=1024
--data_path=../dataset/rlaif.jsonl
--clip_epsilon=0.2
--vf_coef=0.5
--kl_coef=0.02
--gamma=1.0
--lam=0.95
--cliprange_value=0.2
--ppo_update_iters=2
--early_stop_kl=0.25
--mini_batch_size=2
--from_weight=full_sft
--reward_model_path=../../internlm2-1_8b-reward
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-PPO
--use_compile=0 {0,1}
--debug_mode
--debug_interval=20
--debug_log_ratio
--thinking_ratio=0.9
--rollout_engine=torch {torch,sglang}
--sglang_base_url=http://localhost:8998
--sglang_model_path=../model
--sglang_shared_path=./sglang_ckpt_ppo
```

### 24.7 GRPO/CISPO

```text
train_grpo.py
--save_dir=../out
--save_weight=grpo
--epochs=1
--batch_size=2
--learning_rate=3e-7
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=1
--save_interval=10
--hidden_size=768
--num_hidden_layers=8
--use_moe=0 {0,1}
--max_seq_len=768
--max_gen_len=1024
--data_path=../dataset/rlaif.jsonl
--num_generations=6
--beta=0.1
--loss_type=cispo {grpo,cispo}
--epsilon=0.2
--epsilon_high=5.0
--from_weight=full_sft
--reward_model_path=../../internlm2-1_8b-reward
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-GRPO
--use_compile=0 {0,1}
--debug_mode
--debug_interval=20
--thinking_ratio=0.9
--rollout_engine=torch {torch,sglang}
--sglang_base_url=http://localhost:8998
--sglang_model_path=../model
--sglang_shared_path=./sglang_ckpt_grpo
```

### 24.8 Agentic RL

```text
train_agent.py
--save_dir=../out
--save_weight=agent
--epochs=1
--batch_size=2
--learning_rate=3e-7
--device=<cuda:0|cpu>
--dtype=bfloat16
--num_workers=8
--accumulation_steps=1
--grad_clip=1.0
--log_interval=1
--save_interval=10
--hidden_size=768
--num_hidden_layers=8
--use_moe=0 {0,1}
--max_seq_len=1024（当前不执行 prompt 截断）
--max_gen_len=768（每轮生成上限）
--max_total_len=2500（完整轨迹训练 pack 上限）
--data_path=../dataset/agent_rl.jsonl
--num_generations=4
--beta=0.1
--loss_type=cispo {grpo,cispo}
--epsilon=0.2
--epsilon_high=5.0
--from_weight=full_sft
--from_resume=0 {0,1}
--use_wandb
--wandb_project=MiniMind-Agent-RL
--use_compile=0 {0,1}
--debug_mode
--debug_interval=20
--thinking_ratio=0.1
--reward_model_path=../../internlm2-1_8b-reward
--rollout_engine=torch {torch,sglang}
--sglang_base_url=http://localhost:8998
--sglang_model_path=../model
--sglang_shared_path=./sglang_ckpt_agent
```

最大工具轮数 3 没有 CLI 参数。

### 24.9 `eval_llm.py`

```text
--load_from=model
--save_dir=out
--weight=full_sft
--lora_weight=None
--hidden_size=768
--num_hidden_layers=8
--use_moe=0 {0,1}
--inference_rope_scaling
--max_new_tokens=8192
--temperature=0.85
--top_p=0.95
--open_thinking=0
--historys=0
--show_speed=1
--device=<cuda|cpu>
```

### 24.10 `scripts/eval_toolcall.py`

```text
--backend=local {local,api}
--load_from=../model
--save_dir=../out
--weight=full_sft
--hidden_size=768
--num_hidden_layers=8
--use_moe=0 {0,1}
--max_new_tokens=512
--temperature=0.9
--top_p=0.9
--show_speed=0
--device=<cuda|cpu>
--api_base_url=http://localhost:11434/v1
--api_key=sk-123
--api_model=jingyaogong/minimind-3:latest
--stream=1
```

### 24.11 `scripts/serve_openai_api.py`

```text
--load_from=../model
--save_dir=out
--weight=full_sft
--lora_weight=None
--hidden_size=768
--num_hidden_layers=8
--max_seq_len=8192
--use_moe=0 {0,1}
--inference_rope_scaling
--device=<cuda|cpu>
```

`train_tokenizer.py` 和 `convert_model.py` 没有 argparse；`web_demo.py` 通过 UI 控件配置；`chat_api.py` 通过文件内常量配置；`rollout_engine.py`、model 与 Dataset 文件不是独立 CLI。

---

## 25. 推荐的完整学习实践顺序

不要一上来烧 GPU。按下面里程碑推进，每一步都有可验收结果。

### 里程碑 1：先用发布权重推理

任务：

- 下载 Transformers 模型；
- 跑自动 8 问和手动对话；
- 分别打开/关闭 thinking；
- 记录速度与错误样例。

通过标准：

- 能稳定加载和结束；
- 明白 MiniMind 是学习型小模型，不把输出当事实来源；
- 保存一份固定回归问题。

### 里程碑 2：掌握 tokenizer 与模板

任务：

- 打印 0/1/2、Tool/Think token；
- 渲染普通、reasoning、tool、open thinking 四类模板；
- 检查 skip special tokens；
- 理解 131072、32768、训练长度的差别。

通过标准：

- 能手工解释每个角色如何变成 token 流；
- 能指出 Tool/Think 为什么不能被 skip 掉。

### 里程碑 3：读通模型 forward

任务：

- 用小 config 跑 Dense/MoE；
- 打印 Q/K/V 与 cache 形状；
- 比较首轮与 cached 单 token decode；
- 分开观察 CE/aux。

通过标准：

- 能解释 GQA、RoPE、Pre-Norm、SwiGLU、tied embeddings；
- 能解释 `198M-A64M`，不把它误读成 64M 存储参数。

### 里程碑 4：验证数据和 mask

任务：

- 每种 Dataset 取 1 条；
- 解码 IDs 和 active labels；
- 制造超长 user 样本；
- 检查 DPO mask、RLAIF/Agent 末尾占位。

通过标准：

- SFT assistant-only mask 正确；
- 无全 `-100`；
- JSON 字段类型与模板一致。

### 里程碑 5：Pretrain smoke → 正式训练

任务：

- 256 条 smoke 数据跑 1 epoch；
- 验证 checkpoint/resume；
- 再决定 Dense/MoE、340/768、mini/full。

通过标准：

- loss 有限且下降；
- `pretrain_*.pth` 可加载；
- 清楚全局 batch 与显存取舍。

### 里程碑 6：SFT 得到 Zero 对话模型

任务：

- 跑 SFT；
- 固定 50～200 条回归集；
- 检查普通对话、thinking、tool 格式。

通过标准：

- `full_sft_*.pth` 可加载；
- 任务格式明显优于 Pretrain；
- 错误类型被量化而非凭观感。

### 里程碑 7：做一个 LoRA 小实验

任务：

- 准备垂域训练/验证拆分；
- 训练 adapter；
- 比较基模、adapter、merged 三者；
- 检查通用能力遗忘。

通过标准：

- adapter 只有 LoRA key；
- 合并前后输出近似；
- 目标域改善不是靠重复记忆训练题。

### 里程碑 8：选择一个偏好分支

二选一或独立对比：

- 蒸馏：重点看教师质量与 CE/KL；
- DPO：重点看偏好 margin 与通用回归。

不要默认把蒸馏→DPO→PPO 全串联；每次只改变一个变量。

### 里程碑 9：先 GRPO 小实验，再决定 PPO/Agent

任务：

- 用小数据、短生成、debug mode；
- debug mode 查看生成样本和总 reward；
- 给 GRPO 补原始 group reward std 日志；
- 若要分析 reward composition，在奖励函数中分别记录各分项；
- 检查 RM 吞吐、KL、重复和长度；
- Agent 另加日志打印实际多轮 mask。

通过标准：

- reward 有差异；
- 无 NaN/极端 ratio；
- 目标成功率提高且回归集不过度退化；
- 能识别 reward hacking。

### 里程碑 10：转换、API 与部署

任务：

- `.pth` 转 Transformers；
- 对比转换前后；
- 启动 API；
- 再按需要选择 WebUI、vLLM、SGLang、GGUF/Ollama/MNN。

通过标准：

- tokenizer/template 无漂移；
- 接口格式通过自己的客户端测试；
- 公网前补安全边界。

---

## 26. 更深入的源码阅读注记

### 26.1 Tokenizer 底层

```text
normalizer: null
pre-tokenizer: ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=true)
decoder: ByteLevel(add_prefix_space=true, trim_offsets=true, use_regex=true)
post-processor: null
默认 truncation/padding: null
BPE dropout: null
byte_fallback: false
```

虽然 BPE model 没有自身 unk token、byte fallback 也关闭，但词表含完整 256 字节 alphabet，普通 UTF-8 文本仍可逐字节表达；HF 外层把 unk 映射到 ID 0。

模板直接访问 `messages[0]`，空消息不安全。异常 content 中有多个 `</think>` 时，拆分逻辑可能丢失中段。ID 0 同时是 pad、unk、end-of-text，所以凡是用 `input_ids != 0` 做 mask 的流程，也会把正文中真实 ID 0 当 padding。

模板还逆序计算了 `ns.last_query_index`，但后续没有读取它，是不影响当前输出的遗留逻辑。

### 26.2 Model 输出与 Cache 兼容

- 原生 cache 是每层 `(K,V)` 的 list；
- 若传入带 `.layers` 的 Transformers 新式 Cache，当前代码会丢弃并从空 cache 开始；
- `use_cache=False` 时输出仍是由多个 `None` 组成的 list，而非整体 `None`；
- 所有层起始 cache 长度以第一层为准；
- meta device 下若 RoPE buffer 首项为 0，会重新预计算；
- `MoeCausalLMOutputWithPast.hidden_states` 实际放单个最终 Tensor，而 HF 习惯通常是逐层 tuple；
- Dense 也统一返回 MoE output 类型，只是 aux=0；
- `layer_id` 参数接收但未使用；
- `logits_to_keep=0` 通过 `slice(-0,None)` 等价于保留全部 logits。

这些不会阻碍仓库原生训练，却会影响你接入通用 HF Trainer、Cache API 或分析工具。

### 26.3 LoRA 边界

- `apply_lora` 只负责注入，不冻结；训练脚本才按参数名冻结基模；
- rank 不写入 adapter 元数据，加载前必须以同 rank 注入；
- MoE LoRA 仍只训练 attention Q/O，不训练专家；
- 已 half 的基模再注入时需留意 adapter 初始 dtype；
- merge 先在模型 device 上算 `B@A`，再把 base/delta 转 CPU FP16 相加和保存。

### 26.4 RL 数值和吞吐边界

- PPO/GRPO/Agent 没有 FP16 GradScaler；
- reward model 逐回答串行打分；
- SGLang 和 Torch 的采样/同步状态不是严格同分布；
- Agent 重建多轮上下文时会“decode→重新套模板→tokenize”，自定义模板后必须核对 token 前缀和 mask；
- SFT/DPO 的随机空 think 处理会让完全复现实验更难。

---

## 27. 基础依赖清单

当前 `requirements.txt` 的有效条目如下；Torch/torchvision 被注释：

```text
datasets==3.6.0
datasketch==1.6.4
Flask==3.0.3
Flask_Cors==4.0.0
jieba==0.42.1
jsonlines==4.0.0
marshmallow==3.22.0
ngrok==1.4.0
nltk==3.8
numpy==1.26.4
openai==1.59.6
psutil==5.9.8
pydantic==2.11.5
rich==13.7.1
scikit_learn==1.5.1
sentence_transformers==2.3.1
simhash==2.1.2
tiktoken==0.10.0
transformers==4.57.6
jinja2==3.1.2
trl==0.13.0
ujson==5.1.0
wandb==0.18.3
streamlit==1.50.0
einops==0.8.1
swanlab==0.7.11
modelscope==1.37.0
```

文件还重复写了一次 `jsonlines==4.0.0`，并注释了 matplotlib、PEFT、Torch、torchvision。主线手写 LoRA 不依赖 PEFT；主线绘图也不在训练脚本内。

---

## 28. 逐文件覆盖表

| 仓库文件 | 本教程覆盖位置 |
|---|---|
| `model/model_minimind.py` | 第 4、16、26 节 |
| `model/model_lora.py` | 第 9、20、26 节 |
| `model/tokenizer.json` | 第 3、26 节 |
| `model/tokenizer_config.json` | 第 3、5、26 节 |
| `model/__init__.py` | 第 1 节说明为空 |
| `dataset/lm_dataset.py` | 第 5 节 |
| `dataset/dataset.md` | 第 2 节数据放置 |
| `dataset/__init__.py` | 第 1 节说明为空 |
| `trainer/trainer_utils.py` | 第 6 节 |
| `trainer/train_tokenizer.py` | 第 3.6 节；脚本无 CLI 参数 |
| `trainer/train_pretrain.py` | 第 7、24.1 节 |
| `trainer/train_full_sft.py` | 第 8、24.2 节 |
| `trainer/train_lora.py` | 第 9、24.3 节 |
| `trainer/train_distillation.py` | 第 10、24.4 节 |
| `trainer/train_dpo.py` | 第 11、24.5 节 |
| `trainer/rollout_engine.py` | 第 12 节 |
| `trainer/train_ppo.py` | 第 13、24.6 节 |
| `trainer/train_grpo.py` | 第 14、24.7 节 |
| `trainer/train_agent.py` | 第 15、24.8 节 |
| `eval_llm.py` | 第 16、24.9 节 |
| `scripts/eval_toolcall.py` | 第 17、24.10 节 |
| `scripts/serve_openai_api.py` | 第 18、24.11 节 |
| `scripts/chat_api.py` | 第 18.3 节 |
| `scripts/web_demo.py` | 第 19 节 |
| `scripts/convert_model.py` | 第 9.5、20 节 |
| `requirements.txt` | 第 2、27 节 |
| `README.md` | 路线、数据、模型、成本、训练、评测、部署各节 |
| `README_en.md` | README 的英文版本；功能不另起分支 |
| `LICENSE` | 第 2.4、28.1 节；仓库 Work 使用 Apache-2.0 |
| `CODE_OF_CONDUCT.md` | 第 23、28.1 节；Contributor Covenant 2.0 |
| `.gitignore` | 第 6.7、23 节 |

文档图片资产全部是 README 的说明材料，没有运行时代码：

```text
images/logo.png
images/logo2.png
images/minimind-3.gif
images/LLM-structure.jpg
images/LLM-structure-moe.jpg
images/rl-structure.jpg
images/dataset.jpg
images/gpt3_config.png
images/pretrain_loss.jpg
images/sft_loss.jpg
images/grpo_loss.jpg
images/ppo_loss.jpg
images/agent_rl_loss.jpg
images/rope_ppl.png
images/benchmark_radar.jpg
images/agent_webui.jpg
images/with_huggingface.png
images/with_modelscope.png
```

它们分别展示 Logo/演示、Dense/MoE/RL 结构、数据组成、GPT-3 配置参考、各训练曲线、RoPE PPL、benchmark 雷达、Agent WebUI 和社区平台入口。

`.gitignore` 当前忽略：

```text
__pycache__/
*.pyc
.DS_Store
out
website/
docs-minimind/
```

但没有忽略 `checkpoints/`。

### 28.1 许可证与社区治理

仓库 Work 使用 Apache License 2.0。简化理解（不是法律意见）：

- 再分发源码或二进制时应附 Apache-2.0 许可证；
- 修改过的文件应作显著修改说明；
- 应保留适用的版权、专利、商标和 attribution notices；若上游 Work 带 NOTICE，还需按许可证规则处理 NOTICE；
- 许可证提供明确专利授权；若就相关 Work 发起专利诉讼，相关专利许可可能终止；
- 不授予项目名称/商标使用权；
- 软件按现状提供，不作担保，并含责任限制。

本仓库没有受跟踪的 `NOTICE` 文件，但不能据此忽略你引入的第三方 notices。训练数据、奖励模型和外部依赖仍受各自许可证约束，仓库 Apache-2.0 不会覆盖它们。

`CODE_OF_CONDUCT.md` 采用 Contributor Covenant 2.0，适用于项目社区空间以及成员以项目代表身份出现的场景。它规定尊重、包容等正向行为，禁止骚扰、攻击和泄露隐私；执法阶梯包括 correction、warning、temporary ban、permanent ban。当前举报地址仅写成 `.`，实际治理前应补有效且可私密联系的渠道。

### 28.2 引用项目

README 给出的 BibTeX：

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository, accessed 2026}
}
```

---

## 29. 最终完成清单

### 只想学懂

- [ ] 能解释 tokenizer、模板、IDs 0/1/2 和 Tool/Think 标签；
- [ ] 能画出 GQA、RoPE、KV cache、SwiGLU、MoE；
- [ ] 能解释模型内 shift CE 和 assistant-only mask；
- [ ] 能区分 SFT、DPO、PPO、GRPO、CISPO、Agent；
- [ ] 能说明 RL reward 与能力提升不等价。

### 想从零复现

- [ ] `pretrain_t2t_mini` 和 `sft_t2t_mini` 已放入 dataset；
- [ ] mask 抽查无全 `-100`；
- [ ] `pretrain_768[(_moe)].pth` 可加载；
- [ ] `full_sft_768[(_moe)].pth` 可对话；
- [ ] checkpoint 能恢复；
- [ ] 固定回归集已保存；
- [ ] 代码默认与自定义参数有实验记录。

### 想做后训练

- [ ] LoRA 基模一致、adapter key 正确；
- [ ] 蒸馏教师/学生权重和 vocab 一致；
- [ ] DPO chosen/rejected 与 mask 正确；
- [ ] PPO 同时监控 reward/value/KL/clip；
- [ ] GRPO/CISPO 已补原始 group reward std 日志，且指标不退化；
- [ ] Agent 工具 schema、执行器、GT 对齐，并通过额外 mask 日志验收；
- [ ] 对 reward hacking 和通用能力回退做了独立评测。

### 想部署

- [ ] 转换前后输出经过对照；
- [ ] tokenizer/chat template 随模型一起发布；
- [ ] API endpoint 与客户端地址一致；
- [ ] LoRA 文件路径已统一；
- [ ] 去除裸 `eval` 和不可信 HTML；
- [ ] 增加认证、TLS、限流、超时、最大输入/工具轮数；
- [ ] 只从可信来源加载 remote code 和 `.pth`；
- [ ] 量化后重新评测质量。

---

## 30. 一句话总结

MiniMind 的最大价值不是把 64M 模型包装成“大模型替代品”，而是把现代 LLM 的完整学习链条压缩到个人可读、可改、可训练的规模：从 tokenizer、GQA/MoE 和因果语言建模，到 SFT、LoRA、蒸馏、偏好优化、在线 rollout、工具环境、评测与部署。最可靠的学习方法，是每完成一层就检查它的输入契约、目标函数、产物和回归结果，再进入下一层。
