# MiniMind 学习里程碑

> 生成日期：2026-08-24
> 每个里程碑的目标、涉及文件、学习要点和准备工作。

---

## 硬环境前提

| 项目             | 说明                                                                         |
| ---------------- | ---------------------------------------------------------------------------- |
| **Python 3.10+** | 推荐 3.10.16                                                                 |
| **CUDA GPU**     | 建议 NVIDIA GPU（DDP 固定 NCCL）；Dense Pretrain/SFT 可尝试 CPU 但会慢很多   |
| **PyTorch**      | `requirements.txt` 中 torch 被注释，需按自己平台先装                         |
| **依赖包**       | `pip install -r requirements.txt`                                            |
| **工作目录**     | `eval_llm.py` 在根目录跑；训练脚本在 `trainer/` 跑；服务脚本在 `scripts/` 跑 |

---

## 数据下载总览

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 下载，全部放入 `dataset/` 目录：

| 文件                      | 大小  | 用于                 |
| ------------------------- | ----- | -------------------- |
| `pretrain_t2t_mini.jsonl` | 1.2GB | 里程碑 5（最小主线） |
| `sft_t2t_mini.jsonl`      | 1.6GB | 里程碑 6（最小主线） |
| `pretrain_t2t.jsonl`      | 10GB  | 完整预训练           |
| `sft_t2t.jsonl`           | 14GB  | 完整 SFT             |
| `dpo.jsonl`               | 53MB  | 里程碑 8 DPO         |
| `rlaif.jsonl`             | 24MB  | 里程碑 9 GRPO/PPO    |
| `agent_rl.jsonl`          | 86MB  | 里程碑 9 Agent       |
| `agent_rl_math.jsonl`     | 18MB  | 里程碑 9 数学 Agent  |

**最小主线**只需前两个 mini 文件即可完成 Pretrain → SFT → 对话。

---

## 里程碑 1：发布权重推理

**目标：把模型跑起来，建立第一手感。**

| 核心文件                   | 辅助文件                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| [eval_llm.py](eval_llm.py) | [model/tokenizer_config.json](model/tokenizer_config.json)、[scripts/web_demo.py](scripts/web_demo.py) |

**文件作用：** `eval_llm.py` 是推理主脚本，支持原生 torch 权重或 Transformers 格式加载，支持 pretrain/full_sft/rlhf/ppo_actor/grpo 等权重前缀，支持 `--lora_weight` 做基模/adapter/merged 三态对比，控制 `skip_special_tokens`、thinking 开关、stream 输出。

**学什么：**

- 从 HuggingFace / ModelScope 下载预训练权重
- 两种加载方式：原生 torch 权重 vs Transformers 格式
- CLI 多轮对话、stream 输出、`skip_special_tokens` 的效果
- `web_demo.py` 的 Streamlit 交互形态
- 建立"模型能说什么、不能说什么"的基线直觉

**需要准备：**

```bash
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 或 git clone https://huggingface.co/jingyaogong/minimind-3
```

在项目根目录运行 `python eval_llm.py --load_from model`

---

## 里程碑 2：Tokenizer 与模板

**目标：理解模型"眼中"的文本是什么样的。**

| 核心文件                                                                                                 | 辅助文件                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [model/tokenizer_config.json](model/tokenizer_config.json)、[model/tokenizer.json](model/tokenizer.json) | [trainer/train_tokenizer.py](trainer/train_tokenizer.py)、[eval_llm.py](eval_llm.py)、[model/model_minimind.py](model/model_minimind.py) |

**文件作用：** `tokenizer.json` 是 6400 词 BPE 词表本体（36 预留 + 256 字节 + 6108 merge）；`tokenizer_config.json` 含 chat_template（普通/reasoning/tool/open thinking 四类模板）、special tokens 声明、`skip_special_tokens` 行为；`train_tokenizer.py` 是 BPE 训练教学参考，不建议重训。

**学什么：**

- 6400 词 BPE 词表分区逻辑：预留 token → 字节 token → merge token
- `added_tokens` + `special: false` 设计：`<tool_call>` 不可拆分但解码时可见
- `add_bos_token=false` / `add_eos_token=false` 与 ChatML 模板的关系
- 区分三个长度概念：`model_max_length`（131072）、`max_position_embeddings`（32768）、训练 `max_seq_len`
- Chat Template 四类模板的渲染规则，`thinking` / ` response` 如何控制思考链与正文分离
- 可选：阅读 `train_tokenizer.py` 了解 BPE 训练流程

**需要准备：** 无需额外下载，仓库自带。在项目根目录跑 tokenizer 实践脚本即可验证。

---

## 里程碑 3：读通模型 forward

**目标：从代码层面理解一个 Decoder-only LM 的完整前向过程。**

| 核心文件                                           | 辅助文件                                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [model/model_minimind.py](model/model_minimind.py) | [model/model_lora.py](model/model_lora.py)、[trainer/trainer_utils.py](trainer/trainer_utils.py) |

**文件作用：** `model_minimind.py` 是核心模型实现——`MiniMindConfig`（hidden_size、num_hidden_layers、use_moe、max_position_embeddings 等）、`MiniMindForCausalLM` 含 Dense 与 MoE 两种架构的 forward、KV cache、GQA、RoPE、Pre-Norm、SwiGLU、tied embeddings、MoE 的 CE + aux loss；`trainer_utils.py` 的 `get_model_params` 用于参数统计（如 198M-A64M）。

**学什么：**

- `MiniMindConfig` 每个参数的含义与作用
- Dense 完整 forward：embedding → Pre-Norm → Attention（GQA + RoPE + KV Cache）→ FFN（SwiGLU）→ 输出
- MoE 与 Dense 的差异：路由机制、expert 选择、aux loss
- KV Cache 增量推理原理、tied embeddings 实现
- 用 `get_model_params` 解读 198M-A64M 参数分布

**需要准备：** 纯阅读，无需运行。

---

## 里程碑 4：验证数据和 mask

**目标：理解训练数据如何被组织，以及 loss 计算时的 mask 策略。**

| 核心文件                                       | 辅助文件                                 |
| ---------------------------------------------- | ---------------------------------------- |
| [dataset/lm_dataset.py](dataset/lm_dataset.py) | [dataset/dataset.md](dataset/dataset.md) |

**文件作用：** `lm_dataset.py` 包含所有训练阶段的数据集类——SFT assistant-only mask 构造、DPO 的 chosen/rejected mask、RLAIF/Agent 末尾占位、超长 user 样本处理、tool use 数据保留逻辑、`pre_processing_chat` 系统提示注入。

**学什么：**

- `PretrainDataset` 与 `SFTDataset` 的数据构造差异
- SFT 的 assistant-only mask：为什么只对 assistant 部分计算 loss
- DPO 的 chosen/rejected mask 构造
- Agent 数据中 tool use 的保留逻辑
- `pre_processing_chat` 如何注入系统提示、超长 user 样本的处理策略

**需要准备：** 纯阅读，无需运行。

---

## 里程碑 5：Pretrain

**目标：从零开始训练一个语言模型，理解预训练全流程。**

| 核心文件                                               | 辅助文件                                             |
| ------------------------------------------------------ | ---------------------------------------------------- |
| [trainer/train_pretrain.py](trainer/train_pretrain.py) | [trainer/trainer_utils.py](trainer/trainer_utils.py) |

**文件作用：** `train_pretrain.py` 是预训练脚本，支持 Dense/MoE 可选、smoke 数据 1 epoch、checkpoint 保存（`pretrain_*.pth`）与 resume；`trainer_utils.py` 提供 `setup_seed`、checkpoint save/load、resume、自定义 Sampler。

**学什么：**

- 从随机参数开始，让模型学习语言统计规律
- smoke test（小数据 1 epoch）在调通流程中的价值
- checkpoint 保存与 resume 断点续训机制
- DDP 多卡训练基本用法
- loss 下降曲线与预训练收敛速度、`max_seq_len` 对效率与效果的影响

**需要准备：**

- 下载 `pretrain_t2t_mini.jsonl`（1.2GB）→ 放入 `dataset/`
- 在 `trainer/` 目录执行：`python train_pretrain.py`
- 产出权重：`out/pretrain_*.pth`

---

## 里程碑 6：SFT 得到对话模型

**目标：让预训练模型学会"聊天"，理解指令微调的本质。**

| 核心文件                                               | 辅助文件                   |
| ------------------------------------------------------ | -------------------------- |
| [trainer/train_full_sft.py](trainer/train_full_sft.py) | [eval_llm.py](eval_llm.py) |

**文件作用：** `train_full_sft.py` 是全参数 SFT 脚本，产出 `full_sft_*.pth`，支持 thinking/tool 格式训练；`eval_llm.py` 用于对比 Pretrain 与 SFT 后的对话能力差异。

**学什么：**

- SFT 如何将"续写模型"转化为"对话模型"
- ChatML 格式数据如何被拼接成训练样本
- thinking/tool 格式训练对模型行为的影响
- SFT 是后续所有 RL 分支的基线

**需要准备：**

- 下载 `sft_t2t_mini.jsonl`（1.6GB）→ 放入 `dataset/`
- 需要里程碑 5 产出的 `pretrain_*.pth` 或直接下载发布权重
- 在 `trainer/` 目录执行：`python train_full_sft.py`
- 在项目根目录用 `eval_llm.py` 验证对话效果

---

## 里程碑 7：LoRA 小实验

**目标：理解参数高效微调（PEFT）的原理与实践。**

| 核心文件                                                                                   | 辅助文件                   |
| ------------------------------------------------------------------------------------------ | -------------------------- |
| [trainer/train_lora.py](trainer/train_lora.py)、[model/model_lora.py](model/model_lora.py) | [eval_llm.py](eval_llm.py) |

**文件作用：** `model_lora.py` 定义 `LoRA` 层（A 高斯初始化、B 全 0）、`apply_lora`（注入）、`load_lora`（加载）、`merge_lora`（合并回基模）、`save_lora`（保存）；`train_lora.py` 冻结基模、只训 adapter、保存仅含 LoRA key 的权重。

**学什么：**

- LoRA 数学原理：低秩分解、A 高斯初始化、B 全 0 初始化
- `apply_lora` → `merge_lora` → `save_lora` 的完整生命周期
- 冻结基模、只训 adapter 的显存优势
- 用 `eval_llm.py --lora_weight` 做基模 / adapter / merged 三态对比
- LoRA 的 rank、alpha 等超参对效果的影响

**需要准备：**

- 需要 SFT 后的基座模型权重
- 自备 `dataset/lora_medical.jsonl`（垂域数据，仓库不提供）
- 在 `trainer/` 目录执行 `train_lora.py`

---

## 里程碑 8：偏好分支（蒸馏 / DPO）

**目标：理解两种不同的对齐方法，知道各自的适用场景。**

| 核心文件                                                                                                     | 辅助文件                                       |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| [trainer/train_distillation.py](trainer/train_distillation.py)、[trainer/train_dpo.py](trainer/train_dpo.py) | [dataset/lm_dataset.py](dataset/lm_dataset.py) |

**文件作用：** `train_distillation.py` 学生向教师学习，关注 CE/KL loss；`train_dpo.py` 偏好对齐，关注 margin 与通用回归，自带 chosen/rejected mask；两条路线从同一个 SFT 基线出发，不是串联关系。

**学什么：**

- **蒸馏**：学生从教师输出分布中学习，CE loss + KL loss 组合，教师质量决定效果
- **DPO**：从 chosen/rejected 对比中学习偏好，DPO loss 的 margin 机制，不需要显式 reward model
- 对比：蒸馏学"分布"，DPO 学"偏好"

**需要准备：**

- **蒸馏**：需要一个更强的教师模型（如 Qwen3），在 `trainer/` 目录执行 `train_distillation.py`
- **DPO**：下载 `dpo.jsonl`（53MB）→ 放入 `dataset/`；需要 SFT 基线模型；在 `trainer/` 目录执行 `train_dpo.py`

---

## 里程碑 9：GRPO → PPO → Agent

**目标：理解强化学习在 LLM 对齐中的应用，从简单到复杂逐步深入。**

| 核心文件                                                                                                                                                                                               | 辅助文件                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| [trainer/train_grpo.py](trainer/train_grpo.py)、[trainer/train_ppo.py](trainer/train_ppo.py)、[trainer/train_agent.py](trainer/train_agent.py)、[trainer/rollout_engine.py](trainer/rollout_engine.py) | [eval_llm.py](eval_llm.py) |

**文件作用：** `train_grpo.py` 实现短生成、group reward、reward composition；`train_ppo.py` 实现 RM 吞吐、KL、重复与长度监控；`train_agent.py` 实现多轮 tool use mask；`rollout_engine.py` 是三者共用的推理 rollout 引擎，支持本地 transformers 或 sglang 远程加速。

**学什么：**

- **GRPO**：group-based reward 采样与评分，reward composition 加权组合，不需要 critic model
- **PPO**：actor-critic 架构，KL/重复/长度惩罚，与 GRPO 的核心差异（是否需要 critic）
- **Agent**：多轮 tool use 的 RL 训练，tool call 正确性如何被 reward 化，mask 特殊处理
- 共用 `rollout_engine.py`：本地生成 vs SGLang 远程加速
- RL 训练的显存压力（同时驻留多个模型）和训练不稳定性

**需要准备：**

- **GRPO / PPO**：下载 `rlaif.jsonl`（24MB）→ 放入 `dataset/`；需要 SFT 基线模型；PPO 需额外安装 `trl`
- **Agent**：下载 `agent_rl.jsonl`（86MB）+ 可选 `agent_rl_math.jsonl`（18MB）→ 放入 `dataset/`
- ⚠️ PPO/GRPO/Agent 同时驻留多个模型，**显存要求远高于普通 SFT**
- 如用 SGLang 加速 rollout，需额外安装 SGLang

---

## 里程碑 10：转换、API 与部署

**目标：将训练好的模型推向生产，打通最后一公里。**

| 核心文件                                                                                                         | 辅助文件                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/convert_model.py](scripts/convert_model.py)、[scripts/serve_openai_api.py](scripts/serve_openai_api.py) | [scripts/chat_api.py](scripts/chat_api.py)、[scripts/web_demo.py](scripts/web_demo.py)、[scripts/eval_toolcall.py](scripts/eval_toolcall.py) |

**文件作用：** `convert_model.py` 将 `.pth` 原生权重映射到 Qwen3/Qwen3Moe 结构，支持 merge LoRA 后导出；`serve_openai_api.py` 用 uvicorn 提供 chat completions 接口；`chat_api.py` 是极简 OpenAI 客户端示例；`web_demo.py` 是 Streamlit 流式 WebUI；`eval_toolcall.py` 验证工具调用格式正确性。

**学什么：**

- `.pth` → Transformers 格式的权重映射关系，Qwen3/Qwen3Moe 结构对应
- FastAPI + uvicorn 搭建标准 chat completions 接口，请求/响应数据流
- OpenAI SDK 连接自己的模型，适配 Ollama/vLLM 等生态
- Streamlit 流式输出部署形态，Tool Call 格式正确性校验

**需要准备：**

- 在 `scripts/` 目录执行各脚本
- `serve_openai_api.py` 需额外安装 `fastapi` + `uvicorn`
- `web_demo.py` 依赖已含在 `requirements.txt` 中

---

## 跨里程碑复用文件

- **[eval_llm.py](eval_llm.py)** 贯穿 1/2/6/7/9：跑推理（1）→ 看 tokenizer 加载与模板渲染（2）→ 验证 SFT 对话回归（6）→ `--lora_weight` 三态对比（7）→ 检查 RL 后回归退化（9）
- **[trainer/trainer_utils.py](trainer/trainer_utils.py)** 贯穿 3/5/6/7：`get_model_params` 解读参数分布（3）→ checkpoint save/load 与 resume（5/6/7）
- **[trainer/rollout_engine.py](trainer/rollout_engine.py)** 贯穿 9 的三个分支：GRPO/PPO/Agent 共用 rollout 与 reward 收集管线
- **[dataset/lm_dataset.py](dataset/lm_dataset.py)** 贯穿 4/6/8：验证 mask 正确性（4）→ SFT 数据产出（6）→ DPO chosen/rejected mask（8）

---

## 关键提醒

1. **目录切换**：`eval_llm.py` 在根目录跑，训练脚本在 `trainer/` 跑，服务脚本在 `scripts/` 跑，路径都是相对路径，不能混用。
2. **显存**：Dense 模型 Pretrain/SFT 显存要求较低，但 PPO/GRPO/Agent 同时加载多个模型，显存成倍增长。
3. **模型权重**：可以从零 Pretrain，也可以直接下载已发布的 `full_sft` 权重跳到里程碑 6/7/8/9。
4. **LoRA 数据**：`lora_medical.jsonl` 需要自行准备，仓库不提供。
