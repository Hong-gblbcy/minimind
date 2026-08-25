# `test_env.py` 介绍

对应源码：[`test_env.py`](./test_env.py)

## 文件定位

最小化的 PyTorch 运行环境检查脚本，不依赖项目模型或数据。

运行后输出：

- 当前安装的 PyTorch 版本。
- `torch.cuda.is_available()` 的结果，用于判断 CUDA 是否可用。

## 项目脉络

训练或推理失败时，排查顺序通常是“Python 能否导入 PyTorch → PyTorch 版本是否符合依赖 → CUDA 是否被当前 PyTorch 识别 → 再检查模型和数据”。这个脚本只覆盖最前面的环境层，不涉及 MiniMind 代码，因此适合作为安装完成后的第一步检查。

```text
Python 环境
  └─> import torch
        ├─> torch.__version__
        └─> torch.cuda.is_available()
              └─> 决定训练/推理脚本的默认 device
```

## 技术点解释

`torch.__version__` 不只是 PyTorch 的语义版本，某些发行包还会带类似 `+cu12x` 的构建后缀，表示它针对哪一代 CUDA 运行时构建。

`torch.cuda.is_available()` 为 `True` 表示当前 PyTorch 进程能够使用 CUDA。它综合受到以下条件影响：

- 机器是否有受支持的 NVIDIA GPU。
- 驱动是否正确安装并对当前进程可见。
- 安装的 PyTorch 是否为 CUDA 构建，而不是 CPU-only 构建。
- 容器或环境是否正确暴露 GPU。

它为 `False` 并不能单独指出是哪一层出错，也不检查显存是否足够。需要进一步诊断时，可额外查看 `torch.version.cuda`、`torch.cuda.device_count()` 和系统驱动信息。

## 与项目其他文件的关系

多个训练和推理入口都使用类似逻辑：

```python
"cuda:0" if torch.cuda.is_available() else "cpu"
```

因此这里的检查结果会直接影响这些脚本的默认设备选择。不过用户显式传入 `--device` 后，最终行为仍以命令行参数为准。

## 边界

- 不检查 `transformers`、`datasets`、`tokenizers`、FastAPI 等其他依赖。
- 不创建模型，也不执行矩阵运算，无法验证 GPU 计算是否稳定。
- 不检查多 GPU、NCCL 或 DDP；这些需要实际运行 `torchrun` 才能验证。

```bash
python test_env.py
```
