# `dataset/__init__.py` 介绍

对应源码：[`__init__.py`](./__init__.py)

该文件用于将 `dataset` 目录标识为 Python 包。当前文件为空，没有初始化逻辑，也没有统一重导出数据集类；训练脚本直接从 [`lm_dataset.py`](./lm_dataset.py) 导入所需类。

## 项目脉络

训练入口使用下面这种绝对包导入：

```python
from dataset.lm_dataset import PretrainDataset
```

`__init__.py` 使 `dataset` 成为常规 Python 包，从而让解释器可以继续解析其中的 `lm_dataset` 模块。各训练脚本会先把仓库根目录加入 `sys.path`，所以即使当前目录是 `trainer/`，仍能找到这个包。

## 为什么保持为空

包初始化文件可以写统一导出或注册逻辑，但这里刻意没有这样做：

- 导入 `dataset` 本身不会加载 PyTorch、Hugging Face Datasets 或数据文件。
- 不会在包导入阶段修改环境或产生随机行为。
- 数据集的来源清晰，调用方能从导入语句直接看到实现位于 `lm_dataset.py`。

这是一种低副作用设计。若未来希望写成 `from dataset import SFTDataset`，才需要在这里显式重导出；当前代码并未提供这种接口。
