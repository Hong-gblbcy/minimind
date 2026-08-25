# `test_env.py` 介绍

对应源码：[`test_env.py`](./test_env.py)

## 文件定位

最小化的 PyTorch 运行环境检查脚本，不依赖项目模型或数据。

运行后输出：

- 当前安装的 PyTorch 版本。
- `torch.cuda.is_available()` 的结果，用于判断 CUDA 是否可用。

```bash
python test_env.py
```

