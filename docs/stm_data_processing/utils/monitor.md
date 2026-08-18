# Wannier90 收敛监控模块接口文档

## 模块概述

- **模块路径**：`src/stm_data_processing/utils/monitor.py`
- **职责**：从 Wannier90 输出的 `.wout` 文本日志中解析 disenchantment（退纠缠）与 wannierisation（最局域化）两阶段的迭代收敛数据，供监控/可视化收敛过程使用。
- **性质**：`load_wout_file` 为纯函数，仅读取文件，无写文件等副作用，可安全 `import`。

---

## 核心函数

### `load_wout_file(filename)`

加载并解析 Wannier90 `.wout` 文件，返回收敛数据的字典。

```python
def load_wout_file(filename: str) -> dict
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `str` | Wannier90 `.wout` 文件路径 |

**返回**：`dict`，包含以下键：

| 键 | 类型 | 形状/说明 |
|----|------|-----------|
| `disentangle_data` | `np.ndarray` | 形状 `(3, num_iter)`，三行依次为 `[iteration, time, delta]`；无数据时为 `np.array([])` |
| `wannierise_spreads` | `np.ndarray` | 形状 `(num_cycle, num_wann, 1)`，各循环各 Wannier 函数的展宽（spread）；无数据时为 `np.array([])` |
| `wannierise_od` | `np.ndarray` | 形状 `(num_cycle, 1)`，各循环的 `O_D` 值；无数据时为 `np.array([])` |
| `disentangle_tar` | `float \| None` | 退纠缠收敛容差（tolerance）；未解析到时为 `None` |
| `wannierise_tar` | `float \| None` | 最局域化收敛容差（tolerance）；未解析到时为 `None` |

---

## 解析内容说明

`load_wout_file` 分两次读取同一文件：

1. **第一遍**：定位 `"Number of Wannier Functions"` 行，用正则 `:\s*(\d+)` 提取 Wannier 函数总数 `num_wann`。
2. **第二遍**：单次遍历完成全部数据解析，主要匹配的文本标记如下：

| 标记 | 用途 |
|------|------|
| `*--- DISENTANGLE ---*` 头区块 | 识别退纠缠参数区，从 `"Convergence tolerence"` 行提取 `disentangle_tar`（注意 Wannier90 输出原文拼写为 `tolerence`） |
| `*--- WANNIERISE ---*` 头区块 | 识别最局域化参数区，从 `"Convergence tolerence"` 行提取 `wannierise_tar` |
| `"Extraction of optimally-connected subspace"` | 退纠缠迭代区开始标记 |
| `"Final Omega_I"` | 退纠缠迭代区结束标记 |
| 数据行正则 | `^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+\s+([\d.E+-]+)\s+([\d.E+-]+)`，提取 `(迭代号, delta, time)` |
| `"Initial State"` / `"Cycle:"` | 最局域化循环起始与循环编号 |
| `"WF centre and spread"` | 提取各 Wannier 函数展宽（正则 `WF centre and spread\s+\d+\s+\([^)]+\)\s+([\d.]+)`） |
| `"Sum of centres and spreads"` | 单循环展宽读取结束 |
| `"O_D="` 且含 `"<-- DLTA"` | 提取 `O_D` 值（正则 `O_D=\s*([\d.E+-]+)`） |

容差数值解析用正则 `:\s*([\d.E+-]+)`，`float()` 转换失败时置为 `None`。

后处理细节：

- 退纠缠数据按 `np.array([data_iter, data_time, data_delta])` 组装为 `(3, N)`，行序为 `[iteration, time, delta]`。
- 展宽数据 `spreads_data` 重塑为 `(num_cycle, num_wann, 1)`；当存在 `Initial State`（`current_cycle == -1`）且循环数多于 1 时，剔除第一项（Initial State），`od_data` 同步剔除首项。
- `od_data` 重塑为 `(num_cycle, 1)`。

---

## 使用示例

```python
from stm_data_processing.utils.monitor import load_wout_file

data = load_wout_file("wannier90.wout")

# 退纠缠收敛数据：第 0 行迭代号，第 1 行时间，第 2 行 delta
dis = data["disentangle_data"]
if dis.size:
    print("迭代次数:", dis.shape[1])
    print("最后一次 delta:", dis[2, -1])

# 最局域化展宽与 O_D
print("展宽数组形状:", data["wannierise_spreads"].shape)   # (num_cycle, num_wann, 1)
print("O_D 数组形状:", data["wannierise_od"].shape)         # (num_cycle, 1)

# 收敛容差
print("退纠缠容差:", data["disentangle_tar"])
print("最局域化容差:", data["wannierise_tar"])
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `re` | 是 | 正则解析 `.wout` 文本 |
| `pathlib.Path` | 是 | 文件路径处理 |
| `numpy` | 是 | 组装数值数组 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `FileNotFoundError` | `filename` 指向的文件不存在（由 `Path(...).open()` 抛出） |
| `UnicodeDecodeError` | 读取时以 `errors="ignore"` 规避，实际不会抛出 |

> 解析失败（如未找到 `"Number of Wannier Functions"`、容差行）通常不会抛异常，而是保留 `num_wann = 0` 或容差 `None`，返回的空数组 `np.array([])` 需由调用方判空。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`filename` 为有效的 Wannier90 `.wout` 文件路径**
- [ ] **`disentangle_data` 为 `(3, N)` 数组，行序 `[iteration, time, delta]`**
- [ ] **`wannierise_spreads` 形状为 `(num_cycle, num_wann, 1)`，`wannierise_od` 形状为 `(num_cycle, 1)`**
- [ ] **无数据时返回 `np.array([])`，使用前判空（`if dis.size:`）**
- [ ] **容差字段可能为 `None`，展示前判空**
- [ ] **存在 Initial State 时，展宽/O_D 会剔除首项，实际 cycle 数可能比日志中少 1**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/monitor.py`
- 解析目标：Wannier90 `.wout` 文本日志
- 日志级别：无（静默解析）
- **纯函数实现，仅依赖 `re`/`pathlib`/`numpy`，可安全 import**
