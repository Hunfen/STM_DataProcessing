# 磁化率数据 I/O 模块接口文档

## 模块概述

`susceptibility_io.py` 提供磁化率（Lindhard susceptibility）计算结果的 HDF5 落盘与读取功能。它是 `SusceptibilityCalculator_wang2012.calculate()` 计算模块的保存层，同时负责在读取时重建 q 网格、可选地按 `q_range` 扩展/裁剪，并把分数坐标转换为实空间倒空间坐标。

- **模块路径**：`src/stm_data_processing/io/susceptibility_io.py`
- **职责**：
  - `save_susceptibility_to_h5()`：将未扩展的磁化率数组保存到 HDF5（默认网格范围 `[-0.5, 0.5)`）。
  - `load_susceptibility_from_h5()`：读取 HDF5，重建网格，按需扩展，返回与计算模块 `calculate()` 一致的结构。
- **与计算模块的对应关系**：`dft.wannier90.mlwf_susceptibility.SusceptibilityCalculator_wang2012.calculate()` 在 `output_path` 非空时调用 `save_susceptibility_to_h5()`。

**设计约定**：HDF5 只保存原始网格 `[-0.5, 0.5)` 的数据（节省空间）；扩展与实空间坐标转换在加载阶段完成。

---

## 核心函数

### `save_susceptibility_to_h5(...)`

将磁化率结果保存为 HDF5 文件。

```python
def save_susceptibility_to_h5(
    susceptibility: np.ndarray,
    output_path: str,
    module_type: str = "susceptibility",
    bvecs: np.ndarray | None = None,
    eta: float = 5e-3,
    omega_limit: float | None = None,
    resolution: float | None = None,
    nq: int = 256,
    compression: str = "gzip",
    compression_opts: int = 6,
    **metadata_kwargs,
) -> None
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `susceptibility` | `np.ndarray` | 必需 | 磁化率数组，形状 `(nq, nq)`（或更高维） |
| `output_path` | `str` | 必需 | 输出 HDF5 文件路径 |
| `module_type` | `str` | `"susceptibility"` | 模块类型标识 |
| `bvecs` | `np.ndarray \| None` | `None` | 倒格矢，形状 `(2,2)` 或 `(3,3)`；非空时保存为 dataset |
| `eta` | `float` | `5e-3` | 洛伦兹展宽参数 |
| `omega_limit` | `float \| None` | `None` | 积分能量上限（eV），非空时写入属性 |
| `resolution` | `float \| None` | `None` | 积分能量分辨率（eV），非空时写入属性 |
| `nq` | `int` | `256` | 每维 q 点数量 |
| `compression` | `str` | `"gzip"` | 压缩算法 |
| `compression_opts` | `int` | `6` | 压缩级别（0~9） |
| `**metadata_kwargs` | - | - | 额外元数据；`value` 非 `None` 时写入属性（失败仅告警，不中断） |

### `load_susceptibility_from_h5(h5_path, q_range)`

从 HDF5 文件读取磁化率结果。

```python
def load_susceptibility_from_h5(
    h5_path: str,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | dict[str, Any]]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `h5_path` | `str` | 必需 | 输入 HDF5 文件路径 |
| `q_range` | `tuple[float, float] \| None` | `None` | 目标 `(q_min, q_max)`；提供时用 `extend_qpi` 扩展/裁剪 |

**返回**：`dict`，键如下：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `data` | `np.ndarray` | `(Nq, Nq)` | 加载（可能已扩展）的磁化率数组 |
| `q1_grid` | `np.ndarray` | `(Nq, Nq)` | 分数坐标 q1 网格 |
| `q2_grid` | `np.ndarray` | `(Nq, Nq)` | 分数坐标 q2 网格 |
| `qx_grid` | `np.ndarray \| None` | `(Nq, Nq)` | 实空间倒空间坐标（1/Å），`bvecs` 缺失时为 `None` |
| `qy_grid` | `np.ndarray \| None` | `(Nq, Nq)` | 实空间倒空间坐标（1/Å），`bvecs` 缺失时为 `None` |
| `metadata` | `dict` | - | 所有加载的属性与可选数组（见下） |

**`metadata` 字典内容**：

| 键 | 类型 | 说明 |
|----|------|------|
| `module_type` | `str` | 模块类型，缺省 `"susceptibility"` |
| `eta` | `float` | 展宽参数，缺省 `5e-3` |
| `nq` | `int` | 原始 q 网格大小，缺省 `256` |
| `omega_limit` | `float \| None` | 积分能量上限，缺省 `None` |
| `resolution` | `float \| None` | 积分能量分辨率，缺省 `None` |
| `bvecs` | `np.ndarray \| None` | 倒格矢 |
| （其余） | - | 文件属性中未被上述键收录的项（`metadata_extra`，例如 `minit`/`mfin`） |

---

## HDF5 文件结构

`save_susceptibility_to_h5()` 写入的文件布局如下：

**Datasets**：

| 名称 | 存在条件 | 形状 | 说明 |
|------|----------|------|------|
| `susceptibility` | 总是 | `(nq, nq)`（或更高维） | 磁化率数据（未扩展），`compression`/`compression_opts` 压缩 |
| `bvecs` | `bvecs is not None` | `(2,2)` 或 `(3,3)` | 倒格矢 |

**Attributes**：

| 名称 | 存在条件 | 说明 |
|------|----------|------|
| `module_type` | 总是 | 模块类型字符串 |
| `eta` | 总是 | 展宽参数 |
| `nq` | 总是 | 每维 q 点数 |
| `omega_limit` | `omega_limit is not None` | 积分能量上限 |
| `resolution` | `resolution is not None` | 积分能量分辨率 |
| （`**metadata_kwargs`） | `value is not None` | 额外元数据（写入失败仅告警） |

> 注意：`q1_grid`/`q2_grid` **不落盘**。加载时由 `np.linspace(-0.5, 0.5, nq, endpoint=False)` 重建；`qx_grid`/`qy_grid` 由 `frac_to_real_2d` 依 `bvecs` 计算（仅使用 `bvecs[0,0]`、`bvecs[1,0]`、`bvecs[0,1]`、`bvecs[1,1]` 四项投影）。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.io.susceptibility_io import (
    save_susceptibility_to_h5,
    load_susceptibility_from_h5,
)

chi_q = np.random.rand(256, 256)
bvecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# 保存
save_susceptibility_to_h5(
    susceptibility=chi_q,
    output_path="./chi.h5",
    module_type="imag_Lindhard",
    bvecs=bvecs,
    eta=5e-3,
    omega_limit=1.0,
    resolution=0.01,
    nq=256,
)

# 加载（不扩展）
data = load_susceptibility_from_h5("./chi.h5")
print(data["data"].shape)            # (256, 256)
print(data["metadata"]["omega_limit"])  # 1.0

# 加载并扩展/裁剪到 [-0.3, 0.3)
data_ext = load_susceptibility_from_h5("./chi.h5", q_range=(-0.3, 0.3))
print(data_ext["q1_grid"].shape)     # 扩展后网格
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `h5py` | 是 | HDF5 读写 |
| `numpy` | 是 | 网格重建（`linspace`/`meshgrid`） |
| `pathlib.Path` | 是 | 文件大小统计 |
| `extend_qpi`（本包 utils） | 是 | q 网格扩展/裁剪 |
| `frac_to_real_2d`（本包 utils） | 是 | 分数坐标 → 实空间坐标 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `KeyError` | 加载时文件缺少 `susceptibility` dataset |
| `OSError` | 文件无法打开/读取 |

> 额外元数据 `**metadata_kwargs` 写入属性失败时仅记录 `warning`，不抛异常（与 `save_qpi_to_h5` 不同）。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`susceptibility` 为主要数据集名（不是 `qpi_layers`）**
- [ ] **`module_type` 与计算模块约定一致：`SusceptibilityCalculator_wang2012` 使用 `"imag_Lindhard"`**
- [ ] **HDF5 仅保存原始网格 `[-0.5, 0.5)`；扩展通过加载时传 `q_range` 完成**
- [ ] **`load_susceptibility_from_h5()` 返回键与 `SusceptibilityCalculator_wang2012.calculate()` 一致（`data`/`q1_grid`/`q2_grid`/`qx_grid`/`qy_grid`/`metadata`）**
- [ ] **`metadata` 核心字段一致：`module_type`、`eta`、`nq`、`omega_limit`、`resolution`、`bvecs`**
- [ ] **`bvecs` 缺失时 `qx_grid`/`qy_grid` 为 `None`，下游需判空**

---

## 与 SusceptibilityCalculator_wang2012 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| `calculate()` 返回结构与 `load_susceptibility_from_h5()` 一致 | ✅ | 键集合（`data`/`q1_grid`/`q2_grid`/`qx_grid`/`qy_grid`/`metadata`）一致 |
| 保存/加载共享同一网格重建逻辑 | ✅ | 均基于 `[-0.5, 0.5)`、`nq`、`extend_qpi`、`frac_to_real_2d` |
| `module_type` 标识一致 | ⚠️ | `calculate()` 的 metadata 用 `"imag_Lindhard"`；其内部保存调用传入的是 `"Imaginary Lindhard"`（见下） |

**✅ 接口一致性**：`mlwf_susceptibility.calculate()` 内部调用 `save_susceptibility_to_h5()` 时已使用正确的关键字 `output_path=`、`bvecs=`（曾存在 `outpath=`/`bevecs=` 关键字不匹配的历史问题，已修复）。其余 `eta`/`omega_limit`/`resolution`/`nq` 与本函数形参一一对应，`minit`/`mfin` 通过 `**metadata_kwargs` 以属性形式落盘。

---

## 版本信息

- 模块路径：`src/stm_data_processing/io/susceptibility_io.py`
- 数据格式：HDF5（`susceptibility` 主 dataset + 可选 `bvecs`）
- 默认网格：`[-0.5, 0.5)`，`nq=256`，`endpoint=False`
- 日志级别：`INFO`（保存/加载进度、文件大小、形状）；属性写入失败为 `WARNING`
