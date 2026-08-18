# QPI 数据 I/O 模块接口文档

## 模块概述

`qpi_io.py` 提供 QPI（Quasiparticle Interference，准粒子干涉）结果的 HDF5 落盘与读取功能。它是 `JDOSQPI.calculate()` 与 `QPIBorn.calculate()` 等计算模块的保存层，同时负责在读取时重建 q 网格、可选地按 `q_range` 扩展/裁剪，并把分数坐标转换为实空间倒空间坐标。

- **模块路径**：`src/stm_data_processing/io/qpi_io.py`
- **职责**：
  - `save_qpi_to_h5()`：将未扩展的 `qpi_layers` 保存到 HDF5（默认网格范围 `[-0.5, 0.5)`）。
  - `load_qpi_from_h5()`：读取 HDF5，重建网格，按需扩展，返回与计算模块 `calculate()` 完全一致的结构。
- **与计算模块的对应关系**：`stm.qpi_jdos.JDOSQPI.calculate()`（`module_type="jdos"`）与 `stm.qpi_born`（`module_type="born"`）在 `output_path` 非空时调用 `save_qpi_to_h5()`。

**设计约定**：HDF5 只保存原始网格 `[-0.5, 0.5)` 的数据（节省空间）；扩展与实空间坐标转换在加载阶段完成。

---

## 核心函数

### `save_qpi_to_h5(...)`

将 QPI 结果保存为 HDF5 文件。

```python
def save_qpi_to_h5(
    qpi_layers: np.ndarray,
    output_path: str,
    energy_range: float | np.ndarray | list[float],
    module_type: str,
    bvecs: np.ndarray | None = None,
    eta: float = 0.001,
    normalize: bool = True,
    nq: int = 256,
    V: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    bands: str | list[int] | None = None,
    compression: str = "gzip",
    compression_opts: int = 6,
    **metadata_kwargs,
) -> None
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `qpi_layers` | `np.ndarray` | 必需 | QPI 强度数组，形状 `(n_energies, nq, nq)` 或 `(nq, nq)` |
| `output_path` | `str` | 必需 | 输出 HDF5 文件路径 |
| `energy_range` | `float \| np.ndarray \| list[float]` | 必需 | 计算使用的能量值（标量或数组） |
| `module_type` | `str` | 必需 | 模块类型标识，如 `"jdos"`、`"born"` |
| `bvecs` | `np.ndarray \| None` | `None` | 倒格矢，形状 `(3, 3)`；非空时保存为 dataset |
| `eta` | `float` | `0.001` | 洛伦兹展宽参数 |
| `normalize` | `bool` | `True` | 是否对 `qpi_layers` 做了归一化 |
| `nq` | `int` | `256` | 每维 q 点数量 |
| `V` | `np.ndarray \| None` | `None` | 散射势矩阵，非空时保存为 dataset |
| `mask` | `np.ndarray \| None` | `None` | 实空间掩码，非空时保存为 dataset |
| `bands` | `str \| list[int] \| None` | `None` | 能带索引；`"all"` 存为字符串，否则存为 int 数组 |
| `compression` | `str` | `"gzip"` | 压缩算法 |
| `compression_opts` | `int` | `6` | 压缩级别（0~9） |
| `**metadata_kwargs` | - | - | 额外元数据，`value` 非 `None` 时写入属性 |

### `load_qpi_from_h5(h5_path, q_range)`

从 HDF5 文件读取 QPI 结果。

```python
def load_qpi_from_h5(
    h5_path: str,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | dict]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `h5_path` | `str` | 必需 | 输入 HDF5 文件路径 |
| `q_range` | `tuple[float, float] \| None` | `None` | 目标 `(q_min, q_max)`；提供时用 `extend_qpi` 扩展/裁剪 |

**返回**：`dict`，键如下：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `qpi_layers` | `np.ndarray` | `(n_energies, Nq, Nq)` 或 `(Nq, Nq)` | 加载（可能已扩展）的 QPI 强度 |
| `q1_grid` | `np.ndarray` | `(Nq, Nq)` | 分数坐标 q1 网格 |
| `q2_grid` | `np.ndarray` | `(Nq, Nq)` | 分数坐标 q2 网格 |
| `qx_grid` | `np.ndarray \| None` | `(Nq, Nq)` | 实空间倒空间坐标（1/Å），`bvecs` 缺失时为 `None` |
| `qy_grid` | `np.ndarray \| None` | `(Nq, Nq)` | 实空间倒空间坐标（1/Å），`bvecs` 缺失时为 `None` |
| `metadata` | `dict` | - | 所有加载的属性与可选数组（见下） |

**`metadata` 字典内容**：

| 键 | 类型 | 说明 |
|----|------|------|
| `module_type` | `str` | 模块类型，缺省 `"unknown"` |
| `eta` | `float` | 展宽参数，缺省 `0.001` |
| `normalize` | `bool` | 是否归一化，缺省 `True` |
| `nq` | `int` | 原始 q 网格大小，缺省 `256` |
| `energy_range` | 标量或数组 | 计算能量，缺省 `None` |
| `bands` | `None`/`str`/数组 | 能带索引，缺省 `None` |
| `bvecs` | `np.ndarray \| None` | 倒格矢 |
| `V` | `np.ndarray \| None` | 散射势矩阵 |
| `mask` | `np.ndarray \| None` | 实空间掩码 |
| （其余） | - | 文件属性中未被上述键收录的项（`metadata_extra`） |

---

## HDF5 文件结构

`save_qpi_to_h5()` 写入的文件布局如下：

**Datasets**：

| 名称 | 存在条件 | 形状 | 说明 |
|------|----------|------|------|
| `qpi_layers` | 总是 | `(n_energies, nq, nq)` 或 `(nq, nq)` | QPI 强度（未扩展），`compression`/`compression_opts` 压缩 |
| `bvecs` | `bvecs is not None` | `(3, 3)` | 倒格矢 |
| `V` | `V is not None` | 任意 | 散射势矩阵 |
| `mask` | `mask is not None` | 任意 | 实空间掩码 |

**Attributes**：

| 名称 | 存在条件 | 说明 |
|------|----------|------|
| `module_type` | 总是 | 模块类型字符串 |
| `eta` | 总是 | 展宽参数 |
| `normalize` | 总是 | 归一化标志 |
| `nq` | 总是 | 每维 q 点数 |
| `energy_range` | 总是 | 标量或 numpy 数组 |
| `bands` | `bands is not None` | `"all"` 或 int 数组 |
| （`**metadata_kwargs`） | `value is not None` | 额外元数据 |

> 注意：`q1_grid`/`q2_grid` **不落盘**。加载时由 `np.linspace(-0.5, 0.5, nq, endpoint=False)` 重建；`qx_grid`/`qy_grid` 由 `frac_to_real_2d` 依 `bvecs` 计算（仅使用 `bvecs[0,0]`、`bvecs[1,0]`、`bvecs[0,1]`、`bvecs[1,1]` 四项投影）。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.io.qpi_io import save_qpi_to_h5, load_qpi_from_h5

# 假设 qpi_layers 为 (n_energies, nq, nq) 强度
qpi_layers = np.random.rand(3, 256, 256)
bvecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# 保存
save_qpi_to_h5(
    qpi_layers=qpi_layers,
    output_path="./qpi.h5",
    energy_range=np.linspace(-1.0, 1.0, 3),
    module_type="jdos",
    bvecs=bvecs,
    eta=0.001,
    normalize=True,
    nq=256,
)

# 加载（不扩展）
data = load_qpi_from_h5("./qpi.h5")
print(data["qpi_layers"].shape)  # (3, 256, 256)
print(data["metadata"]["module_type"])  # 'jdos'

# 加载并扩展/裁剪到 [-0.3, 0.3)
data_ext = load_qpi_from_h5("./qpi.h5", q_range=(-0.3, 0.3))
print(data_ext["q1_grid"].shape)  # 扩展后网格
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
| `KeyError` | 加载时文件缺少 `qpi_layers` dataset |
| `OSError` | 文件无法打开/读取 |

> 其余可选 dataset（`bvecs`/`V`/`mask`）与属性（`module_type`/`eta` 等）均以 `.get()` 或缺省值兜底，不会因缺失而报错。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`energy_range` 可为标量或数组；落盘时标量原样存储、数组转为 `np.array`**
- [ ] **`module_type` 与计算模块约定一致：`JDOSQPI` → `"jdos"`，Born → `"born"`**
- [ ] **HDF5 仅保存原始网格 `[-0.5, 0.5)`；扩展通过加载时传 `q_range` 完成**
- [ ] **`load_qpi_from_h5()` 返回键与 `JDOSQPI.calculate()` / `QPIBorn.calculate()` 完全一致（`qpi_layers`/`q1_grid`/`q2_grid`/`qx_grid`/`qy_grid`/`metadata`）**
- [ ] **`metadata` 字段名一致：`module_type`、`eta`、`normalize`、`nq`、`energy_range`、`bands`、`bvecs`、`V`、`mask`**
- [ ] **`bvecs` 缺失时 `qx_grid`/`qy_grid` 为 `None`，下游需判空**

---

## 与 JDOSQPI / QPIBorn 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| `calculate()` 返回结构与 `load_qpi_from_h5()` 一致 | ✅ | 键集合与字段名均一致 |
| `save_qpi_to_h5()` 被计算模块正确调用 | ✅ | `qpi_jdos`/`qpi_born` 均以匹配的关键字调用（`output_path`、`bvecs`、`V`、`nq`、`eta`、`normalize`、`bands`） |
| HDF5 保存原始网格，加载时扩展 | ✅ | 节省存储空间，接口语义一致 |
| `extend_qpi`/`frac_to_real_2d` 处理逻辑一致 | ✅ | 计算模块与加载层使用同一工具函数 |

---

## 版本信息

- 模块路径：`src/stm_data_processing/io/qpi_io.py`
- 数据格式：HDF5（`qpi_layers` 主 dataset + 可选 `bvecs`/`V`/`mask`）
- 默认网格：`[-0.5, 0.5)`，`nq=256`，`endpoint=False`
- 日志级别：`INFO`（保存/加载进度、文件大小、形状）
