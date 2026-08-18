# EK2D 能带数据 I/O 模块接口文档

## 模块概述

`ek2d_io.py` 提供二维能带（E-k）轮廓图数据的 HDF5 落盘与读取功能。它是 `EK2DCalculator.calculate()` 计算模块的保存层，负责保存原始布里渊区 `[-0.5, 0.5)` 的能带数据，并在加载时按 `k_range` 做周期性扩展、把分数坐标转换为实空间倒空间坐标。

- **模块路径**：`src/stm_data_processing/io/ek2d_io.py`
- **职责**：
  - `save_ek2d()`：将 `(num_wann, nk, nk)` 的能带与 k 网格保存到 HDF5（含重初始化所需元数据）。
  - `load_ek2d()`：读取 HDF5，重建网格，按需扩展，返回字典结构。
  - `_extend_ek2d_static()`：把原始 BZ 轮廓图周期性扩展到任意分数 k 窗口。
- **与计算模块的对应关系**：`dft.wannier90.mlwf_ek2d.EK2DCalculator.calculate(save_to_file=...)` 内部调用 `EK2DIO.save_ek2d()`，并在 `k_range` 非空时调用 `EK2DIO._extend_ek2d_static()`。

**设计约定**：HDF5 只保存原始 BZ `[-0.5, 0.5)` 数据（节省空间）；扩展与实空间坐标转换在加载阶段完成。

---

## 核心类：`EK2DIO`

### 类定义

```python
class EK2DIO:
    """A class for handling I/O operations of 2D band structure data."""
```

### 静态方法

#### `save_ek2d(energies, k1_grid, k2_grid, filename, mlwf_hamiltonian)`

保存能带数据到 HDF5 文件。

```python
@staticmethod
def save_ek2d(
    energies: np.ndarray,
    k1_grid: np.ndarray,
    k2_grid: np.ndarray,
    filename: str,
    mlwf_hamiltonian: MLWFHamiltonian,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `energies` | `np.ndarray` | `(num_wann, nk, nk)` 能带能量（eV） |
| `k1_grid` | `np.ndarray` | `(nk, nk)` 分数坐标 k1 网格 |
| `k2_grid` | `np.ndarray` | `(nk, nk)` 分数坐标 k2 网格 |
| `filename` | `str` | 输出路径；无 `.h5`/`.hdf5` 扩展名时自动追加 `.h5` |
| `mlwf_hamiltonian` | `MLWFHamiltonian` | 哈密顿量实例，自动提取其 `bvecs`、`folder`、`seedname` |

> ⚠️ 参数名为 `mlwf_hamiltonian`。本方法内部访问 `mlwf_hamiltonian.bvecs`、`.folder`、`.seedname` 三个属性。

#### `load_ek2d(filename, k_range)`

从 HDF5 文件读取能带数据。

```python
@staticmethod
def load_ek2d(
    filename: str,
    k_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | None]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `filename` | `str` | 必需 | HDF5 文件路径 |
| `k_range` | `tuple[float, float] \| None` | `None` | 目标 `(kmin, kmax)`；提供时用 `_extend_ek2d_static` 扩展 |

**返回**：`dict[str, np.ndarray | None]`，键如下：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `energies` | `np.ndarray` | `(num_wann, Nk1, Nk2)` | 能带能量（eV） |
| `kx` | `np.ndarray \| None` | `(Nk1, Nk2)` | 实空间 k 坐标（1/Å），`bvecs` 缺失时为 `None` |
| `ky` | `np.ndarray \| None` | `(Nk1, Nk2)` | 实空间 k 坐标（1/Å），`bvecs` 缺失时为 `None` |
| `k1_grid` | `np.ndarray` | `(Nk1, Nk2)` | 分数坐标 k1 网格 |
| `k2_grid` | `np.ndarray` | `(Nk1, Nk2)` | 分数坐标 k2 网格 |
| `bvecs` | `np.ndarray \| None` | `(3, 3)` | 倒格矢（1/Å） |
| `metadata` | `dict` | - | 文件属性字典（`dict(f.attrs)`） |

#### `_extend_ek2d_static(ek2d, k_range)`

将能带轮廓图从原始 BZ `[-0.5, 0.5)` 周期性扩展到任意分数 k 窗口 `[kmin, kmax)`。

```python
@staticmethod
def _extend_ek2d_static(
    ek2d: dict[str, np.ndarray],
    k_range: tuple[float, float],
) -> dict[str, np.ndarray]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ek2d` | `dict[str, np.ndarray]` | 含 `energies`、`k1_grid`、`k2_grid` 三键的字典 |
| `k_range` | `tuple[float, float]` | `(kmin, kmax)` 目标分数 k 窗口 |

**返回**：同结构的字典，含扩展并裁剪后的 `energies`、`k1_grid`、`k2_grid`。

---

## HDF5 文件结构

`save_ek2d()` 写入的文件布局如下：

**Datasets**：

| 名称 | 存在条件 | 形状 | 说明 |
|------|----------|------|------|
| `energies` | 总是 | `(num_wann, nk, nk)` | 能带能量，gzip 压缩（级别 4） |
| `k1_grid` | 总是 | `(nk, nk)` | 分数 k1 网格，gzip 压缩（级别 4） |
| `k2_grid` | 总是 | `(nk, nk)` | 分数 k2 网格，gzip 压缩（级别 4） |
| `bvecs` | `bvecs is not None` | `(3, 3)` | 倒格矢 |

**Attributes**：

| 名称 | 存在条件 | 说明 |
|------|----------|------|
| `num_wann` | 总是 | Wannier 函数数量 |
| `nk` | 总是 | 每维 k 点数 |
| `total_points` | 总是 | `nk * nk` |
| `units_energy` | 总是 | `"eV"` |
| `units_k_frac` | 总是 | `"reciprocal lattice units"` |
| `creation_date` | 总是 | 创建时间 `%Y-%m-%d %H:%M:%S` |
| `generator` | 总是 | `"EK2DCalculator"` |
| `folder` | `folder is not None` | 解析后的绝对路径字符串 |
| `seedname` | `seedname is not None` | seedname 字符串 |

> 加载时 `metadata = dict(f.attrs)`，即上述全部属性。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.ek2d_io import EK2DIO

# 假设已有哈密顿量与能带数据
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")
energies = np.zeros((ham.num_wann, 256, 256))
k1, k2 = np.meshgrid(
    np.linspace(-0.5, 0.5, 256, endpoint=False),
    np.linspace(-0.5, 0.5, 256, endpoint=False),
    indexing="ij",
)

# 保存（注意关键字名 mlwf_hamiltonian）
EK2DIO.save_ek2d(
    energies=energies,
    k1_grid=k1,
    k2_grid=k2,
    filename="./ek2d.h5",
    mlwf_hamiltonian=ham,
)

# 加载（不扩展）
data = EK2DIO.load_ek2d("./ek2d.h5")
print(data["energies"].shape)      # (num_wann, 256, 256)
print(data["metadata"]["generator"])  # 'EK2DCalculator'

# 加载并扩展到 [-1.0, 1.0)
data_ext = EK2DIO.load_ek2d("./ek2d.h5", k_range=(-1.0, 1.0))
print(data_ext["energies"].shape)  # (num_wann, 512, 512)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `h5py` | 是 | HDF5 读写 |
| `numpy` | 是 | 网格重建与 `np.ix_` 裁剪 |
| `pathlib.Path` | 是 | 路径解析与存在性判断 |
| `MLWFHamiltonian` | 是 | `save_ek2d` 入参类型（提取 `bvecs`/`folder`/`seedname`） |
| `frac_to_real_2d`（本包 utils） | 是 | 分数坐标 → 实空间坐标 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `FileNotFoundError` | `load_ek2d` 时文件不存在 |
| `ValueError` | `load_ek2d` 时缺少必需 dataset（`energies`/`k1_grid`/`k2_grid`）或读取失败（`Exception` 统一包装为 `ValueError`） |

> 扩展名非 `.h5`/`.hdf5` 时，`load_ek2d` 仅记录 `warning`，不阻止加载。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`save_ek2d` 的第五个参数关键字为 `mlwf_hamiltonian`（不是 `hamiltonian`）**
- [ ] **`mlwf_hamiltonian` 需提供 `bvecs`、`folder`、`seedname` 三个属性（`MLWFHamiltonian` 满足）**
- [ ] **`energies` 形状为 `(num_wann, nk, nk)`，`k1_grid`/`k2_grid` 为 `(nk, nk)`**
- [ ] **HDF5 仅保存原始 BZ `[-0.5, 0.5)`；扩展通过加载时传 `k_range` 完成**
- [ ] **`load_ek2d` 返回 7 个键（含 `metadata`）；`EK2DCalculator.calculate()` 返回 6 个键（无 `metadata`）——两者并非逐键一致**
- [ ] **`bvecs` 缺失时 `kx`/`ky` 为 `None`，下游需判空**

---

## 与 EK2DCalculator 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| 返回结构 | ⚠️ | `load_ek2d()` 比 `EK2DCalculator.calculate()` 多一个 `metadata` 键（`calculate()` 不返回 `metadata`） |
| HDF5 保存原始网格，返回/加载时扩展 | ✅ | 两者共享 `_extend_ek2d_static` 与 `frac_to_real_2d` |
| `k_range` 处理逻辑一致 | ✅ | 均调用 `EK2DIO._extend_ek2d_static` |
| `frac_to_real_2d` 转换一致 | ✅ | 均从分数坐标转实空间坐标 |

**✅ 接口一致性**：`EK2DCalculator.calculate()` 内部调用 `EK2DIO.save_ek2d()` 时使用 `mlwf_hamiltonian=self.ham`，与本方法签名一致（曾存在 `hamiltonian=` 关键字不匹配的历史问题，已于 0.3.0 后修复）。直接调用本模块时同样使用 `mlwf_hamiltonian=` 关键字。

---

## 版本信息

- 模块路径：`src/stm_data_processing/io/ek2d_io.py`
- 数据格式：HDF5（`energies`/`k1_grid`/`k2_grid` 主 dataset + 可选 `bvecs`）
- 压缩：gzip 级别 4
- 默认网格：原始 BZ `[-0.5, 0.5)`
- 日志级别：`INFO`（保存/加载进度、形状、能量范围、耗时）
