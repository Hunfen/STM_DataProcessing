# 晶格加载模块接口文档

## 模块概述

`lattice_loader.py` 提供从倒格矢构造 `LATTICE` 对象的工厂能力。它可以从 Wannier90 `.wout` 文件、OpenMX `.out` 文件，或直接传入的 numpy 数组读取/接收倒格矢，并据此初始化一个 `LATTICE` 实例。

- **模块路径**：`src/stm_data_processing/io/lattice_loader.py`
- **职责**：解析倒格矢（`bvecs`），返回由实空间矢量（`avecs`）与倒格矢（`bvecs`）共同初始化好的 `LATTICE` 对象。
- **与计算模块的对应关系**：`Wannier90HRLoader._load_bvecs()` 内部调用 `LatticeLoader.create_lattice()` 提取倒格矢；下游计算模块通过 `hamiltonian.bvecs` 使用倒格矢完成分数坐标 → 实空间坐标转换。

**构造约定**：`LATTICE` 对象由解析得到的倒格矢 `bvecs` 反推实空间矢量 `avecs`（满足 `a_i · b_j = 2π δ_ij`）。

---

## 核心类：`LatticeLoader`

### 类定义

```python
class LatticeLoader:
    """Factory class for creating LATTICE instances from reciprocal lattice vectors."""
```

### 静态方法

#### `create_lattice(filename, bvecs_array)`

由倒格矢创建 `LATTICE` 实例。

```python
@staticmethod
def create_lattice(
    filename: str | None = None,
    bvecs_array: np.ndarray | None = None,
) -> LATTICE
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `filename` | `str \| None` | Wannier90 `.wout` 或 OpenMX `.out` 文件路径；提供后自动读取倒格矢 |
| `bvecs_array` | `np.ndarray \| None` | 直接提供倒格矢，形状 `(2, 2)`（2D，第三行/列补零）或 `(3, 3)`（3D），单位 1/Å |

**优先级**：`bvecs_array` 优先于 `filename`；若两者皆未提供，抛出 `ValueError`。

**返回**：`LATTICE` —— 完全初始化的晶格对象（含 `avecs`、`bvecs`）。

---

## 私有方法（不建议直接调用）

| 方法 | 签名 | 说明 |
|------|------|------|
| `_set_bvecs_from_array` | `(bvecs_array) -> np.ndarray` | 将 `(2,2)` 或 `(3,3)` 数组标准化为 `(3,3)`；`(2,2)` 时第三行/列补零 |
| `_load_reciprocal_vectors` | `(filename) -> np.ndarray \| None` | 按扩展名解析 `.wout` / `.out`，返回 `(3,3)` 倒格矢或 `None` |
| `_parse_wannier_vector_line` | `(line) -> np.ndarray` | 解析 Wannier90 矢量行，如 `b_1  1.474634  0.851380  0.000000` |
| `_parse_openmx_vector_line` | `(line) -> np.ndarray` | 解析 OpenMX 矢量行，如 `#  Reciprocal vector b1 (1/Ang): 2.55414 1.47463 0.00000` |

### 输入文件解析要点

- **`.wout`（Wannier90）**：定位含 `Reciprocal-Space Vectors` 或 `Reciprocal Vectors` 的行，随后读取 `b_1`、`b_2`、`b_3` 三行。
- **`.out`（OpenMX）**：定位 `Reciprocal vector b1/b2/b3` 行，三行齐备即停止扫描。
- 若三者缺一（如 2D 体系仅两个非零矢量但文件未给出 `b_3`），`_load_reciprocal_vectors` 返回 `None`，进而 `create_lattice` 抛出 `ValueError`。

---

## 返回的 `LATTICE` 对象

`create_lattice` 返回 `stm_data_processing.utils.lattice.LATTICE` 实例。与加载相关的关键属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `bvecs` | `np.ndarray` | 倒格矢 `(3, 3)`，按行存储：`bvecs[0]=b1`、`bvecs[1]=b2`、`bvecs[2]=b3` |
| `avecs` | `np.ndarray` | 实空间格矢 `(3, 3)`，由 `bvecs` 反推得到，按行存储 |
| `b1`/`b2`/`b3` | `np.ndarray` | 单个倒格矢（`bvecs` 各行的便捷属性） |
| `a1`/`a2`/`a3` | `np.ndarray` | 单个实空间格矢（`avecs` 各行的便捷属性） |
| `volume` | `float` | 元胞体积（由 `avecs` 计算） |
| `reciprocal_volume` | `float` | 倒空间元胞体积（由 `bvecs` 计算） |

> 注意：`LATTICE.__init__` 会校验 `bvecs` 为 `(3,3)` 且非奇异（行列式绝对值 ≥ `1e-12`），否则抛出 `ValueError`。因此 `(2,2)` 补零得到的奇异 `(3,3)` 矩阵在 3D `LATTICE` 构造阶段可能被拒绝。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.io.lattice_loader import LatticeLoader

# 方式 1：从 .wout 文件读取
lattice = LatticeLoader.create_lattice(filename="./wannier/silicon.wout")
print(lattice.bvecs)  # (3, 3) 倒格矢

# 方式 2：从 OpenMX .out 文件读取
lattice = LatticeLoader.create_lattice(filename="./openmx/system.out")

# 方式 3：直接传数组（(2,2) 自动补零为 (3,3)）
b2d = np.array([[1.0, 0.0], [0.0, 1.0]])
lattice = LatticeLoader.create_lattice(bvecs_array=b2d)

# 方式 4：直接传 (3,3) 数组
b3d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
lattice = LatticeLoader.create_lattice(bvecs_array=b3d)

# 提取倒格矢供下游使用
bvecs = lattice.bvecs
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组构造、`vstack`、形状校验 |
| `pathlib.Path` | 是 | 扩展名判断与文件打开 |
| `LATTICE`（本包 utils） | 是 | 返回对象类型 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `filename` 与 `bvecs_array` 均未提供；解析/设置倒格矢失败（返回 `None`）；`bvecs_array` 形状不是 `(2,2)` 或 `(3,3)` |
| `TypeError` | `bvecs_array` 不是 numpy 数组 |
| `FileNotFoundError` | 指定的文件不存在 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`filename` 与 `bvecs_array` 至少提供一个，二者同时提供时以 `bvecs_array` 为准**
- [ ] **`bvecs_array` 形状须为 `(2, 2)` 或 `(3, 3)`，单位 1/Å**
- [ ] **返回对象为 `LATTICE`，倒格矢通过 `lattice.bvecs`（`(3,3)` 按行存储）访问**
- [ ] **`(2,2)` 输入会补零为 `(3,3)`，可能导致 `LATTICE` 构造时的奇异矩阵校验失败**
- [ ] **`.wout`/`.out` 若无法解析出完整的 `b_1/b_2/b_3`，会返回 `None` 并抛出 `ValueError`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/io/lattice_loader.py`
- 输入格式：Wannier90 `.wout`、OpenMX `.out`、numpy 数组
- 输出类型：`LATTICE`（`stm_data_processing.utils.lattice`）
- 日志级别：无（静默，异常直接抛出）
