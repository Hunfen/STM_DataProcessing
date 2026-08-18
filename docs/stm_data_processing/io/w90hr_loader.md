# Wannier90 HR 数据加载模块接口文档

## 模块概述

`w90hr_loader.py` 提供 Wannier90 `*_hr.dat` / `*_hr.h5` 实空间哈密顿量数据的读取功能，并负责从 `.wout` / `.out` 文件中提取倒格矢。它是整个 MLWF 计算链的数据入口层，供 `MLWFHamiltonian` 等计算模块消费。

- **模块路径**：`src/stm_data_processing/io/w90hr_loader.py`
- **职责**：解析 Wannier90 HR 文件，返回统一结构的字典（包含 `num_wann`、`r_list`、`h_list`、`ndegen`、`h_list_flat`、`bvecs`）。
- **与计算模块的对应关系**：`MLWFHamiltonian.from_seedname(folder, seedname)` 内部调用 `Wannier90HRLoader.load(folder, seedname)` 获取数据，再经 `from_arrays()` 构造哈密顿量实例。

**文件格式优先级**：若 `{seedname}_hr.h5` 存在，则优先加载 HDF5 格式；否则回退到 `{seedname}_hr.dat` 文本格式。

---

## 核心类：`Wannier90HRLoader`

### 类定义

```python
class Wannier90HRLoader:
    """Loader for Wannier90 HR Hamiltonian data files."""
```

### 公开静态方法

#### `load(folder, seedname)`

加载 Wannier90 HR 哈密顿量数据（`.h5` 优先，`.dat` 回退）。

```python
@staticmethod
def load(folder: str | Path, seedname: str) -> dict[str, Any]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder` | `str \| Path` | 存放 HR 文件的目录 |
| `seedname` | `str` | Wannier90 文件基名（不含扩展名） |

**返回**：`dict[str, Any]`，键如下：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `num_wann` | `int` | - | Wannier 函数数量 |
| `r_list` | `np.ndarray` | `(nrpts, 3)` | 实空间格矢索引（`.dat` 路径为 `int32`） |
| `h_list` | `np.ndarray` | `(nrpts, num_wann, num_wann)` | 每个 R 点的哈密顿量矩阵（`.dat` 路径为 `complex128`） |
| `ndegen` | `np.ndarray` | `(nrpts,)` | 每个 R 点的简并因子（`.dat` 路径为 `float64`） |
| `h_list_flat` | `np.ndarray` | `(nrpts, num_wann*num_wann)` | 展平后的哈密顿量矩阵 |
| `bvecs` | `np.ndarray \| None` | `(3, 3)` 或 `None` | 倒格矢（若能从 `.wout`/`.out` 提取） |

> `h_list_flat` 由 `h_list.reshape(len(r_list), num_wann * num_wann)` 得到，直接匹配 `MLWFHamiltonian.__init__` 的 `h_list_flat` 入参。

---

## 私有方法（不建议直接调用）

| 方法 | 签名 | 说明 |
|------|------|------|
| `_load_h5` | `(folder, seedname) -> dict` | 从 `{seedname}_hr.h5` 读取 `num_wann`、`r_list`、`h_list`、`ndegen`，并附加 `bvecs` 与 `h_list_flat` |
| `_load_hr` | `(folder, seedname) -> dict` | 从 `{seedname}_hr.dat` 解析数据，附加 `bvecs` 与 `h_list_flat` |
| `_load_bvecs` | `(folder, seedname) -> np.ndarray \| None` | 依次尝试 `{seedname}.wout`、`{seedname}.out`，借助 `LatticeLoader.create_lattice` 提取倒格矢 |
| `_parse_hr_file` | `(filename) -> (num_wann, r_list, h_list, ndegen)` | 解析 `_hr.dat` 文本，返回 `(int, (nrpts,3) int32, (nrpts,nw,nw) complex128, (nrpts,) float64)` |

### `.dat` 文件格式要点

`_parse_hr_file` 兼容两种文件头：

- **带注释头**（首行以 `written on` 开头）：第 2 行为 `num_wann`，第 3 行为 `nrpts`。
- **无注释头**：首行即为 `num_wann`，次行为 `nrpts`。

随后是 `nrpts` 个简并因子（`ndegen`，按 15 个/行排版），再跟 `nrpts * num_wann * num_wann` 行矩阵元，每行 7 列：

```
r1 r2 r3 m n Re(H) Im(H)
```

其中 `(r1, r2, r3)` 为格矢索引，`(m, n)` 为 1-based Wannier 轨道下标，最终写入 `h_list[i, m-1, n-1] = Re + 1j*Im`。

---

## 使用示例

```python
from stm_data_processing.io.w90hr_loader import Wannier90HRLoader

# 直接加载（.h5 优先，.dat 回退）
data = Wannier90HRLoader.load(folder="./wannier", seedname="silicon")

num_wann = data["num_wann"]
r_list   = data["r_list"]      # (nrpts, 3)
h_flat   = data["h_list_flat"] # (nrpts, num_wann*num_wann)
bvecs    = data["bvecs"]       # (3, 3) 或 None

# 更常见用法：经 MLWFHamiltonian 间接加载
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `h5py` | 是 | 读取 `*_hr.h5` 文件 |
| `numpy` | 是 | 数组构造与 reshape/stack |
| `pathlib.Path` | 是 | 路径拼接与存在性判断 |
| `LatticeLoader`（本包） | 是 | 从 `.wout`/`.out` 提取倒格矢 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `FileNotFoundError` | 目标 `*_hr.h5` 或 `*_hr.dat` 不存在（`_load_h5`/`_load_hr` 内） |
| `RuntimeError` | `.dat` 解析时提前 EOF、`ndegen` 数量不足、矩阵元行缺列，或读到的 R 点数与 `nrpts` 不一致 |

> 注意：若 `*_hr.h5` 与 `*_hr.dat` 均不存在，`load()` 会先判定 `.h5` 不存在后进入 `_load_hr`，由后者抛出 `FileNotFoundError`。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`folder` 指向同时包含 `{seedname}_hr.dat`（或 `_hr.h5`）的目录**
- [ ] **`seedname` 不含扩展名**
- [ ] **返回字典的 `h_list_flat` 形状为 `(nrpts, num_wann*num_wann)`，可直接传给 `MLWFHamiltonian.from_arrays()`**
- [ ] **`bvecs` 可能为 `None`（当目录下无 `.wout`/`.out` 时），下游需判空**
- [ ] **`.dat` 路径下 `r_list` 为 `int32`、`ndegen` 为 `float64`、`h_list` 为 `complex128`**
- [ ] **`ndegen` 用于 H(k) 计算时的权重 `1/ndegen(R)`（见 `MLWFHamiltonian.hk()`）**

---

## 与 MLWFHamiltonian 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| `load()` 返回的键被 `from_seedname()` 完整消费 | ✅ | `num_wann`/`r_list`/`h_list_flat`/`ndegen`/`bvecs` 一一对应 |
| `h_list_flat` 形状匹配 `_validate_data` 期望 | ✅ | `(nrpts, num_wann*num_wann)` |
| `bvecs` 可缺省 | ✅ | `from_arrays()` 的 `bvecs` 参数默认 `None` |

---

## 版本信息

- 模块路径：`src/stm_data_processing/io/w90hr_loader.py`
- 数据来源格式：Wannier90 `*_hr.dat` 文本 / `*_hr.h5` HDF5
- 倒格矢来源：`*.wout`（Wannier90）或 `*.out`（OpenMX），经 `LatticeLoader` 解析
- 日志级别：`INFO`（记录加载格式、`num_wann`、`nrpts`）
