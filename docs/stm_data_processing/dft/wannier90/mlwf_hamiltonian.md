# MLWF Hamiltonian 模块接口文档

## 模块概述

`mlwf_hamiltonian.py` 提供 Wannier90 HR Hamiltonian 计算功能，支持 CPU/GPU 双后端，用于从 Wannier90 文件计算任意 k 点的哈密顿量 H(k)。

**后端选择**: 模块导入时自动检测 CuPy 可用性，无需配置环境变量。

**输入格式**: k 点坐标统一使用 `(N, 3)` 形状的 `np.ndarray`。

---

## 核心类：`MLWFHamiltonian`

### 类定义

```python
class MLWFHamiltonian:
    """Wannier90 HR Hamiltonian handler."""
```

### 构造函数

```python
def __init__(
    self,
    num_wann: int,
    r_list: np.ndarray,
    h_list_flat: np.ndarray,
    ndegen: np.ndarray,
    bvecs: np.ndarray | None = None,
)
```

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `num_wann` | `int` | - | Wannier 函数数量 |
| `r_list` | `np.ndarray` | `(nrpts, 3)` | 晶格矢量索引，dtype=int32 |
| `h_list_flat` | `np.ndarray` | `(nrpts, num_wann*num_wann)` | 展平的哈密顿矩阵，dtype=complex128 |
| `ndegen` | `np.ndarray` | `(nrpts,)` | 简并度因子，dtype=float64 |
| `bvecs` | `np.ndarray \| None` | `(3, 3)` | 倒格矢（可选） |

---

### 类方法

#### `from_seedname(folder, seedname)`

从 Wannier90 文件加载并构造实例。

```python
@classmethod
def from_seedname(cls, folder: str, seedname: str) -> "MLWFHamiltonian"
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder` | `str` | 包含 HR 文件的目录路径 |
| `seedname` | `str` | Wannier90 文件基础名（无扩展名） |

**返回**: `MLWFHamiltonian` 实例

**示例**:
```python
ham = MLWFHamiltonian.from_seedname("/path/to/wannier", "material")
```

---

#### `from_arrays(...)`

从原始数组构造实例。

```python
@classmethod
def from_arrays(
    cls,
    num_wann: int,
    r_list: np.ndarray,
    h_list_flat: np.ndarray,
    ndegen: np.ndarray,
    bvecs: np.ndarray | None = None,
) -> "MLWFHamiltonian"
```

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `num_wann` | `int` | - | Wannier 函数数量 |
| `r_list` | `np.ndarray` | `(nrpts, 3)` | 晶格矢量索引 |
| `h_list_flat` | `np.ndarray` | `(nrpts, num_wann*num_wann)` | 展平的哈密顿矩阵 |
| `ndegen` | `np.ndarray` | `(nrpts,)` | 简并度因子 |
| `bvecs` | `np.ndarray \| None` | `(3, 3)` | 倒格矢（可选） |

**返回**: `MLWFHamiltonian` 实例

---

### 实例方法

#### `hk(k_frac)`

计算 k 点的哈密顿量 H(k)。

```python
def hk(self, k_frac: np.ndarray) -> np.ndarray | cp.ndarray
```

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `k_frac` | `np.ndarray` | `(N, 3)` | 分数坐标 k 点，**必须为 2D 数组** |

**返回**:
- 形状：`(N, num_wann, num_wann)`
- 类型：`BACKEND == "cpu"` → `np.ndarray`，`BACKEND == "gpu"` → `cp.ndarray`

**⚠️ 重要**: 
- 不支持 tuple 或 `(3,)` 形状输入
- 单 k 点请使用 `(1, 3)` 形状
- 返回值不自动压缩维度

**示例**:
```python
# ✅ 单 k 点 (1, 3)
k_point = np.array([[0.0, 0.0, 0.0]])
h_k = ham.hk(k_point)  # shape: (1, nw, nw)

# ✅ 批量 k 点 (N, 3)
k_points = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
h_batch = ham.hk(k_points)  # shape: (3, nw, nw)

# ❌ 不支持
ham.hk((0, 0, 0))              # tuple 不支持
ham.hk(np.array([0, 0, 0]))    # (3,) 不支持
```

---

## 后端检测与检查

### 自动检测逻辑

模块导入时自动检测，**无需设置环境变量**：

```
CuPy 可导入 且 CUDA 设备可用 → BACKEND = "gpu"
否则 → BACKEND = "cpu"
```

### 公开常量：`BACKEND`

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND

print(BACKEND)  # 'cpu' 或 'gpu'
```

| 常量 | 类型 | 说明 |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | 当前激活的计算后端（导入时确定） |

**用途**: 外部代码根据 `BACKEND` 决定计算策略或数据处理方式。

**⚠️ 注意**: 
- `BACKEND` 在模块导入时确定，运行时不会改变
- 不要尝试通过环境变量修改后端
- GPU 模式下 `hk()` 返回 `cp.ndarray`，可能需要 `cp.asnumpy()` 转换

---

## 属性

### 公开只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `num_wann` | `int` | Wannier 函数数 |
| `r_list` | `np.ndarray` | 晶格矢量列表 `(nrpts, 3)` |
| `h_list_flat` | `np.ndarray` | 展平哈密顿量 `(nrpts, nw*nw)` |
| `ndegen` | `np.ndarray` | 简并度 `(nrpts,)` |
| `bvecs` | `np.ndarray \| None` | 倒格矢 `(3, 3)` |
| `folder` | `str \| None` | 仅 `from_seedname` 创建时存在 |
| `seedname` | `str \| None` | 仅 `from_seedname` 创建时存在 |

### 内部属性（GPU 缓存）

| 属性 | 类型 | 说明 |
|------|------|------|
| `_r_list_gpu` | `cp.ndarray \| None` | GPU 上的 r_list 缓存 |
| `_ndegen_gpu` | `cp.ndarray \| None` | GPU 上的 ndegen 缓存 |
| `_h_flat_gpu` | `cp.ndarray \| None` | GPU 上的 h_list_flat 缓存 |

---

## 内部方法（不建议直接调用）

| 方法 | 说明 |
|------|------|
| `_hk_cpu(k_frac)` | CPU 后端计算，输入 `(N, 3)`，返回 `(N, nw, nw)` |
| `_hk_gpu(k_frac)` | GPU 后端计算，输入 `(N, 3)`，返回 `(N, nw, nw)` |
| `_ensure_gpu_cache()` | 确保 GPU 缓存已填充，首次 GPU 计算时自动调用 |
| `_validate_data(...)` | 静态方法，验证输入数据形状和类型 |
| `_as_k_array(k_frac)` | 静态方法，标准化 k 点输入为 `(N, 3)` 数组（内部辅助） |

---

## 数学公式

H(k) 计算公式：

```
H(k) = Σ_R exp(2π i R·k) / ndegen(R) * H(R)
```

实现采用扁平化收缩：
```
weights(k,R) @ H_flat(R, nw*nw) → H_flat(k, nw*nw) → reshape(nw, nw)
```

其中 `weights(k,R) = exp(2π i k·R) / ndegen(R)`，形状 `(N, nrpts)`。

---

## 使用示例

### 基础用法

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
import numpy as np

# 从文件加载
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# 计算 Γ 点 (注意使用 (1, 3) 形状)
k_gamma = np.array([[0.0, 0.0, 0.0]])
h_gamma = ham.hk(k_gamma)  # shape: (1, nw, nw)
h_gamma_0 = h_gamma[0]     # 取第一个 k 点的结果 (nw, nw)

# 计算 k 点路径
k_path = np.array([
    [0.0, 0.0, 0.0],   # Γ
    [0.5, 0.0, 0.0],   # X
    [0.5, 0.5, 0.0],   # M
    [0.0, 0.0, 0.0],   # Γ
])
h_path = ham.hk(k_path)  # shape: (4, nw, nw)
```

### 根据 BACKEND 决定计算策略

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND, MLWFHamiltonian
import cupy as cp
import numpy as np

ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

if BACKEND == "gpu":
    # GPU 后端：可批量计算更多 k 点以充分利用并行
    k_points = np.random.rand(10000, 3)
    h_results = ham.hk(k_points)  # 返回 cp.ndarray, shape: (10000, nw, nw)
    h_results = cp.asnumpy(h_results)  # 如需 numpy 则转换
else:
    # CPU 后端：分批计算避免内存压力
    k_points = np.random.rand(1000, 3)
    h_results = ham.hk(k_points)  # 返回 np.ndarray, shape: (1000, nw, nw)
```

### 直接构造（已有数据）

```python
ham = MLWFHamiltonian(
    num_wann=10,
    r_list=r_list,          # (nrpts, 3)
    h_list_flat=h_flat,     # (nrpts, 100)
    ndegen=ndegen,          # (nrpts,)
    bvecs=bvecs,            # (3, 3) 可选
)
```

### 检查后端状态

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND

print(f"Using backend: {BACKEND}")

# 根据后端调整后续处理
if BACKEND == "gpu":
    # GPU 路径：数据可能需要保持在 GPU 上
    pass
else:
    # CPU 路径：直接使用 numpy
    pass
```

### 辅助函数：构造 k 点数组

```python
def make_k_points(*k_list: tuple[float, float, float]) -> np.ndarray:
    """Convert k-point tuples to (N, 3) array."""
    return np.array(k_list, dtype=np.float64)

# 使用
k_points = make_k_points(
    (0, 0, 0),
    (0.5, 0, 0),
    (0, 0.5, 0),
)
h_results = ham.hk(k_points)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心计算 |
| `cupy` | 否 | GPU 加速（可选，有则自动启用） |
| `logging` | 是 | 日志记录 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `k_frac` 形状不是 `(N, 3)`、数据形状不匹配、num_wann 无效 |
| `RuntimeError` | GPU 后端激活但 CuPy/CUDA 不可用 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`k_frac` 输入形状必须为 `(N, 3)` 的 2D numpy 数组**
- [ ] **单 k 点使用 `(1, 3)` 形状，而非 `(3,)`**
- [ ] 返回数组形状为 `(N, num_wann, num_wann)`
- [ ] **通过 `BACKEND` 常量检查当前后端，决定计算策略**
- [ ] **GPU 模式下返回类型为 `cp.ndarray`，必要时用 `cp.asnumpy()` 转换**
- [ ] 批量计算时保持输入输出维度一致
- [ ] 使用 `from_seedname` 时确保文件路径正确
- [ ] 直接构造时验证所有数组形状匹配
- [ ] **注意：`BACKEND` 在模块导入时确定，运行时不会改变**

---

## 版本信息

- 模块路径：`src/STM_DataProcessing/src/stm_data_processing/dft/wannier90/mlwf_hamiltonian.py`
- 后端检测：导入时自动完成（CuPy 可用性检测）
- 日志级别：`INFO` 用于计算通知
- **无环境变量配置，后端自动选择**
- **输入格式：统一为 `(N, 3)` 的 numpy 数组**