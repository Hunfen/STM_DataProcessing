# MLWF Green's Function 模块接口文档

## 模块概述

`mlwf_gk.py` 提供基于 Wannier90 MLWF Hamiltonian 的格林函数计算功能，支持 CPU/GPU 双后端，用于计算 G(k, ω) = (ω + iη - H(k))⁻¹。

**后端选择**: 继承自 `mlwf_hamiltonian` 模块，通过 `BACKEND` 常量自动确定，无需额外配置。

**输入格式**: 哈密顿量 `hk` 支持单点 `(nw, nw)` 或批量 `(N, nw, nw)` 形状。

---

## 核心类：`GreenFunction`

### 类定义

```python
class GreenFunction:
    """Green's function calculator for MLWF Hamiltonian."""
```

### 构造函数

```python
def __init__(
    self,
    mlwf_hamiltonian: MLWFHamiltonian,
    eta: float = 0.001,
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mlwf_hamiltonian` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `eta` | `float` | 展宽参数（虚部能量），默认 0.001 |

**内部属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `eta` | `float` | 展宽参数 |
| `num_wann` | `int` | Wannier 函数数量（从 `ham` 继承） |

---

### 实例方法

#### `compute_green(hk, omega)`

计算格林函数 G(k, ω)。

```python
def compute_green(self, hk: Any, omega: float) -> Any
```

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `hk` | `np.ndarray \| cp.ndarray` | `(nw, nw)` 或 `(N, nw, nw)` | 哈密顿量矩阵 |
| `omega` | `float` | - | 能量值 |

**返回**:
- 形状：与 `hk` 相同
- 类型：`BACKEND == "cpu"` → `np.ndarray`，`BACKEND == "gpu"` → `cp.ndarray`

**示例**:
```python
# ✅ 单 k 点哈密顿量
h_k = ham.hk(np.array([[0.0, 0.0, 0.0]]))[0]  # shape: (nw, nw)
gf = green.compute_green(h_k, omega=0.5)  # shape: (nw, nw)

# ✅ 批量 k 点哈密顿量
h_batch = ham.hk(k_points)  # shape: (N, nw, nw)
gf_batch = green.compute_green(h_batch, omega=0.5)  # shape: (N, nw, nw)
```

---

## 后端检测与检查

### 后端继承逻辑

模块导入时从 `mlwf_hamiltonian` 继承后端状态：

```
mlwf_hamiltonian.BACKEND == "gpu" → 使用 CuPy
mlwf_hamiltonian.BACKEND == "cpu" → 使用 NumPy
```

### 公开常量：`BACKEND`

```python
from stm_data_processing.dft.wannier90.mlwf_gk import BACKEND

print(BACKEND)  # 'cpu' 或 'gpu'
```

| 常量 | 类型 | 说明 |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | 当前激活的计算后端（从 `mlwf_hamiltonian` 导入） |

**用途**: 外部代码根据 `BACKEND` 决定数据处理方式（如 `cp.asnumpy()` 转换）。

**⚠️ 注意**: 
- `BACKEND` 在模块导入时确定，运行时不会改变
- GPU 模式下 `compute_green()` 返回 `cp.ndarray`，可能需要 `cp.asnumpy()` 转换
- 无需设置环境变量，后端由 `mlwf_hamiltonian` 自动检测

---

## 内部属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `eta` | `float` | 展宽参数 |
| `num_wann` | `int` | Wannier 函数数量 |

---

## 数学公式

格林函数计算公式：

```
G(k, ω) = (ω + iη - H(k))⁻¹
```

其中：
- `ω` : 能量值（实数）
- `η` : 展宽参数（`self.eta`）
- `H(k)` : k 点哈密顿量

实现采用线性方程组求解：
```
(ω + iη - H(k)) · G(k, ω) = I
```

---

## 使用示例

### 基础用法

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
import numpy as np

# 加载哈密顿量
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# 创建格林函数计算器
green = GreenFunction(ham, eta=0.001)

# 计算 Γ 点格林函数
k_gamma = np.array([[0.0, 0.0, 0.0]])
h_gamma = ham.hk(k_gamma)[0]  # shape: (nw, nw)
g_gamma = green.compute_green(h_gamma, omega=0.5)  # shape: (nw, nw)
```

### 批量 k 点计算

```python
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction, BACKEND
import cupy as cp
import numpy as np

ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")
green = GreenFunction(ham, eta=0.001)

# 批量 k 点
k_points = np.random.rand(100, 3)
h_batch = ham.hk(k_points)  # shape: (100, nw, nw)

# 批量计算格林函数
g_batch = green.compute_green(h_batch, omega=0.5)  # shape: (100, nw, nw)

# GPU 模式下转换回 numpy（如需要）
if BACKEND == "gpu":
    g_batch = cp.asnumpy(g_batch)
```

### 根据 BACKEND 决定计算策略

```python
from stm_data_processing.dft.wannier90.mlwf_gk import BACKEND, GreenFunction

green = GreenFunction(ham, eta=0.001)

if BACKEND == "gpu":
    # GPU 后端：可批量计算更多 k 点
    k_points = np.random.rand(10000, 3)
    h_batch = ham.hk(k_points)
    g_batch = green.compute_green(h_batch, omega=0.5)
    g_batch = cp.asnumpy(g_batch)  # 如需 numpy 则转换
else:
    # CPU 后端：分批计算避免内存压力
    k_points = np.random.rand(1000, 3)
    h_batch = ham.hk(k_points)
    g_batch = green.compute_green(h_batch, omega=0.5)
```

### 能量扫描计算

```python
from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
import numpy as np

green = GreenFunction(ham, eta=0.001)

# 能量范围
omega_values = np.linspace(-1.0, 1.0, 100)

# 单 k 点能量扫描
k_point = np.array([[0.0, 0.0, 0.0]])
h_k = ham.hk(k_point)[0]

g_spectrum = []
for omega in omega_values:
    g_omega = green.compute_green(h_k, omega=omega)
    g_spectrum.append(g_omega)

g_spectrum = np.array(g_spectrum)  # shape: (100, nw, nw)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心计算 |
| `cupy` | 否 | GPU 加速（可选，由 `mlwf_hamiltonian` 检测） |
| `typing` | 是 | 类型注解 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `hk` 形状不匹配、矩阵求逆失败 |
| `RuntimeError` | GPU 后端激活但 CuPy/CUDA 不可用（由 `mlwf_hamiltonian` 捕获） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`hk` 输入形状为 `(nw, nw)` 或 `(N, nw, nw)`**
- [ ] **返回数组形状与输入 `hk` 一致**
- [ ] **通过 `BACKEND` 常量检查当前后端，决定数据处理方式**
- [ ] **GPU 模式下返回类型为 `cp.ndarray`，必要时用 `cp.asnumpy()` 转换**
- [ ] **`eta` 参数为正实数，典型值 0.001~0.01**
- [ ] **`omega` 参数为实数能量值**
- [ ] **`GreenFunction` 必须关联有效的 `MLWFHamiltonian` 实例**
- [ ] **注意：`BACKEND` 在模块导入时确定，运行时不会改变**
- [ ] **批量计算时保持输入输出维度一致**

---

## 与 mlwf_hamiltonian 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| 使用 `BACKEND` 常量判断后端 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| `cp` 根据 `BACKEND` 条件导入 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| 输入输出类型与后端匹配 | ✅ | CPU→`np.ndarray`, GPU→`cp.ndarray` |
| 支持批量 k 点计算 | ✅ | 与 `ham.hk()` 接口对齐 |
| 无额外安全检查 | ✅ | 依赖 `mlwf_hamiltonian` 的后端检测 |

---

## 版本信息

- 模块路径：`src/STM_DataProcessing/src/stm_data_processing/dft/wannier90/mlwf_gk.py`
- 后端检测：继承自 `mlwf_hamiltonian`（导入时自动完成）
- 日志级别：无（静默计算）
- **无环境变量配置，后端自动继承**
- **输入格式：`(nw, nw)` 或 `(N, nw, nw)` 的 numpy/cupy 数组**