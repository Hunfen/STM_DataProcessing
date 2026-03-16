# JDOS QPI 模块接口文档

## 模块概述

`qpi_jdos.py` 提供基于 Wannier90 MLWF Hamiltonian 的 JDOS (Joint Density of States) QPI (Quasiparticle Interference) 计算功能，支持 CPU/GPU 双后端。

**后端选择**: 继承自 `mlwf_hamiltonian` 模块，通过 `BACKEND` 常量自动确定，无需额外配置。

**k 网格**: 固定为 `[-0.5, 0.5)` 范围，`nkx = nky = nk`。

**数据保存策略**: 
- HDF5 保存时使用原始网格 `[-0.5, 0.5)`（节省存储空间）
- `calculate()` 返回时根据 `q_range` 参数扩展/裁剪（方便后续使用）
- 返回结构与 `load_qpi_from_h5()` 完全一致

---

## 核心类：`JDOSQPI`

### 类定义

```python
class JDOSQPI:
    """Class for calculating JDOS QPI (Quasiparticle Interference)."""
```

### 构造函数

```python
def __init__(
    self,
    hamiltonian: MLWFHamiltonian,
    nk: int = 256,
    eta: float = 0.001,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `hamiltonian` | `MLWFHamiltonian` | 已初始化的哈密顿量实例 |
| `nk` | `int` | k 点网格数量 (nkx = nky = nk)，默认 256 |
| `eta` | `float` | 谱展宽参数，默认 0.001 |

**内部属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `num_wann` | `int` | Wannier 函数数量 |
| `nk` | `int` | k 网格大小 |
| `eta` | `float` | 展宽参数 |
| `eigenvalues` | `np.ndarray` | 本征值 `(nk, nk, num_wann)` |
| `k1_grid`, `k2_grid` | `np.ndarray` | k 点网格 `(nk, nk)` |
| `q1_grid`, `q2_grid` | `np.ndarray` | q 点网格 `(nk, nk)` |

**⚠️ 初始化检查**:
- `hamiltonian.num_wann` 必须已初始化且为正数
- 否则抛出 `ValueError`

---

### 实例方法

#### `calculate(energy_range, q_range, normalize, output_path)`

统一 QPI 计算器，自动选择 CPU/GPU 后端。

```python
def calculate(
    self,
    energy_range: float | np.ndarray | list[float],
    q_range: tuple[float, float] | None = (-0.5, 0.5),
    normalize: bool = True,
    output_path: str | None = None,
) -> dict[str, Any]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `energy_range` | `float`, `list[float]`, 或 `np.ndarray` | 目标能量值 (eV) |
| `q_range` | `tuple[float, float]` 或 `None` | q 范围 `[q_min, q_max]`，默认 `(-0.5, 0.5)` |
| `normalize` | `bool` | 是否归一化 QPI 强度，默认 `True` |
| `output_path` | `str` 或 `None` | HDF5 输出路径 (可选) |

**返回**: `dict[str, Any]` 包含以下键：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `qpi_layers` | `np.ndarray` | `(n_energies, nq, nq)` | QPI 强度图（可能已扩展） |
| `q1_grid`, `q2_grid` | `np.ndarray` | `(nq, nq)` | 分数坐标 q 网格（可能已扩展） |
| `qx_grid`, `qy_grid` | `np.ndarray` | `(nq, nq)` | 实空间 q 坐标（如果 bvecs 可用） |
| `metadata` | `dict` | - | 计算参数和元数据 |

**`metadata` 字典内容**:

| 键 | 类型 | 说明 |
|----|------|------|
| `module_type` | `str` | 模块类型标识，固定为 `"jdos"` |
| `eta` | `float` | 谱展宽参数 |
| `normalize` | `bool` | 是否进行了归一化 |
| `nq` | `int` | 原始 q 网格大小 |
| `energy_range` | `np.ndarray` | 计算使用的能量数组 |
| `bands` | `None` 或 `np.ndarray` | 能带索引（JDOS 模式下为 `None`） |
| `bvecs` | `np.ndarray` 或 `None` | 倒格矢 `(3, 3)` |
| `V` | `np.ndarray` 或 `None` | 散射势矩阵（JDOS 模式下为 `None`） |
| `mask` | `np.ndarray` 或 `None` | 实空间掩码（JDOS 模式下为 `None`） |

**⚠️ 重要说明**:
- 返回结构与 `qpi_io.load_qpi_from_h5()` 完全一致
- HDF5 文件保存的是**原始网格** `[-0.5, 0.5)` 数据
- `calculate()` 返回的是**扩展后**的网格（根据 `q_range` 参数）
- 这种设计既节省存储空间，又方便用户使用

**示例**:
```python
# 单能量
result = qpi.calculate(energy_range=0.5)
qpi_data = result["qpi_layers"]
bvecs = result["metadata"]["bvecs"]

# 多能量
result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 50))
qpi_data = result["qpi_layers"]  # shape: (50, nq, nq)

# 自定义 q 范围
result = qpi.calculate(energy_range=0.5, q_range=(-0.3, 0.3))

# 保存结果（保存原始网格，返回扩展网格）
result = qpi.calculate(energy_range=0.5, output_path="./qpi.h5")

# 从文件加载（结构相同）
from stm_data_processing.io.qpi_io import load_qpi_from_h5
loaded = load_qpi_from_h5("./qpi.h5", q_range=(-0.3, 0.3))
# loaded 与 result 结构完全一致
```

---

## 后端检测与检查

### 后端继承逻辑

模块导入时从 `mlwf_hamiltonian` 继承后端状态：

```python
from ..dft.wannier90.mlwf_hamiltonian import BACKEND

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None
```

### 公开常量：`BACKEND`

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND

print(BACKEND)  # 'cpu' 或 'gpu'
```

| 常量 | 类型 | 说明 |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | 当前激活的计算后端（从 `mlwf_hamiltonian` 导入） |

**⚠️ 注意**: 
- `BACKEND` 在模块导入时确定，运行时不会改变
- GPU 模式下内部使用 `cp.ndarray`，但最终返回 `np.ndarray`
- 无需设置环境变量，后端由 `mlwf_hamiltonian` 自动检测

---

## 内部方法（不建议直接调用）

| 方法 | 说明 |
|------|------|
| `_initialize_eigenvalues()` | 对角化哈密顿量，返回 `(nk, nk, num_wann)` 本征值 |
| `_compute_spectral_function(energy, use_gpu)` | 计算谱函数 A(k, E) |
| `_compute_jdos_cpu(energy_array, normalize)` | CPU 后端 QPI 计算 |
| `_compute_jdos_cuda(energy_array, normalize)` | GPU 后端 QPI 计算（批量处理） |

---

## 数学公式

### 谱函数

```
A(k, E) = Σ_n (1/π) * η / [(E - ε_n(k))² + η²]
```

### JDOS QPI

```
QPI(q, E) = FFT⁻¹[ |FFT[A(k, E)]|² ]
```

实现步骤：
1. 计算谱函数 `A(k, E)`
2. 傅里叶变换到实空间：`A(r, E) = FFT[A(k, E)]`
3. 计算自相关：`|A(r, E)|²`
4. 逆傅里叶变换：`QPI(q, E) = FFT⁻¹[|A(r, E)|²]`
5. `fftshift` 将零频移到中心

---

## 使用示例

### 基础用法

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.stm.qpi_jdos import JDOSQPI
import numpy as np

# 加载哈密顿量
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# 创建 QPI 计算器
qpi = JDOSQPI(ham, nk=256, eta=0.001)

# 计算单能量 QPI
result = qpi.calculate(energy_range=0.5)
qpi_data = result["qpi_layers"]  # shape: (1, 256, 256)
bvecs = result["metadata"]["bvecs"]  # 从 metadata 访问

# 计算多能量 QPI
energies = np.linspace(-1.0, 1.0, 50)
result = qpi.calculate(energy_range=energies)
qpi_data = result["qpi_layers"]  # shape: (50, 256, 256)
```

### 根据 BACKEND 决定计算策略

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND
from stm_data_processing.stm.qpi_jdos import JDOSQPI

qpi = JDOSQPI(ham, nk=256)

if BACKEND == "gpu":
    # GPU 后端：可处理更大网格和更多能量点
    qpi = JDOSQPI(ham, nk=512, eta=0.001)
    result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 100))
else:
    # CPU 后端：使用较小网格避免内存压力
    qpi = JDOSQPI(ham, nk=256, eta=0.001)
    result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 50))
```

### 保存结果到 HDF5

```python
from stm_data_processing.stm.qpi_jdos import JDOSQPI

qpi = JDOSQPI(ham, nk=256)

result = qpi.calculate(
    energy_range=np.linspace(-1.0, 1.0, 50),
    output_path="./output/qpi_data.h5",
)

# 保存的 HDF5 包含原始网格 [-0.5, 0.5)
# result 返回的是根据 q_range 扩展后的数据
```

### 自定义 q 范围裁剪

```python
from stm_data_processing.stm.qpi_jdos import JDOSQPI

qpi = JDOSQPI(ham, nk=256)

# 裁剪到 [-0.3, 0.3] 范围
result = qpi.calculate(
    energy_range=0.5,
    q_range=(-0.3, 0.3),
)

# 不裁剪（使用完整 [-0.5, 0.5) 范围）
result = qpi.calculate(
    energy_range=0.5,
    q_range=None,
)

# 访问元数据
print(f"Module type: {result['metadata']['module_type']}")
print(f"Eta: {result['metadata']['eta']}")
print(f"Energy range: {result['metadata']['energy_range']}")
```

### 与 load_qpi_from_h5 接口对齐

```python
from stm_data_processing.stm.qpi_jdos import JDOSQPI
from stm_data_processing.io.qpi_io import load_qpi_from_h5

# 计算并保存
qpi = JDOSQPI(ham, nk=256)
result_calc = qpi.calculate(
    energy_range=0.5,
    q_range=(-0.3, 0.3),
    output_path="./qpi.h5",
)

# 从文件加载（使用相同的 q_range）
result_load = load_qpi_from_h5("./qpi.h5", q_range=(-0.3, 0.3))

# 两者结构完全一致
assert result_calc.keys() == result_load.keys()
assert result_calc["qpi_layers"].shape == result_load["qpi_layers"].shape
assert "metadata" in result_calc and "metadata" in result_load
```

---

## GPU 批量优化

GPU 模式下自动进行内存管理和批量处理：

| 参数 | 说明 |
|------|------|
| `_CUDA_SAFETY_FRACTION` | GPU 内存安全使用比例 (0.75) |
| `_CUDA_HARD_MAX_BATCH` | 最大批量大小 (32) |

**自动调整逻辑**:
- 根据可用 GPU 内存计算最优批量大小
- 大网格 (≥1024) 限制批量 ≤ 8
- 中等网格 (≥512) 限制批量 ≤ 16
- 每批次后释放 GPU 内存

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心计算 |
| `cupy` | 否 | GPU 加速（可选，由 `mlwf_hamiltonian` 检测） |
| `scipy.fft` | 是 | FFT 计算 |
| `MLWFHamiltonian` | 是 | 哈密顿量计算 |
| `h5py` | 是 | HDF5 文件保存（通过 `qpi_io`） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `hamiltonian.num_wann` 未初始化或非正数 |
| `RuntimeError` | GPU 后端激活但 CuPy 不可用 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`hamiltonian` 必须是已初始化的 `MLWFHamiltonian` 实例**
- [ ] **`hamiltonian.num_wann` 必须为正数**
- [ ] **通过 `BACKEND` 常量检查当前后端，决定计算策略**
- [ ] **`nk` 参数为正整数，典型值 256~512**
- [ ] **`eta` 参数为正实数，典型值 0.001~0.01**
- [ ] **`energy_range` 会被转换为 1D `np.float64` 数组**
- [ ] **返回结果始终为 `np.ndarray`（GPU 内部计算后转换）**
- [ ] **`q_range=None` 时不进行裁剪**
- [ ] **注意：`BACKEND` 在模块导入时确定，运行时不会改变**
- [ ] **HDF5 保存需要 `output_path` 参数**
- [ ] **访问 `bvecs` 等元数据需通过 `result["metadata"]["bvecs"]`**
- [ ] **返回结构与 `load_qpi_from_h5()` 完全一致**

---

## 与 mlwf_hamiltonian 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| 使用 `BACKEND` 常量判断后端 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| `cp` 根据 `BACKEND` 条件导入 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| 无 `ham._use_cuda()` 调用 | ✅ | 使用 `BACKEND == "gpu"` 替代 |
| 无实例后端状态缓存 | ✅ | 直接使用模块级 `BACKEND` |
| 返回类型统一为 `np.ndarray` | ✅ | GPU 内部计算后转换 |

---

## 与 qpi_io 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| `calculate()` 返回结构与 `load_qpi_from_h5()` 一致 | ✅ | 均包含 `metadata` 字典 |
| `metadata` 字段名称一致 | ✅ | `module_type`, `eta`, `normalize`, `nq`, `energy_range`, `bands`, `bvecs`, `V`, `mask` |
| HDF5 保存原始网格，返回扩展网格 | ✅ | 节省存储空间，方便使用 |
| `module_type` 标识 | ✅ | JDOS 模块固定为 `"jdos"` |
| `q_range` 处理逻辑一致 | ✅ | 均使用 `extend_qpi` 进行扩展/裁剪 |
| `frac_to_real_2d` 转换一致 | ✅ | 均从分数坐标转换到实空间坐标 |

---

## 版本信息

- 模块路径：`src/STM_DataProcessing/src/stm_data_processing/stm/qpi_jdos.py`
- 后端检测：继承自 `mlwf_hamiltonian`（导入时自动完成）
- 日志级别：`INFO` 用于计算进度通知
- **无环境变量配置，后端自动继承**
- **k 网格范围：固定为 `[-0.5, 0.5)`**
- **输出格式：始终返回 `np.ndarray`**
- **返回结构：与 `load_qpi_from_h5()` 完全对齐**
- **保存策略：HDF5 保存原始网格，返回扩展网格**

---

## 数据流示意图

```
┌─────────────────────────────────────────────────────────────────┐
│                        calculate()                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 计算 QPI (原始网格 [-0.5, 0.5))                               │
│  2. 保存 HDF5 (原始网格，节省空间)                                  │
│  3. 根据 q_range 扩展/裁剪                                        │
│  4. 返回 dict {qpi_layers, q1_grid, q2_grid, qx_grid, qy_grid,   │
│                metadata}                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    load_qpi_from_h5()                           │
├─────────────────────────────────────────────────────────────────┤
│  1. 加载 HDF5 (原始网格)                                          │
│  2. 根据 q_range 扩展/裁剪                                        │
│  3. 返回 dict {qpi_layers, q1_grid, q2_grid, qx_grid, qy_grid,   │
│                metadata}                                        │
└─────────────────────────────────────────────────────────────────┘

✅ 两者返回结构完全一致，可互换使用
```