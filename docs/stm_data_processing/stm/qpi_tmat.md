# T 矩阵 QPI 模块接口文档

## 模块概述

`qpi_tmat.py` 提供基于 Wannier90 MLWF Hamiltonian 的 T 矩阵 QPI（Quasiparticle Interference）计算框架。

> **⚠️ 存根状态声明**：本模块当前为**存根（stub）**。构造函数 `__init__` 已完整实现（包含 `num_wann` 校验与格林函数初始化），但计算核心 `_compute_tmat()` 与公开接口 `calculate()` **尚未实现**，两者均只返回 `None`。**请勿依赖本模块进行实际 QPI 计算**，且不要基于文档虚构其未实现的功能。

**依赖关系**: 依赖 `mlwf_gk.GreenFunction`（格林函数）与 `mlwf_hamiltonian.MLWFHamiltonian`（哈密顿量）。

---

## 核心类：`TmatQPI`

### 类定义

```python
class TmatQPI:
    """Class for calculating T-matrix QPI (Quasiparticle Interference)."""
```

### 构造函数

```python
def __init__(
    self,
    hamiltonian: MLWFHamiltonian,
    nk: int = 128,
    eta: float = 0.001,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `hamiltonian` | `MLWFHamiltonian` | 已初始化的哈密顿量实例 |
| `nk` | `int` | k 点网格数量，默认 128（经 `int()` 强制转换为整型） |
| `eta` | `float` | 谱展宽参数，默认 0.001（经 `float()` 强制转换） |

**内部属性**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `num_wann` | `int \| None` | Wannier 函数数量（从 `hamiltonian.num_wann` 复制，已校验为正数） |
| `nk` | `int` | k 网格大小（`int(nk)`） |
| `eta` | `float` | 展宽参数（`float(eta)`） |
| `gf` | `GreenFunction` | 格林函数计算器，以 `GreenFunction(hamiltonian, eta=eta)` 构造 |

**⚠️ 初始化检查**（构造函数内已完成）:
- 若 `hamiltonian` 没有 `num_wann` 属性，或 `hamiltonian.num_wann is None`，抛出 `ValueError("Invalid MLWFHamiltonian: num_wann is not initialized.")`
- 若 `hamiltonian.num_wann <= 0`，抛出 `ValueError("Invalid MLWFHamiltonian: num_wann must be positive, got {num_wann}.")`

---

### 实例方法

#### `_compute_tmat()`

计算 T 矩阵的核心方法（**未实现**）。

```python
def _compute_tmat():
    return None
```

**状态**: 存根。函数体仅 `return None`，无任何计算逻辑。

**⚠️ 注意**: 该方法签名**未声明 `self` 参数**（定义为无参函数）。作为实例方法调用时会触发 `TypeError`（Python 自动传入实例导致参数数量不匹配）。

#### `calculate()`

公开 QPI 计算接口（**未实现**）。

```python
def calculate():
    return None
```

**状态**: 存根。函数体仅 `return None`，无任何计算逻辑。

**⚠️ 注意**: 该方法签名同样**未声明 `self` 参数**。当前直接调用 `instance.calculate()` 会触发 `TypeError: calculate() takes 0 positional arguments but 1 was given`。

---

## 数学公式

T 矩阵 QPI 的理论公式（参考实现，本模块**尚未落地**）：

```
T(ω) = V · (1 - G(ω) · V)⁻¹
```

其中：
- `V` : 散射势矩阵
- `G(ω)` : 格林函数（由 `self.gf` 提供）

> 上述公式仅为领域内 T 矩阵法的通用形式，**并未出现在当前源码中**。源码中 `_compute_tmat()` 与 `calculate()` 均无实现，仅返回 `None`。

---

## 使用示例

### 构造实例（当前唯一可用的功能）

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.stm.qpi_tmat import TmatQPI

# 加载哈密顿量
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# 构造 T 矩阵 QPI 计算器（构造函数已完整实现）
tmat = TmatQPI(ham, nk=128, eta=0.001)

# 检查初始化后的属性
print(tmat.num_wann)   # 正整数的 Wannier 函数数量
print(tmat.nk)         # 128
print(tmat.eta)        # 0.001
print(tmat.gf)         # GreenFunction 实例
```

### ⚠️ 不要调用计算接口

```python
# ❌ 当前不可用：calculate() 为存根且签名缺少 self
result = tmat.calculate()  # 返回 None 或 TypeError（签名无 self）
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `GreenFunction` | 是 | 格林函数计算（`stm_data_processing.dft.wannier90.mlwf_gk`） |
| `MLWFHamiltonian` | 是 | 哈密顿量类型（`stm_data_processing.dft.wannier90.mlwf_hamiltonian`） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `hamiltonian` 缺少 `num_wann` 属性或 `num_wann is None` |
| `ValueError` | `hamiltonian.num_wann <= 0`（非正数） |
| `TypeError` | 调用 `_compute_tmat()` / `calculate()` 时（方法签名未声明 `self`） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`hamiltonian` 必须是已初始化的 `MLWFHamiltonian` 实例**
- [ ] **`hamiltonian.num_wann` 必须已初始化且为正数，否则抛出 `ValueError`**
- [ ] **`nk` 会被 `int()` 强制转换为整型，`eta` 会被 `float()` 强制转换**
- [ ] **⚠️ 本模块计算核心未实现，`_compute_tmat()` 与 `calculate()` 仅返回 `None`**
- [ ] **⚠️ `_compute_tmat()` 与 `calculate()` 签名未声明 `self`，实例调用会触发 `TypeError`**
- [ ] **构造实例（校验 `num_wann`、初始化 `gf`）是本模块当前唯一可用的功能**
- [ ] **不要基于文档假设存在尚未实现的返回结构或计算逻辑**

---

## 版本信息

- 模块路径：`src/stm_data_processing/stm/qpi_tmat.py`
- 实现状态：**存根（stub）**，计算核心未实现
- 类数量：1（`TmatQPI`）
- 日志级别：无（静默）
- 后端：通过 `GreenFunction` 间接继承 `mlwf_hamiltonian` 的后端检测
