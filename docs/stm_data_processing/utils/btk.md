# BTK 隧道谱模型接口文档

## 模块概述

模块路径：`src/stm_data_processing/utils/btk.py`

`btk.py` 实现 Blonder–Tinkham–Klapwijk（BTK）模型，用于计算超导隧穿谱的微分电导，支持零温与有限温度两种情形。典型应用于 STM 隧道谱（微分电导 dI/dV）拟合。

**单位约定**：

- 能隙 `Delta`、能量 `E`、展宽 `Gamma`：eV
- 温度 `T`：K

模块常量 `kB = 8.617333262e-5`（eV/K）。

---

## 核心类：`BTK`

```python
class BTK:
    """Blonder-Tinkham-Klapwijk (BTK) model for tunneling spectroscopy."""
```

### 构造函数

```python
def __init__(self, Delta, Z, Gamma=1e-6)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Delta` | `float` | — | 超导能隙（eV） |
| `Z` | `float` | — | 势垒强度参数（无量纲） |
| `Gamma` | `float` | `1e-6` | 展宽参数（eV） |

实例属性：`Delta`、`Z`、`Gamma`。

---

### 实例方法

| 方法 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `sigma_zero_T` | `sigma_zero_T(E)` | `ndarray` | 零温电导 σ(E) |
| `sigma_finite_T` | `sigma_finite_T(E, T)` | `ndarray` | 有限温度电导 σ(E, T) |
| `spectrum` | `spectrum(E_min=-5, E_max=5, n_points=500, T=0)` | `(E, sigma)` | 生成完整谱 |
| `update_params` | `update_params(Delta=None, Z=None, Gamma=None)` | `None` | 更新模型参数 |

#### `sigma_zero_T(E)`

零温 BTK 电导。

```python
def sigma_zero_T(self, E)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `E` | `array_like` | 能量值（eV） |

**返回**：归一化电导 σ(E)（`ndarray`，与输入同形）。

- 亚能隙区（`|E| < Delta`）：`A = Delta² / [E² + (Delta² - E²)(1 + 2Z²)²]`，`σ = 2A`。
- 超能隙区（`|E| ≥ Delta`）：见「数学公式」节，`σ = Re(1 + A - B)`。

#### `sigma_finite_T(E, T)`

通过热展宽计算有限温度电导。

```python
def sigma_finite_T(self, E, T)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `E` | `array_like` | 能量值（eV） |
| `T` | `float` | 温度（K） |

**返回**：归一化电导 σ(E, T)（`ndarray`）。实现为 σ₀(E) 与费米函数导数的卷积（`np.convolve(..., mode="same")`）。

#### `spectrum(E_min=-5, E_max=5, n_points=500, T=0)`

生成完整电导谱。

```python
def spectrum(self, E_min=-5, E_max=5, n_points=500, T=0)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `E_min` | `float` | `-5` | 最小能量（eV） |
| `E_max` | `float` | `5` | 最大能量（eV） |
| `n_points` | `int` | `500` | 能量点数 |
| `T` | `float` | `0` | 温度（K） |

**返回**：`(E, sigma)` 元组，`E = np.linspace(E_min, E_max, n_points)`；`T == 0` 时调用 `sigma_zero_T`，否则调用 `sigma_finite_T`。

#### `update_params(Delta=None, Z=None, Gamma=None)`

```python
def update_params(self, Delta=None, Z=None, Gamma=None)
```

只更新传入的非 `None` 参数，返回 `None`。

---

## 数学公式

### 零温电导（亚能隙区 |E| < Δ）

```
A = Δ² / [E² + (Δ² - E²)·(1 + 2Z²)²]
σ = 2A
```

### 零温电导（超能隙区 |E| ≥ Δ）

引入复能量展宽 `Γ`：

```
Ẽ = E + iΓ
u² = 0.5·(1 + √(Ẽ² - Δ²) / Ẽ)
v² = 0.5·(1 - √(Ẽ² - Δ²) / Ẽ)

γ  = u² + (u² - v²)·Z²
A  = u²v² / γ²
B  = (u² - v²)²·Z²·(1 + Z²) / γ²

σ  = Re(1 + A - B)
```

其中 √(Ẽ² - Δ²) 在 `Re(Ẽ) < 0` 时取负号分支。

### 有限温度热展宽

```
σ(E, T) = ∫ σ₀(E') · (-∂f/∂E)(E - E') dE'
```

数值实现：先算 `σ₀ = sigma_zero_T(E)`，再构造

```
β = 1/(kB·T),  x = βE/2
sech² = 4·exp(-2|x|) / (1 + exp(-2|x|))²
dfdE = (β/4)·sech²
```

将 `dfdE` 归一化后与 `σ₀` 做 `np.convolve(..., mode="same")` 并乘以能量步长 `dE`。

---

## 使用示例

```python
import numpy as np
import matplotlib.pyplot as plt
from stm_data_processing.utils.btk import BTK

# 构造模型：能隙 1.5 meV，势垒 Z=0.5
btk = BTK(Delta=0.0015, Z=0.5, Gamma=1e-6)

# 零温谱
E0, sigma0 = btk.spectrum(E_min=-0.01, E_max=0.01, n_points=1000, T=0)

# 有限温度谱（T=4.2 K）
E4, sigma4 = btk.spectrum(E_min=-0.01, E_max=0.01, n_points=1000, T=4.2)

# 直接计算指定能量点的零温电导
E = np.linspace(-0.005, 0.005, 101)
s = btk.sigma_zero_T(E)

# 更新参数后重新计算
btk.update_params(Z=0.7, Delta=0.002)
E2, s2 = btk.spectrum()
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组运算与卷积 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| （无显式校验） | 构造函数与方法未内置参数校验；注意 `T=0` 时 `sigma_finite_T` 会除零（`β = 1/(kB·T)`），此时应使用 `sigma_zero_T` 或 `spectrum(T=0)` |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`Delta`、`Gamma`、`E` 单位为 eV，`T` 单位为 K，`Z` 无量纲**
- [ ] **`spectrum(T=0)` 自动走零温路径，`T>0` 走有限温度路径**
- [ ] **`sigma_zero_T` 亚能隙区（|E| < Δ）与超能隙区（|E| ≥ Δ）采用不同公式**
- [ ] **`sigma_finite_T` 内部依赖 `sigma_zero_T` 并做卷积，勿在 `T=0` 时调用**
- [ ] **`update_params` 仅更新非 `None` 参数，未传入的参数保持不变**
- [ ] **`spectrum` 返回 `(E, sigma)` 元组，E 由 `np.linspace` 生成**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/btk.py`
- 模型：Blonder–Tinkham–Klapwijk（BTK）
- 温度处理：零温解析 + 有限温度费米函数导数卷积
- 常量：`kB = 8.617333262e-5` eV/K
