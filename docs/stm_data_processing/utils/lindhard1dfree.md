# 一维自由电子 Lindhard 响应函数模块接口文档

## 模块概述

模块路径：`src/stm_data_processing/utils/lindhard1dfree.py`

`lindhard1dfree.py` 计算一维自由电子气的 Lindhard 响应函数 χ(q, ω+iη)，以及辅助的自由电子色散与费米-狄拉克分布函数。用于研究一维电子气的静态/动态电荷响应。

**单位约定**：

- 波矢 `q`、`k`、费米波矢 `k_f`：Å⁻¹
- 能量 `E`、化学势 `μ`、能量转移 `ω`、展宽 `η`：eV
- 温度：K

**响应函数定义**：

```
χ(q, ω+iη) = ∫ dk [f(E_k) - f(E_{k+q})] / [E_k - E_{k+q} + ω + iη]
```

---

## 模块常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `HBAR_Js` | `1.054571817e-34` | 约化普朗克常数 ħ（J·s） |
| `ME_kg` | `9.1093837015e-31` | 电子静质量（kg） |
| `EV_TO_J` | `1.602176634e-19` | 1 eV 对应焦耳 |
| `ANGSTROM_TO_M` | `1e-10` | 1 Å = 1e-10 m |
| `H2OVER2M_EVA2` | `ħ²/(2m_e)`（eV·Å²） | 预计算因子，用于色散 `E = H2OVER2M_EVA2 · k²` |
| `KB_EVK` | `8.617333262159999e-5` | 玻尔兹曼常数（eV/K） |

---

## 模块级函数

### `free_electron_energy(k)`

```python
def free_electron_energy(k: float) -> float
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `k` | `float` | 波矢（Å⁻¹） |

**返回**：动能 `E(k) = ħ²k²/(2m)`（eV），即 `H2OVER2M_EVA2 * k * k`。

### `fermi_dirac_from_energy(e, mu, t)`

数值稳定的费米-狄拉克分布。

```python
def fermi_dirac_from_energy(e: float, mu: float, t: float) -> float
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `e` | `float` | 能量（eV） |
| `mu` | `float` | 化学势（eV） |
| `t` | `float` | 温度（K） |

**返回**：`f(E)`。

- `t <= 0`：阶跃函数（`e < mu` → 1，否则 0）。
- `t > 0`：`x = (e - mu)/(kB·t)`；`x > 50` 返回 `exp(-x)`，`x < -50` 返回 1，否则 `1/(1 + exp(x))`。

---

## 核心类：`Lindhard1DFreeElectron`

```python
class Lindhard1DFreeElectron:
    """Compute the 1D Lindhard function χ(q, ω+iη) for a free electron gas."""
```

### 构造函数

```python
def __init__(
    self,
    q_max: float,
    k_f: float,
    temperature: float,
    omega: float,
    eta: float,
    q_points: int = 200,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `q_max` | `float` | — | 最大 q（q ∈ [0, q_max]，Å⁻¹），必须 > 0 |
| `k_f` | `float` | — | 费米波矢（Å⁻¹） |
| `temperature` | `float` | — | 温度（K） |
| `omega` | `float` | — | 能量转移（eV） |
| `eta` | `float` | — | 展宽参数（eV），必须 > 0 |
| `q_points` | `int` | `200` | q 点数目，必须 ≥ 2 |

构造时生成 `q_array`（`np.linspace(0, q_max, q_points)`），并逐点计算 `chi_array`。

### 实例属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `q_max` | `float` | 最大 q |
| `k_f` | `float` | 费米波矢 |
| `temperature` | `float` | 温度 |
| `omega` | `float` | 能量转移 |
| `eta` | `float` | 展宽 |
| `q_points` | `int` | q 点数目 |
| `q_array` | `np.ndarray (q_points,)` | q 网格（0 到 q_max） |
| `chi_array` | `np.ndarray (q_points,)` | 复值 χ 数组（dtype=complex） |

### 方法

#### `_compute_lindhard_q(q)`

```python
def _compute_lindhard_q(self, q: float) -> complex
```

计算单个 q 的 χ(q, ω+iη)。为内部方法（构造时自动调用）。

实现要点：

- 化学势 `μ = a·k_f²`（`a = H2OVER2M_EVA2`）。
- `q ≈ 0` 时用 `1e-14` 防止除零。
- 被积函数 `[f(E_k) - f(E_{k+q})] / [E_k - E_{k+q} + ω + iη]`。
- 积分上限 `k_max` 由 `k_f`、`q`、温度/展宽共同估计。
- 在 `-k_f, k_f, -q-k_f, -q+k_f`、极点 `k_pole = ω/(2aq) - q/2` 等处分段，用 `scipy.integrate.quad` 对实部/虚部分别积分。

---

## 数学公式

### 自由电子色散

```
E(k) = ħ²k² / (2m)
```

实现为 `E(k) = H2OVER2M_EVA2 · k²`，其中

```
H2OVER2M_EVA2 = ħ²·(1 Å⁻¹ → m⁻¹)² / (2 m_e · e) = ħ²·1e20 / (2 m_e · EV_TO_J)
```

### 费米-狄拉克分布

```
f(E) = 1 / (1 + exp((E - μ) / (kB·T)))
```

### Lindhard 响应函数

```
χ(q, ω+iη) = ∫ dk [f(E_k) - f(E_{k+q})] / [E_k - E_{k+q} + ω + iη]

E_k = a·k²,  E_{k+q} = a·(k+q)²,  μ = a·k_f²
```

数值实现按实部/虚部分段数值积分（`scipy.integrate.quad`），在奇点附近通过插值点分段提高精度。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.utils.lindhard1dfree import (
    Lindhard1DFreeElectron,
    free_electron_energy,
    fermi_dirac_from_energy,
)

# 自由电子色散
E = free_electron_energy(k=1.0)      # eV

# 费米-狄拉克分布（T=300 K, μ=0 eV）
f = fermi_dirac_from_energy(0.0, mu=0.0, t=300.0)

# 构造 Lindhard 响应函数计算器
lin = Lindhard1DFreeElectron(
    q_max=1.0,        # Å⁻¹
    k_f=0.5,          # Å⁻¹
    temperature=300.0,  # K
    omega=0.1,        # eV
    eta=0.01,         # eV
    q_points=200,
)

# q 网格与 χ 数组（复值）
q = lin.q_array
chi = lin.chi_array

re_chi = np.real(chi)   # 实部
im_chi = np.imag(chi)   # 虚部
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组与数值运算 |
| `scipy.integrate.quad` | 是 | 分段数值积分 |
| `itertools.pairwise` | 是 | 积分区间成对遍历 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `q_max <= 0` |
| `ValueError` | `eta <= 0` |
| `ValueError` | `q_points < 2` |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`q_max > 0`、`eta > 0`、`q_points ≥ 2`**
- [ ] **波矢单位 Å⁻¹，能量单位 eV，温度单位 K**
- [ ] **`chi_array` 为复值数组，需分别取 `np.real` / `np.imag`**
- [ ] **`q_array` 覆盖 `[0, q_max]`，共 `q_points` 个点**
- [ ] **χ 定义含 `+iη`（`denom = Ek - Ekq + ω + iη`）**
- [ ] **化学势 μ = a·k_f²（由 `k_f` 决定，非独立参数）**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/lindhard1dfree.py`
- 物理常量：CODATA 2018
- 积分方案：`scipy.integrate.quad` 分段实/虚部分别积分
- 精度相关：`quad` 使用 `limit=400, epsabs=1e-10, epsrel=1e-8`
