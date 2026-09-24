# Usadel 扩散极限超导能隙 / STS 隧道谱接口文档

## 模块概述

模块路径：`src/stm_data_processing/utils/usadel.py`

实现**扩散极限（脏极限，`l << xi`）**的 1D 稳态 Usadel 方程求解器，用于计算超导能隙剖面、局域态密度与 STS 隧道谱。数学规范见 [`docs/design/usadel_gap_model.md`](../../design/usadel_gap_model.md)（v1.1）。

**与 `btk.py` 的分工（不可互换）**：

| | `btk.py`（BTK） | `usadel.py`（本模块） |
|---|---|---|
| 输运机制 | 弹道极限 `l >> xi` | 扩散极限 `l << xi` |
| 界面 | delta 势垒，单个无量纲 `Z` | `Delta(x)` 剖面（S/N 界面为宽 `0.05*xi` 的平滑过渡） |
| 输出 | 电导 `sigma(E)`（Sharvin 归一化） | 态密度 `N(E, x)`、能隙剖面 `Delta(x)`、`dI/dV` |

**单位约定**（与 `btk.py` 一致）：

- 能量 `Delta0`、`E`、`Gamma`：eV
- 温度 `T`、`Tc`：K
- 长度 `x`、`d`、`L_S`：nm
- 扩散常数 `D`：`nm^2/s`；若传入值 `< 1e12` 会被当作 `m^2/s` 自动换算成 `nm^2/s`

常量：`KB = 8.617333262e-5`（eV/K）、`HBAR = 6.582119569e-16`（eV·s）、`WEAK_COUPLING_RATIO = pi / e^gamma ≈ 1.7639`。

**约定（对称 `tau1`）**：`Delta_hat = Delta*tau1`、`G_hat = cos(theta)*tau3 + sin(theta)*tau1`，归一化 `g^2 + f^2 = 1`，`N(E, x) = Re cos(theta)`。松原分支的 `theta` 为实数，实时分支为复数。

---

## 模块级函数

| 函数 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `bcs_dos` | `bcs_dos(energy, Delta, Gamma=0.0)` | `ndarray` | 解析 BCS 态密度（含 Dynes 展宽）：`Gamma=0` 时 `abs(E)>Delta` 为 `abs(E)/sqrt(E^2-Delta^2)`、亚能隙为 0 |
| `bcs_gap` | `bcs_gap(Tc, T)` | `float` | 弱耦合自洽能隙（`ln(Tc/T)` 正则化 + 二分）；`T -> 0` 给 `1.7639*k_B*Tc`，`T >= Tc` 给 0 |
| `fermi_derivative` | `fermi_derivative(E, T)` | `ndarray` | `-df/dE` |

内部求解器（未列入 `__all__`，供 `Usadel1D` 与回归脚本调用）：

- `solve_retarded_sn(d, E, D, Delta0, Gamma=..., L_S=..., n_nodes=...)` —— S/N 几何的实时推迟解，返回 `(x, N(x))`
- `solve_matsubara_sn(...)` —— S/N 几何的松原解（自洽能隙剖面）

---

## 核心类：`Usadel1D`

```python
class Usadel1D:
    def __init__(self, Delta0, D, T=0.0, Tc=None, Gamma=0.0)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `Delta0` | `float` | — | 超导能隙（eV） |
| `D` | `float` | — | 扩散常数（`nm^2/s`，或 `< 1e12` 时按 `m^2/s` 换算） |
| `T` | `float` | `0.0` | 温度（K） |
| `Tc` | `float` | `None` | 临界温度（K）；缺省由弱耦合关系 `Delta0 = 1.7639*k_B*Tc` 反推 |
| `Gamma` | `float` | `0.0` | Dynes 展宽（eV），可被各方法的同名参数覆盖 |

### 属性与方法

| 成员 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `xi` | 属性 | `float` | 脏极限相干长度 `sqrt(HBAR*D/Delta0)`（nm） |
| `thouless` | `thouless(L)` | `float` | Thouless 能量 `HBAR*D/L^2`（eV） |
| `solve_gap` | `solve_gap(geometry="bulk", d=None, L_S=None, T=None, n_nodes=300, max_iter=500, tol=1e-6, damping=0.5)` | `(x, delta_x)` | 自洽求解 `Delta(x)`（松原分支） |
| `dos` | `dos(energy, position=0.0, geometry="bulk", d=None, L_S=None, Gamma=None, n_nodes=300)` | `ndarray` 或 `float` | 局域态密度 `N(E, x)` |
| `tunnel_spectrum` | `tunnel_spectrum(V, position=0.0, geometry="bulk", d=None, L_S=None, T=None, Gamma=None)` | `(V, dIdV)` | STS 隧道谱 |

### 几何

| `geometry` | 含义 | 走哪条路径 |
|---|---|---|
| `"bulk"` | 均匀超导体 `Delta(x) = Delta0` | **解析路径**：直接调用 `bcs_dos`（不经数值求解器） |
| `"sn"` | 1D S/N 双层：S 区 (`x < 0`) `Delta = Delta0`，N 区 (`0 < x < d`) `Delta = 0`，S/N 界面在 `x = 0`，N 外端 `x = d` 为无电流边界 | **数值路径**：实时 Riccati `gamma = tan(theta/2)` + `scipy.integrate.solve_bvp` |

- `L_S` 为 S 区厚度，缺省 `10*xi`；要检验"数值解在深 S 区复现块体 BCS"，请用 `geometry="sn"` 并取足够大的 `L_S`（回归脚本里的数值路径用例用 `L_S = 30*xi`，在 `x = -15*xi` 取值）。
- `d` 在 `"sn"` 几何下必填（N 层厚度，nm）。

---

## 数学公式

### 松原分支（自洽能隙）

```
hbar*D * theta'' = 2*omega_n*sin(theta) - 2*Delta(x)*cos(theta)        # theta 实
omega_n = (2n+1)*pi*k_B*T
```

弱耦合自洽方程（`ln(Tc/T)` 正则化，见设计文档 §4.1）：

```
Delta(x)*ln(Tc/T) = 2*pi*k_B*T * sum_{omega_n>0} [ Delta(x)/omega_n - f(omega_n,x) ]
f(omega_n,x) = sin(theta(omega_n,x))      # 块体: Delta/sqrt(omega_n^2+Delta^2)
```

### 实时推迟分支（态密度）

```
hbar*D * theta'' + 2i*E*sin(theta) + 2*Delta(x)*cos(theta) = 0         # theta 复
N(E, x) = Re cos(theta^R(E, x))                                        # |E| >> Delta 时 -> 1
```

`|E| < Delta` 的推迟分支（`E -> E + i0+`）在实现中显式处理；`theta/Riccati` 参量化在 `|E| < Delta` 处避免了复 `theta` 的 `cosh/sinh` 溢出。

### STS 隧道谱

```
dI/dV(V) ∝ integral dE  N(E, x_surf) * (-df/dE)(E - eV)
```

`T = 0` 时退化为 `dI/dV(V) = N(eV)`；有限温度用自适应求积（能分辨能隙边的 `1/sqrt` 奇异性）。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.utils.usadel import Usadel1D, bcs_dos, bcs_gap

# 1 meV 能隙、D = 1e16 nm^2/s 的脏超导体
u = Usadel1D(Delta0=1e-3, D=1e16, T=0.0)
print(u.xi)                       # 81.1 nm
print(u.Tc)                       # 6.579 K（弱耦合反推）

# 均匀能隙剖面（松原分支，自洽）
x, delta_x = u.solve_gap(geometry="bulk")

# STS 谱（正常金属针尖，含温度与 Dynes 展宽）
V = np.linspace(-3e-3, 3e-3, 401)
V, didv = u.tunnel_spectrum(V, T=4.2, Gamma=1e-5)

# N 区外端的诱导 minigap（数值路径，必须用 sn 几何）
N_edge = u.dos(0.0, position=200.0, geometry="sn", d=200.0, Gamma=1e-6)

# 解析 BCS 参照
E = np.linspace(-3e-3, 3e-3, 401)
N = bcs_dos(E, Delta=1e-3, Gamma=1e-5)
```

典型数值（来自独立验证报告 `var/usadel_verify/report.md`）：

- `Delta0/(k_B*Tc) = 1.763878`（弱耦合理论值 1.763877）
- S/N 双层长结极限 minigap `E_g -> 0.78 * E_Th`（Zhou et al. 1998），`E_Th = HBAR*D/d^2`；有限 `d` 时偏大（交接区）
- `T -> 0`、`Gamma -> 0` 时 `dI/dV(V) = N(eV)`

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组运算 |
| `scipy` | 是 | `integrate.solve_bvp`（实时 BVP）、`integrate.quad`（有限温卷积） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `geometry="sn"` 但 `d` 为 `None`；或 `geometry` 不是 `"bulk"`/`"sn"` |
| `RuntimeError` | `solve_bvp` 不收敛（不静默返回错误结果） |

---

## 已知限制（重要，与设计文档 §0.3 / §6.4 对应）

1. 单自旋、无相位/电流、无非平衡分布函数；**不含环境 `P(E)` 效应**（能隙边整体位移可能来自环境，见设计文档 §6.4）。
2. S/N 界面是宽度 `0.05*xi` 的平滑 `Delta` 过渡（等价于透明阶跃界面）；**未实现 Kupriyanov–Lukichev 界面电阻**（设计文档 §5.2 / §9-1）。
3. `solve_gap(geometry="sn")` 的松原求和截断在 `omegac = 20*k_B*Tc`，自洽深 S 区 `Delta` 相对 `omegac = 1000` 的块体值偏离约 0.24%。
4. 实时 BVP 在 `|E|` 恰为 `Delta0` 且 `Gamma -> 0` 时数值脆弱（回归用 `Gamma >= 1e-9`，且锚点避开该点）。
5. `geometry="bulk"` 走解析路径，**不能**用来检验数值求解器；数值求解器由 `sn` 几何与回归用例 `test_numerical_bulk_reproduction` 覆盖。
6. `theta -> 0`（正常态）会让空间自洽的 S/N 求解退化，因此该路径与块体能隙方程分开实现。

---

## 回归与验证

| 产物 | 说明 |
|------|------|
| `tests/regression/check_usadel.py` | 6 项回归：`test_a1_bulk_bcs_dos_analytic_path`、`test_a2_gap_equation`、`test_a3_self_consistency_analytic_path`、`test_a4_tunneling_spectrum_analytic_path`、`test_a5_sn_bilayer`、`test_numerical_bulk_reproduction` |
| 运行 | `.venv/bin/python tests/regression/check_usadel.py`，或 `.venv/bin/python -m pytest -q -k usadel` |
| 独立验证 | `var/usadel_verify/report.md`（V1–V6，含"改坏数值路径必须变红"的反面检查） |
| 评审 | `var/usadel_review/report.md`（verdict=pass） |
| 演示图 | `.venv/bin/python scripts/usadel_sts_demo.py` → `var/usadel_demo/`（5 张 PNG + README） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`Delta0`、`E`、`Gamma` 单位 eV，`T`/`Tc` 单位 K，`D` 单位 `nm^2/s`，长度单位 nm**
- [ ] **`D < 1e12` 会被当成 `m^2/s` 自动换算**——若确实需要极小的 `nm^2/s` 值，请显式换算后再传入
- [ ] **`geometry="bulk"` 是解析路径**；要检验或依赖数值求解器，请用 `geometry="sn"`
- [ ] **`geometry="sn"` 必须传 `d`**；`L_S` 缺省 `10*xi`，检验深 S 块体极限时用 `L_S >= 30*xi`
- [ ] **`T=0` 时 `tunnel_spectrum` 退化为 `N(eV)`**；有限 `T` 才会做费米卷积
- [ ] **亚能隙 `N = 0` 只在 `T = 0` 且 `Gamma = 0` 时严格成立**
- [ ] **`solve_gap` 返回 `(x, delta_x)`；`dos`/`tunnel_spectrum` 对 `bulk` 几何返回解析值**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/usadel.py`
- 模型：1D 稳态 Usadel（扩散极限），对称 `tau1` 约定
- 数值方案：松原分支实 `theta` 不动点自洽 + 实时分支 Riccati `gamma = tan(theta/2)` + `solve_bvp`
- 首次实现：2026-09-23（AgentTeams 团队 `usadel-sts` 的 t1）；同日加固 t4（门覆盖 + 演示图诚实性），`usadel.py` 字节未变
- 数学规范：[`docs/design/usadel_gap_model.md`](../../design/usadel_gap_model.md)
