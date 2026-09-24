# Usadel 方程：扩散极限下的超导能隙与隧道谱（数学描述）

> **状态**：v1.2 —— 数学部分经 v1/v1.1 定稿，**已按本规范实现并通过独立验证与评审**（实现状态见 §13）。
> §9 明确列出需要与原始文献逐条核对的条目（含所有约定敏感的前因子），§13 给出每条当前状态。
> 阅读对象：熟悉 BCS / BTK 的人；本稿不重复讲解 BCS 基础。

---

## 0. 目标、边界与与现有模块的分工

### 0.1 这份文档要定义什么

一个**扩散极限（脏极限）**超导能隙模拟器的完整数学描述：

| 输入 | 中间量 | 输出 |
|---|---|---|
| 能隙 `Delta0`（或 `Tc`）、扩散常数 `D`、结几何与界面透明率（`R_B`）、温度 `T` | 复准经典格林函数 `G(x, E)`；自洽场 `Delta(x)` | 局域态密度 `N(E, x)`；隧道谱 `dI/dV(V)`；能隙压低/近邻效应剖面 |

### 0.2 与 `utils/btk.py` 的分工（不可互换）

| | BTK（现有 `btk.py`） | Usadel（本稿） |
|---|---|---|
| 输运机制 | **弹道**（`l >> xi`，洁净） | **扩散**（`l << xi`，脏） |
| 界面 | 理想化 delta 势垒，单个无量纲 `Z` | 界面面积电阻 `R_B`，走 Kupriyanov–Lukichev 边界条件 |
| 自由度 | 散射振幅 `A`、`B`，有解析解 | 各向同性 `G`，需求解边值问题 |
| 参数 | `Delta, Z, Gamma, T` | `Delta0/Tc, D, T, R_B, 几何` |

**判据**：比较平均自由程 `l` 与相干长度 `xi`。`l >> xi` 用 BTK；`l << xi` 用 Usadel；两者之间（如 Pb 的 `xi0 ~ 80 nm`、`l ~ 10–100 nm`）两个极限都不严格，只能选更接近的一个并记录误差方向。

### 0.3 本稿不覆盖

- **多重 Andreev 反射（MAR）亚谐波结构**：需要含时/Keldysh Usadel（带分布函数 `f̂`），稳态方程拿不到。
- **自旋/多带的具体数值方案**：只给出扩展入口（Riccati 参数化），不展开。
- **非平衡分布函数**：只列出方程形式，不做数值实现。
- **环境效应（`P(E)` 理论）**：v1 不实现；但 §6.4 给出判据、关键公式与验证锚点。**低温低偏压下能隙边的整体位移、零偏压凹陷往往来自环境，而不是 `Gamma`**，不要用调 `Gamma` 去硬凑。

---

## 1. 约定与符号

### 1.1 Nambu 与自旋空间

基矢 `(u, v) = (电子, 空穴)`，Pauli 矩阵：

```
tau0 = [[1, 0], [0, 1]]      tau1 = [[0, 1], [1, 0]]
tau2 = [[0, -i], [i, 0]]     tau3 = [[1, 0], [0, -1]]
J    = i * tau2 = [[0, 1], [-1, 0]]        (反对称，{tau3, J} = 0，J^2 = -tau0)
```

本稿**固定**采用**对称（`tau1`）异常分量约定**：

```
Delta_hat(x) = Delta(x) * tau1
G_hat = g * tau3 + f * tau1          # 即 [[g, f], [f, -g]]
```

**选它的理由**（这条不是审美问题）：只有在这个约定下，松原分支的旋转角 `theta` 才是**实数**，方程才写成文献里常见的
`hbar*D*theta'' = 2*omega_n*sin(theta) - 2*Delta*cos(theta)`，块体解才给出标准的 `sqrt(omega_n^2 + Delta^2)`。

反对称约定（`G_hat = g*tau3 + f*J`、`Delta_hat = i*Delta*tau2`）与之相差一个规范变换：它把 `i` 搬进异常分量
（松原 `f = -i*Delta/sqrt(omega_n^2 + Delta^2)`），`theta` 不再是实角，若仍按实角求解会得到 `sqrt(omega_n^2 - Delta^2)`
这种错误结果。两套约定**物理等价**，但混用必然出错——所有 `i` 的位置见 §9-2。

### 1.2 归一化与格林函数

- 归一化条件：`G_hat^2 = tau0`，在本稿的对称约定下即 `g^2 + f*f~ = 1`（注意：反对称约定写成 `g^2 - f*f~ = 1`，两者不可混用）。
- 平衡态只需推迟分量 `G_hat^R(E)`；超前分量由 `G_hat^A = -tau3 (G_hat^R)^dagger tau3` 给出。
- 松原分量 `G_hat(omega_n)`（`omega_n = (2n+1) pi k_B T`）用于自洽求解能隙；实能量分量 `G_hat^R(E)` 用于态密度与谱。
- `g`、`f` 的实/虚结构由解析性决定（见 §3.2 的块体解），不要在推导前先假设。

局域态密度（LDOS）：

```
N(E, x) = (1/2) * Re Tr[ tau3 * G_hat^R(E, x) ]
```

在 §3 的 theta 参数化下即为 `N(E, x) = Re cos(theta^R(E, x))`。

### 1.3 单位与常量（与 `btk.py` 保持一致）

| 量 | 单位 | 说明 |
|---|---|---|
| 能量 `E`、`Delta`、`omega_n` | eV | 与 `btk.py` 一致 |
| 温度 `T` | K | `k_B = 8.617333262e-5` eV/K（沿用 `btk.py` 的常量） |
| 长度 `x` | nm | 与 STM 数据尺度一致 |
| 扩散常数 `D` | nm^2 / s | 也可输入 `m^2/s`，构造时显式换算 |
| `hbar` | 6.582119569e-16 eV·s | 因能量用 eV，`hbar*D` 的量纲是 eV·nm^2/s |

### 1.4 特征尺度

```
xi      = sqrt( hbar * D / Delta0 )        脏极限相干长度
xi_T    = sqrt( hbar * D / (2*pi*k_B*T) )  温度相干长度
E_Th    = hbar * D / L^2                    Thouless 能量
tau_D   = L^2 / D                           扩散时间
```

Usadel 方程的适用性要求 `l << xi`（且 `l <<` 结的最小几何尺度）。

---

## 2. 从 Eilenberger 到 Usadel（为什么要这样退化）

准经典格林函数 `g_hat(omega_n, r, v_F)` 满足 **Eilenberger 方程**（沿费米速度方向的一阶输运方程）：

```
i*hbar*v_F . grad g_hat + [ omega_n*tau3 + Delta_hat , g_hat ] = 0        (Matsubara)
```

（本稿约定下块体解 `g_hat ∝ omega_n*tau3 + Delta*tau1` 自洽；符号/相位约定见 §9-2。）

在**强杂质散射**极限 `l << xi` 下，`g_hat` 对费米面方向几乎各向同性，可展开为

```
g_hat(r, v_F) = G_hat(r) + v_hat . g_hat_1(r) + ...          |v_hat| = 1
```

把各向异性部分 `g_hat_1 ~ l * grad G_hat` 消去，即得 `G_hat(r)` 的 **Usadel 方程**（§3）。这一步同时给出适用边界：

- **成立**：`l << xi`、`l <<` 结尺度；
- **不成立**：`l >> xi`（弹道，用 BTK）、强相位相干的长结中弹道式共振。

---

## 3. 核心方程

### 3.1 矩阵形式

**Matsubara（求 `Delta(x)` 用）**：

```
hbar*D * grad . ( G_hat * grad G_hat )  =  [ omega_n * tau3 + Delta_hat(x) , G_hat ]
G_hat^2 = tau0 ,   omega_n = (2n+1) * pi * k_B * T ,   n = 0, 1, 2, ...
```

**实时推迟（求态密度/谱用）**：把 `omega_n -> -i*(E + i*0^+)` 代入**方程**（注意：代入方程合法，代入松原的数值解不合法，见下面的坑）：

```
hbar*D * grad . ( G_hat^R * grad G_hat^R )  =  [ -i*E * tau3 + Delta_hat(x) , G_hat^R ]
```

**注意（解析延拓的坑）**：`omega_n -> -iE` 对**方程**没问题（方程对 `omega_n` 解析）；但对**解**不行——`|E| < Delta` 时该替换会落到 `omega_n = ±i*Delta` 支割点的另一侧，`sqrt(E^2 - Delta^2)` 必须按推迟分支（`E -> E + i0^+`）取值，否则态密度会取到错误的分支。§8 的 R-5 专门验证这一点。

### 3.2 无自旋 1D 的 theta 参数化

定义角度 `theta`（松原为实、实时为复）：

```
G_hat = cos(theta) * tau3 + sin(theta) * tau1
```

它**自动满足** `G_hat^2 = tau0`（`cos^2 + sin^2 = 1`、`{tau3, tau1} = 0`）。代入 §3.1：

| 分支 | 方程 | `theta` |
|---|---|---|
| Matsubara | `hbar*D * theta'' = 2*omega_n*sin(theta) - 2*Delta(x)*cos(theta)` | 实 |
| 实时推迟 | `hbar*D * theta'' + 2i*E*sin(theta) + 2*Delta(x)*cos(theta) = 0` | 复 |

**推导要点**（便于逐行复核）：

```
G_hat * dG_hat/dx = i*theta'*tau2            # tau0 分量 ∝ d(cos^2+sin^2)/dx = 0，自动消失
grad.(G_hat grad G_hat) = i*theta''*tau2
[omega_n*tau3, G_hat]  = 2i*omega_n*sin(theta)*tau2
[Delta*tau1,  G_hat]   = -2i*Delta*cos(theta)*tau2
```

两边同除 `i`：所有项都落在 `tau2` 方向上，方程自洽（若异常分量写成 `J`，这一步会漏出 `i`，见 §9-2）。

**块体检验**（`theta'' = 0`）：

- Matsubara：`tan(theta) = Delta / omega_n`，于是 `g = cos(theta) = omega_n/sqrt(omega_n^2 + Delta^2)`、
  `f = sin(theta) = Delta/sqrt(omega_n^2 + Delta^2)`——标准 BCS 松原格林函数，且 `g^2 + f^2 = 1`。
- 实时推迟：`tan(theta) = Delta / (-i*E) = i*Delta/E`，于是

```
g^R = cos(theta) = E/sqrt(E^2 - Delta^2)              (|E| > Delta，推迟分支)
N(E) = Re cos(theta)
     = abs(E)/sqrt(E^2 - Delta^2)     (|E| > Delta)
     = 0                              (|E| < Delta)          # 标准 BCS 态密度
```

  其中 `|E| < Delta` 时 `g^R = -i*E/sqrt(Delta^2 - E^2)`、`f^R = i*Delta/sqrt(Delta^2 - E^2)`（取实部后为 0）。

### 3.3 Riccati 参数化（一般情形的入口）

一般（含自旋、多带、自洽场连续变化）情形下推荐用 Riccati 参数化：把 `G_hat` 用 `2x2`（自旋空间为 `4x4`）矩阵 `gamma` 表示，归一化条件自动满足，数值上避免了 `theta -> 0`（正常态解）附近的退化。

- v1 **只给出结论与用途**，精确形式与对应的传输方程留到 v2（列入 §9-3），并需与 Schopohl–Maki 原始文献对齐。

---

## 4. 自洽能隙方程

### 4.1 弱耦合、无截断的 T_c 正则化形式

用 `ln(Tc/T)` 形式可以同时消掉截断 `omega_c` 与耦合常数 `N(0)V`：

```
Delta(x) * ln(Tc / T) = 2*pi*k_B*T * sum_{omega_n > 0} [ Delta(x)/omega_n - f(omega_n, x) ]
```

其中 `f(omega_n, x)` 是 §3.2 的异常分量（`sin(theta)`）。

**块体自检**：把 `f = Delta/sqrt(omega_n^2 + Delta^2)` 代回，得到的是标准 BCS 方程

```
ln(Tc/T) = 2*pi*k_B*T * sum_{n>=0} [ 1/omega_n - 1/sqrt(omega_n^2 + Delta^2) ]
```

再由弱耦合关系 `2*pi*k_B*T_c * sum 1/omega_n = ln(1.13 * hbar*omega_c / (k_B*T_c))` 即回到常见的 `N(0)V` 形式。`T -> 0` 时该式给出 `Delta0 = 1.764 * k_B * T_c`（见 §8 R-3）。

### 4.2 什么时候必须自洽

- **不必**：远离界面、弱耦合、只关心单一位置谱形（例如 STM 在岛中心取谱）→ 直接固定 `Delta(x) = Delta0`，只解 §3 的线性（给定 `Delta`）边值问题。
- **必须**：界面附近 `Delta(x)` 被压低、S/N 双层或 SNS 的能隙剖面、近邻效应、`Delta` 的空间梯度导致的谱权重转移。

### 4.3 已知的数值陷阱

- `Delta ≡ 0` 是自洽迭代的**平凡不动点**（尤其 `T -> T_c` 附近）。必须用 `Delta0 != 0` 的初值，或在 v2 加入线性化稳定性分析。
- 松原求和截断与 `Delta` 的正则化必须用同一套约定（本稿统一用 `ln(Tc/T)` 形式）。

---

## 5. 边界条件

### 5.1 外边界（真空/绝缘）

```
grad G_hat . n = 0            # 无电流
```

### 5.2 内部界面：Kupriyanov–Lukichev（KL）

两块脏金属（或脏 N 与 S）在 `x = 0` 处相接，界面面积电阻 `R_B`（单位 `Ohm·nm^2`），两侧解记为 `G_hat_L`、`G_hat_R`：

```
sigma * ( G_hat * dG_hat/dx )|_interface  =  (1 / (2*R_B)) * [ G_hat_L , G_hat_R ]
```

- 形状（`[G_hat_L, G_hat_R]` 对易子形式）是 KL 条件的标准结构；**前因子与对易子次序属于约定敏感项**，见 §9-1。
- 无量纲透明率参数（Usadel 侧的"势垒强度"，对应 BTK 的 `Z`）：

```
zeta = R_B / R_N          R_N = L / (sigma * A)   （电极的正常态电阻）
```

`zeta << 1` 透明界面；`zeta >> 1` 隧穿极限。
- **自标定方式**：前因子由 §8 的 R-1（正常态下界面面电导必须恰好等于 `1/R_B`）与 R-2（隧穿极限 I_cR_N 的 Ambegaokar–Baratoff 值）反标定，而不是靠记诵。

### 5.3 与 BTK 的 `Z` 的关系（开放问题）

`Z`（弹道、delta 势垒）与 `zeta`（扩散、界面电阻）**没有一般的一一对应**：前者是势垒强度，后者是界面电阻与电极电阻之比，两者只在特定极限下近似挂钩。本稿**不断言**换算公式，列入 §9-4 待推导/核对。

---

## 6. 观测量

### 6.1 局域态密度

```
N(E, x) = Re cos( theta^R(E, x) )         # |E| >> Delta 时 -> 1
```

### 6.2 隧道谱（STM，正常金属针尖）

针尖态密度视为常数、弱耦合、含仪器/寿命展宽 `Gamma`：

```
dI/dV(V) ∝ integral dE  N(E, x_surf) * ( -df/dE )( E - eV )        f: 费米函数
```

- `Gamma`（Dynes 型展宽）在 v1 中用最简单的替换 `E -> E + i*Gamma` 实现，与 `btk.py` 的 `Gamma` 语义保持一致以便对照。
- 若针尖本身有能隙（超导针尖），则是两个 DOS 的卷积——v2 再谈。
- **前提声明（不要跳过）**：上式是"单粒子 + 弱耦合 + 无环境"近似。有相互作用时**隧道谱不等于态密度**：Ingold–Nazarov 附录 A.5 对此有明确警告（原话："we are quite pessimistic about the principal possibility to measure the electron density of states in the presence of interaction"）。因此拟合得到的 `N(E)` 只能叫"隧道谱反演出的有效态密度"。
- 若环境阻抗不可忽略，应改用 §6.4 的 `P(E)` 卷积，而不是继续调 `Gamma`。

### 6.3 点接触/扩散结电导

与 BTK 的直接对应量是"扩散结的微分电导"，需要由界面流（而非 §6.2 的隧道卷积）得到。**v1 不实现**，列在 §10 的路线图中；实现时应与 §8 R-4 的"极限对照"一起验收。

### 6.4 环境效应：为什么实测谱不等于理想卷积（`P(E)` 理论）

参考：G.-L. Ingold & Yu. V. Nazarov, *Charge Tunneling Rates in Ultrasmall Junctions*（1992，见 §11-7）。
**这不是 Usadel 的内容**，而是"隧道电子与外部电路交换能量"的独立一层物理：隧道事件不满足严格能量守恒，而是被能量交换概率 `P(E)` 加权。对 STM 低温低偏压、能隙边附近的谱形，这一层经常是主导的系统误差。

**定义**（原文 (49)(50)）：

```
P(E) = (1/(2*pi*hbar)) * integral dt * exp[ J(t) + i*E*t/hbar ]
J(t) = < [ phi_tilde(t) - phi_tilde(0) ] * phi_tilde(0) >        # 相位-相位关联函数
```

**三条与阻抗无关的恒等式**（原文 (65)(66)(72)，最适合做代码自检）：

```
integral dE P(E)          = 1
integral dE E * P(E)      = E_c = e^2 / (2*C)          # 平均能量交换 = 结的充电能
P(-E) = exp(-beta*E) * P(E)                            # 细致平衡；T = 0 时 P(E<0) = 0
```

**含环境的准粒子隧道电流**（原文 (153)/(157)；`I_qp,0` 是无环境时的理想结果）：

```
I_qp(V)   = integral dE  [ (1 - exp(-beta*e*V)) / (1 - exp(-beta*E)) ] * P(e*V - E) * I_qp,0(E/e)

I_qp,0(V) = (1/(e*R_T)) * integral dE * ( N_S(E)*N_S(E + e*V) / N(0)^2 ) * [ f(E) - f(E + e*V) ]     # (155)
```

`N_S(E)/N(0)` 就是 §3.2 的 BCS 态密度（原文 (151)，与我们的 `N(E)` 同式）；若一个电极为正常金属（STM 正常针尖），把其中一个 `N_S` 换成常数即回到 §6.2。

**四个必须记住的极限**：

| 环境 | `P(E)` | 谱会变成什么样 |
|---|---|---|
| 无环境阻抗 | `P(E) = delta(E)` | 严格回到 §6.2 的理想卷积（**这是本模型的默认假设**） |
| 低阻抗 `R << R_K`（`g = R_K/R >> 1`） | 窄峰 + 幂律尾巴 | 峰位基本不动；靠近能隙边处出现幂律 `dI/dV ∝ (e*V - 2*Delta)^(2/g - 1)`（原文 (158)），即 Altshuler–Aronov 型零偏压反常 |
| 高阻抗 `R >> R_K`、`T -> 0` | `P(E) = delta(E - E_c)`（原文 (86)） | **整条谱沿电压位移 `E_c/e`**；正常结 `I(V) = (e*V - E_c)*Theta(e*V - E_c)/R_T`（原文 (87)） |
| 高阻抗 + 有限 `T` | 高斯 `exp[-(E-E_c)^2/(4*E_c*k_B*T)] / sqrt(4*pi*E_c*k_B*T)`（原文 (85)） | 库仑阻塞被热抹平 |

- `E_c = e^2/(2C)`；`R_K = h/e^2 ≈ 25.8 kOhm`。**判据是把环境阻抗与 `R_K` 比**（不是与隧道电阻 `R_T` 比）。
- 原文 Fig. 13/14 是 `Delta = 2*E_c` 时超导结 I-V 被环境扭曲的例子：Fig. 13 扫 `g = R_K/R`（从 `∞` 到 0），Fig. 14 是有限温度（`k_B*T/E_c = 0.25`），其中虚线明确标出 `I_qp,0` 与**整条曲线位移 `E_c/e`** 后的曲线——两图对照可以分清"整体位移"与"热抹平"两种效应。
- **对拟合的实操结论**：先用 §6.2 的理想式拟合 `(Delta, Gamma, T)`；若残差集中在"能隙边整体位移"或"零偏压附近的凹陷/反常"，再引入 `P(E)`（需要环境参数 `R`、`C`，或完整 `Z_t(omega)`），**不要用加大 `Gamma` 去吸收它**。
- 只要 `E_c ≪ Delta` 且 `R ≪ R_K`，`P(E) ≈ delta(E)`，本节可以整节跳过——这是判断"要不要做这层"的唯一标准。

---

## 7. 数值方案（拟）

### 7.1 两条分支

1. **松原分支（求 `Delta(x)`）**：`omega_n` 求和（`ln(Tc/T)` 正则化后无需硬截断，实际取到 `omega_n ~ 10-20 k_B*T_c` 即收敛）；1D 空间网格在界面附近加密；`theta` 为实 → 二阶常微分方程边值问题（BVP）。
2. **实时分支（求 `N(E)`）**：给定 `Delta(x)` 后，逐能量点求复 `theta` 的 BVP。

### 7.2 求解器

- BVP：`scipy.integrate.solve_bvp`，或对 `theta'` 做打靶（shooting）。
- 自洽循环：外层 `Delta(x)`（简单/Anderson 混合，阻尼 0.1–0.3），内层 BVP；收敛判据 `max|dDelta| < 1e-6 * Delta0`。
- 能量网格：`|E|` 接近 `Delta` 处加密（支割附近梯度大）。

### 7.3 实现层面的已知坑

- `theta -> 0`（正常态解）在界面附近数值退化 → v2 换 Riccati。
- `Delta ≡ 0` 平凡不动点（§4.3）。
- `|E| < Delta` 时 `theta` 为复，`cos/sin` 的解析性（推迟分支 `E -> E + i0^+`）要显式处理，不能靠 `arccos` 的主值。
- 若后续要并行/长结：注意本项目已有的 Apple Accelerate `zgemm` 大矩阵段错误坑（见项目记忆），BVP 逐步求解不涉及批量 GEMM，但网格/能量扫描的批量实现要分块。

---

## 8. 验证锚点（每条都要能跑出数来）

| 编号 | 锚点 | 判据 |
|---|---|---|
| R-1 | 正常态极限 `Delta = 0` | `N(E) ≡ 1`；任意 `zeta` 下界面面电导 = `1/R_B` |
| R-2 | 块体 BCS | `N(E) = abs(E)/sqrt(E^2-Delta^2)`（`abs(E) > Delta`），`N = 0`（`abs(E) < Delta`），峰位在 `±Delta` |
| R-3 | 能隙方程 | `T -> 0` 时 `Delta0 = 1.764 k_B T_c`；`Delta(T)` 落在 BCS 普适曲线上 |
| R-4 | 长 SNS minigap | `L >> xi`、透明界面：`E_g ≈ 3.1 E_Th`（`E_Th = hbar*D/L^2`；精确常数与 `L` 定义挂钩，v2 核） |
| R-5 | 松原/实时一致性 | 松原解的正确延拓与实时解给出的 `N(E)` 在 `abs(E) > Delta` 一致到 ~1e-3 |
| R-6 | Ambegaokar–Baratoff（若实现相位/电流） | `I_c R_N = (pi*Delta/(2e)) * tanh( Delta/(2 k_B T) )` |
| R-7 | **反面锚点** | 弹道极限（`l >> xi`）下 Usadel 必然给错——记录为适用域边界，**不设"通过"判据**，禁止为了绿灯把它包装成通过 |
| R-8 | `P(E)` 的通性（若实现 §6.4） | `integral dE P(E) = 1`；`integral dE E*P(E) = E_c`；`P(-E) = exp(-beta*E)P(E)`；高阻抗 `T -> 0` 下正常结 `I(V) = (e*V - E_c)Theta(e*V - E_c)/R_T`，超导结谱相对 `I_qp,0` 整体位移 `E_c/e` |
| R-9 | 无环境退化 | 取 `P(E) = delta(E)` 时 (157) 必须**严格**退回 §6.2 的理想卷积（数值上做退化检查，不允许差一个常数） |

---

## 9. v1 待核对清单（请审阅时重点看这几条）

| # | 条目 | 为什么需要核 | 怎么核 |
|---|---|---|---|
| 9-1 | KL 条件的前因子（`1/2`?）与对易子次序 `[G_L, G_R]` vs `[G_R, G_L]`，以及面积/长度归一化 | 不同文献的 `sigma`、`R_B` 归一化不同，差一个常数就会让透明率标错 | 对 Kupriyanov–Lukichev (1988) 原文与 Belzig 等综述；再由 R-1、R-6 反标定 |
| 9-2 | 异常分量相位约定：本稿选对称 `tau1`（`Delta_hat = Delta*tau1`，松原 `theta` 为实、块体 `sqrt(omega_n^2+Delta^2)`）；备选反对称 `J = i*tau2`（`Delta_hat = i*Delta*tau2`，`f` 带 `i`、`theta` 不再是实角）。另需对齐 Eilenberger/Usadel 方程里 `i` 与 `omega_n` 的位置 | 约定不同会让 `Delta` 的符号、态密度峰位、松原 `theta` 的实性全部改变；本稿已按 τ̂₁ 约定给出推导与块体检验 | 与 Belzig 等综述、Usadel/Eilenberger 原文逐条对齐一次 |
| 9-3 | Riccati 参数化与自旋/多带扩展的精确形式 | v1 只给结论，未展开 | Schopohl–Maki 原始文献 |
| 9-4 | BTK 的 `Z` 与 KL 的 `zeta` 的对应关系 | 直接影响"什么时候该用哪个模型"的判断 | 专门推导或在明确极限下数值对照 |
| 9-5 | 正则化：`ln(Tc/T)` 形式 vs 显式 `omega_c` + `N(0)V` 的数值等价性 | 两者在强耦合/近邻效应下的截断误差不同 | 数值对照 |
| 9-6 | `Gamma`（Dynes）的引入方式 | 与 `btk.py` 的 `Gamma` 语义需一致，否则两个模块的拟合结果无法对照 | 用同一组 `(Delta, Gamma, T)` 对拍 BTK 与 Usadel 的隧穿极限 |
| 9-7 | R-4 的常数 3.1 与 `L`/`xi` 定义 | 记忆型常数，必须落到文献公式 | 查 SNS minigap 的原始结果 |
| 9-8 | 是否把环境/`P(E)` 纳入本模型（v1 默认**不含**）；若纳入需新增 `R`、`C`（或 `Z_t(omega)`）参数 | BTK 与 Usadel 都**不含环境**；不纳入时"能隙边整体位移"会被误当成 `Delta` 或 `Gamma` 的拟合误差 | 用实际 STM 线路估 `E_c = e^2/(2C)` 与 `R`，与 `Delta`、`R_K` 比；`E_c ≪ Delta` 且 `R ≪ R_K` 就整节跳过 |
| 9-9 | §6.2 的"隧道谱 ∝ 态密度"前提声明是否足够（Ingold–Nazarov 附录 A.5 的警告） | 关系到能否把拟合出的 `N(E)` 直接解释为态密度；有相互作用时这个解释有原理性限制 | 在 §6.2/§6.4 明确写出适用条件，并在报告里用"有效态密度"措辞 |

---

## 10. 模块接口草案（未来实现，供讨论）

单位与 `btk.py` 一致（eV / K / nm），API 草案：

```python
class Usadel1D:
    def __init__(self, Delta0, D, T, Tc=None, Gamma=0.0, ...): ...
    def solve_gap(self, geometry, interface, ...) -> "delta_x"     # 自洽 Delta(x)（松原分支）
    def dos(self, energy, position) -> "N"                          # 局域态密度（实时分支）
    def tunnel_spectrum(self, V, position=None) -> "(V, dIdV)"      # 隧道谱（§6.2）
```

路线图（v2+，按性价比排序）：

1. 固定 `Delta` 的 1D 实时解 + `N(E,x)` + 隧道谱（最小可用版）；
2. 自洽 `Delta(x)`（松原分支）+ S/N、SNS 能隙剖面；
3. Riccati 参数化（数值稳定性）；
4. 点接触/扩散结电导（§6.3）；
5. 含时/Keldysh Usadel（MAR、非平衡分布函数）——**另立文档**。

**不重复造轮子的选项**（实现前先评估）：JYU 的 `usadel1`（1D，含自旋与非平衡分布函数；注意其年代久远，Python 3.14 兼容性要先验）；`SuperConga` 是 2D **Eilenberger**（不是 Usadel），只在需要 2D/涡旋几何时才考虑。

---

## 11. 参考文献（v2 逐条核对卷号页码）

1. K. D. Usadel, *Generalized diffusion equation for superconducting alloys*, Phys. Rev. Lett. **25**, 507 (1970).
2. G. Eilenberger, Z. Phys. **214**, 195 (1968).
3. M. Yu. Kupriyanov, V. F. Lukichev, Sov. Phys. JETP **67**, 1163 (1988). （KL 边界条件）
4. W. Belzig, F. K. Wilhelm, C. Bruder, G. Schön, A. D. Zaikin, Superlattices Microstruct. **25**, 1251 (1999). （准经典理论综述）
5. A. A. Golubov, M. Yu. Kupriyanov, E. Il'ichev, Rev. Mod. Phys. **76**, 411 (2004). （约瑟夫森结电流相位关系）
6. P. Virtanen 等，`usadel1`（JYU，1D Usadel + 分布函数）：<https://gitlab.jyu.fi/jyucmt/usadel1>
7. G.-L. Ingold, Yu. V. Nazarov, *Charge Tunneling Rates in Ultrasmall Junctions*, in: H. Grabert, M. H. Devoret (eds.), *Single Charge Tunneling: Coulomb Blockade Phenomena in Nanostructures*, NATO ASI Series B: Physics **294**, Plenum Press, New York (1992), Ch. 2, pp. 21–107（含附录 A "Microscopic foundation", pp. 91–107）。
   - 本仓库解析件：`/Users/hunfen/Zotero/llm-for-zotero-mineru/4499/full.md`（注意：该解析件覆盖**整本书**，第 2 章对应 `full.md` 第 717–3296 行；`P(E)` 定义/通性在 §3.2.3–§3.4、高阻抗极限在 §3.7、含环境准粒子电流在 §5.4、微观基础与"隧道谱≠态密度"的警告在附录 A）。

---

## 12. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-09-23 | 首版草案：约定、矩阵与 theta 方程、自洽方程、KL 边界、观测量、数值方案、验证锚点、待核对清单 |
| v1.1 | 2026-09-23 | 参考 Ingold–Nazarov (1992)：新增 §6.4 环境效应/`P(E)` 理论（判据、`P(E)` 三条恒等式、含环境准粒子电流式、四个极限）；§6.2 补"隧道谱≠态密度"的前提声明；新增 R-8/R-9 与待核对项 9-8/9-9；补参考文献 7 |
| v1.2 | 2026-09-23 | 本规范落地为 `src/stm_data_processing/utils/usadel.py`（接口文档 `docs/stm_data_processing/utils/usadel.md`）；新增 §13 实现与验证状态（质量记录、§8 锚点落地情况、§9 逐条状态） |

---

## 13. 实现与验证状态（2026-09-23）

**已实现**：`src/stm_data_processing/utils/usadel.py` —— 1D 稳态 Usadel（本节 §1 约定 / §3 方程 / §4 自洽方程 / §6 观测量），对称 `tau1` 约定，松原分支实 `theta` 自洽 + 实时分支 Riccati `gamma = tan(theta/2)` + `solve_bvp`。
配套：接口文档 [`../stm_data_processing/utils/usadel.md`](../stm_data_processing/utils/usadel.md)、回归门 `tests/regression/check_usadel.py`（6 项）、演示图 `scripts/usadel_sts_demo.py` → `var/usadel_demo/`。

**质量记录**：

| 环节 | 产物 | 结论 |
|---|---|---|
| 独立验证 | `var/usadel_verify/report.md`（V1–V6） | 全部 passed；含"改坏数值路径必须变红"的反面检查 |
| 评审 | `var/usadel_review/report.md` | verdict = **pass**（独立重推导 Riccati 变换、逐像素复核演示图） |
| 加固 | `tests/regression/check_usadel.py` | 补了走**数值路径**的块体极限用例；原 A1/A3/A4 因 `geometry="bulk"` 直接返回解析值而属自证，已改名为 `*_analytic_path` |
| 演示图加固 | `scripts/usadel_sts_demo.py` + `var/usadel_demo/` | 三轮（t4/t6/t7）：补 `0.99 Tc` 曲线与相干峰截断标注、fig03 展宽 `0.1 Delta0 -> 1e-3 Delta0`（minigap 才可见）、fig05 四条曲线四种线型 + `(1+Z^2)` 同口径 + 图例移出峰区、fig03 加 `E_g` 放大面板、标注统一移到坐标区外的底部留白 |
| 图像验收 | 独立图像 agent 的**像素级复核**（PNG 解码 + 刻度线反算数据坐标） | 抓出三条真缺陷并确认修复：fig05 只有 3 种线型（A-R8 不达标）、fig05 隙缘峰无标注截断、fig03 标注被图例截断。结论：fig05 四线型 / 截断标注（数字独立复算 4.000 与 19.964）/ 图例距峰 41 px / fig03 `E_g` 标记 均 VERIFIED。**演示图这类"给人看"的交付物，用像素复核而不是实现者自述来验收** |

**§8 锚点的落地情况**：R-1 / R-2 / R-3 由解析与数值两条路径共同覆盖（数值路径：深 S `sn` 几何，`L_S = 30 xi`）；R-4 实测 S/N 双层长结极限 `E_g -> 0.78 E_Th`（Zhou et al. 1998），与本节所记的 SNS `3.12 E_Th` 通过 factor-4 几何关系（SNS 长度 `L` ↔ SN 长度 `L/2`）一致；R-5 由数值块体复现覆盖；**R-6 未实现**（无相位/电流）；R-7 作为适用域边界记录；**R-8 / R-9（`P(E)` 环境）未实现**。

**§9 待核对清单的当前状态**：

| # | 状态 |
|---|---|
| 9-1 KL 前因子 | **未落地**：v1 用宽度 `0.05 xi` 的平滑 `Delta` 过渡代替 KL 界面，前因子仍未标定；要支持界面电阻须先补 KL |
| 9-2 约定对齐 | **已闭合**：实现按对称 `tau1` 约定；评审独立重推导 Riccati 变换逐位一致 |
| 9-3 Riccati 形式 | **已闭合（实现层）**：`gamma = tan(theta/2)` + `solve_bvp`；与 Schopohl–Maki 原文的逐条对齐仍可后续补 |
| 9-4 BTK 的 `Z` ↔ KL 的 `zeta` | **仍未闭合**（依赖 KL 界面实现） |
| 9-5 正则化等价性 | **部分闭合**：实现用 `ln(Tc/T)` 形式；验证以独立积分式参考比对 `Delta(T)`，5 个 `T/Tc` 点偏差 < 1.4e-4 |
| 9-6 Dynes `Gamma` 语义 | **已核清，但两模块不一致**（重要）：`usadel.bcs_dos` 用 `E -> E + i*Gamma`（全能量范围，**亚能隙会被填隙**，Dynes 式）；而 `btk.py` 的 `Gamma` 只出现在超能隙分支，**亚能隙不填隙**。跨模块对照拟合前必须先统一这一点 |
| 9-7 minigap 常数 | **已闭合**：S/N 双层长结 `0.78 E_Th` ↔ SNS `3.12 E_Th`，经 factor-4 几何关系 |
| 9-8 是否纳入环境 `P(E)` | **仍开放**（v1 不含；判据见 §6.4：`E_c << Delta` 且 `R << R_K` 时可整节跳过） |
| 9-9 "隧道谱 ≠ 态密度" | **已写进文档**：§6.2 与接口文档均采用"隧道谱反演出的有效态密度"措辞 |

**已知数值限制**（细节见接口文档）：`solve_gap(geometry="sn")` 的松原求和截断 `omegac = 20 k_B Tc`（自洽深 S 区 `Delta` 相对 bulk 偏约 0.24%）；实时 BVP 在 `abs(E) = Delta0` 且 `Gamma -> 0` 时脆弱；A5 minigap 阈值步长 `0.0025 Delta0`。
