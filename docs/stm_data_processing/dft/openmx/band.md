# OpenMX Band 解析模块接口文档

## 模块概述

`band.py` 提供 OpenMX 能带结构文件（`.Band`）的解析功能，将 OpenMX 输出的离散能带数据转换为便于绘图的结构：沿高对称路径的累计 k 距离、以费米能级为零点的能带能量，以及高对称点的刻度位置与标签。

**输入格式**: OpenMX 的 `.Band` 文件（文本格式，以 ISO-8859-1 编码读取），其结构为：

- 第 1 行：能带数 `nband`、自旋相关标志、费米能级 μ（Hartree）
- 第 2 行：倒格矢 b1、b2、b3（各 3 个分量，单位 a.u.）
- 第 3 行：k 路径条数 `n_kpaths`
- 后续若干行：每条 k 路径一行，格式为 `num_kpts start_frac_x start_frac_y start_frac_z end_frac_x end_frac_y end_frac_z start_label end_label`
- 再后续：能带数据——每个 k 点为一行 `nband kx ky kz`（kx/ky/kz 为分数坐标），其后紧跟 `nband` 个能量本征值（单位 Hartree）

**输出格式**: 一个 `dict`，包含 `dist`、`bands`、`tick_pos`、`tick_label`、`kpts_frac`、`kpts_cart`、`fermi_energy`、`n_bands` 八个键。

**单位约定**: 能量由 Hartree 转换为 eV 并以 E_F = 0 为参考；k 距离与倒格矢由 a.u. 转换为 Å 与 1/Å。

---

## 核心函数

### `parse_dft_band_data`

```python
def parse_dft_band_data(
    fname_band: str | None = None,
    folder: str | None = None,
    systemname: str | None = None,
) -> dict[str, np.ndarray | list]
```

解析 OpenMX `.Band` 文件，返回能带结构数据字典。

| 参数 | 类型 | 说明 |
|------|------|------|
| `fname_band` | `str \| None` | `.Band` 文件路径（如 `C6LiC6.Band`）。若为 `None` 且给定 `folder`/`systemname`，则自动构造 `folder/systemname.Band` |
| `folder` | `str \| None` | 存放 OpenMX 文件的目录 |
| `systemname` | `str \| None` | OpenMX 系统名（System.Name） |

**返回**: `dict[str, np.ndarray | list]`，键说明如下：

| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `dist` | `np.ndarray` | `(nk_total,)` | 沿 k 路径的累计距离（1/Å） |
| `bands` | `np.ndarray` | `(nk_total, nband)` | 能带能量（eV，E_F = 0） |
| `tick_pos` | `list[float]` | `(n_tick,)` | 高对称点在 `dist` 上的 x 坐标 |
| `tick_label` | `list[str]` | `(n_tick,)` | 高对称点标签（与 `tick_pos` 等长） |
| `kpts_frac` | `np.ndarray` | `(nk_total, 3)` | 分数 k 坐标 |
| `kpts_cart` | `np.ndarray` | `(nk_total, 3)` | 笛卡尔 k 坐标（1/Å） |
| `fermi_energy` | `float` | - | 费米能级（eV） |
| `n_bands` | `int` | - | 能带数 |

---

### `openmx_band_analysis`

```python
def openmx_band_analysis(
    band_file: str | None = None,
    folder: str | None = None,
    systemname: str | None = None,
) -> dict[str, np.ndarray | list]
```

解析 OpenMX 能带结构数据，是 `parse_dft_band_data` 的便捷封装。

| 参数 | 类型 | 说明 |
|------|------|------|
| `band_file` | `str \| None` | `.Band` 文件路径。若提供，优先级最高 |
| `folder` | `str \| None` | 存放 OpenMX 文件的目录 |
| `systemname` | `str \| None` | OpenMX 系统名 |

**返回**: 与 `parse_dft_band_data` 完全相同的字典结构。

**参数解析逻辑**:

1. 若 `band_file` 不为 `None`，直接调用 `parse_dft_band_data(fname_band=band_file)`；
2. 否则若 `folder` 与 `systemname` 均不为 `None`，调用 `parse_dft_band_data(folder=folder, systemname=systemname)`；
3. 否则抛出 `ValueError`。

---

## 数学公式

能量与费米能级的单位换算（Hartree → eV）：

```
E(eV)   = (ε_au − μ_au) × h2ev
E_F(eV) = μ_au × h2ev
```

其中 `h2ev = 27.211386245988`。倒格矢由 a.u. 转换为 1/Å：

```
b_i(1/Å) = b_i(au) × au2ang
```

其中 `au2ang = 1.8897261254578281`。笛卡尔 k 坐标由分数坐标得到：

```
k_cart = k_frac · b，  b = [b1; b2; b3]
```

沿路径的累计 k 距离：

```
dist_0 = 0
dist_j = dist_{j−1} + ‖k_cart,j − k_cart,j−1‖
```

---

## 使用示例

### 直接指定 `.Band` 文件

```python
from stm_data_processing.dft.openmx.band import parse_dft_band_data

data = parse_dft_band_data(fname_band="./C6LiC6.Band")

print(data["bands"].shape)       # (nk_total, n_bands)
print(data["fermi_energy"])      # 费米能级 (eV)
print(data["tick_label"])        # 高对称点标签，如 ['G', 'K', 'M', 'G']
print(data["tick_pos"])          # 对应累计距离位置
```

### 通过目录与系统名自动构造路径

```python
from stm_data_processing.dft.openmx.band import parse_dft_band_data

# 等价于读取 ./work/C6LiC6.Band
data = parse_dft_band_data(folder="./work", systemname="C6LiC6")
```

### 使用便捷封装

```python
from stm_data_processing.dft.openmx.band import openmx_band_analysis

data = openmx_band_analysis(band_file="./C6LiC6.Band")
# 或
data = openmx_band_analysis(folder="./work", systemname="C6LiC6")
```

### 绘制能带图

```python
import matplotlib.pyplot as plt
from stm_data_processing.dft.openmx.band import openmx_band_analysis

data = openmx_band_analysis(band_file="./C6LiC6.Band")

plt.figure(figsize=(6, 8))
for n in range(data["n_bands"]):
    plt.plot(data["dist"], data["bands"][:, n], "k-", lw=1.0)

for x in data["tick_pos"][1:-1]:
    plt.axvline(x, color="gray", lw=0.5, ls="--")
plt.xticks(data["tick_pos"], data["tick_label"])
plt.ylabel("Energy (eV)")
plt.axhline(0.0, color="gray", lw=0.5)
plt.show()
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 数组运算、向量/矩阵乘法 |
| `pathlib` | 是 | 文件路径处理（标准库） |
| `logging` | 是 | 日志输出（标准库） |
| `contextlib` | 是 | `suppress` 忽略非数值 token（标准库） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | 未提供 `fname_band` 且未同时提供 `folder` 与 `systemname`；`.Band` 文件为空；文件头或倒格矢行、k 路径行格式非法；`openmx_band_analysis` 中既未提供 `band_file` 也未提供 `folder`+`systemname` |
| `FileNotFoundError` | 指定的 `.Band` 文件不存在 |
| `RuntimeError` | 高对称点位置数与标签数不一致（`tick_pos` 与 `tick_label` 长度不匹配） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`parse_dft_band_data` 需提供 `fname_band`，或同时提供 `folder` 与 `systemname`**
- [ ] **返回字典键为 `dist`/`bands`/`tick_pos`/`tick_label`/`kpts_frac`/`kpts_cart`/`fermi_energy`/`n_bands`**
- [ ] **`bands` 以 E_F = 0 为参考，单位 eV，形状 `(nk_total, nband)`**
- [ ] **`dist`、`kpts_cart` 单位为 1/Å（由 a.u. 转换而来）**
- [ ] **`tick_pos` 与 `tick_label` 等长，用于 `plt.xticks`**
- [ ] **`.Band` 文件头第 1 行含能带数与费米能级（Hartree），第 2 行含倒格矢（a.u.）**

---

## 版本信息

- 模块路径：`src/stm_data_processing/dft/openmx/band.py`
- 日志级别：`INFO`（自动探测/指定文件路径、能带数、费米能级、k 点数、能量范围）
- 单位换算常量：`h2ev = 27.211386245988`（Hartree→eV）、`au2ang = 1.8897261254578281`（Bohr→Å）
- **无环境变量配置**
- **输入格式：OpenMX `.Band` 文本文件**
- **输出格式：能带数据字典，能量以 E_F = 0 为参考**
