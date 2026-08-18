# QPI 工具函数模块接口文档

## 模块概述

- **模块路径**：`src/stm_data_processing/utils/miscellaneous.py`
- **职责**：提供 QPI（Quasiparticle Interference）数据处理中频繁复用的底层工具函数：分数坐标 → 实空间倒格坐标转换、FFT 倒空间范围计算、QPI 扩展/裁剪、k → q 映射以及费米-狄拉克分布。
- **后端**：导入时从 `stm_data_processing.config` 读取 `BACKEND`；`BACKEND == "gpu"` 时条件导入 `cupy`（别名 `cp`），否则 `cp = None`。
- **性质**：全部为纯函数，无文件读写等副作用，可安全 `import`。

---

## 核心函数

### `frac_to_real_2d(grid1, grid2, bvecs)`

将无量纲分数坐标网格映射为物理倒空间坐标（Å⁻¹）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `grid1`, `grid2` | `np.ndarray` | 同形状 2D 数组，沿倒格矢 b₁、b₂ 方向的分数坐标系数 |
| `bvecs` | `np.ndarray \| None` | 倒格基矢 `(2, 2)` 或 `(3, 3)`，单位 Å⁻¹，仅用前两行两列；`None` 时跳过转换 |

**返回**：`(gridx, gridy)`，物理倒空间坐标网格（Å⁻¹）；`bvecs` 为 `None` 时二者均为 `None`。转换关系：`gridx = grid1·bvecs[0,0] + grid2·bvecs[1,0]`，`gridy = grid1·bvecs[0,1] + grid2·bvecs[1,1]`。

### `fft_q_limits(frame_size_nm, n)`

计算 FFT 输出倒空间范围（Å⁻¹）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `frame_size_nm` | `float` | 实空间扫描框边长（nm，假定正方形） |
| `n` | `int` | 每维像素数（`n × n` 网格） |

**返回**：`([qx_min, qx_max], [qy_min, qy_max])`，均为 `[-q_max, q_max]`（Å⁻¹）。公式：`q_max = π·n / (frame_size_nm·10.0)`（系数 10 用于 nm → Å）。

### `extend_qpi(qpi_layers, q1_base, q2_base, qmin, qmax)`

周期平移扩展 QPI 数据并精确裁剪到 `[qmin, qmax)`，严格保持 q 点密度。支持 2D `(nk, nk)` 与 3D `(nband, nk, nk)`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `qpi_layers` | `np.ndarray` | QPI 数据，形状 `(nk, nk)` 或 `(nband, nk, nk)` |
| `q1_base`, `q2_base` | `np.ndarray` | 基础分数坐标网格 `(nk, nk)` |
| `qmin`, `qmax` | `float` | 分数坐标裁剪边界（左闭右开） |

**返回**：`(qpi_ext, q1_ext, q2_ext)`，扩展并裁剪后的 QPI 数组与坐标网格，维度与输入一致（2D 进 2D 出）。

### `crop_qpi(qpi_layers, q1_base, q2_base, qmin, qmax)`

`extend_qpi` 的逆操作，将扩展数据裁剪回基本布里渊区（或指定范围）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `qpi_layers` | `np.ndarray` | `extend_qpi` 输出的扩展 QPI 数据 |
| `q1_base`, `q2_base` | `np.ndarray` | `extend_qpi` 输出的扩展坐标网格 |
| `qmin`, `qmax` | `float` | 目标裁剪边界（恢复原数据通常取 `-0.5, 0.5`） |

**返回**：`(qpi_crop, q1_crop, q2_crop)`。若裁剪范围内无数据点则抛 `ValueError`。

### `k_to_q(k1_grid, k2_grid)`

将 k 空间网格转为以零为中心的无量纲 q 空间网格。

| 参数 | 类型 | 说明 |
|------|------|------|
| `k1_grid`, `k2_grid` | `np.ndarray` | 2D k 空间坐标网格 `(nkx, nky)` |

**返回**：`(q1_grid, q2_grid)`，同形状无量纲 q 网格（`meshgrid(..., indexing="ij")`）。

### `fermi(e, mu=0, T=1.5)`

费米-狄拉克分布（CPU/NumPy 版）。

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `e` | `array_like` | - | 能量值 |
| `mu` | `float` | `0` | 化学势（单位同 `e`） |
| `T` | `float` | `1.5` | 温度（K） |

**返回**：`array_like` 占据数。`T <= 1e-12` 时退化为 `(e < mu).astype(float)`。

### `fermi_cuda(energies, mu, T)`

费米-狄拉克分布（GPU/CuPy 版），仅 `BACKEND == "gpu"` 时 `cp` 可用。

| 参数 | 类型 | 说明 |
|------|------|------|
| `energies` | `cp.ndarray` | 能量值（eV） |
| `mu` | `float` | 化学势（eV） |
| `T` | `float` | 温度（K） |

**返回**：`cp.ndarray`。`T <= 1e-12` 时返回 `cp.where(energies <= mu, 1.0, 0.0)`，并对 `x` 裁剪到 `[-50, 50]` 防 `exp` 溢出。

---

## 数学公式

### 费米-狄拉克分布

```
f(e) = 1 / (1 + exp((e - μ) / k_B T))，k_B = 8.617333262145e-5 eV/K（硬编码）
```

- `fermi` 用 `scipy.special.expit(-x)`，`x = (e - mu)/kT`，即 `expit(-x) = 1/(1 + exp(x))`。
- `fermi_cuda` 直接算 `1/(1 + exp(clip(x, -50, 50)))`。

> ⚠️ T=0 边界差异：`fermi` 用严格 `<`（`e == mu` → 0），`fermi_cuda` 用 `<=`（`e == mu` → 1.0）。

### k → q 映射

```
dk1 = k1_grid[1,0] - k1_grid[0,0]（单行时 1.0）；dk2 = k2_grid[0,1] - k2_grid[0,0]（单列时 1.0）
q1_vals = (arange(nkx) - nkx//2) · dk1
q2_vals = (arange(nky) - nky//2) · dk2
(q1_grid, q2_grid) = meshgrid(q1_vals, q2_vals, indexing="ij")
```

### QPI 扩展 / 裁剪索引逻辑

1. **扩展**：`n_min = floor(qmin + 0.5)`、`n_max = ceil(qmax - 0.5)`、`shifts = arange(n_min, n_max + 1)`；`(nk, nk)` 块沿两方向重复 `len(shifts)` 次得 `nq_big = nq·len(shifts)`，坐标块为 `q1_base + sx`、`q2_base + sy`。
2. **裁剪**：`mask_x = (q1_big[:,0] >= qmin) & (q1_big[:,0] < qmax)`，`mask_y` 同理作用于 `q2_big[0,:]`，左闭右开。

`crop_qpi` 复用同一掩码逻辑直接裁剪以恢复原密度。

---

## 使用示例

```python
import numpy as np
from stm_data_processing.utils.miscellaneous import extend_qpi, crop_qpi, fermi, k_to_q

# QPI 扩展后裁剪回基本布里渊区
nk = 32
q1, q2 = np.meshgrid(np.linspace(-0.5, 0.5, nk), np.linspace(-0.5, 0.5, nk), indexing="ij")
qpi = np.random.rand(nk, nk)
qpi_ext, q1_ext, q2_ext = extend_qpi(qpi, q1, q2, -1.5, 1.5)
qpi_back, q1_back, q2_back = crop_qpi(qpi_ext, q1_ext, q2_ext, -0.5, 0.5)  # (32, 32)

# k → q
k1, k2 = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8), indexing="ij")
qq1, qq2 = k_to_q(k1, k2)

# 费米分布
occ = fermi(np.linspace(-0.1, 0.1, 5), mu=0.0, T=4.2)
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心数组计算 |
| `scipy.special.expit` | 是 | `fermi` 的 sigmoid 实现 |
| `cupy` | 否 | GPU 后端（仅 `BACKEND == "gpu"`） |
| `stm_data_processing.config` | 是 | 提供 `BACKEND` 常量 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `extend_qpi`/`crop_qpi` 中 `qpi_layers` 维度非 2 或 3 |
| `ValueError` | `crop_qpi` 在 `[qmin, qmax)` 内无数据点 |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`qpi_layers` 形状为 `(nk, nk)` 或 `(nband, nk, nk)`，否则抛 `ValueError`**
- [ ] **`extend_qpi` 输出维度与输入一致（2D 进 2D 出）**
- [ ] **`qmin`/`qmax` 为分数坐标，边界左闭右开 `[qmin, qmax)`**
- [ ] **`crop_qpi` 的 `qmin`/`qmax` 须与扩展网格覆盖范围匹配**
- [ ] **`bvecs` 传 `None` 时 `frac_to_real_2d` 返回 `(None, None)`，调用方需判空**
- [ ] **`fermi` 温度单位为 K，化学势单位与能量一致**
- [ ] **`fermi_cuda` 仅接受 `cp.ndarray`，需 `BACKEND == "gpu"` 才有 `cp` 可用**
- [ ] **T=0 时 `fermi`（`<`）与 `fermi_cuda`（`<=`）在 `e == mu` 处结果不同**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/miscellaneous.py`
- 后端：继承自 `stm_data_processing.config.BACKEND`（导入时自动检测，`"cpu"` 或 `"gpu"`）
- 日志级别：无（静默计算）
- **无环境变量配置，纯函数实现，可安全 import**
