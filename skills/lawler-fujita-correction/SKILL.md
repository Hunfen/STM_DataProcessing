---
name: lawler-fujita-correction
description: Lawler–Fujita 晶格相位矫正（Fujita et al., PNAS 2014, SI §4）：正方形 STM 拓扑 CSV + 视场 L（nm）→ 在 1x1 环上做双（三）方向 lock-in 得到局域晶格相位，解出逐点位移场 u(r)（nm）并按 u 重采样整张图；输出矫正 CSV/HDF5 数组（corrected + fft2）/复数 FFT2 npy/预览图/报告 + 相位·幅度·位移·掩码的单一 `<stem>_lf.h5`（每张图另有 png 预览），并可导出「可迁移矫正包」（变换 JSON + u 场 h5）套用到同一扫描的其它图（含不同像素数）。当用户给出拓扑图 CSV 与图像边长（nm）并要求 Lawler–Fujita / 位移场 / 局域相位矫正、或要求把某张图的位移场矫正套用到另一张图时使用。
---

# lawler-fujita-correction（Lawler–Fujita 晶格相位矫正，v1.0）

**范围**：本 skill 只做**几何矫正数学**——局域 lock-in 相位提取、位移场求解、按位移场重采样、
矫正后自检与可迁移包导出。**物理解释由使用者进行**：脚本、图、log 与文档都不给物理结论。

**与 affine-correction 的分工**：`affine-correction` 用**一个全局对称正定拉伸**（仿射、6 参数以内）
去掉整幅图的畸变；本 skill 用**逐点位移场 u(r)**（每个像素两个自由度）去掉任意缓慢畸变——
对应 Fujita 2014 SI §4 的算法。二者输出契约同类（矫正 CSV + 复数 FFT2 + 报告 + 契约行），
可接同一套下游（例如 `phase-analysis`）。

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | 正方形拓扑数值矩阵 CSV/txt（可含 NaN）。**第一行 = 扫描起始行 = 图像顶部**（脚本内部 `flipud(subtractMeanPlane(...))` 转到数组坐标，与 affine skill 一致） |
| 视场 L (nm) | 正方形扫描边长，**必须显式给出**（文件名里的标注不可信） |
| 参考晶格 | 六方 1x1 环，晶格常数 `a`（默认 0.246 nm）；**半径取理想值 `2L/(√3a)` px，取向取数据自身**（见 §2.3） |
| 锚定环 | 数据锚定峰检测（`bragg_peak.detect_bragg_peaks`，与 affine skill 同一入口）找到的、半径与理想 1x1 环相符（`--ring-tol`，默认 10%）且 ≥6 成员的环 |
| λ（lock-in 尺度） | `--lambda-nm`，默认 30.0 nm：q 空间高斯低通 `exp(-λ²k²/2)` 保留的畸变尺度 |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`。**不要用 `uv run`** |
| 绘图 | `text.usetex=False`（mpl 3.10 + TeX Live 2026 有编码 bug），mathtext cm + Palatino |

## 2 矫正流程（`scripts/stm_lf_correct.py`）

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 拟合 + 写出可迁移包
.venv/bin/python skills/lawler-fujita-correction/scripts/stm_lf_correct.py \
  INPUT.csv -L 50 -o OUT --a 0.246 --lambda-nm 30 \
  --save-transform OUT/bundle.json

# 第二段（可选）：把同一位移场套用到同一扫描的另一张图（可不同像素数）
.venv/bin/python skills/lawler-fujita-correction/scripts/stm_lf_apply.py \
  TARGET.csv --transform OUT/bundle.json -L 50 -o OUT_TARGET
```

### 2.1 模型与约定（脚本内固定，自检端到端验证）

```
T(r)     = ideal(r - u(r))                  实测图；u = 晶格位移场
psi_i(r) = lowpass[ T(r) exp(-i Q_i . r) ]  第 i 个方向的复 lock-in 场
theta_i  = arg psi_i(r) = -Q_i . u(r)       局域晶格相位
u(r)     = Q^-1 (theta_bar - theta(r)),  theta_bar = 0（规范：有效像素上相位均值为 0）
corrected(r) = T(r + u(r))                  沿位移场重采样（即把畸变点阵映回理想点阵）
```

`Q` 是 2×2 矩阵，两行是两个参考波矢（rad/nm，`K = 2π q_px / L`）；`theta` 是两相位列向量。
**符号约定由自检固定**（要求矫正后图像就是理想点阵），不是任选：φ 与 u 的符号对调也能自洽，
但只有 `u = Q⁻¹(θ̄ − θ)` + `T(r + u)` 这一组能让矫正后的 1×1 环各向异性从 1.7 % 降到 0.01 %。

### 2.2 lock-in（局域相位）

对每个方向：`psi = lowpass[T · exp(-i Q·r)]`，低通是 **q 空间高斯 `exp(-λ²|k|²/2)`**，
即论文中半径 `δq = 1/λ` 的圆盘（λ = 保留的畸变实空间尺度）。实现细节：

- 低通在**未位移的 FFT 索引网格**上用环绕距离做高斯权重，避免 `fftshift` 插值误差；
- 相位解缠用**最小二乘（Poisson + DCT，Neumann 边界）**，不用逐行逐列顺序解缠：复数场存在
  零点（相位结点），顺序解缠会把一个 2π 位错沿整行传出去，最小二乘把影响限制在局部；
- 解缠前把幅度低于阈值的像素用最近可靠像素的**折叠相位**填充（同 §2.4 掩码），避免结点处
  的伪相位进入解缠。

**可解性条件**（写进 SKILL.md 是必要的，否则 λ 会被误用）：
lock-in 只能保留满足 `|∇θ| = |K||∇u| < 2π/λ` 的畸变结构，即

```
|grad u|  <  2 pi / (|K| lambda)        (|K| = 4 pi / (sqrt(3) a) = 29.49 nm^-1, a = 0.246 nm)
```

λ = 30 nm 对应 **0.7 % 应变**上限；要保留更大的畸变必须减小 λ。λ 也不能太小：q 空间掩码半径
`1/λ` 一旦接近邻近 Bragg 峰间距 `|K|`（约 4.7 cycles/nm），邻近环的信号会漏进来（实测 λ ≲ 1 nm
时第三个方向的相位出现 100° 量级错误）。

### 2.3 参考六方（Q_a、Q_b、Q_c 的来源）

- 数据锚定检测给出 1x1 环的 6 个成员（测得峰位，被平均畸变整体平移过）；
- **半径取理想值**（由 `--a` 决定），**取向取数据**：把三个成员相对「恰好 120° 间隔」的平均偏差
  加到最强成员的方位角上，得到参考六方的取向；
- 三个方向按 `--pair-angle`（默认 120°）**严格等间隔**放置，于是 `Q_a + Q_b + Q_c = 0`
  **精确成立**，第三方向自检检验的是 lock-in 而不是参考几何；
- `--pair-angle 60` 时不存在第三个方向：`-(θ_a + θ_b)` 不是任何环成员的相位，此时 log 与报告
  明说只有 **2** 个 lock-in 方向、间距 60°、`Q_a + Q_b` 不闭合，报告 `third_direction.available = false`，
  也不写 `theta_c` / `amplitude_c` 产物。

log 与报告同时给出**测得峰位**与**实际使用的参考波矢**（`q_measured_px` / `q_a_px` 等），
便于核对参考与数据的差异。

### 2.4 相位规范与仿射规范自由度

`θ̄ = 0`：有效像素上的局域相位均值被减掉，等价于「理想点阵在原点有原子峰（平均 Bragg 峰虚部为 0）」。

**规范自由度（必须写明）**：位移场在「刚性平移 + 参考六方的一个均匀应变」下不可分辨——
数据只给出参考取向，平均剪切会把它偏一点。因此

```
u(r)  ~  u_true(r) + a + S r      (a 平动, S 均匀应变，由参考选择固定)
```

自检因此同时报「原始误差」与「去掉该仿射规范后的误差」（实测 0.054 nm vs 0.0031 nm）。
真实数据里这意味着：本 skill 给出的是**相对参考晶格**的位移场，参考的均匀应变部分由调用者
用 `--a`（半径）与数据取向共同确定，不是绝对应变测量。

### 2.5 有效掩码与边界

- **掩码**：`|psi_i| < amplitude_fraction × median(|psi_i|)`（`--amplitude-fraction`，默认 0.10）
  的像素标为无效（三个方向取交集）；无效像素不施加位移（原样拷贝），掩码随包传递。
- **边界**：FFT 把图像当周期函数，靠近边界的解调相位会混入对面边缘的数据，实测污染深度约
  `0.65 λ` px。因此**边界不作硬性剔除**（那会在大 λ 时把整幅图判死），而是在第三方向诊断里
  剔除 `0.65 λ` 的边界带（报告中 `wrapped_rms_deg_interior` 与 `wrapped_rms_deg` 都给）。
  **λ 必须远小于视场 L**：λ = 30 nm 配 50 nm 视场时，只有中心区域可信。
- **带宽自检**：参考波矢（理想半径）与**测得**峰位之间总有偏移
  `offset = |q_measured − q_ref|/L`（来自数据的平均应变）。lock-in 掩码半径是 `1/λ`，
  因此必须 `λ · offset < 1`，否则掩码里根本没有真正的 Bragg 峰，解出的相位/位移场不是晶格畸变。
  脚本在 log 打印 `offset` 与 `λ·offset`，超过 1 时报 WARNING，并在报告里记
  `lockin.reference_offset_cycles_per_nm` / `band_margin_lambda_times_offset` / `band_warning`。

## 3 输出与接口契约

输出（`-o OUT`）：

| 产物 | 说明 |
| --- | --- |
| `<stem>_corrected.csv` | 矫正后拓扑（正方形，`%.10e` 逗号分隔，NaN 填充） |
| `<stem>_corrected.h5` | HDF5 数组产物：`corrected`（`float64`，与 CSV 同一数组）+ `fft2`（`complex128`，与 `.npy` 同一数组）；本 skill 自包含的 `scripts/h5io.py` 按仓库 h5 约定写出（`schema_version = 1`、`generator`、`creation_date`，显式 chunk ≤ 1 MiB + `gzip`/4 + `track_times=False`；两个数组无物理单位故无 `units`） |
| `<stem>_corrected_fft2.npy` | 复数 FFT2（`complex128`，fftshift，Hanning 窗 + NaN 平面填充，与 affine skill 同口径）。**保留**：`skills/phase-analysis/` 通过 `--fft2` 读它，phase-analysis 不在本次迁移范围 |
| `<stem>_corrected.png` / `<stem>_corrected_fft.png` | gwyddion 拓扑图 / inferno 对数 FFT 图 |
| `correction.log` | 控制台报告（含下面两条契约行） |
| `correction_report.json` | 机器可读报告（见下） |

**跨 skill 契约行（勿改）**：`correction.log` 必须同时含

```
# canvas <n> x <n> px, field of view <L> nm (<L/n> nm/px)
# corrected canvas: <n_out> x <n_out> px, field of view <X> nm (<X/n_out> nm/px), NaN <..> %
```

`X = L · n_out / n`（nm/px 不变）；下游 `phase-analysis --size-nm-from-log` 解析第二条。

**Lawler–Fujita 附加产物**：相位/幅度/位移/掩码全部是**一个 HDF5 文件** `<stem>_lf.h5` 的
数据集（**文件名一律带 `_lf_` 标记**），每张图另有 `<stem>_lf_<name>.png` 预览；
不再写 per-map `.npy`。

| 数据集（`<stem>_lf.h5`） | 内容 | `units` | png 预览 |
| --- | --- | --- | --- |
| `theta_a` / `theta_b` / `theta_c` | 解缠后的局域相位（弧度，θ̄ = 0 规范） | `rad` | 同一相位折叠到 (−180, 180] 度（twilight 循环色标） |
| `amplitude_a` / `amplitude_b` / `amplitude_c` | lock-in 幅度 `|psi_i|` | 无（与输入拓扑同单位，非物理单位） | inferno |
| `u_x` / `u_y` | 位移场分量（**nm**；像素轴 x = 列、y = 行） | `nm` | RdBu_r（对称色标） |
| `mask` | 有效掩码（0/1，`float64`） | 无（无量纲） | gray |

三方向模式（默认 `--pair-angle 120`）写全 9 个数据集（9 张 png）；`--pair-angle 60` 时只有两个方向，
因此**不写** `theta_c` / `amplitude_c`（7 个数据集、7 张 png），log 与报告也会写明只有两个
lock-in 方向、间距 60°、不存在第三方向。核心产物 `_corrected.*`、`correction.log`、
`correction_report.json`、包文件名不受影响。

**`correction_report.json` 字段**（要点）：

| 字段 | 含义 |
| --- | --- |
| `skill` / `skill_version` / `input` / `canvas_px` / `field_of_view_nm` / `nm_per_px` | 溯源 |
| `a_nm` / `ideal_radius_px` / `ideal_radius_nm_inv` | 参考环 |
| `rings_before` / `anchor_ring` | 检测到的环表与锚定环 |
| `q_a_px` / `q_b_px` / `q_c_px` / `q_*_nm_inv` / `q_matrix_px` / `q_measured_px` | 参考波矢（px 与 rad/nm）与测得峰位 |
| `hexagon_orientation_deg` / `hexagon_member_deviation_deg` | 参考取向与成员偏离 120° 的量 |
| `lambda_nm` / `amplitude_fraction` / `lockin.*` | λ、幅度阈值、`amplitude_threshold`、`mask_coverage_fraction`、`max_phase_step_rad` |
| `gauge` | 规范说明（θ̄ = 0） |
| `displacement.u_stats_nm` | u_x/u_y/norm 的 rms 与 max（nm，有效像素上） |
| `third_direction.*` | `wrapped_rms_deg`（全掩码）、`wrapped_rms_deg_interior`、`interior_margin_px`、`interior_margin_rule`、`interior_pixels`；60° 配对时 `available = false` + `reason` |
| `residual_self_check.before/after` | 矫正前后重检测的 1x1 环各向异性（min/max 半径）与 1x1/r3 环对比值 |
| `canvas_growth_px` / `max_displacement_px` / `n_out` / `corrected_field_of_view_nm` / `corrected_nm_per_px` / `nan_fraction` | 画布与产物 |
| `method` / `fallback` / `fallback_reason` | `lawler_fujita`，或 `identity_fallback` + 原因 |
| `lf_artifacts` | 每张 LF 图的登记项：`h5`（`<stem>_lf.h5` 路径）+ `dataset`（数据集名）+ `png`（预览路径） |
| `written` / `transfer_constraints` | 核心产物路径与迁移约束说明 |

**回退语义**：找不到可用 1x1 环、或有效掩码占比低于 `--min-coverage`（默认 0.50）时，
`method = "identity_fallback"`、`fallback = true`：**不做任何矫正**，把输入原样（在其自身画布与
视场上、不补 NaN）复制输出，并在 log 打 WARNING。此时不写相位/位移产物，也不写 u 场 h5。

## 4 可迁移矫正包（`--save-transform FILE`）

Lawler–Fujita 矫正**是可以迁移的**，但迁移的是**稠密位移场**而不是全局矩阵——论文 SI 本身就把
同一张形貌图得到的矫正套用到同步测量的谱学数据上。包 = 一个 JSON + 一个 h5（同名、换后缀）：

```json
{
  "schema": "lawler-fujita-correction-transform", "schema_version": 1,
  "source_report": "...", "source_input": "...",
  "q_a_px": [...], "q_b_px": [...], "q_a_nm_inv": [...], "q_b_nm_inv": [...],
  "q_matrix_px": [[...], [...]], "lambda_nm": 30.0, "a_nm": 0.246,
  "amplitude_fraction": 0.10, "amplitude_threshold": 0.049,
  "gauge": "theta_bar = 0 ...",
  "n_px_reference": 1024, "n_out_reference": 1070,
  "field_of_view_nm_reference": 50.0, "nm_per_px_reference": 0.048828,
  "pad": 10, "order": 3, "method": "lawler_fujita", "fallback": false,
  "mask_coverage_fraction": 0.999, "u_stats_nm": {...},
  "u_field_file": "bundle.h5", "transfer_constraints": {...}, "usage": "apply with: ..."
}
```

h5 内容（`scripts/h5io.py`，仓库 h5 约定）：`u_x`、`u_y`（nm，参考网格，`units = "nm"`）、
`valid`（bool 掩码，无量纲，**无** `units` 属性）、以及 `field_of_view_nm_reference`（`units = "nm"`）、
`n_px_reference`（像素数，不是物理量，**无** `units` 属性）、`nm_per_px`（像素尺度，nm/px，
`units = "nm/px"`）。HDF5 的标量不能分块/过滤，因此这三个数存成**单元素数据集**（名字与 npz
时代一致），`u_field_file` 指向该 `.h5`，`stm_lf_apply.py` 从这里读场（不再是 `np.load` 的 npz）。

**迁移约束（`stm_lf_apply.py` 编码并写进报告）**：

| 约束 | 行为 |
| --- | --- |
| 同一扫描/同一视场 | 要求 `|L_target − L_ref| / L_ref` 小于 `--fov-tol`（默认 1e-3），超出打 WARNING 并记 `fov_warning = true`（不中止） |
| 不同像素数 | 支持：u 是物理量（nm），按各自的 nm/px 插值到目标像素坐标；**网格完全相同时直接逐位使用**（因此同图 apply 与拟合段逐字节一致） |
| 目标图无需有晶格 | 这正是迁移的用途（同步测量的谱学图等）：apply 段**不重新检测峰、不重跑自检** |
| 掩码传递 | `valid` 随场一起重采样；无效像素不施加位移 |
| 回退包 | `method = "identity_fallback"` 时目标图原样复制 + log WARNING |

## 5 一键 self-test（20 项）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/lawler-fujita-correction/scripts/selftest.py \
  --workdir data_processing/_skill_work/selftest_lf
# 可选：--size N、--keep、--stm-lib DIR
```

合成真值：a = 0.246 nm、L = 50 nm 的六方点阵（1x1 三成员 + r3 三成员），注入**已知**光滑位移场
（1.0 nm 高斯包 + 0.4 nm 线性漂移，零均值）。注入幅度受**可见性条件**约束：数据锚定检测要求
Bragg 峰不被畸变抹开，峰宽 ≈ `|∇u| × 半径`，而环聚类容差是半径的 2–3 %，因此 `|∇u|` 必须 ≲ 2 %；
在 50 nm 视场内 1.0 nm 变化已经接近该上限（实测 `|∇u|` 到 0.025 仍可检出、0.03 起环碎裂）。

覆盖与验收阈值（实测 **20/20** 通过）：

| # | 检查 | 验收阈值 |
| --- | --- | --- |
| 1 | 注入场在检测可见范围内（λ_test = λ_band/3 落在 1.5–6 nm） | λ_test ∈ (1.5, 6.0) nm |
| 2 | 拟合跑通并写出 `correction_report.json` | 退出码 0 + 报告写出 |
| 3 | 报告含全部 Lawler–Fujita 字段 | 必填字段齐全 |
| 4 | log 含两条契约行 | `# canvas ...` 与 `# corrected canvas: ...` |
| 5 | 相位/幅度/位移/掩码 = **一个** `<stem>_lf.h5` 的 9 个数据集（含 `units`）+ 每张一张 png，且报告 `lf_artifacts` 登记 `h5`/`dataset`/`png` | h5 约定完全符合（root 三属性、`gzip`/4、约定 chunk、`track_times=False`、units 正确）+ 登记项齐全 |
| 6 | 矫正 CSV + 复数 FFT2 + 两张预览图 | 形状 = 矫正画布、`complex128` |
| 6b | LF 产物**字面文件名**在磁盘上存在：`<stem>_lf.h5` + 9 个 `<stem>_lf_<map>.png`，且**不留任何 per-map `.npy`** | 10 个名字全在 + 0 个遗留 npy |
| 6c | `<stem>_corrected.h5` 的 h5 约定 + 数组逐位一致 | `fft2` 与 `.npy` **逐字节**相同（`tobytes()`，含 dtype）、`corrected` 以 `%.10e` 重渲染后与 CSV 文本逐字符相同（`float64`） |
| 7 | 恢复位移场与注入场一致 | 误差 rms < max(10 % × 注入 rms, 0.1 nm)（实测 0.054 nm） |
| 8 | 去掉参考仿射规范后一致 | 同上阈值（实测 0.0031 nm，即 1.7 %） |
| 9 | 该检查非平凡 | 注入 rms > 阈值（实测 0.184 nm > 0.10 nm） |
| 10 | 矫正后 1x1 环各向异性不劣于输入 | after ≤ 1.05 × before（实测 1.72 % → 0.012 %） |
| 11 | 1x1/r3 环对比值回到理想 √3 | 偏差 ≤ 1 %（实测 −0.0009 %） |
| 12 | 第三方向一致性（θ_a+θ_b+θ_c mod 2π） | 边界带（0.65λ）以内 wrapped rms < 2°（实测 0.18°；全掩码 7.41°，由边界污染主导） |
| 13 | 可迁移包 JSON + u 场 **h5** 写出 | schema/schema_version 正确 + h5 约定合规 + `u_field_file` 指向 `.h5` + 无 `.npz` 遗留 |
| 13b | `<stem>_lf.h5` 与包 h5 的 `u_x`/`u_y`/`mask` 逐位相同 | 三组 `np.array_equal` 全真 |
| 14 | 同网格 apply 复现拟合段 | `allclose(rtol=atol=1e-10)` 且 NaN 掩码相同（实测 max|Δ| = 0） |
| 14b | apply 段自身写出的 `<stem>_corrected.h5` 约定合规，且 `corrected`/`fft2` 与拟合段**逐字节**相同 | h5 约定 + `tobytes()` 字节比较（dtype 相同） |
| 15 | 不同像素数（800）apply 重采样位移场 | 退出码 0、`n_px = 800`、`u_field_rescaled = true`、契约行在 |
| 16 | 无晶格输入 → 回退 | `fallback = true`、`method = identity_fallback`、log 含 WARNING、输入原样复制 |

另有 1 条 **informational** 输出：默认 λ = 30 nm 在合成数据上只保留 0.117 nm（63.7 %）的位移，
说明默认低通对应的是远小于 1 % 应变的畸变（不参与判定）。

## 6 已知限制

1. **相位结点**：`|psi|` 很小的像素相位不可靠（掩码剔除 + 解缠填充），掩码覆盖率会随 λ 与噪声下降。
2. **边界**：解调相位在边界 0.65λ 带内被周期化 FFT 污染；λ 越接近视场，可信区域越小。
3. **参考规范**：位移场含一个「刚性平移 + 均匀应变」自由度（§2.4），绝对应变需外部参考。
4. **可见性**：应变 ≳ 2–3 % 时数据锚定检测找不到成形的环 → 回退（`identity_fallback`）；
   本 skill 不是为大应变/大转角场景设计的。
5. **不重新检测**：`stm_lf_apply.py` 只做重采样，不重跑检测与自检，包的可信度等于来源图。

## 7 修改记录

2026-09（HDF5 数组产物，本版）：全部数组侧产物改为 **HDF5 优先**——新增自包含模块
`scripts/h5io.py`（仓库 h5 约定：`SCHEMA_VERSION = 1`、`gzip`/4、显式 chunk ≤ 1 MiB、
root `schema_version`/`generator`/`creation_date`、`units` 属性；skill **不** import
`stm_data_processing`）；`<stem>_corrected.h5`（`corrected` + `fft2`）；9 张 per-map
`<stem>_lf_<name>.npy`（theta_a/b/c、amplitude_a/b/c、u_x、u_y、mask）**合并为一个**
`<stem>_lf.h5`（同名数据集，相位 `units = "rad"`、位移 `units = "nm"`、掩码无量纲），
每张图的 png 预览保留；`--save-transform` 的 u 场由 `<bundle>.npz` 改为 `<bundle>.h5`
（同样 6 个成员），bundle JSON 的 `u_field_file` 指向 `.h5`，`stm_lf_apply.py` 改为从 h5 读场。
**`<stem>_corrected_fft2.npy` 保留**：`skills/phase-analysis/` 用它做 `--fft2` 输入。
CSV/PNG 名称与字节不变；`correction_report.json` 只有 `lf_artifacts` 的登记项随产物改名
（`npy` 键 → `h5` + `dataset` 键），log 只有 `# LF artifacts:` 一行把过时的 `(npy+png)` 改成
`(h5+png) ... in <stem>_lf.h5`（其余字段、图、契约行与逐行数值都逐字节不变）。
self-test 17 → 20 项。见 `CHANGES.md`。
2026-09（v1.0）：首个版本。Fujita et al., PNAS 2014 SI §4 的双（三）方向 lock-in 位移场矫正：
最小二乘相位解缠、理想半径 + 数据取向的参考六方、`θ̄ = 0` 规范、按 u 重采样（画布 `n + 2(⌈max|u|⌉ + pad)`、
矫正后视场 `L·n_out/n`）、相位/幅度/位移/掩码 9 张 npy+png 产物、边界带内第三方向一致性诊断、
可迁移包（JSON + u 场 npz）与 `stm_lf_apply.py`（同网格逐位复现、不同像素数按物理位移重采样、
回退包原样复制），以及 16 项一键 self-test。见 `CHANGES.md`。
