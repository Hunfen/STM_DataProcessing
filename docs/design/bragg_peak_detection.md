# FFT Bragg 点检测 / 亚像素定位 / 晶格约束精修 —— 设计规格（bragg_peak_detection）

- 任务：t4（团队 `bragg-peak-finder`，需求阶段 round 1）
- 状态：**定案**（无开放问题；每个选择都有审计证据 + 本机实测阈值）
- 交付物：`docs/design/bragg_peak_detection.md`（本文）、`src/stm_data_processing/utils/bragg_peak_detection.py`（t5 实现）、`scripts/regression/check_bragg_peak_detection.py`（t5 实现 / t6 验证）
- 输入证据（三份审计报告，全部可复跑）：
  - `tmp_verify/bragg_audit/lfa_audit.md`（NEtCAT_LFA，318 行，简称 **LFA**）
  - `tmp_verify/bragg_audit/radsym_audit.md`（TrackingGUI 径向对称/高斯 MLE，306 行，简称 **RAD**）
  - `tmp_verify/bragg_audit/project_audit.md`（本项目现状管线与集成约定，267 行，简称 **PROJ**）
  - 本文引用的新实测：`tmp_verify/bragg_spec/probe_spec_v2.py`、`probe_sigma_cal.py`、`check_label_ambiguity.py`（本文 §8，简称 **PROBE**）
- 语言约定：正文中文；代码/公式/API/断言全英文；实现与回归脚本内不得出现中文。

---

## 1. 目标、范围与非目标

### 1.1 目标（可度量）

对一张实空间 STM 拓扑图（或用户提供的复数 FFT2），

1. **尽可能多地找出**图中真实存在的 Bragg 点（含弱环、近邻超结构卫星），不设常数硬上限；
2. 每个点给出**亚像素 q**（FFT 像素单位与 nm⁻¹ 双单位）、**强度/SNR**、**逐点不确定度**（1σ + 2×2 协方差）、**(h,k) 晶格指数**；
3. 拟合**共享倒格基**（2×2，nm⁻¹）及其 4×4 协方差；在给定参考晶格时给出**affine 矩阵 M 及其协方差**与派生的旋转/主拉伸/线性应变；
4. 用**合成成真值**基准与回归脚本把上述指标钉死为可断言的阈值。

### 1.2 落位与集成约束（PROJ §3）

| 约束 | 规定 |
|---|---|
| 模块路径 | `src/stm_data_processing/utils/bragg_peak_detection.py` |
| `utils/__init__.py` | **不修改**（现状只导出 5 个模块）；用 `from stm_data_processing.utils.bragg_peak_detection import ...` 导入 |
| 依赖 | 仅 `numpy` + `scipy`（`scipy.optimize.curve_fit`、`scipy.linalg.polar`、`scipy.ndimage.maximum_filter`）；**不** import matplotlib |
| 单位 | 对外统一 **nm⁻¹**；内部显式区分 `q_px`（FFT 像素）与 `q_nm_inv` |
| ruff | `.venv/bin/python -m ruff check src/stm_data_processing/utils/` 必须保持 **0 错误**（旧 22 条基线已由 commit `21f67e7` 清零，无可用额度；本机 2026-09-17 复核 `All checks passed!`） |
| 回归脚本 | `scripts/regression/check_bragg_peak_detection.py`，`.venv/bin/python` 运行，全过 `exit 0`、否则 `exit 1`；固定随机种子 |
| 真实数据 | `/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/topo0002.sxm`（512²、30 nm）**只读**；工作区内无 corrected CSV/FFT 产物（PROJ §0.6），故实测必须走 raw_data |

### 1.3 Non-goals（明确不做，避免范围蔓延）

1. **不修改任何现有模块**（含 `lattice_operations.py`、`miscellaneous.py`）**与 skill 脚本**（`~/.agents/skills/stm-topo-phase-analysis/scripts/*.py`）；本模块是纯新增工具。
2. **不做实空间原子定位**（AtomMapper 类功能）、不做 GUI。
3. **不用频率域零填充**（理由见 §2 D2）。
4. **不用 Poisson 高斯 MLE**（在 `|FFT|` 上发散，RAD §5.2/§5.5-7）。
5. **不用径向对称法作为主定位器**（RAD §5.2 F1/F2、§4.2 C/D；作为已评估并否决的备选记录在案，本轮不实现）。
6. **不做窗核形状匹配（Hann-kernel）拟合**（RAD §5.2：F1 略优 0.0112 vs 0.0121 bin，但 F5 晶格无序时优势消失 0.0504 vs 0.0484，且要求"已知窗函数 + 相关长度 ≫ 视场"，与真实扫描常态不符）。
7. **不做不确定度向 Kekulé 相位等下游量的传播**（属 skill 脚本职责，本轮只保证逐点 σ_q 与 affine 协方差可用）。
8. 不做 Kekulé / r3 峰选择；除 §4.4(a) 的对称类推断外不做其他物理建模。

---

## 2. 决策摘要（每条含证据与精度论证）

| # | 决策 | 证据（审计/实测） |
|---|---|---|
| D1 | 实空间加 **Hann 外积窗** → `fftshift(fft2)` → 在 `|F|`（幅度域）上检测与拟合 | 与本项目 skill 管线一致（`stm_topo_correct.py:180-181`、`stm_phase_analysis.py:70-71`）；LFA 默认 Hann（`fft_engine.py:19-23,68-89`）；RAD §5.1–5.2 全部 FFT 实测均以 Hann 窗为准 |
| D2 | **不使用零填充**；亚像素精度全部由参数化拟合提供 | LFA §2：全局零填充 ×4 + argmax 的 rms = 0.1216 px，正是 0.25 px 网格的量化极限（0.072 px/轴），而高斯 LSQ 在原生网格上 0.0061–0.1151 px；且频域插值样本完全相关，会让"逐点协方差"失去可标定性。PROBE：原生网格 + 7×7 拟合在 SNR≥50 时 rms = 0.019 px，远优于零填充极限 |
| D3 | 检测 = **稳健噪声尺度（Rayleigh）+ SNR 阈值 + 尺度自适应非极大抑制 + 无硬上限** | LFA B1/B2/B3：全局 `percentile` 无法兼顾（pct=99.9 漏 2% 弱环；无 cap → 105–208 假阳性；`window=41` 把 25 px 间距卫星召回压到 1/3）；PROJ B1：真实图默认 `window=41,pct=99.7,cap=12` 只返回 2–4 峰，`pct=98.0`→44 峰、`window=11`→26 峰 → 产能由两个写死常数决定 |
| D4 | 亚像素定位 = **log 幅度 3×3 抛物线定初值 + 有界旋转椭圆高斯 LSQ（含平面背景）** | LFA §2：2D 高斯 LSQ 最优；5×5 与 15×15 差 ≤4%（0.0062 vs 0.0061 px）；`parabola_3x3_log` 偏置 <0.003 px 而线性抛物线偏置 +0.049 px、DFT 上采样 −0.039 px、质心 +0.03 px；RAD §4.2：椭圆高斯 LSQ 1.08–1.28×CRLB（各向异性时优于径向对称 1.69×） |
| D5 | 背景项 = **平面 `a + b·dx + c·dy`**（非单常数） | PROJ H4：真实 FFT 峰叠加在 1/f 本底与邻峰尾巴上；RAD §5.5-4：非平坦背景是质心类方法崩溃主因（patch 21 → 0.67 bin），也是径向对称在低 SNR 落后 1.9× 的主因 |
| D6 | 高斯模型**默认含 xy 交叉项 ρ**（全协方差椭圆） | RAD §4.2 D：真 ρ=0.5 时全协方差 1.24×CRLB vs 无 ρ 椭圆 1.36×；C：ρ=0 时两者持平（1.19 vs 1.21）→ 多 1 个参数的代价可忽略；各向异性来源为非可分离窗/非方形有效孔径（F3）与剪切畸变 |
| D7 | 拟合参数**全部有界**，σ 初值由二阶矩给出 | LFA I7/§2.4：`p0.σ = roi_radius` 且无界 → patch 7×7 时 **20% 实现发散**（最大 ~40 px）却返回 `success=True` 与 1e28 的 σ；本项目基线同样无界且静默回退整数像素（PROJ H4/M5） |
| D8 | **逐点协方差必留**：`curve_fit(..., sigma=σ_n, absolute_sigma=True)`，σ_n 取 |FFT| 本底的 **Rayleigh 稳健尺度 MAD/0.4486** | LFA I4/E2：逐点协方差存在却不用于加权/代价矩阵 = 白丢 2× 精度；PROJ H1：`pcov` 被显式丢弃（`stm_topo_correct.py:112`、`stm_phase_analysis.py:122`）。PROBE 标定：MAD/σ_Rayleigh = 0.448641（数值 4e6 样本），用该尺度后 pull = 1.07–1.77（校准良好） |
| D9 | 报告 σ = `sqrt(σ_fit² + σ_model_floor²)`，`σ_model_floor = 0.006 px`；**不叠加 1/√12 地板** | LFA §2.5/E1：LFA 的 1/√12 地板使报出 σ 比真实误差大 3.5–70×（pull 0.015–0.28）；地板值取自 RAD §5.2 无噪相位扫描：高斯 LSQ 的峰形失配系统偏差 0.002–0.006 bin（Hann 窗核 vs 高斯，残差 27.2%） |
| D10 | 全局阶段 = **加权最小二乘（GLS）拟合 2×2 倒格基 B（等价 affine），随后 predict→verify→refit 迭代** | LFA E2：无权 OLS 旋转 rms 0.0709° → GLS 0.0359°、主拉伸 1.75e-3 → 0.95e-3（**2× 提升**）；PROBE §C 复现：GLS 0.0353° vs OLS 0.0952° |
| D11 | **不内建对称性硬约束**（不做"六方平均半径 + 精确 R60"） | LFA I2/E4：硬约束把 2% 单轴应变平均掉，偏差 2.5 pm = 统计误差 0.107 pm 的 **24 倍**；无约束 6 点 LSQ 偏差 ≤4e-6 nm |
| D12 | 只把 **±q 中的一个**计入独立测量，`n_independent` 显式输出 | RAD §5.3：实图 `|FFT|` 严格偶对称（实测 3.2e-17），配对平均的降噪收益**恰为 0**（估计差 4.9e-11 bin）；精度增益只能来自独立 G 矢量数 |
| D13 | 每个峰带**质量状态**，禁止静默回退 | LFA I7（`success=True` 却发散）、PROJ M5（静默回退整数像素）、PROJ B2（`max(0,py-half)` 环绕切片 → 实测返回 `|q| = 957 px`，而谱半径仅 362 px；`(256,1)/(256,511)` 直接抛 ValueError、`(256,8)` 返回伪值 −247.5 px） |
| D14 | 全流程**确定性**：无随机步骤；若实现可选 MC，必须 `seed` 固定 | LFA I8：MC 回退用未设种子的 `default_rng()` → 不确定度不可复现 |
| D15 | `affine`（M）**只在给定参考晶格时可报告**；其旋转角在未给定参考取向时**只能定到点群模数**，主拉伸/|det M| 是不变量 | PROBE `check_label_ambiguity.py`（本轮新实测）：同一批观测在两种合法指数标号下，拟合出的 M 相差**恰好 120.0005°**（六方点群元素），而奇异值完全一致（1.028619/0.971740 vs 1.028607/0.971743）、χ² 近乎相同（2.19）→ 数据无法区分标号；LFA 把这一模糊性藏在"六方硬约束"里（D11 的坑），本模块显式暴露 |
| D16 | 量纲与坐标系：`q_px` = fftshift 后有符号像素偏移 `(qx, qy)`；`q_nm_inv = q_px · 2π/size_nm`；`q_nyq = (n/2)·dq = π·n/size_nm` | PROJ §1.3 L204 `pixel_to_q = 2π/size_nm`；`miscellaneous.fft_q_limits` 同含 2π（本机复核：`fft_q_limits(30,256)[0][1]·10 == π·256/30 == 26.8083` ✓）；同时避免 LFA I1 的 0.5 px 双约定（统一"index = 像元中心，无 ±0.5"） |
| D17 | 复用 `lattice_operations` 生成理想倒格点池，**不修改该模块** | PROJ §3.4 实测 (h,k) 复原误差 6e-18；本机复核：`LatticeLoader.create_lattice(bvecs_array=B_nm_inv)` + `LatticeOperations.get_bragg_points_in_circle(q_max, include_origin=False)` → 72 点，`round(pts.T @ inv(B))` 最大误差 **8.9e-16** |

### 2.1 预期精度：相对 CRLB 与现有 `detect_peaks` 基线

| 环节 | 现有基线（`stm_topo_correct.detect_peaks` / `stm_phase_analysis.subpixel_q`） | 本规格 | 证据 |
|---|---|---|---|
| 单点定位（噪声效率） | 15×15 无界轴对齐高斯，实测**等价于** LFA 的旋转椭圆高斯（regime 1/2/3：0.0061/0.0239/0.1148 px），即 1.08–1.28×CRLB——**定位不是现状瓶颈** | 有界旋转椭圆高斯 + 平面背景，7×7：**1.07–1.31×CRLB**（与基线同级），PROBE 在 FFT 峰上 rms₂D = 0.0186 px @ SNR≥50 | LFA §2 三 regime 表；RAD §4.2 A–E；PROBE S1/S2 |
| 单点不确定度 | **无**（`pcov` 在 `stm_topo_correct.py:112`、`stm_phase_analysis.py:122` 被丢弃） | 逐点 1σ + 2×2 协方差，pull 中位数 1.07–1.77（校准到 ~1），量级 0.008–0.016 px @SNR≥50（**不用** 1/√12 地板：那会让 σ 大 3.5–70×） | PROJ H1；LFA §2.5/E1；PROBE |
| 检测完整度 | 常数 `window=41, pct=99.7, n_keep=12`：真实图 **2–4 峰**；合成三环图漏掉 2% 弱环；`window=41` 合并 25 px 卫星（召回 1/3） | SNR 自适应 + footprint 自适应 + 无上限：`ladder` 各环（SNR ≥ 5）recall **1.000**、FPR 0.00 | PROJ B1；LFA B1/B2/B3；PROBE `probe_per_ring.py` |
| 假阳性 | percentile 一旦放宽即 105–1467 个假阳性（靠常数 cap 才压住） | 晶格一致性验证（`lattice_tolerance=3σ`）在合成场景 FPR = 0.00 | LFA B2；PROBE |
| 晶格/affine | 只解 2 点闭式：方格子静默接受 45° 峰对、弱环接受 1.71° 峰对、无协方差 | GLS（4 参数）+ 迭代 + 4×4 协方差：异方差场景旋转 rms **0.0353°**（无权 OLS 0.0952°）、主拉伸误差 8e-5–1.08e-3 | PROJ B3；LFA E2/I4；PROBE §C |
| 全局约束的额外增益 | 无（无 (h,k)、无预测-验证） | `q_model` 的 rms₂D 比逐点小 **2–4×**（0.0078 vs 0.0179 px @S2；0.0364 vs 0.0602 @S3），理论增益 ∝ 1/√n_independent（±q 不重复计数） | PROBE M12；RAD §5.3/§6 |

> 结论：本规格的定位精度与现状基线**同级**（1.1–1.3×CRLB，FFT 峰上 0.019 px @SNR≥50），真正的精度与可用性增益来自 (i) 检测完整度（不再被常数钉死）、(ii) 逐点不确定度（可加权、可证伪）、(iii) 全局晶格约束（把 q 的不确定度再压低 2–4× 并给出 (h,k) 与 affine 协方差）。

---

## 3. 约定与公式

### 3.1 FFT 与坐标（全模块唯一约定，实现必须照抄）

- 输入实空间图 `image` 形状 `(n, n)`，物理边长 `size_nm = L`（nm，方形视场）。
- 预处理：`subtract_plane=True` 时先用 `stm_data_processing.utils.plot_funcs.subtractMeanPlane` 的等价实现（最佳拟合平面扣除）**或**直接调用该函数（只读 import）；NaN 按 §4.1 处理。
- 窗：`window = np.outer(np.hanning(n), np.hanning(n))`（可换成 hamming/blackman 或用户传入的 2D 数组；传 `None` 时不加窗）。
- 谱：`fft2 = np.fft.fftshift(np.fft.fft2(image * window))`，幅值 `magnitude = np.abs(fft2)`。
- 索引约定：数组下标 `[row, col] = [y, x]`，中心 `(n//2, n//2)`。像素 `(row, col)` 对应

```
q_px = (qx_px, qy_px) = (col - n//2, row - n//2)        # (x, y) 分量序
q_nm_inv = q_px * dq,      dq = 2*pi / L                # nm^-1
q_nyq_nm_inv = (n/2) * dq = pi * n / L                  # nm^-1
```

- **所有对外的二维向量一律 `(qx, qy)` 分量序**（与 `lattice_operations.get_bragg_points_in_circle` 的 `(2, N)` 列序一致），数组内一律 `[row=y, col=x]`；docstring 必须写明这两句。
- 与现有模块的一致性断言（回归脚本 R13）：`miscellaneous.fft_q_limits(L, n)[0][1] * 10 == q_nyq_nm_inv`。

### 3.2 倒格基与 affine（行向量约定）

```
B (2x2, nm^-1), rows = b1, b2:
    q_model(h,k) = (h, k) @ B            # 1x2 行向量

给定理想参考基 B_ideal（由 LatticeSpec 生成）:
    q_obs = q_ideal @ M        =>   M = inv(B_ideal) @ B_fit
    vec_row(M) = (b_ideal^-1 (x) I2) @ vec_row(B_fit)      # 协方差传播
```

- `vec_row(B) = (b1x, b1y, b2x, b2y)`（行优先展平，4 维）；`Cov(vec_row(M)) = (B_ideal^-1 ⊗ I2) Cov(vec_row(B)) (B_ideal^-1 ⊗ I2)^T`。
- 畸变分解 `M = R·U`（`scipy.linalg.polar`）：旋转角 `θ = atan2(R[0,1], R[0,0])`（行向量约定；回归脚本 R9 用植入的已知旋转验证该约定），主拉伸 = `M` 的奇异值，线性应变 = **`polar(M)[1] − I`**（右拉伸张量减单位阵）。它与 `0.5(M + Mᵀ) − I` 一阶等价（相差 `O(ω·ε)`，即"旋转 × 应变"量级，远小于应变本身），选前者的理由是**它在点群重标号下不变**：`M → DM`、`D` 正交 ⇒ `U` 不变；而 `0.5(M + Mᵀ) − I` 在重标号下会变（与 D15 的标号不变量口径一致）。
- 当参考基由数据推断（`lattice=None` 且推断为六方/四方）时，拟合把 gauge 固定为 `|det M| = 1`：此时 `principal_stretches`/`linearized_strain` 描述的是**去迹（偏斜）畸变**，其行列式为 1；等比的"整体收缩/膨胀"在**无外部标定时不可观测**（绝对尺度已归入 `bvecs_nm_inv`），用户若关心绝对尺度必须给出 `a_nm`/`bvecs_nm_inv`。该 gauge 对 `B_fit`、模型位置、`(h,k)`、χ²/rms 严格不变（在最终 affine 上重施一次以保证精确）；与 `polar(M)[1] − I` 兼容：`det(polar(M)[1]) = |det M| = 1` ⇒ 线性应变无迹。
- 由 PROBE 的 `check_label_ambiguity.py`：`θ` 与 `M` 在**未给定参考取向**时只能确定到点群模数（六方 60°、四方 90°、斜方 180°）；主拉伸与 `|det M|` 对任何幺模重标号不变，回归只对它们做绝对断言（§7 R8/R9）。

### 3.3 像素 ↔ 物理换算表

| 量 | 定义 | 例（n=256, L=30 nm） |
|---|---|---|
| `dq` | `2π/L` | 0.20944 nm⁻¹/px |
| `q_nyq` | `π n / L` | 26.81 nm⁻¹（=128 px） |
| 峰位置 `q_px` | 亚像素、fftshift 中心偏移 | 17.32 px（a = 2 nm 六方首环） |
| 峰位置 `q_nm_inv` | `q_px·dq` | 3.6276 nm⁻¹ |
| 不确定度 | `σ_q_px`（px）与 `σ_q_nm_inv = σ_q_px·dq` | 0.02 px ≙ 0.0042 nm⁻¹ |
| 石墨烯标定注意 | a=0.246 nm → \|b\|=29.5 nm⁻¹，在 512 px/50 nm 场内目标半径 234.7 px（近 Nyquist），100 nm 场直接超 Nyquist（PROJ §4） | 规格基准统一用 a=2.0 nm，避免 Nyquist 边界歧义 |

---

## 4. 算法栈

### 4.1 Stage 0：预处理与 FFT

```
compute_fft2(image, size_nm, *, window="hann", subtract_plane=True, nan_policy="plane")
```

1. 输入校验：`image.ndim == 2`；`image.shape[0] == image.shape[1]`；`size_nm > 0`；否则 `ValueError`（英文消息）。非 `float64` 输入转 `float64`。
2. NaN 处理（`nan_policy="plane"`）：对 NaN 像素用平面拟合值填充（与 `subtractMeanPlane` 同源的稳健平面：对非 NaN 像素做一次 `lstsq` 平面拟合）；`nan_policy="raise"` 时直接 `ValueError`。理由：PROJ M3（corrected CSV 含 NaN 是常态，现状直接把 NaN 喂进 `argmax`/拟合）。
3. 若 `subtract_plane`：扣除最佳拟合平面（唯一允许的既有函数复用点：`stm_data_processing.utils.plot_funcs.subtractMeanPlane`，只读调用）。
4. 加窗与 FFT（§3.1）。**不做零填充**（D2）。
5. 返回 `fftshift` 后的**复数** `fft2`（下游 `detect_from_fft2` 直接复用；`|F|` 由调用方/内部取模）。

### 4.2 Stage 1：候选检测（"尽可能多找点"的第一半）

```
DQ:
  dc_radius_px = dc_mask_frac * n                      # default 0.03 (PROJ M2: 半径可配)
  magnitude    = |fft2|
  sigma_n      = robust_rayleigh_scale(magnitude)      # two-pass, see below
  snr_map      = (magnitude - median(magnitude)) / sigma_n
  footprint_px = adaptive_footprint(...)               # see below
  local_max    = magnitude == maximum_filter(magnitude, size=footprint_px, mode="nearest")
  mask         = local_max & (snr_map >= min_snr) & (r_px > dc_radius_px) \
                 & (r_px <= q_max_px)                  # 默认 q_max_px = 0.95*(n//2)
  candidates   = coordinates of mask, sorted by magnitude desc
  if max_peaks is not None: candidates = candidates[:max_peaks]     # default None = no cap
```

`q_max_px`（默认 `0.95*(n//2)`）**同时**约束候选掩模与 §4.4(c) 的预测池：2D fftshift 网格的**角落**半径可达 `(n/2)·√2`（n=256 → 181 px）> `q_nyq` 半径 `n/2 = 128 px`，若掩模只用 DC 掩模，角落高频候选会被检出并报出，"所有 `|q| ≤ q_nyq`"（§6.3 F3、§7 R12）就不可能成立。`q_max_px > n//2` 仅供测试（如 F3 的边界构造）。

1. **噪声尺度（两遍法）**：第一遍在环带 `3·dc_radius_px ≤ r ≤ 0.97·(n/2)` 上取 `MAD = median(|X − median(X)|)`，`σ_n = MAD / 0.448641`（Rayleigh 尺度，常数由 PROBE 数值标定：`MAD/σ_Rayleigh = 0.448641`）。第二遍从环带中剔除所有第一遍候选极大值 ±2 px 邻域后重算，避免把峰本身算进噪声。**禁止**全局 `percentile` 阈值（LFA B1/B2）。
2. **自适应 footprint**：用 SNR 最高的 `min(20, len(candidates))` 个候选，在 5×5 窗口内以半高全宽估计两轴宽度，取中位数 `w_half`，则 `footprint_px = clip(round(1.5 * w_half_half_max), 3, 11)`（初始候选用固定 3 探测）。理由：LFA B3（常数 41 把 25 px 卫星召回压到 1/3）；RAD §5.2-4（FFT 峰主瓣仅 ~3 bin，7–11 px patch 已足够）；RAD §3.1（Python 翻译版把合并半径放大到 `nsize/2` 导致密峰漏检，本模块不照抄）。
3. **DC 掩模**：`|q| < dc_radius_px` 的点**直接排除**（不是置 0）。理由：PROJ H2（现状把 DC 圆盘置 0 反而在谱面留下人造陡边）。
4. **不设常数上限**：`max_peaks=None`（默认）。理由：LFA B1（`cap=12` → 12/30 召回；`cap=18` → 18/18）。
5. **±q 镜像配对**：对每个候选 `(i, j)`，其共轭点为 `((n-i) % n, (n-j) % n)`。由于 `|F|` 严格偶对称（RAD §5.3：3.2e-17），
   - 只在**半平面**代表点（`qy_px > 0`，或 `qy_px == 0 且 qx_px > 0`）上做后续晶格拟合；
   - 每个峰在输出里带 `conjugate_index`（其 −q 伙伴在 `peaks` 中的下标）与 `independent` 标志；
   - **禁止**把 ±q 当作两个独立测量（RAD §5.3：配对平均收益恰为 0），也禁止对 ±q 求平均。
6. **边缘与去重**：候选若无法容纳完整 patch（`patch_half` 见 §4.3）则标记 `quality="edge"`（保留在输出中、默认不参与晶格 GLS，见 §4.4）；两个候选的亚像素中心距离 `< 1.0 px` 时按 `chi2_reduced` 小的保留（去重阈值是常数，理由：峰宽自适应的是 footprint，去重只需排除同一极值的重复）。

### 4.3 Stage 2：亚像素定位（逐候选）

模型（**9 参数**，全部作用在幅度域 `|F|`）：

```
I(y, x) = A * exp( -0.5 * Q ) + b0 + bx*(x-x0) + by*(y-y0)
Q = [ dx^2/sx^2 - 2*rho*dx*dy/(sx*sy) + dy^2/sy^2 ] / (1 - rho^2),  dx = x-x0, dy = y-y0
params = (A, x0, y0, sx, sy, rho, b0, bx, by)      # 9 params; x0,y0 are patch-local px
```

- **patch 尺寸**：`patch_half = clip(ceil(1.5 * σ̂), 3, 7)`（`σ̂` 为该峰的二阶矩宽度初值，见下）；默认下限 3 → 7×7（49 点、dof = 49 − 9 = 40）。理由：LFA §2.1（5×5 与 15×15 等价，15×15 贵 9×）；RAD §5.2-4（7–11 足够）；PROJ baseline 的 15×15 偏大且是边界事故的来源（B2）。
- **初值**：
  - 中心：patch 内 3×3 **log 幅度抛物线**（`_log_parabola`，偏移 clip ±1 px），再在 ±2 px 内 snap 到局部极大以避免抛物线落在次极值上（PROBE 实现同此）。
  - `σ`：以 `max(patch − min(patch), 0)` 为权重的二阶矩（分别对 x、y），clip 到 `[0.7, patch_half + 0.5]`。**禁止** LFA 的 `p0.σ = roi_radius`（I7 的 20% 发散根因）。
  - `A`：`max(patch) − min(patch)`；`b0 = min(patch)`；`bx = by = 0`。
- **界**：`x0, y0 ∈ 初值 ± 2.0 px`；`sx, sy ∈ [0.4, patch_half + 0.5]`；`rho ∈ [-0.9, 0.9]`；`A ∈ (0, ∞)`；`b0, bx, by` 无界。
- **噪声权重**：`curve_fit(..., sigma=np.full(patch.size, sigma_n), absolute_sigma=True)`，`maxfev=4000`。`sigma_n` 用 §4.2 的全局稳健 Rayleigh 尺度（FFT 本底在 DC 之外平稳；比 patch-local MAD 更稳定，且不会被强峰自身污染）。理由 D8。
- **质量门（全部满足才算 `ok`，否则降级，绝不静默）**：
  1. `curve_fit` 未抛异常；
  2. 参数未贴界（位置、`sx`、`sy` 均不在界上，容差 1e-6/1e-9）；
  3. 协方差对角元有限且 ≥0，`σ_q` 有限且 `< 1.0 px`；
  4. 拟合中心落在 patch 内；
  5. `chi2_reduced = Σ(r/σ_n)² / dof` 有限（仅作诊断与协方差放大参考，**不**作为硬拒绝条件——Hann 窗核形状失配在 SNR 高时会让 χ²_red > 1，见 LFA §2.3/RAD §5.1）。
  失败时降级为 **log 幅度 3×3 抛物线**结果，`method="parabola_log"`、`quality="coarse"`、σ 取 `sqrt(抛物线协方差估计 + σ_model_floor²)`（抛物线协方差由三点抛物线公式的误差传播给出；若不可得则取 `0.15 px` 并如实标注）；再失败则 `method="max_pixel"`、`quality="failed"`、σ = `1/√12 px ≈ 0.2887 px`（**唯一允许使用量化地板的情形，且必须显式标注**）。
- **逐点输出**：亚像素 `q_px`（含 `q_nm_inv`）、振幅 `A`、`snr = magnitude[峰心]/σ_n`、`sigma_q_px`、`cov_q_px`、`chi2_reduced`、`n_pixels`、`method`、`quality`。

`σ_q` 合成：`sigma_q_px = sqrt(σ_fit² + sigma_model_floor_px²)`，`sigma_model_floor_px = 0.006`（默认，可配）。

### 4.4 Stage 3：晶格约束全局精修（"尽可能多找点"的第二半）

输入：已定位的峰（半平面代表点集合）；输出：`LatticeFit` 与每个峰的 `(h,k)`、模型位置、残差。

**（a）理想点池与指数分配**

- 给定 `LatticeSpec`（`a_nm` + `symmetry` 或直接 `bvecs_nm_inv`）→ 构造 `B_ideal`（行基，nm⁻¹）：六方 `b1 = b(1,0)`、`b2 = b(cos60°, sin60°)`，`b = 4π/(√3 a)`；四方 `b1 = (2π/a, 0)`、`b2 = (0, 2π/a)`；斜方用 `bvecs_nm_inv`。**所有基矢都含 `2π`、单位 nm⁻¹**，与 §3.1 的 `dq = 2π/L`、与六方式的 `4π/(√3 a)` 自洽；LFA 的无 `2π` 约定（hex `(2/√3)/a`、square `1/a`）**不得照抄**（照抄会使理想池整体差 `2π` 倍、无任何观测点可标）。
- 点池生成**复用** `lattice_operations`（D17）：

```python
from stm_data_processing.io.lattice_loader import LatticeLoader
from stm_data_processing.utils.lattice_operations import LatticeOperations
lat  = LatticeLoader.create_lattice(bvecs_array=B_ideal)           # rows b1,b2, nm^-1
pts  = LatticeOperations(lat).get_bragg_points_in_circle(q_max, include_origin=False)  # (2,N)
hk   = np.round(pts.T @ np.linalg.inv(B_ideal)).astype(int)        # (N,2) 复原指数
assert np.abs(pts.T @ np.linalg.inv(B_ideal) - hk).max() < 1e-9    # 实测 8.9e-16
```

  该函数不返回指数，故用 `pts.T @ inv(B_ideal)` 复原（PROJ §3.4 与本次复核一致）。
- **两种标号路径**：
  - `orientation_deg` **已给**：用"最近理想点"分配（理想池与观测都在同一取向附近，畸变 < 10% 时无歧义）→ 匈牙利/最近邻迭代：分配 → GLS 拟合 → 重分配，最多 3 轮。
  - `orientation_deg = None`（默认）：用**半径门 + 首环角序规范标号**：取独立峰中 `| |q| / |b1_ideal| − 1 | < 0.15` 的首环点，按极角排序，与首环六个理想指数 `(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)` 做 6 个循环偏移 × 2 个手向共 **12 种对齐**，各自拟合 M 并用加权 χ² 选最优，随后按最近预测扩充分配。该规范标号是**确定性的**，但旋转角只能定到点群模数（D15）。
  - **斜方 + 调用方给 `bvecs_nm_inv` 且 `orientation_deg = None`**：斜方的标号模糊性是 **GL(2,Z)**（不是点群）——`M → B_ref⁻¹ G B_ref` 一般**非正交**，会污染应变本身，故必须用先验定标号。实现取**最小畸变约定（least-distortion gauge）**：候选集 = `SNR 最强的 min(6, m) 个峰` **∪** `{每个参考基行按半径 |q| 最近的 3 个峰}`（半径匹配对旋转不变、对百分之几的畸变稳健），每个候选按 `(1,0)/(0,1)` 两种顺序生成闭式 `M₀`；整轮 expand+refit 后按 `d(M) = max(|λ₁−1|, |λ₂−1|)`（`λ` = `M` 的奇异值）排序，平局取"标号点数更多"、再取"加权 χ² 更小"。`meta["oblique_gauge"] = "least_distortion"`、`meta["oblique_gauge_margin"] = d₂/d₁`（=`1` 表示 gauge 未识别，用户应显式给 `orientation_deg`），`rotation_is_absolute` 在该情形保持 `False`；由此得到的 `rotation_deg`/`principal_stretches` 是**该约定下的量，不是绝对取向**。`orientation_deg` 已知时走全池最近邻分配（容差 `0.3·|b1|`），即为绝对标号。
    为什么候选集必须含"半径匹配"这一支：强峰排序由**幅值 × 相位干涉**共同决定，**不保证**包含参考行 `b1`/`b2` 的像——在本规格的 gauge 判例上，真参考行按实测 SNR 排 **19 / 34**（回归脚本用例为 **9–11**），故严格"只取 top-6 强峰"**不可达**正确 gauge（A-10 原文的"top-6"措辞作废，见 A-13）。并集是原候选集的**严格超集**（只增候选、不减 top-6 项），排序键与验收容差（主拉伸 ≤2e-3、旋转 ≤0.05°）**均未改动**。
- 无 `LatticeSpec` 时（`lattice=None`）：从强峰（`snr ≥ min_snr_seed`）按**全环**多重数（= 2 × 半平面代表点数，见 §4.2-5 与 D12）推断对称类（≥5 → hexagonal；==4 → square；≤2 → oblique；==3 → 同时试 square/hexagonal），各候选类并行验证后按**加权 `χ²_red` 最小者胜、平局取 `|det B|` 大者**排序；`==3` 一档在加倍计数下**不可达**（半平面 `k` 个点 → 全环 `2k` 恒为偶数，例如 3 → 6 归 hexagonal），仅当 `detect_from_fft2` 收到非厄米谱时触发，此时同时试 square 与 hexagonal。生成 `B_ideal` 后走同一流程；`affine` 置 `None`（无参考基，M 无定义），仅输出 `bvecs_nm_inv` 与其协方差。**推理 oblique 的种子**取最短的两个强非共线峰（半径优先、以中位 SNR 为强度下限）：按 SNR 排序会取到高阶向量、生成**子晶格**并大量漏标（实测标号率 0.541 → 1.000），而该路径按 §4.4(d) 不发布畸变量，故只影响 `(h,k)` 标号，取最短向量是标准原胞约定。推断结果写入 `meta["inferred_symmetry"]`。

**（b）GLS 拟合（D10）**

对分配集合 `{(hk_i, q_i, Σ_i)}` 解

```
min_{M or B}  Σ_i || (hk_i @ B_ideal @ M) - q_i) · W_i ||²         # W_i = Σ_i^{-1/2}（对称平方根）
Σ_i' = Σ_i + sigma_model_floor_px² · I2        # 总位置协方差（含模型地板）= GLS 的权重矩阵（Σ_i'^-1）
```

实现：把每个峰的 2 个分量按 `W_i` 白化后堆成 `(2N × 4)` 设计阵 `D` 与 `(2N,)` 右端 `r`，`np.linalg.lstsq(D, r)` 解 `vec_row(M)`；`Cov = pinv(Dᵀ D) · max(1, χ²_red)`，`χ²_red = ||r − D·m||² / (2N − 4)`。**判据用加权 χ²_red，绝不用无权 RMSE**（LFA E2：无权残差 RMSE 反而更小，因为它就是 OLS 的目标函数）。
- `B_fit = B_ideal @ M`；`Cov(vec_row B_fit)` 由线性传播得到（行优先展平满足 `vec_row(A·Y) = (A ⊗ I2) vec_row(Y)`，故 `vec_row(B_fit) = (B_ideal ⊗ I2) vec_row(M)`）。
- **字段口径（数值零变化，仅表示层）**：报出的 `cov_q_px` **就是**上式含模型地板的**总**位置协方差 `Σ_i'`，因此 `sigma_q_px = sqrt(diag(cov_q_px))` **恒等**成立、`cov_q_nm_inv = cov_q_px · dq²`；GLS 直接以该矩阵为权重，**不再**在地板之外重复叠加一次。旧规格把 `cov_q_px` 写成"纯拟合协方差"、`sigma_q_px` 另含地板，两者在高 SNR 峰上可差一个数量级；外部使用者拿 `cov_q_px` 自行做加权或误差传播会系统性低估不确定度（MF-2 按此修正，权重矩阵与修正前逐位相同）。
- 约束：`|det B| > 1e-9`、`χ²_red` 有限、`N ≥ 3` 且至少 2 个非共线独立点；否则 `converged=False` 并给出 `ValueError` 级诊断（英文）而不是静默返回（对比 LFA E3：4 点时静默给出 −30° 的错误 affine 与 11–13 px RMSE；PROJ B3：方格子输入静默把 45° 拉成 60°）。

**（c）predict → verify → refine 迭代（"尽可能多找点"）**

```
for it in range(max_iterations):            # default 3
    pool = [(h,k) for h in [-h_max,h_max], k in [-h_max,h_max], (h,k)!=(0,0)
            if dc_radius_px < |hk @ B_ideal @ M| / dq <= q_max_px]     # px; q_max_px default 0.95*(n/2)
    for (h,k) in pool not already matched (dist >= 1.5 px):
        q_pred = hk @ B_ideal @ M
        if max(|F|) in 3x3 around round(q_pred) / sigma_n < min_snr_verify:  continue   # default 3.5
        p = localize(patch centred on round(q_pred), position bounds +-1.5 px)
        sigma_pred = sqrt(diag(J Cov_M J^T)),  J = [[h,0,k,0],[0,h,0,k]]
        accept if |q_fit - q_pred|_2 <= lattice_tolerance * hypot(sigma_total)  # default 3.0
    if nothing accepted: break
    refit (b) with the enlarged set
```

- 当 `lattice=None`（无参考基）时，伪码里的 `B_ideal @ M` 一律用第 (b) 步拟合出的 `B_fit` 代替（同一个 4 参数线性模型，只是基底约定不同）。
- 迭代新增的点与首轮点同等对待（都过 §4.3 的质量门）。停止条件：无新点接受、或达到 `max_iterations`。
- **晶格是过滤器而不是截断器**：不被任一整数指数解释的候选**不丢弃**，而是保留在 `peaks` 中、`index_hk=None`、`quality="unmatched"`（它们可能是超结构卫星、moire、或非本晶格的实特征）。理由：LFA B1/B2/B3 显示"多找点"与"高精度"是两件事，不能用一个常数 cap 表达；同时 RAD §5.5-1 指出检测/合并偏差远大于定位偏差，丢点比丢精度更致命。
- 每个峰的最终字段：`index_hk`（若与模型一致，即 `residual ≤ lattice_tolerance·σ_total`）、`q_model_px`（模型预测位置）、`sigma_q_model_px`（由 `Cov_M` 传播）、`residual_px`。
- **推荐的下游用法**：与晶格一致（`quality="ok"` 且 `index_hk` 非空）的峰，其**最终 q 用模型预测值** `q_model_px`，其不确定度显著小于逐点拟合（PROBE：模型预测 rms 0.0078 px vs 逐点 0.0179 px；理论上 ∝ σ_q/√n_independent）。本规格**不**提供"逐点值 × 模型值"的朴素加权平均（两者由同一批数据导出、强相关，朴素 BLUE 会过度自信），要么用逐点值（`q_px`），要么用模型值（`q_model_px`），两者都在输出里。

**（d）非对称性与不变量**

- 不施加任何度量对称约束（D11）；`symmetry` 只用于生成理想池与给出旋转角的模数。
- 输出 `LatticeFit.rotation_mod_deg`（hexagonal 60 / square 90 / oblique 180 或 360）与 `rotation_is_absolute`（是否给了 `orientation_deg`）。回归脚本按压模量断言旋转（§7 R9），并对主拉伸做绝对断言（§7 R8）。
- **推理 oblique（`lattice=None` 且推断为斜方）不发布畸变量**：`principal_stretches`/`linearized_strain`/`rotation_deg` 置 `None`、`strain_gauge = "undefined"`。理由：该路径的参考基由数据派生（`B = [q₁; q₂]`、标号 (1,0)/(0,1) ⇒ `M ≡ I`），发布的形变量恒为"无畸变"而无信息；`bvecs_nm_inv`/`cov_bvecs_nm_inv`/`(h,k)` 等几何 payload 照常保留。

**（e）晶格拟合全局质量门（A-0..A-9）**

三条件合取（记 `B_fit` 为拟合倒格基、`dq = 2π/L`、`dof = 2N_labelled − 4`）：

| # | 判据 | 默认上限 | 推导要点 |
|---|---|---|---|
| G-1 | `pool_spacing_px >= min_pool_spacing_px` | `3.0 px` | `pool_spacing_px = min_{hk≠0,\|h\|,\|k\|≤2} \|hk @ B_fit\| / dq`（最短倒格矢 = 首环半径）。两个晶格点若比非极大抑制 footprint 下限（3 px）还近，它们在检测阶段**不可能**被独立检出，用这样的基矢解释已检出峰必然是退化拟合。传 `0.0` 关闭。 |
| G-2 | `rms_residual_px <= residual_max_px` | 派生 `min(1.0 px, 0.25·pool_spacing_px)` | 双重理由：残差须小于**一个谱 bin**（1 px，否则 `(h,k)` 指认本身处在量化误差量级）；且 ≤ **最近邻间距的 1/4**（Voronoi 半间距之上再留 2× 余量，保证"指认到正确格点比指认到任何其他格点更近"）。取更紧者；显式传入数值时派生规则不再生效。 |
| G-3 | `chi2_reduced <= chi2_red_max` | `25.0` | `χ²_red` = 声明 1σ 被低估倍数的平方 ⇒ 取 `s = 5`（典型残差 ≤ 5σ_q）得 25。传 `math.inf` 显式关闭（关闭后 S-2 类风险自担）。 |

`fit_ok := G-1 ∧ G-2 ∧ G-3 ∧`（既有的）`rank = 4 ∧ isfinite(chi2_reduced) ∧ |det(B_fit)| > 1e-9`；`quality` = `"ok"`，或失败的门按 **G-1 → G-2 → G-3** 的顺序用 `+` 连接（如 `spacing_below_min+residual_above_max+chi2_above_max`）；`meta["lattice_error"]` 写入可读理由串。

**单位与统计量口径（只改措辞与单位，阈值不变）**

- `rms_residual_px` 的单位是 **FFT 像素**：模型预测 `hk @ B_fit` 是 nm⁻¹，必须**除以 `dq`** 换算后再取 RMS；它与逐点 `residual_px` 同单位，且**接受与拒绝两种结果下都必须等于被标号峰逐点残差的 RMS**（两态恒等是本设计的不变量；三个门参数只决定接受/拒绝，不改变门时刻的标号集与残差）。同理 `residual_max_px`、`pool_spacing_px`、`sigma_q_px` 也都是像素。
- `consistent_fraction` := 被标号峰中满足
  `residual_px <= lattice_tolerance · hypot(sigma_q_px)`
  的比例——**只用逐点测量 σ**（`sigma_q_px`），**不含** `sigma_q_model_px`；用**调用方**的 `lattice_tolerance`；单位 px；在门看到的那批标号峰上、**清空标签之前**计算，故**被拒的拟合也照常报告**该值（可审计）。它**不参与**门。
  为什么排除模型 σ：`sigma_q_model_px` 随 `cov_bvecs_nm_inv` 变化，而后者被 GLS 按 `max(1, χ²_red)` 放大 ⇒ 拟合越差、窗口越宽、"一致比例"越高，统计量会**自我认证**（实测在被拒拟合上可达 0.86–1.00，而按本口径为 0.14–0.56，与"三例全拒"自洽）。
  `_prediction_round` 的**逐点接受判据**仍用组合 σ（含预测不确定度，见 (c)），这与本统计量是**两个不同定义**，分别实现、分别写入 docstring，不得混用。
- **测量量 vs 门限（字段分组）**：`LatticeFit` 里 `chi2_reduced`/`rms_residual_px`/`pool_spacing_px`/`consistent_fraction`/`n_independent` 是**测量量**；`residual_max_px` 是 G-2 **实际采用的门限**（由测量派生或显式传入，故留在结果对象内使判决可复算）。另两个门限是**调用方输入**，只在 `meta` 回显：`meta["min_pool_spacing_px"]`、`meta["chi2_red_max"]`（连同 `meta["pool_spacing_px"]`、`meta["residual_max_px"]`、`meta["fit_ok"]`、`meta["lattice_quality"]`、`meta["lattice_error"]`）⇒ 任何一次判决都可仅凭 `(result.lattice, result.meta)` 复算。

**（f）拒绝语义（`fit_ok=False` 时做什么 / 不做什么）**

- **做**：保留 `result.peaks`（这些峰确实被检出——晶格是过滤器而不是截断器）；保留 `LatticeFit` 对象及其诊断量（`bvecs_nm_inv`、`cov_bvecs_nm_inv`、`chi2_reduced`、`rms_residual_px`、`pool_spacing_px`、`residual_max_px`、`consistent_fraction`、`n_independent`、`symmetry`、`n_iterations`）。
- **不做（全部收回）**：对**每一个**峰 `index_hk = None`、`q_model_px = None`、`sigma_q_model_px = None`、`residual_px = None`（其 `quality` 落到既有的 `"unmatched"`）；`affine = None`、`cov_affine = None`、`rotation_deg = None`、`rotation_sigma_deg = None`、`principal_stretches = None`、`linearized_strain = None`、`strain_gauge = "undefined"`（`rotation_mod_deg` 保留为类别模数，无歧义）。
- 理由：仅打 `fit_ok` 标记仍会把错误的 `(h,k)`/affine 交给下游（LFA E3/E4 的教训：错误的 39.7° 基矢照样被当成"六方"），所以拒绝必须**收回全部模型声明**、只留诊断量。同一"清空"辅助函数在三处调用：无候选种子/无解、候选种子回滚、质量门拒绝。

### 4.5 不确定度（三层，全部落在数据结构里）

| 层 | 量 | 来源 | 备注 |
|---|---|---|---|
| 单点 | `sigma_q_px`（1σ，逐轴）、`cov_q_px`(2×2) | **总**位置协方差 `cov_q_px = cov_fit + sigma_model_floor_px²·I`（`absolute_sigma=True` 的 `curve_fit` 协方差 ⊕ 模型地板 0.006 px），`sigma_q_px = sqrt(diag(cov_q_px))` **恒等**；它同时就是 §4.4(b) GLS 使用的权重矩阵 | **不叠加** 1/√12；仅 `method="max_pixel"` 的失败点使用 1/√12 并标注 |
| 点 vs 模型 | `residual_px`、`sigma_q_model_px` | `J Cov_M Jᵀ`（`J = [[h,0,k,0],[0,h,0,k]]`） | 用于 `lattice_tolerance` 接受判据与离群标记 |
| 晶格 | `cov_bvecs_nm_inv`(4×4)、`cov_affine`(4×4)、`rotation_sigma_deg`、`principal_stretches`、`linearized_strain` | GLS 正规方程协方差 × max(1, χ²_red)，再经 `B_ideal` 线性传播与数值 Jacobian 传播 | 派生标量的 1σ 用中心差分 Jacobian（步长 1e-6 相对） |

### 4.6 已评估并否决的方案（记录在案，不再讨论）

| 方案 | 否决理由（证据） |
|---|---|
| 零填充 + argmax / 零填充 + 抛物线 | 量化极限 rms 0.1216 px（LFA §2），且插值样本相关破坏协方差可标定性 |
| 全局 percentile 阈值 | 在真实数据上阈值扫描使峰数 4↔44 跳变（PROJ B1/H2），且与假阳性不可兼得（LFA B2） |
| 常数非极大抑制窗口（41/21） | LFA B3 卫星召回 1/3；PROJ B1 window 11→26 峰、21→6 峰、41→4 峰 |
| 常数检测上限 `n_keep=12` | LFA B1：cap 决定召回（12/30 vs 30/30） |
| 无界高斯 + `p0.σ = roi_radius` | LFA I7：20% 发散、σ≈1e28、`success=True` |
| Poisson 高斯 MLE | RAD §5.2/§5.5-7：在 `|FFT|` 上 12–27 bin 偏置（病态） |
| 径向对称中心法 | RAD §5.2 F1/F2：高 SNR 差 1.2–1.5×、低 SNR 1.9×；§5.4：`meand2` 只是弱序数指标（ρ=+0.34），无协方差 |
| 质心 / 整数极大值 / 线性 3×3 抛物线 / DFT 上采样 | LFA §2.3：系统偏置 +0.28~+0.049 px、rms 0.29–0.60 px，不可用于定量 q |
| ±q 配对平均 | RAD §5.3：收益恰为 0（4.9e-11 bin） |
| 六方硬约束（平均半径 + R60） | LFA E4：把 2% 应变平均掉（偏差 24σ） |
| 无权 OLS 拟合 affine | LFA E2：比 GLS 差 2× |

---

## 5. 公开 API（`src/stm_data_processing/utils/bragg_peak_detection.py`）

```python
"""FFT Bragg peak detection, sub-pixel localization and lattice-constrained
refinement for STM topographs.  All vectors are (qx, qy); array indices are
[row=y, col=x].  Reciprocal-space units are nm^-1 unless a name says *_px."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LatticeSpec",
    "BraggPeak",
    "LatticeFit",
    "BraggDetectionResult",
    "compute_fft2",
    "detect_bragg_peaks",
    "BraggPeakDetector",
]


@dataclass(frozen=True)
class LatticeSpec:
    """Ideal reference reciprocal lattice used to label detected peaks.

    a_nm            real-space lattice constant in nm (hexagonal/square).
    symmetry        "hexagonal" | "square" | "oblique".
    bvecs_nm_inv    (2, 2) rows b1, b2 in nm^-1; overrides a_nm/symmetry.
    orientation_deg reference orientation in degrees; None = unknown, in which
                    case the index labelling is canonical (see the design spec)
                    and the affine rotation is defined only modulo the point
                    group angle reported in LatticeFit.rotation_mod_deg.
    h_max           maximum |h|, |k| used to build the prediction pool.
    """
    a_nm: float | None = None
    symmetry: str = "hexagonal"
    bvecs_nm_inv: np.ndarray | None = None
    orientation_deg: float | None = None
    h_max: int = 8


@dataclass(frozen=True)
class BraggPeak:
    """One detected Bragg peak.

    q_px, q_nm_inv            sub-pixel position, (qx, qy).
    sigma_q_px, sigma_q_nm_inv  per-axis 1-sigma.
    cov_q_px, cov_q_nm_inv      (2, 2) covariance.
    amplitude, snr              fitted amplitude and |F|_peak / noise_sigma.
    index_hk                    (h, k) once consistent with the fitted lattice.
    q_model_px, sigma_q_model_px  lattice-model position and its 1-sigma.
    residual_px                 |q_measured - q_model| in px.
    chi2_reduced, n_pixels, method, quality
    conjugate_index             index of the -q partner inside result.peaks.
    independent                 True for the half-plane representative.
    """
    q_px: tuple[float, float]
    q_nm_inv: tuple[float, float]
    sigma_q_px: tuple[float, float]
    sigma_q_nm_inv: tuple[float, float]
    cov_q_px: np.ndarray
    cov_q_nm_inv: np.ndarray
    amplitude: float
    snr: float
    index_hk: tuple[int, int] | None = None
    q_model_px: tuple[float, float] | None = None
    sigma_q_model_px: tuple[float, float] | None = None
    residual_px: float | None = None
    chi2_reduced: float = float("nan")
    n_pixels: int = 0
    method: str = "gaussian"          # "gaussian" | "parabola_log" | "max_pixel"
    quality: str = "ok"               # "ok" | "coarse" | "edge" | "unmatched" | "failed"
    conjugate_index: int | None = None
    independent: bool = True


@dataclass(frozen=True)
class LatticeFit:
    """Fitted reciprocal basis and (optionally) the affine distortion.

    bvecs_nm_inv         (2, 2) rows b1, b2 in nm^-1.
    cov_bvecs_nm_inv     (4, 4), vec order (b1x, b1y, b2x, b2y); the source of
                         every sigma_q_model_px through J_B cov J_B^T with
                         J_B = [[h, 0, k, 0], [0, h, 0, k]].
    affine               (2, 2) M with q_observed = q_ideal @ M; None when no
                         reference lattice was supplied (an inferred reference
                         still yields bvecs, rotation and the strain below).
    cov_affine           (4, 4) row-major covariance of M.
    rotation_deg         polar-decomposition rotation angle of M, degrees; with
                         an inferred reference it is the canonical labelling and
                         is only defined modulo rotation_mod_deg.
    rotation_sigma_deg   1-sigma of rotation_deg (numerical Jacobian).
    rotation_mod_deg     point-group modulus of rotation_deg (60/90/180/360).
    rotation_is_absolute True when LatticeSpec.orientation_deg was supplied.
    principal_stretches  singular values of M (ascending); invariant under the
                         labelling ambiguity of design D15.
    linearized_strain    (2, 2) polar(M)[1] - I, the right stretch tensor minus
                         the identity; invariant under a left point-group
                         relabelling (0.5 * (M + M.T) - I would not be).  When no
                         reference lattice was supplied the fit is gauge-fixed
                         to |det M| = 1, so only the deviatoric shape distortion
                         is reported and the absolute scale lives in bvecs.
    n_independent, chi2_reduced (weighted), rms_residual_px,
    n_iterations, symmetry

    Acceptance gates (design ruling A-0..A-9, all three must hold):

    fit_ok              True when the fit passed every gate below *and* the
                        rank/|det B|/finiteness checks; when False the fit is
                        reported for diagnosis only: every peak loses its
                        index_hk/q_model_px/sigma_q_model_px/residual_px and
                        affine/cov_affine/rotation*/stretch/strain are None.
    quality             "ok", or the failed gates joined by "+"
                        (spacing_below_min / residual_above_max / chi2_above_max).
    The trailing fields split into **measured quantities** -- chi2_reduced,
    rms_residual_px, pool_spacing_px, consistent_fraction, n_independent -- and
    the **threshold actually applied** by G-2 (residual_max_px).  The other two
    gate thresholds are caller inputs and are echoed in ``meta`` instead of being
    copied into the frozen result: ``meta["min_pool_spacing_px"]`` and
    ``meta["chi2_red_max"]``.  Together with ``meta["pool_spacing_px"]`` and
    ``meta["residual_max_px"]`` this lets every verdict be recomputed from
    ``(result.lattice, result.meta)`` alone.

    pool_spacing_px     measured G-1 quantity: ``min_{hk != 0, |h|, |k| <= 2}
                        |(h, k) @ B_fit| / dq`` in FFT pixels; identical to
                        ``meta["pool_spacing_px"]``.
    rms_residual_px     measured G-2 quantity, in **pixels** (the nm^-1 model
                        prediction is converted with ``dq`` before the RMS); it
                        equals the per-peak residual RMS of the labelled peaks in
                        both the accepted and the rejected state.
    residual_max_px     G-2 threshold actually applied (explicit, or derived as
                        ``min(1.0 px, 0.25 * pool_spacing_px)``).
    consistent_fraction fraction of labelled peaks with
                        ``residual_px <= lattice_tolerance * hypot(sigma_q_px)``
                        -- measurement sigma only, in pixels, on the labelled set
                        the gates see (so a rejected fit still reports it).
                        Reported for auditing only; it never gates the fit.
                        The model-prediction sigma is excluded on purpose: it
                        scales with ``cov_bvecs``, which the GLS inflates by
                        ``max(1, chi2_reduced)``, so including it would make a
                        worse fit look *more* consistent (measured 0.86-1.00 on
                        fits that the gates reject).
    strain_gauge        which reference fixes the strain scale: "absolute"
                        (caller-supplied reference lattice), "deviatoric" (an
                        inferred hexagonal/square reference, gauge-fixed to
                        |det M| = 1 so only the shape distortion is reported) or
                        "undefined" (no reference with a meaningful ideal shape
                        -- an inferred oblique lattice -- or a rejected fit; the
                        shape fields are then None).  It is independent of
                        rotation_is_absolute: a caller-supplied basis without an
                        orientation has an absolute strain but a modulo-point-
                        group rotation.
    """
    bvecs_nm_inv: np.ndarray
    cov_bvecs_nm_inv: np.ndarray
    affine: np.ndarray | None
    cov_affine: np.ndarray | None
    rotation_deg: float | None
    rotation_sigma_deg: float | None
    rotation_mod_deg: float
    rotation_is_absolute: bool
    principal_stretches: tuple[float, float] | None
    linearized_strain: np.ndarray | None
    n_independent: int
    chi2_reduced: float
    rms_residual_px: float
    n_iterations: int
    symmetry: str
    fit_ok: bool = True
    quality: str = "ok"
    pool_spacing_px: float = 0.0
    residual_max_px: float = 0.0
    consistent_fraction: float = 0.0
    strain_gauge: str = "absolute"


@dataclass(frozen=True)
class BraggDetectionResult:
    peaks: tuple[BraggPeak, ...]
    lattice: LatticeFit | None
    size_nm: float
    n_px: int
    dq_nm_inv: float
    q_nyquist_nm_inv: float
    noise_sigma: float
    n_candidates: int
    fft2: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def compute_fft2(
    image: np.ndarray,
    size_nm: float,
    *,
    window: str | np.ndarray | None = "hann",
    subtract_plane: bool = True,
    nan_policy: str = "plane",
) -> np.ndarray:
    """Return fftshift(fft2(windowed, plane-subtracted image)) as complex128. ..."""


def detect_bragg_peaks(
    image: np.ndarray,
    size_nm: float,
    *,
    lattice: LatticeSpec | None = None,
    min_snr: float = 5.0,
    min_snr_seed: float = 8.0,
    min_snr_verify: float = 3.5,
    dc_mask_frac: float = 0.03,
    footprint: int | None = None,
    patch_half: int | None = None,
    max_peaks: int | None = None,
    q_max_px: float | None = None,
    lattice_tolerance: float = 3.0,
    max_iterations: int = 3,
    sigma_model_floor_px: float = 0.006,
    chi2_red_max: float = 25.0,
    residual_max_px: float | None = None,
    min_pool_spacing_px: float = 3.0,
    return_fft2: bool = False,
) -> BraggDetectionResult:
    """Detect, localize and lattice-refine Bragg peaks. ..."""


class BraggPeakDetector:
    """Reusable detector: same keyword arguments as detect_bragg_peaks."""

    def __init__(self, **kwargs) -> None: ...
    def detect(self, image: np.ndarray, size_nm: float) -> BraggDetectionResult: ...
    def detect_from_fft2(
        self, fft2: np.ndarray, size_nm: float, *, return_fft2: bool = False
    ) -> BraggDetectionResult: ...
```

语义补充（实现必须遵守）：

1. `detect_from_fft2` 接受 **已 fftshift 的复数谱**（§3.1 约定），`size_nm` 仍需给出以换算 nm⁻¹；它不做加窗/去平面（调用方负责或先用 `compute_fft2`）。
2. `footprint=None` / `patch_half=None` 触发 §4.2/§4.3 的自适应规则；显式传整数则覆盖（供消融实验与验证用）。
3. `q_max_px=None` → `0.95 * n // 2`；`max_peaks=None` → 无上限。
4. 返回的 `peaks` 按 `snr` 降序；`independent=False` 的点是 −q 伙伴，其位置由镜像索引直接给出（不得独立重新拟合，避免 0.5 px 之类的双约定误差，LFA I1）。
5. 任何 `quality != "ok"` 的情形都必须在 `peaks` 中可见，且 `method` 如实标注（PROJ M5）。
6. 模块级只暴露 `__all__` 中的 9 个名字；其余为私有（`_` 前缀）。**不**修改 `utils/__init__.py`。
7. **三个质量门 keyword-only 参数**（§4.4(e)，默认值即规格值，全部在 `meta` 回显）：
   - `chi2_red_max: float = 25.0`（G-3 上限；传 `math.inf` 显式关闭，关闭后 S-2 类风险自担）；
   - `residual_max_px: float | None = None`（G-2 上限；`None` ⇒ 派生 `min(1.0, 0.25·pool_spacing_px)`，显式数值则派生规则不再生效）；
   - `min_pool_spacing_px: float = 3.0`（G-1 下限，px；传 `0.0` 关闭）。
   `BraggPeakDetector.__init__` 通过 `**kwargs` 透传同一组参数；三个参数均为 keyword-only 且有默认值 ⇒ 既有调用不受影响。**行为变更（有意为之）**：对"此前会返回错误晶格"的真实数据（§6.4 的三张图属此类），现在返回 `lattice` 但 `fit_ok=False`、并按 §4.4(f) 收回全部模型声明；合成基准路径的数值与判定不变。
8. **`meta` 决策通道**：`pool_spacing_px`、`residual_max_px`、`fit_ok`、`lattice_quality`、`lattice_error`，以及两个门限回显 `min_pool_spacing_px`、`chi2_red_max`，连同 `inferred_symmetry`、`oblique_gauge`/`oblique_gauge_margin`、`reference_gauge`、`nan_filled`、`degenerate_input` 等诊断键；判决可仅凭 `(result.lattice, result.meta)` 复算。

---

## 6. 基准协议（合成成真值）

### 6.1 合成图构造（固定种子 `20260917`，全部可复现）

```
n = 256 ; L = 30.0 nm ; dq = 2*pi/L ; q_nyq = pi*n/L = 26.8083 nm^-1
a = 2.0 nm (hexagonal) ; b = 4*pi/(sqrt(3)*a) = 3.6276 nm^-1
B_ideal rows: b1 = b*(1,0), b2 = b*(cos60, sin60)          # px 单位除以 dq: |b1| = 17.32 px
M_true = [[1.02, 0.03], [0.01, 0.98]]                      # 非对称，可捕获 x/y 转置与符号错
真值集合: 所有 (h,k) != (0,0) 且 |(h,k) @ B_ideal @ M_true| <= 0.5*q_nyq   (n=256 时 54 个点)
相位 phi_G ~ Uniform(0, 2*pi)，每场景固定种子
image(y,x) = Σ_G A_G * cos(2*pi*(qx_G*x + qy_G*y)/n + phi_G) [+ 高斯包络] + sigma_img * N(0,1)
magnitude  = |fftshift(fft2(image * outer(hann, hann)))|
sigma_n    = MAD/0.448641（环带稳健 Rayleigh 尺度）
SNR_G      = magnitude[round(q_G)] / sigma_n
```

- 两种振幅剖面：`formfactor`：`A_G = exp(-(|q_G|_px/40)^2)`（动态范围 ~50×）；`ladder`：按理想环给 `{首环 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.02}`（LFA B2 的阶梯，用于召回/假阳性）。
- 场景 SNR 定义：**最强 Bragg 峰的 SNR**（参考峰 = 首环中最强者），通过缩放信号振幅设定（噪声实现固定），`probe_spec_v2.scaled_image` 同此。
- 真值 q 精确已知：Hann 窗只改变峰形（窗核 ⊛ 包络变换），**不移动峰的解析位置**，故真值就是 `(h,k)@B_ideal@M_true`（PROBE 无噪实测证明：不变量与相位无关，且真值标号拟合残差 1.97e-4）。

### 6.2 指标与通过阈值（阈值 = 本机实测值 + 余量；括号内为 PROBE 实测）

| # | 指标 | 场景 | 阈值 | 实测 |
|---|---|---|---|---|
| M1 | 逐环 recall，环 SNR ≥ 50 | ladder，最强 SNR ≥ 150 | **每环 ≥ 0.95** | 1.000（SNR 199 / 107 / 51 三环） |
| M2 | 逐环 recall，环 SNR ≥ 15 | 同上 | **每环 ≥ 0.90** | 1.000（SNR 20 环） |
| M3 | 逐环 recall，环 SNR ≥ 5 | 同上 | **每环 ≥ 0.75**；汇总（环 SNR ≥ 8）**≥ 0.90** | 1.000（SNR 8.2 环）；SNR ≈2.9 环不计入（实测 0.056，允许漏） |
| M4 | 逐环 recall，环 SNR ≥ 15 | 低 SNR（最强峰 SNR ≈ 30） | **每环 ≥ 0.85** | 1.000（SNR 31 / 17.3 环）；SNR 7.2 环 0.750 低于阈值、不计入判定 |
| M5 | 假阳性率 FPR = FP/(FP+TP) | 全部场景 | **≤ 0.05**（低 SNR 场景 ≤ 0.10） | 0.00 |
| M6 | 定位 rms₂D（SNR ≥ 50 检出峰） | formfactor，最强 SNR ≥ 100 | **≤ 0.05 px** | 0.0186（另 0.0245 含包络） |
| M7 | 定位 rms₂D（SNR 15–50） | formfactor，最强 SNR ≥ 100 | **≤ 0.12 px** | 0.0565（另 0.0723 含包络） |
| M7b | 定位 rms₂D（SNR 15–50） | 低 SNR（最强峰 SNR ≈ 30） | **≤ 0.15 px** | 0.0602 |
| M8 | 定位 rms₂D（SNR < 15） | 同上（宽松兜底） | **≤ 0.40 px** | 0.224 |
| M9 | 逐轴偏置（SNR ≥ 50） | formfactor | **≤ 0.010 px** | < 0.001 |
| M10 | σ 标定：中位数 pull（逐轴 \|Δq\|/σ） | SNR ≥ 50 与 15–50 两类 | **∈ [0.5, 2.0]** | 1.07–1.77 |
| M11 | σ 量级：中位数 σ_q（SNR ≥ 50） | formfactor | **≤ 0.05 px**（直接否决 1/√12 地板：0.2887 px） | 0.0082–0.0162 |
| M12 | 模型预测精度 rms₂D(q_model) | 全部场景 | **≤ 0.05 px 且 < 逐点 rms₂D** | 0.0078–0.0364 vs 0.0179–0.0602 |
| M13 | 主拉伸绝对误差 \|Δλ\| | 全部场景 | **≤ 2e-3** | 8e-5 – 1.08e-3 |
| M14 | 旋转误差（按点群模 60°） | 全部场景 | **≤ 0.05°** | 0.0012–0.0218 |
| M15 | 异方差场景旋转 rms（3 点 σ=0.03 px + 3 点 σ=0.2 px，500 次） | 直接采样仿真 | **≤ 0.05°**（GLS 通过、无权 OLS 不通过） | GLS 0.0353°，OLS 0.0952° |
| M16 | 异方差场景主拉伸 rms | 同上 | **≤ 1.2e-3** | GLS 8.9e-4，OLS 2.5e-3 |
| M17 | 迭代增益：predict→verify 新增点 | 低 SNR ladder | **≥ 1 个点/次运行**（机制必须真实工作） | 7.5 个点/次（S3） |
| M18 | 运行时间 | 整个回归脚本 | **≤ 180 s** | 探针 4 场景（含 18 次实现）≈ 90 s |
| M19 | 晶格质量门（§4.4(e)：G-1/G-2/G-3 合取） | 合成基准四场景 + 失配规格场景 | **合成四场景 `fit_ok=True` 且 `quality="ok"`（实测 χ² 1.34–1.84、rms 0.0594/0.1191/0.2013/0.0931 px、`pool_spacing_px` 16.84–16.85、`consistent_fraction` 1.000）；失配规格场景 `fit_ok=False` 且每个峰 `index_hk is None`** | 新增验收项（不修改任何既有 M 阈值）；真实数据基线见 §6.4 |

> M1–M4 的**召回按环统计**（同一环内所有点的理论振幅相同；六方环 6 或 12 个点，±q 都计入真值），每环的 SNR 取该环真值位置上 `|F|/sigma_n` 的中位数；只有 **SNR ≥ 阈值的环**参与判定，其余环只需打印（`ladder` 剖面的 0.02 环在最强 SNR=200 时 SNR≈2.9，低于 `min_snr_verify=3.5`，**允许全漏**）。回归脚本必须打印每环的 `(amp, n_truth, SNR_median, recall)` 四元组。实测（§8）：`ladder` 最强 SNR=200 时，SNR 199/107/51/20/8.2 五环 recall 全为 **1.000**，SNR 2.9 环 0.056。

### 6.3 附加场景（功能与鲁棒性）

| # | 场景 | 阈值/断言 | 依据 |
|---|---|---|---|
| F1 | 近邻卫星：在一首环峰 6 px 处置一独立峰（amp 0.3） | 两点都被检出，且定位误差 ≤ 0.05 px | LFA B3（常数 41 会合并）；PROBE footprint 自适应 |
| F2 | NaN 区域（10% 随机像素置 NaN） | **(a)** `nan_policy="plane"` 必须与"预先按同一最佳拟合平面填充"的输入**逐位一致**（≤0.02 px，实测 0.00e+00）；**(b)** 鲁棒性：与未损坏参考图**逐峰漂移 ≤ max(0.05 px, 3σ_q)**（用每峰自己的不确定度解释它自己的位移）、峰数差 ≤ 20%、不抛异常；**(c)** `nan_policy="raise"` 在该输入上必须抛 `ValueError` | PROJ M3；D13（不静默产生垃圾位移）。注：旧文字"与无 NaN 版本 rms 差 ≤ 0.02 px"指代不明且物理上不可达（10% 像素被抹除本身就是信息损失，弱峰必然移动，实测中位漂移 0.0336 px、最差 0.285 px 且集中在 SNR≈5 的峰上），故拆为上述三段，0.02 px 只适用于 (a) |
| F3 | 谱边界峰（|q| 距 Nyquist < patch） | 不抛异常、**不产生谱外坐标**（全部 `|q| ≤ q_nyq·(1+1e-9)`），边界点 `quality="edge"` | PROJ B2（`|q|=957 px`；`(256,1)` ValueError） |
| F4 | ±q 镜像一致性 | 每个非 DC 峰的镜像点存在，且 `| q(−G) + q(G) | ≤ 1e-6 px`；`n_independent == n_non_dc // 2` | RAD §5.3（3.2e-17） |
| F5 | (h,k) 复原（集成） | 用 `lattice_operations` 生成池后 `round(pts.T @ inv(B))` 最大误差 **≤ 1e-9** | PROJ §3.4；本机复核 8.9e-16 |
| F6 | 单位一致性（集成） | `miscellaneous.fft_q_limits(L, n)[0][1] * 10 == q_nyquist_nm_inv`（相对误差 ≤ 1e-12） | 本机复核通过 |
| F7 | 确定性 | 同输入两次调用，`q_px`、`sigma_q_px`、`bvecs_nm_inv` 逐位相同（`np.array_equal`） | LFA I8 |
| F8 | 真实数据（只读 smoke） | `topo0002.sxm` 加载→检测→不抛异常；打印峰数、`|q|`、σ 中位数；断言 `quality="ok"` 的峰 σ 均为有限正数且 `|q| ≤ q_nyq`；文件缺失时打印 `[SKIP]` 且不判失败 | PROJ §4（工作区无 FFT 产物，必须走 raw_data） |

---

### 6.4 真实数据质量门基线（A-6，v5 冻结产物）

以下三张 Si(111)-Pb 岛的 Nanonis 扫描（`/Users/hunfen/Documents/论文/Si111_Pb_islands/raw_data/2025-07-09/`，**只读**）在**默认门**下的实测结果，作为真实数据的验收基线。**三例均被拒**（`fit_ok=False`、标号 0）——这是设计意图：它们各自表达了"退化/失配拟合"，宁可拒也不能把错误基矢派发给下游。

| 图像 | n | L (nm) | dq (nm⁻¹) | χ²_red | rms (px) | `pool_spacing_px` | 首环夹角 (°) | `consistent_fraction` | 拒因（`quality`） | `fit_ok` | 标号 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| topo0002.sxm | 512 | 30 | 0.209440 | 6708 | **1.4925** | 9.312 | 41.57 | 0.1389 | `residual_above_max+chi2_above_max` | False | 0 |
| topo0011.sxm | 512 | 30 | 0.209440 | 8.617e4 | **3.1366** | 10.003 | 43.21 | 0.1944 | `residual_above_max+chi2_above_max` | False | 0 |
| topo0005.sxm | 128 | 900 | 0.006981 | 39.26 | **1.1037** | 0.250 | 65.38 | 0.5556 | `spacing_below_min+residual_above_max+chi2_above_max` | False | 0 |

**rms 列的换算脚注**：三行 rms 按**各自**的 `1/dq` 换算（L = 30 / 30 / 900 nm，dq = 0.2094395 / 0.2094395 / 0.0069813 nm⁻¹，因子 4.775 / 4.775 / 143.239）；**标号集与拟合未变**，与早期记录（0.313 / 0.657 / 0.0077）的差异纯为旧版把 nm⁻¹ 当 px 报告的单位缺陷。（曾有一处推断按统一 dq=0.2094395 换算 topo0005 得 0.0368 px 并据此认为"标号集变了"，该推断不成立。）

**门行为说明**：单位修复后 G-2 不再被放松 `1/dq`，故 topo0002/0011 **额外**触发 `residual_above_max`（1.49 / 3.14 > 上限 1.0），topo0005 为 G-1+G-2+G-3 三重触发；`fit_ok` 三例仍为 False ⇒ **判决不变**。合成基准（同一次运行）四场景 `fit_ok=True`：χ² 1.34–1.84、rms 0.0594/0.1191/0.2013/0.0931 px、`pool_spacing_px` 16.84–16.85、`consistent_fraction` 1.000（G-1 余量 5.6×、G-2 余量 11–80×、G-3 余量 15–19×）。

**对上层使用的含义（预期管理）**：真实 STM 拓扑图直接以默认门跑，**很可能被拒**——被拒时应先做数据侧提质（更大场/更平整、线/平面扣除、降低扫描噪声、确认晶格常数与对称类正确），或用 `LatticeSpec` 显式给出参考基与取向；**不要**为了让它"通过"而放宽门限（三个门限参数暴露出来是给消融实验用的，不是默认调参项）。真实数据断言不进回归脚本（§7 R16 只做不变量 smoke），本节数值属集成证据。

---

## 7. 回归脚本规格：`scripts/regression/check_bragg_peak_detection.py`

**形态**（照抄现有 8 套模板，见 `check_mlwf_im_susceptibility.py` / `check_nanonis_sxm.py`）：

```python
"""Regression checks for stm_data_processing.utils.bragg_peak_detection.

Check list:
  (R1) detection completeness on a synthetic ladder (recall >= ...)
  (R2) false positive rate on the same images
  (R3) sub-pixel localization RMS per SNR class
  (R4) per-axis bias
  (R5) uncertainty calibration (median per-axis pull in [0.5, 2.0])
  (R6) reported sigma magnitude (median sigma_q <= 0.05 px at SNR >= 50)
  (R7) lattice-model prediction beats the per-peak fit
  (R8) affine principal stretches (abs err <= 2e-3)
  (R9) affine rotation, modulo the point-group angle (<= 0.05 deg)
  (R10) heteroscedastic GLS gain over unweighted OLS (rotation rms <= 0.05 deg)
  (R11) predict-verify loop adds peaks on a low-SNR ladder
  (R12) boundary / NaN / |q| <= q_nyquist robustness
  (R13) +/-q mirror pairing and independent counting
  (R14) lattice_operations index recovery and fft_q_limits unit cross-check
  (R15) determinism (two runs identical)
  (R16) real-data smoke on 2025-07-09/topo0002.sxm (read-only, [SKIP] if absent)
  (R17) a near-neighbour satellite 6 px from a first-ring peak is resolved

Run from the repository root:
    .venv/bin/python scripts/regression/check_bragg_peak_detection.py
"""
```

- 结构：`def check_rN() -> None:` + 裸 `assert`；`main()` 遍历 `checks`，`[PASS]/[FAIL]` 打印，末尾 `RESULT: ALL CHECKS PASSED` 或 `raise SystemExit(1)`。
- 所有合成场景用 `np.random.default_rng(20260917)`（或片段化固定种子）；真实数据用 `NanonisFileLoader`（`stm_data_processing.io.nanonis_loader`）只读打开；**不 import matplotlib**（避免 MPLCONFIGDIR 噪声）。
- 运行时间预算 ≤ 180 s；回归脚本必须打印 M1–M18 的**实测数值**（供 t6 抄进验证报告与本文档的数值对账）。
- 断言阈值全部取 §6.2/§6.3 的数值；**任何阈值都不得放宽**（放宽即视为规格变更，需要重开需求流程）。

---

## 8. 复现证据（本文引用的本机实测）

| 脚本 | 作用 | 关键输出 |
|---|---|---|
| `tmp_verify/bragg_spec/probe_spec_v2.py`（+ `.json`/`v2_stdout.txt`） | 按本规格实现算法栈（SNR 自适应检测 + 有界旋转高斯 LSQ + GLS + predict/verify/refit），在 4 个合成场景上量化 M1–M17 | S1/S2/S4（高 SNR）逐环 recall 1.000（SNR 199/107/51/20/8.2 五环）；rms₂D 0.018–0.024 px（SNR≥50）；pull 1.07–1.77；σ 中位数 0.008–0.016 px；FPR 0.00；主拉伸误差 8e-5–3.5e-4；旋转（模 60°）0.0012–0.0077°；模型预测 rms 0.0078–0.0152 px；S3（低 SNR，最强 SNR 30）环 SNR 31/17.3 recall 1.000（SNR 7.2 环 0.750）、FPR 0.00、迭代新增 7.5 点 |
| `tmp_verify/bragg_spec/probe_per_ring.py` | `ladder` 逐环召回与 SNR（最强 SNR=200 与 30 两个工作点） | SNR=200：amp 1.0/0.5/0.25/0.1/0.05（SNR 199.3/106.8/50.6/19.7/8.2）recall **1.000**，amp 0.02（SNR 2.9）0.056。SNR=30：amp 1.0/0.5（SNR 31.0/17.3）recall **1.000**，amp 0.25（SNR 7.2）0.750，amp 0.1（SNR 2.8）0.500 |
| `tmp_verify/bragg_spec/probe_sigma_cal.py` | 标定 σ 配方：MAD→Rayleigh 常数、`absolute_sigma` 取舍 | `MAD/σ_Rayleigh = 0.448641`；用该尺度（`absolute_sigma=True`）pull = 0.88–0.98，而 `absolute_sigma=False` 的 pull = 1.7–2.6（过度自谦） |
| `tmp_verify/bragg_spec/check_label_ambiguity.py` | 验证 affine/旋转的标号模糊性 | 真标号拟合 M 相对误差 **1.97e-4**；自动标号与真标号之差为点群元素（旋转差 **−120.0005°**，模 60° 后 0.0005°），奇异值完全一致（1.028619/0.971740） |
| （内联复核） | `lattice_operations` 复用与单位一致性 | 72 点池；(h,k) 复原最大误差 8.9e-16；`fft_q_limits(30,256)[0][1]*10 == π·256/30` |
| （内联复核） | ruff 基线 | `ruff check src/stm_data_processing/utils/` → `All checks passed!` |

> 说明：PROBE 是**规格可行性探针**（用原库/独立实现量化阈值），不是交付实现；t5 的实现必须独立写成模块，t6 用回归脚本复核。若 t5 的实测数值系统性差于 PROBE（例如 rms₂D 超过阈值的 2 倍），按 §7 最后一条处理：先修实现，不得改阈值。

---

## 9. 与 `lattice_operations.py` 的复用关系（PROJ §3.4 + 本机复核）

| 需求 | 复用方式 | 不做什么 |
|---|---|---|
| 生成理想倒格点池（给定 B_ideal、q_max） | `LatticeLoader.create_lattice(bvecs_array=B_ideal)` → `LatticeOperations(lat).get_bragg_points_in_circle(q_max, include_origin=False)` | 不修改 `lattice_operations.py` / `lattice_loader.py`；不改 `utils/__init__.py` |
| 复原 (h,k) | `hk = np.round(pts.T @ np.linalg.inv(B_ideal)).astype(int)`（该函数不返回指数；`get_bragg_points_supercell_in_1x1_fftshift(..., return_mn=True)` 是超胞 1×1 折叠语义，用途不同） | 不新写第二套"理想点池"生成器（除非 `lattice_operations` 抛错，此时允许回退到模块内 5 行枚举并在 `meta` 标注） |
| 单位一致性 | `bvecs_array` 传 **nm⁻¹**（该函数只做 2πδ 一致性校验、不校验单位；PROJ §3.3 实测传入 nm⁻¹ 得到 `avecs = 0.384 nm` 正确）；并用 `miscellaneous.fft_q_limits`（Å⁻¹）做 ×10 交叉断言 | 不在模块内混用 Å⁻¹/nm⁻¹；对外只承诺 nm⁻¹ |
| 平面扣除 | 复用 `utils.plot_funcs.subtractMeanPlane` | 不修改该函数 |

---

## 10. 风险与处置（无开放问题）

| 风险 | 处置（已定案） |
|---|---|
| real-data 的"真值"不可得（PROJ §4：真实峰是 ~4.5 nm⁻¹ 的大尺度特征，不一定是 1×1/7×7 原子峰） | 真值只在合成场景断言；真实数据只做 smoke 不变量断言（F8）；定量结论全部来自合成成真值 |
| 无 `LatticeSpec` 时的对称类推断可能误判（首环缺峰） | 推断是 best-effort：候选类并行验证、按"验证通过点数"取胜、平局取 |det B| 大者；结果写入 `meta["inferred_symmetry"]`，用户可用 `LatticeSpec` 覆盖；`affine` 在此路径下为 `None`（避免给出无定义的畸变） |
| 标号模糊性导致 rotation 误读 | 显式输出 `rotation_mod_deg` 与 `rotation_is_absolute`；文档与 docstring 明写"未给参考取向时旋转只到点群模数"；回归按压模数断言 |
| 弱峰定位误差大（SNR < 15 时 rms₂D 0.22 px） | 逐点 σ_q 如实给出并由 M10 标定；弱峰若与晶格一致应改用 `q_model_px`（M12 要求模型预测优于逐点） |
| 窗口形状失配（Hann 核非高斯，残差 27.2%） | 用模型地板 0.006 px 覆盖系统项（D9）；不引入窗核模型（non-goal 6） |
| patch 越过 DC 掩模/谱边界 | 显式拒绝：`quality="edge"`，不参与晶格 GLS，绝不环绕切片（D13/F3） |
| 稀疏点（<3 独立点）时无法解 affine | `lattice=None` 或 `converged=False` + 英文异常/诊断；绝不静默返回错误 affine（对比 LFA E3、PROJ B3） |

---

## 11. 下游交接清单（t5 → t6 → t7）

1. t5 交付：`src/stm_data_processing/utils/bragg_peak_detection.py`（API 与 §5 逐字一致）、`scripts/regression/check_bragg_peak_detection.py`（§7 的 R1–R16 全部实现）；改动范围仅这两个新增文件 + 可选的 `docs/stm_data_processing/utils/bragg_peak_detection.md` 接口文档（如需，按 `docs/stm_data_processing/README.md` 的索引风格）。
2. t5 必须给出：`.venv/bin/python scripts/regression/check_bragg_peak_detection.py` 的 exit 0 实测输出，以及 `ruff check src/stm_data_processing/utils/` 的 `All checks passed!`。
3. t6 独立验证：复跑回归脚本、复跑既有 8 套 `check_*.py`（不得回归失败）、在 `topo0002.sxm` 上做真实数据实测并记录峰数/`|q|`/σ；对 §6.2 的 M1–M17 与 PROBE 的对应数值做**数量级对账**（若实测优于 PROBE 需说明原因）。
4. t7 评审：核对实现与本文档逐条一致（特别是 §4.2 噪声/自适应规则、§4.3 界与质量门、§4.4 GLS+迭代、§4.5 三层不确定度、§5 API 字段名与单位），核对 non-goals 未被越界实现。
