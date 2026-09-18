# FFT Bragg 峰检测 —— 设计规格（模块化 `bragg_peak` 包）

> 本规格**替代**旧单体版规格（2026-09，`src/stm_data_processing/utils/bragg_peak_detection.py` 2515 行及其"三质量门拒收真实数据"设计）。重构由 AgentTeams 团队 bragg-peak-rewrite 完成并经两轮评审（t1 实现 → t2 独立验证 → t3 评审 needs_revision → t4 修复 → t5 复审 **pass**）；本文与 `src/stm_data_processing/utils/bragg_peak/` 的最终实现一致。

## 1. 目标、范围与非目标

**目标**：对一张 STM 拓扑图（方形数值矩阵 + 场边长 nm）做四件事——① 检出尽可能多的 Bragg 峰（石墨烯 1×1 环及其谐波环、更弱的高阶 Bragg 点）；② 每个峰亚像素 `q` 定位并给出逐点不确定度；③ 拟合倒格基矢（GLS 2×2 affine）并给每个峰整数指数 `(h, k)`；④ 报告畸变（旋转、主拉伸）。**真实数据在默认参数下必须直接可用**——不存在"默认拒绝"的语义。

**非目标**：超结构/摩尔纹峰的独立建模（以未标号峰保留在结果中）；扫描畸变矫正；相位分析（另有 skill/stm-topo-phase-analysis）；晶格结构解析之外的推理（斜方只保留最简单的两点种子）。

## 2. 模块结构与预算

包位于 `src/stm_data_processing/utils/bragg_peak/`；旧路径 `bragg_peak_detection.py` 是 29 行兼容 shim（re-export，不实现）。

| 模块 | 职责 | 行数（≤350） |
|---|---|---|
| `models.py` | `LatticeSpec` / `BraggPeak` / `LatticeFit` / `BraggDetectionResult` 数据容器与全局约定 | 124 |
| `preprocess.py` | 输入校验、NaN 平面填充、去平面、窗函数、`load_image` ASCII 矩阵读取 | 116 |
| `fft.py` | `compute_fft2`（Hann 窗 + 去平面 + fftshift 复数谱）、幅度、q 轴换算 | 45 |
| `detect.py` | 径向背景估计（稳健 Rayleigh 尺度）、SNR 图、自适应 footprint NMS、候选列表 | 166 |
| `localize.py` | 亚像素定位：log-parabola 3×3 种子 + 2D 高斯 LSQ（含协方差），退化到整数极大 | 289 |
| `lattice_fit.py` | 理想基矢、两点斜方种子、整数池标号、GLS affine 拟合、旋转/主拉伸 | 284 |
| `rings.py` | 六方环聚类（60° 三元组）、参考环推断（谐波阶梯规则）、`ring_model` 旋转搜索 | 247 |
| `pipeline.py` | `detect_bragg_peaks` / `BraggPeakDetector` 编排、退化输入与失败分支 | 295 |
| `__init__.py` | 公开 API re-export | 29 |
| **合计** | | **1595（≤1600）** |

依赖方向无环：`models ← {preprocess, fft, detect, localize, lattice_fit}`；`rings → lattice_fit`；`lattice_fit, rings, detect, fft, localize → pipeline`。包内模块名用 `lattice_fit.py`（**不是** `lattice.py`——后者是 `utils/lattice.py` 的 3D 晶体学 `LATTICE` 类，与 Bragg 包无关，改名即为此避撞）。回归脚本 `scripts/regression/check_bragg_peak_detection.py` 593 行（≤700）。

## 3. 全局约定（实现必须照抄）

- 所有 2-D 矢量是 `(qx, qy)`；数组索引是 `[row=y, col=x]`。
- `_px` 位置是**有符号 fftshift 偏移**：`q_px = index − n // 2`；`q_nm_inv = q_px · 2π/size_nm`；`q_nyquist = π·n/size_nm`。
- `snr` 是**检测**信噪比：相对径向 |FFT| 背景的 z 分数（§4 S2），**不是** `|F|/noise_sigma`。
- `cov_q_px` 是**总**位置协方差（高斯 LSQ 协方差 + 模型地板 `sigma_model_floor_px²·I`），GLS 晶格拟合直接用它加权；`sigma_q_px = sqrt(diag(cov_q_px))` 恒成立。
- 半平面：只对 `+q` 半平面（`qy > 0` 或 `qy == 0 ∧ qx > 0`）的代表独立拟合；其镜像伙伴 `conjugate_index` 由 `|F(−q)| = |F(q)|` 给出，`independent=False`。
- **失败语义（核心）**：`fit_ok` 只报告 GLS 收敛，不是数据质量门；任何失败分支都**保留全部峰的位置、不确定度与诊断量**，只清空模型声明（`index_hk`/`q_model_px`/`sigma_q_model_px`/`residual_px` 置 `None`、`lattice=None`）。不存在"门拒收"路径。

## 4. 算法栈

### S0 预处理（preprocess）
`validate_image`（方阵、尺寸校验）→ `fill_nan_with_plane`（`nan_policy="plane"` 默认，非有限像素填最佳拟合平面值）→ `subtract_plane`（去线性背景斜坡）→ `window_array`（默认 Hann）。

### S1 FFT（fft）
`compute_fft2` = `fftshift(fft2(窗内积))`，complex128；输出复数谱 + 幅度谱 + q 轴。

### S2 候选检测（detect）
统计量是**径向** SNR：

```
z(r) = (|F(r)| − level(r)) / sigma(r)
```

`level`/`sigma` 取 2 px 宽径向 bin 的运行中位数与 MAD→Rayleigh 尺度（`_MAD_OVER_RAYLEIGH = 0.448641`）。径向归一化对真实数据是必须的：STM 谱背景随 |q| 陡降且带 1/f 脊，全局单一噪声尺度既会淹掉弱的高阶 Bragg 点、又会让低 |q| 脊霸占排序。流程：DC 区排除（`dc_mask_frac=0.03`）→ `q_max_px`（默认 `0.95·n/2`）→ `maximum_filter` 自适应 footprint 非极大值抑制 → 候选（`min_snr=4.0`，上限 `max_candidates=2048`，按 SNR 截断）。旧单体在 30 nm 图上产出 5042 个峰，本设计在同样输入上产出 48 个峰（4066 预截断候选 → 2048 上限）。

### S3 亚像素定位（localize）
对每个候选：3×3 log-幅度抛物面给出种子 → 带平面背景的 2D 高斯 LSQ（有界、旋转协方差）→ 位置 + 2×2 协方差；边界/陡背景候选合法地退化为整数极大（`method="max_pixel"`，量化 σ = 1/√12）。真实数据实测 σ_q ≈ 0.006 px。

### S4 环聚类与晶格（rings + lattice）
- **`ring_clusters`**：一个环 = 三个测得峰位于 `θ, θ+60°, θ+120°`、半径容差内（真实环是椭圆，30 nm 标准图三成员半径差 2.1%，故容差取松）。每个环报告半径、SNR 与成员三元组。
- **`ring_model`（R-1 修复点）**：对环的三个成员，在 `_FIRST_RING_LABELS` 的 **3 个循环旋转**上各做一次 GLS，每次都锚定在**承担 (1,0) 标号成员的真实（未折叠）arctan2 角**上，取 χ² 最小者为解。修复前用折叠角（`angle % 180`）排序 + 固定标签序，轴对齐晶格在 7/20 种子丢失晶格（废 affine，χ²≈1.2e8）；修复后 12/12 通过，R1.8 把该失败签名钉死为回归项。
- **`reference_model`（lattice=None 推断）**：逐一试每个环的 `ring_model`，按其整数标号能解释的环数打分；取**最大**的、能解释 ≥`_LADDER_MIN_RINGS` 个环的环为参考（"自带谐波阶梯的最大环"）。该规则解决嵌套歧义：√3×√3 R30° 超结构环位于 |b|/√3，其反射包含基本晶格全部反射，只数"能解释的反射数"永远偏好更细的胞；100 nm 标准图上最强环（273.5 px = 17.2 nm⁻¹）正是 √3R30 环，而 1×1 环（474 px）带自己的谐波阶梯，正确胜出。显式给 `LatticeSpec` 时以其为参考（绝对标号）。
- **`fit_lattice`**：`q_obs = (h,k) @ b_ref @ M` 的加权 GLS，输出 `bvecs_nm_inv` + 4×4 协方差、`affine`/`cov_affine`、旋转（模点群 60°/90°/180°）、主拉伸；`match_labels` 在容差（2%·|q| 或 2 px 下限）内给整数标号。
- **失败分支**：候选为空/无环 → `lattice=None`、空峰集、`meta["degenerate_input"/"lattice_error"]`；拟合不收敛或秩亏 → `lattice=None` + 清空全部标签，峰照常上报。

## 5. 公开 API

| 入口 | 说明 |
|---|---|
| `detect_bragg_peaks(image, size_nm, *, lattice=None, min_snr=4.0, dc_mask_frac=0.03, q_max_px=None, max_candidates=2048, max_peaks=None, footprint=None, patch_half=3, sigma_model_floor_px=0.006, window="hann", subtract_plane=True, nan_policy="plane", return_fft2=False)` | 一步完成 S0→S4 |
| `BraggPeakDetector(**kwargs)` + `.detect(image, size_nm)` / `.detect_from_fft2(fft2, size_nm, *, return_fft2=False)` | 复用配置；`detect_from_fft2` 接受已 fftshift 的复数谱 |
| `compute_fft2(image, size_nm, *, window="hann", subtract_plane=True, nan_policy="plane")` | 只做预处理 + FFT |
| `load_image(path)` | 读 tab/逗号/空白分隔的方阵 ASCII（`.txt`/`.csv`） |

`BraggDetectionResult` 关键字段：`peaks`（按 SNR 降序，含 −q 镜像伙伴）、`lattice`（`LatticeFit` 或 None）、`size_nm`/`n_px`/`dq_nm_inv`/`q_nyquist_nm_inv`、`noise_sigma`、`n_candidates`、`fft2`（仅 `return_fft2=True`）、`meta`（环表、`basis_source`、`harmonic_ladder`、各拟合诊断）。

## 6. 验收与回归（`scripts/regression/check_bragg_peak_detection.py`，35/35，exit 0）

- **R1 合成真值**：六方 a=0.246 nm、n=512、L=30 nm、固定种子多壳层——一环召回 6/6、SNR≥30 峰 rms₂D ≤ 0.1 px、|b₁| 误差 ≤ 0.5%（实测 0.002 %）。**R1.7**：12 个轴对齐取向（每 30°）扫面，每个都必须拟合出晶格并正确标号（实测 12/12，最差 |b₁| 0.009 %、成员残差 0.009 px）。**R1.8**：跨 ±q 边界的对抗环——旧"钉死锚 + 固定标签序"配对 χ²=3.2e7 无标号，新旋转搜索 χ²=2.6e-24 三成员全标号 0.000 px 残差。
- **R2 标准数据验收**（数据只读、md5 前后一致、文件缺失时优雅 SKIP）：两套标准图各断言——6 个一环峰（公共半径 r₁、RMS 与最大偏差 ≤2%、角距 60±4°）、r₁∈[0.93,1.07]·29.49 nm⁻¹、≥4 个更弱 Bragg 峰（|q|∈[1.2r₁, 3r₁]）、`fit_ok=True`、反演 a 与 0.246 nm 偏差 ≤±7 %、单图 ≤60 s。
- **R3 证据**：每套数据一张 log 幅度 FFT + 峰叠加 PNG（一环青色 `#33d1ff`、弱峰橙色 `#ff9f40`）写入 `tmp_verify/bragg_rewrite/`。

复现命令：

```bash
.venv/bin/python scripts/regression/check_bragg_peak_detection.py   # 35/35，约 33 s
for f in scripts/regression/check_*.py; do .venv/bin/python "$f" || exit 1; done
.venv/bin/python -m ruff check src/stm_data_processing/utils/ scripts/regression/check_bragg_peak_detection.py
```

## 7. 标准数据实测基线（t5 复审钉死修订版实测，默认调用、不调参）

| 量 | topo0009.txt（2048²，100 nm） | 20251117_topo4_30nm.csv（1024²，30 nm） |
|---|---|---|
| 一环半径 r₁ | 473.70 px = 29.7635 nm⁻¹（1.009× 理想 29.4927） | 135.78 px = 28.4378 nm⁻¹（0.964×） |
| 一环峰 | 6 个，标号 ±(1,0), ±(0,1), ±(1,−1) | 6 个，同上 |
| 半径散布 rms / 最大偏差 / 全距 | 0.81 % / 1.10 % / 1.92 % | 0.90 % / 1.24 % / 2.12 %（真实椭圆） |
| 角距（最大偏差） | 58.90/60.24/60.86°（1.10°） | 59.94/58.81/61.25°（1.25°） |
| 更弱标号峰（[1.2r₁, 3r₁]） | 10 | 14 |
| `fit_ok` / 参考来源 / 拟合 rms | True / `harmonic_ladder` / 1.245 px | True / `harmonic_ladder` / 0.244 px |
| 反演 a | 0.24316 nm（−1.15 %） | 0.25738 nm（+4.62 %） |
| 峰数 / 预截断候选 / 环数 | 42 / 22524 / 7 | 48 / 4066 / 8 |
| 运行时间 | 14.3 s | 14.3 s |
| 数据 md5 | `0ffeb0a562b9c6de6cfe6dfeb03108b2`（不变） | `e1536e87510a2e56433d0d0cf8b4e6ad`（不变） |

## 8. 已知限制（诚实记录）

1. **参考环"最大谐波阶梯"规则**对带强 2× 谐波阶梯（多壳层落在 q_max 内）的人造晶格可能取到因子 2 的谐波环（如 a=1.5 nm、n=256、L=30 nm 时 |b₁|=2× 真值）。真实石墨烯数据与 R1 不受影响；R1.7 用 a=0.5 nm（两环）配置回避。候选修法：要求所选晶格解释含最强峰的环。
2. **R-6（测试加固建议，非阻塞）**：R1.7 扫面只变取向不变噪声种子，旧代码在该配置下也 12/12；判别守卫是 R1.8。建议后续把噪声种子加入扫面变量（种子变扫下旧代码 10/12 vs 新 12/12）。
3. `max_peaks=0` 视为"无上限"（退化输入，不单独处理）。
4. spread 断言口径：RMS ≤2 % 与最大偏差 ≤2 % 为断言项，全距（离群敏感）只打印——30 nm 图一环是真实椭圆（全距 2.12 %）。
5. 斜方对称只支持两点种子，不发布斜方应变口径（无参考基时形变是重言式）。

## 9. 与旧单体的对照（已移除项）

| 旧机制 | 处置 |
|---|---|
| G-1/G-2/G-3 三质量门、"拒收真实数据即设计意图" | **删除**；`fit_ok`=收敛，失败不清峰 |
| χ² 门拒收语义（`chi2_red_max` 等 3 个门参数） | 删除；χ² 只进诊断 |
| predict→verify→localize→refit 循环 | 删除；单次 GLS |
| 斜方 gauge 机制（`oblique_gauge`/margin） | 删除；只留两点种子 |
| `_seed_assignments` 全循环×手性打分 | 等价物重写为 `ring_model` 三循环旋转 + 真角锚定（R-1 修复） |
| 7 个旧调参 kwarg（`min_snr_seed`/`min_snr_verify`/`lattice_tolerance`/`max_iterations`/`residual_max_px`/`min_pool_spacing_px`/`chi2_red_max`） | 删除（仓库内 0 调用方） |
| `io.lattice_loader.LatticeLoader` 依赖 | 解除 |
| 旧回归 R1–R25 契约 | 由新 R1/R1.7/R1.8/R2/R3（35 检查）替代 |

## 10. 审查与证据链

- t5 复审报告（含全部实测数字与探测）：`tmp_verify/bragg_rewrite/review.md`；首轮报告存档 `review_round1.md`。
- t2 独立验证：`tmp_verify/bragg_rewrite/verification_report.md`（命令、行数预算、md5 前后一致、逐项 PASS）。
- 实现说明与 API smoke：`tmp_verify/bragg_rewrite/implementation_notes.md`、`api_smoke.py`。
- `tmp_verify/` 已 gitignore，仅本地；以上数字均可用 §6 命令复现。
