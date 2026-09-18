---
name: stm-topo-phase-analysis
description: STM 拓扑图 CSV 全流程处理：Bragg 峰仿射矫正（调用 STM_DataProcessing 的 bragg_peak 包，显式 LatticeSpec，石墨烯 a=0.246 nm、无旋转对称拉伸）、gwyddion/inferno 绘制、r3 Bragg 峰 mask 复数 iFFT 相位分析与 Kekulé 相位提取（6 个 r3 峰分别做，q 由包内 17×17 高斯亚像素定位给出）。当用户提供 STM 拓扑图 CSV 及图像边长（nm），要求矫正、绘图或相位分析时使用。
---

# STM Topo CSV → Correction → Phase Analysis

## 输入

- 拓扑图 CSV/txt：正方形数值矩阵，可含 NaN；**第一行 = 扫描起始行 = 图像顶部**
- 图像边长 L (nm)：正方形扫描
- 可选：石墨烯晶格常数 a（默认 0.246 nm）、mask 半径百分比（默认 5%）

## 依赖与执行环境

- 峰检测、亚像素定位、对称正定拉伸矩阵与重采样**全部来自库内包** `stm_data_processing.utils.bragg_peak`（`detect_bragg_peaks` / `correct_bragg_peaks` / `compute_fft2`）；本 skill 只负责读入、预处理（去平面 + 上下翻转）、显式声明参考晶格与绘图
- 参考晶格必须**显式给出** `LatticeSpec(a_nm=a, symmetry="hexagonal", bvecs_nm_inv=None)`（不做点群自动推断）。**峰检测本身仍走包内数据锚定路径**（`detect_bragg_peaks(..., lattice=None)`：把六方基矢锚在第一环上），再用显式 spec 做几何校正；用 spec 直接做检测会在数据相对理想晶格偏离超过匹配容差时锁到错峰（实测 30 nm 数据：spec 检测把 141 px 的弱峰当 (1,0)，真实 135 px 环漏标；数据锚定则正确标出 134.6/135.3/137.5 px 三峰）。
- spec 的**取向**取数据自身拟合出的 (1,0) 方向（`lattice_orientation_deg`）：对称拉伸无旋转，若把未旋转的理想基矢直接喂给拟合，取向差会被吸收成假剪切
- 本机系统 Python 无 numpy/matplotlib，必须在库目录下用 uv 环境执行：
  `cd /path/to/STM_DataProcessing && uv run python <script> ...`（脚本默认 `--stm-lib` 指向本库 `src`）
- 字体：**禁用 `text.usetex`**（本机 matplotlib 3.10 + TeX Live 2026 有编码 bug，`$-\pi$` 会被渲成 `hBc\pi`）；用 `mathtext.fontset='cm'` + `font.serif=['Palatino']` 代替

## 流程

### 第一步：矫正（scripts/stm_topo_correct.py）

```bash
cd /path/to/STM_DataProcessing
uv run python skills/stm-topo-phase-analysis/scripts/stm_topo_correct.py \
    topo0009.txt -L 100 -o data_processing
# tab 分隔文件（如 20251117_topo4_30nm.csv）加 --delimiter '\t'，逗号 CSV 加 --delimiter ','
# 需要时用 --peaks Q1X Q1Y Q2X Q2Y 手动指定锚定峰对（q 像素偏移），--list-peaks 打印峰表
```

步骤：
1. 读入（默认自动识别 tab/逗号/空白分隔）→ `flipud(subtractMeanPlane(data))`（去平面 + 上下翻转，保证 origin=lower 方向正确）
2. `detect_bragg_peaks(topo, L, patch_half=8)`：包内完成 Hanning 窗 FFT、径向背景 SNR 峰检测、17×17 高斯亚像素定位、环聚类与晶格拟合/标注（第一环 = 石墨烯 1×1 环）
3. 取该检测拟合出的 (1,0) 方向作为 spec 取向 → `LatticeSpec(a_nm=a, symmetry="hexagonal", orientation_deg=θ)`
4. `correct_bragg_peaks(topo, L, lattice=spec, result=<上面的检测>)`：包内用**全部带标签独立峰**加权最小二乘解对称正定拉伸 `M`（纯拉伸、零旋转，`q_obs = q_ideal @ M`，理想基矢按 spec 的 a 与 θ 给出），检查正定性，失败时回退双矢量闭式解并在 `meta` 中记录
5. 重采样：`scipy.ndimage.affine_transform(order=3, cval=NaN)`，画布由包按变换后的四角 + pad 给出（`n_out`），**校正后视场 = L · n_out / n**（nm/px 尺度不变）
6. 打印：`|b1|` 校正前实测值 / 理想值、`M`、画布边长、NaN 比例，以及**对校正图重新检测**得到的 `|b1|` 校正后实测值

**可用范围与已知限制**：矫正必须先有**带标签**的晶格，而标号受包内检测自带的容差限制——环聚类要求 60° 三重峰的半径彼此一致到 3 %（`rings._RING_RADIUS_TOL`），`match_labels` 要求峰位与本征 |q| 相差不超过 2 %（另有 2 px 下限）。因此**各向异性大约 ≤ 3 % 才可用**。更强各向异性（以及天生各向异性、没有 60° 三重峰的矩形/方晶格）会连晶格都拟合不出来：`method="identity_fallback"`、`fallback=True`、`n_labelled=0`、`affine_q=affine_image=I`，**什么都没矫正**——此时返回数组只是把原数据重铺到更大的 NaN 画布上（n 为偶数时中心对齐还会落在半像素偏移上：实测 n=512 时 offset=−10.5 px、与原数据最大差 2.04 而信号幅度 0.68；奇数 n 才逐点复原到 1e-15），不要把它当成已矫正图像，日志给出 `no positive-definite stretch; correction skipped`，检测侧给出 `lattice_error="no hexagonal ring found in the candidate set"`；实测 5 %/3 % 各向异性合成几何即落到该回退。该情形**不静默**：`meta` 与警告日志都会显示，本脚本也会打印 fallback 标记——**信任矫正结果前务必先检查 `meta['method']` / 打印的 fallback 标记**。更强各向异性与畸变矩形/方晶格需要宽容差对应路径（路线图 L1，见 `docs/design/bragg_peak_detection.md` 第 9 节）。

输出：
- `<stem>_corrected.csv` — 矫正后数据（正方形，NaN 填充）
- `<stem>_corrected_fft2.npy` — 复数 FFT2（complex128，fftshift，DC 居中；NaN 按最佳拟合平面填充后加 Hanning 窗，`subtract_plane=False`）
- `<stem>_corrected.png` — 拓扑图（gwyddion colormap，origin=lower，NaN 灰色）
- `<stem>_corrected_fft.png` — FFT 图（inferno + log + 百分位归一化 [5, 99.5]）

### 第二步：相位分析（scripts/stm_phase_analysis.py）

```bash
cd /path/to/STM_DataProcessing
uv run python skills/stm-topo-phase-analysis/scripts/stm_phase_analysis.py \
    tmp_verify/bragg_correct/topo0009_corrected.csv -o OUT_DIR --pct 5 --bins 1024
# 可选 -L/--size-nm 只影响包内检测/标注（FFT 本身与尺寸无关），默认 100
```

三种情况（建议输出到独立文件夹并按类分目录）：
1. **nomask**：直接对复数 FFT2 本身（q 空间）画 amplitude（log, inferno）+ phase（seismic），不做 mask 不做 iFFT
2. **r3all**：6 个 r3 内圈峰一起 mask，mask 与补集分别复数 `ifft2(ifftshift(...))` → amplitude（log, gwyddion）+ phase（seismic），NaN 灰色，complex128 存 npy
3. **per-peak（核心）**：6 个 r3 峰**分别**做 Kekulé 相位分析：
   - r3 峰位置与亚像素 q：由包内 `detect_bragg_peaks(..., patch_half=8)`（17×17 高斯定位器）给出；峰号按 qy 升序（12 点钟起顺时针）r3p0..r3p5
     ⚠️ **禁止做全局平面修正**：曾用 unwrap+平面拟合扣梯度，拟合吸收 16px 伪梯度，把 Z3 三峰抹平（0.16 vs 0.15 背景）；正确做法是 q 从峰位一次取准，不做任何二次修正
   - 单峰 mask（半径 p%×N）→ 复数 iFFT → ψ(r) = Δ e^{i(q·r+φ)}
   - φ(r) = angle(ψ) − q·r (mod 2π)（坐标用居中坐标）
   - amplitude 加权直方图（360 bins，p50 阈值像素），Z3 标记线 0/2π/3/4π/3
   - φ(r) 相位图（hsv）

输出结构（每个峰 r3p{i}，12 点钟为 r3p0，顺时针；文件名与旧版一致）：
- `r3p{i}_in.png` `r3p{i}_out.png` — mask/补集 amplitude+phase
- `r3p{i}_mask_{in,out}_ifft.npy` — complex128
- `r3p{i}_kekule_phi.png` — φ(r) 图 + φ 直方图
- `r3_peaks_kekule_phi_hist.png` — 6 峰 φ 直方图 2×3 汇总
- `r3_peaks_kekule_phi_map.png` — 6 峰 φ(r) 图 2×3 汇总
- `all_cases_phase_histograms.png` — 0/π 直方图（第一行 OUT+nomask，第二行 IN）
- `nomask.png`、`r3all_in.png`、`r3all_out.png` + `r3all_mask_{in,out}_ifft.npy`
- `combined_kekule_phi.png`、`combined_theta_field.npy`、`combined_product_psi.npy`

## 物理要点（Kekulé 分析）

- **0/π 双峰与 Z3 锁定不冲突**：含 Friedel 对的 iFFT 是实数 ρ₃ = 2|Δ|cos(q·r+φ)，像素相位只有 0（正）/π（负）——实数信号的 Z2 折叠，永远是两值
- Z3 信息在**单侧**分量：φ(r) 直方图应有 0/2π/3/4π/3 三峰（三 domain 锁定）或单峰（单 domain 主导）
- 实数测量有 φ → -φ 简并：Friedel 对的两个峰（如 r3p0 与 r3p5）测到的锁定值互补（240° ↔ 120°）
- ±π 是同一相位被直方图首尾 bin 切开

## 关键约定

- 拓扑图统一 gwyddion colormap；FFT 图 inferno；phase seismic；amplitude log 尺度
- 全程复数 complex128，iFFT 结果不取 `.real`
- 绘图屏蔽 NaN：`np.where(valid, arr, np.nan)` + colormap `set_bad(grey)`
- 峰编号按 r3 内圈 12 点钟起顺时针：r3p0..r3p5
- 参考晶格显式声明：`LatticeSpec(a_nm=a, symmetry="hexagonal")`（a 默认 0.246 nm），不做点群自动推断

## 修改记录

2026-09：初版（topo0009 实测流程）；增加 6 峰 Kekulé 相位分析（exact-q 方法，无全局平面修正）。
2026-09（入库）：skill 移入仓库 `skills/stm-topo-phase-analysis/`，两个脚本改为调用 `stm_data_processing.utils.bragg_peak` 包（峰检测/亚像素定位/对称正定拉伸/重采样），删除各自重复实现的 `detect_peaks`/`gauss2d`/`subpixel_q`/`symmetric_stretch`；矫正脚本新增 `--list-peaks`，相位脚本新增可选 `-L/--size-nm`。矫正流程：包内数据锚定检测 → 取数据 (1,0) 方向构造显式 `LatticeSpec(a_nm=a, orientation_deg=θ)` → 几何校正。
