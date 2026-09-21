---
name: phase-analysis
description: STM 拓扑图（矫正后或任意正方形 CSV）的双环相位数学分析 v3：Gaussian 窗局域引擎（local-q-map，λ=3.0 nm，θ=arg ψ≈+φ 无斜坡）给出 1x1 与 r3 两族逐反射相位统计（幅度加权圆均值/中位数/半峰宽/簇/幅度/三重积/折叠分布，含 Friedel 恒等式自检与 r0 规范固定），并做论文式成对相位差分析（组内 3 对/环 + 跨环 6 对：D 场、归一化幅度差场、2D 直方图）；完整图集 31/环 + 跨环 19 = 81 张与一键数学 self-test。只做几何与相位数学，不做物理解释。当用户给出拓扑图 CSV 与图像边长（nm），要求两族环的相位/相位统计/成对相位差时使用。
---

# phase-analysis（STM 拓扑图：双环相位数学，v3）

**范围**：本 skill 只做**相位数学**——局域解调场、圆统计量、规范固定、成对相位差、图集与自检。
（几何矫正已拆分为独立 skill `affine-correction`。）**物理解释由使用者进行**：本 skill 的
脚本、图标题、log 与文档都不给出物理结论，也不使用任何 k 空间/高对称点类称呼。

两个环只有两个名字，归属只看**半径比**：

| 名字 | 定义 |
| --- | --- |
| `ring_1x1` | 参考环：它的六个峰做最小二乘定出晶格原点 r0，相位以它为规范 |
| `ring_r3` | 半径是 `ring_1x1` 半径的 **1/√3** 的环（容差可配，默认 3 %） |

---

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | 正方形拓扑数值矩阵 CSV/txt（可含 NaN）。**第一行 = 扫描起始行 = 图像顶部**。典型输入 = `affine-correction` 的矫正产物（见 §2）；也接受任意方形 CSV（不经矫正，配 `-L` 或 `--size-nm-from-log`） |
| 视场 L (nm) | 正方形扫描边长，**必须显式给出**（文件名里的标注不可信） |
| 分隔符 | 自动识别；显式用 `--delimiter ','` / `--delimiter '\t'` |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`。**不要用 `uv run`**（会写仓库 `.venv`） |
| 解调引擎 | **`skills/local-q-map/scripts/localqmap.py` 的 `demodulate`**，本 skill 经 `phasepipe.gaussian_field` 调用（同目录安装即可；不复制、不修改引擎代码） |
| 绘图 | `text.usetex=False`（mpl 3.10 + TeX Live 2026 有编码 bug），mathtext cm + Palatino |

## 2 输入来源：矫正产物（affine-correction skill）

几何矫正已拆分为独立 skill `affine-correction`（`skills/affine-correction/scripts/stm_topo_correct.py`）：
读入 → 预处理（`flipud(subtractMeanPlane(...))`）→ 数据锚定峰检测 → 对称正定拉伸（纯拉伸、零旋转）
→ 重采样。其产物里本 skill 消费两个：

| 产物 | 本 skill 的用法 |
| --- | --- |
| `<stem>_corrected.csv` | 输入图像（正方形、NaN 填充） |
| `correction.log` | `--size-nm-from-log` 解析矫正后画布视场（依赖 `corrected canvas ... field of view X nm` 行，见 §3） |
| `correction_report.json` | 机器可读报告（`affine_q`、锚定自检等；本 skill 不消费） |

矫正的锚定环显式指定（`--anchor-ring 1x1|r3`）、锚定自检（1:√3 环对 + 全局拉伸尺度双 tell-tale）、
`identity_fallback` 限制等见 `affine-correction` 的 SKILL.md。

## 3 相位分析（`scripts/stm_phase_analysis.py`）

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  skills/phase-analysis/scripts/stm_phase_analysis.py \
  OUT/INPUT_corrected.csv -o OUT/phase --size-nm-from-log OUT/correction.log \
  --lambda-nm 3.0 --gate p50 --anchor auto
```

**`--size-nm-from-log` 读的是矫正后画布的视场**：`affine-correction` 的 `stm_topo_correct.py` 产出的
log 里有两处
`field of view <value> nm`——第一处是**输入画布**（`# canvas ... field of view 50 nm`），
第二处是**矫正后画布**（`# corrected canvas: ... field of view 51.7090 nm`）。本脚本分析的是
**矫正后的 CSV**，所以解析按以下优先级取值：
① 优先取带 `corrected canvas` 标签的那一行；
② 没有该行时取 log 里**最后一处**匹配（单行 log 即那一行）；
③ 一处都没有 → 报错退出。
取值后 log 打印一行来源说明（如
`# field of view from log: 51.7090 nm (corrected canvas line; .../correction.log)`），
`phase_stats.json` 的 `field_of_view_source` 同文。手工指定 `-L` 时以命令行值为准（同一行也打印来源）。

### 3.1 解调引擎与相位约定（全脚本唯一一套定义）

```
ψ_q(r) = FFT⁻¹{ FFT[ T(r)·e^(−i q·r) ] · e^(−λ²|k|²/2) }        λ = --lambda-nm（默认 3.0 nm）
θ_q(r) = arg ψ_q(r)  ≈  +φ_q(r)                                    （无 q·r 斜坡、无每峰常数）
```

引擎来自同仓库的 `local-q-map` skill（`localqmap.demodulate`），`λ_px = λ_nm / (L/N)`；
本 skill **不含**任何硬圆 mask 引擎、不含 `--pct`、不含 `mask_radius`。**v2 的
`φ(r) = angle(ψ) − (2π/N) q·(r − c)` 约定已废除**（`--pct` 一并废除）。

三条恒等式（self-test 逐条实测，全部机器精度）：

| 恒等式 | 含义 / 用途 |
| --- | --- |
| `Σ_r ψ_q(r) = Σ_r T(r) e^(−i q·r)` | 窗的 `k = 0` 权重**恰为 1** ⇒ 等式与 λ 无关。**逐峰相位值 = 全画布幅度加权圆均值 = `arg Σ_r T(r) e^(−i q·r)`** |
| `ψ_{−q}(r) = conj(ψ_q(r))` | 实图像的 Friedel 关系（引擎级） |
| `ψ'_q(r) = e^(−i q·δ) ψ_q(r − δ)` | 平移律（`T'(r) = T(r − δ)`）：整幅平移 δ 后**场本身跟着平移**，且每个像素的相位再加常数 `−q·δ`；当场为常数（单个恰好落在 q 的平面波）时退化为“相位整体加 `−q·δ`”，self-test 正是在平面波上实测这一退化形式 |

| 量 | 定义（唯一口径） |
| --- | --- |
| **逐峰相位（唯一口径）** | **像素集合 = 全画布的全部有效（非 NaN）像素**，权重 `|ψ(r)|`：`θ_val = arg Σ_{r ∈ 全画布} ψ_q(r) = arg Σ_r T(r) e^(−i q·r)`（与 λ 无关，恒等式给出）。**不是**实空间圆盘子集、**不是**加门样本。`p50` 门**只用于分布形状展示** |
| 相位中位数 / IQR / FWHM | **加门**样本（默认 `p50`）的形状描述量；引用必须同时给出 `--gate` |
| FWHM 两种口径 | `fwhm_deg` = 平滑直方图的半高全宽（默认 2° 圆高斯平滑、0.1° 分箱）；`fwhm_deconv_deg` = 高斯等效去卷积 `√(FWHM² − (2.3548σ)²)`，不满足时为 NaN |
| 簇数 / 簇心 / 簇宽 | `phasemath.cluster_list`（平滑直方图严格局部极大 ≥ `min_frac`=0.25 倍峰值、抛物线亚分箱、中心 30° 内贪心合并、±15° 窗口内幅度加权圆均值迭代）；**簇宽** = 该簇 ±15° 内样本的幅度加权圆标准差 |
| 幅度统计 | 未加门的幅度中位数与 FWHM（加门后中位数按定义就是门阈值） |
| Friedel 对和 | `θ(q) + θ(−q)`（模 360°）：**实图像恒等式**，只作自检；管线按**检出的**波矢量配对，故残余受 `|Σq|` 限制（每对都输出 `|Σq|`） |
| 三独立值 | 波矢和为零的三元组（另一三元组是它的对径镜像）；给出三值、各自 mod 120°、和（mod 360° 与 mod 120°）、镜像值、到 0/120/240 阶梯的距离 |
| 三重积 θ | 三独立反射的逐像素相位之和（默认把 Σq 投影到零；`--no-project-q` 保留检出值），给 θ 的均值/中位数/R/FWHM/簇数/镜像值/`3θ mod 360°`，并附 `|q_sum|`。**注意**：Σq ≠ 0 时三独立场各自带残余载波，θ 会出现跨画布斜坡 `(2π/N)(Σq)·r`（未投影时跨画布跨度 = `360·|Σq|` 度，脚本逐环打印 `theta_ramp_span_deg_if_unprojected`） |
| 120° 折叠分布 | `θ mod 120°` 映射回整圆后的中位数/R/FWHM/簇数（对全局 120° 重标号不变） |
| 集中度 R | 幅度加权圆均值的模长 |
| 周期性边界 | 引擎乘上载波后再做 FFT，**没有加窗（无 Hann 切趾）**：画布不是周期图时，边界约 `λ_px` 宽的带进入结果（这是"逐峰相位 = 全画布精确求和"的代价）。真实数据上该带占比 ≈ 4–5 %（topo0009: λ_px≈63 px / 2117 px） |

**参考环（ring_1x1）的选择**：`--anchor auto`（默认）→ `strongest` → `outer` / `inner` →
`radius`（配 `--reference-radius-px`），方法学在 `phasepipe.choose_rings` 里，log 打印实际走的路径。
选定参考环后，**r3 环由参考半径的 1/√3 定位**（`--r3-tol`，默认 3 %）。

**「未检出 r3 圈」分支**：定位不到就**如实输出**——log 写 `ring pair: NOT FOUND`、
`# the r3 ring is NOT reported ...`，写 `phase_stats.json`（`status="r3_not_found"`）与
`ring_candidates.csv`，**退出码 2，不画任何图**（不猜、不硬套）。

**规范固定（gauge fix）**：用 `ring_1x1` 六个峰的未加门相位做
`θ_j = (2π/N) q_j·r0 + c (mod 2π)` 的加权最小二乘（多起点 Gauss–Newton，返回全部局部极小）。
约定：`θ̃ = θ − (2π/N) q·r0 − c`（**另一符号约定只差一个常旋转**，簇数/FWHM/R 完全相同）。

**原点拟合口径（写进 `phase_stats.json` 的 `gauge.fit_convention` / `triple_fit`）**：本 skill 用
**六峰**（三对 Friedel，三峰携带 +Φ、镜像携带 −Φ）对**同一个公共偏移 + 一个原点**做最小二乘。
六个方程只有三个未知量 ⇒ 只有当六条相位能被单一原点表示时残差才为 0；因此**六峰 rms 是模型一致性
诊断量**（参考环不均匀、畸变、或合成图案的镜像反号都会把它抬起来）。作为对照同时报
**q-sum-zero 三峰拟合**：它 3 方程 3 未知、**rms 恒为 0**，因此**没有**诊断信息，只用于给出
"另一个合法分支"。
六峰解可以落在离解调中心任意多个**直接点阵**矢量处（解集 = {最佳解} ⊕ 点阵平移）；
遗留分支把 `ring_r3` 相位移动 `(2π/N) q·L`，而 `3Δ` 与其到阶梯的距离不变。

**报告 r0 必须附全四项**（写进 `phase_stats.json` 的 `gauge`）：

1. **残差 RMS**（`rms_deg`）与参考环六峰 gauge 后偏离 0 的 rms（`reference_rms_deg`）；
2. **局部极小个数**（`n_minima`）与**分支表**（每个分支的 `r0`、`c`、rms、是否点阵平移）；
3. **解集说明**：解集 = {最佳 `r0`} ⊕ {参考环的**相位点阵**（`q·L ∈ Nℤ`）}——点阵平移给六条模型
   相位各加 2π 的整数倍，**残差逐位不变**，拟合永远分不开（结构性退化，不是数值误差）；
4. **诱导位移律**：点阵平移对另一族每峰的位移是 `(2π/N) q·L`；脚本对当前数据**实测**并输出
   `induced_shifts_are_multiple_of_120deg`——**不要假定 mod 120° 可比**。

⇒ **跨峰差（spread）与 120° 折叠分布的位置都随分支变**；随分支**严格不变**的只有闭合和
（三独立相位之和、逐像素 θ）与每峰分布形状（R/FWHM/簇数/簇宽）。

**随原点漂移的量**：单峰原始相位按 `θ → θ − (2π/N) q·δ` 漂移。self-test 在**周期画布**（波矢全为
FFT 整数 bin）上实测该律残差 ≤ 1e-9°、并实测"原点跟着图像走时 gauge 后相位不变"（≤ 1e-9°）；
在**非周期合成画布**上如实报告漂移率（可达 ~20 °/px）与对精确律的残差（~0.8°，来源是周期性边界带）
——此时 **严格不变的量**：逐像素三重积 θ、集中度 R、加门后 FWHM、簇数。

**单峰绝对相位的系统带**：由参考环残差反推逐峰散布 `σ_peak ≈ RMS_res/0.707`，
再除以 3 得 √3 配对反射的系统带（精确关系 `std(Δφ_r3) = σ_peak/3`）；该带**不随像素数下降**。

**峰位不确定度**：混合相位会让检出的峰位偏移；脚本输出每族的**六边形残差**
（`hexagon_residual_rms_px` / `..._max_px`）与 `|Σq|`。默认 `--project-q` 把三独立波矢之和
**投影到零**，去掉 θ 的净斜坡；但**不清除逐峰偏置**。

**三重积的镜像二义**：另一闭合三元组是当前三元组的对径，故**有符号 θ 与 `3θ mod 360°` 会镜像**，
而**到 0/120/240 阶梯的距离不变**——脚本对每一条 θ 输出同时给出镜像值。

**多成分判别边界（纯数学结论，self-test 实测）**：

| 配置 | 相干度 | 三独立值和 | 簇数 | 说明 |
| --- | --- | --- | --- | --- |
| 1 个成分 | 1.000 | 落在 0/120/240 阶梯上 | 1 | 恒等 |
| 2 个**等权**成分、相差 120° | 0.500 | **离阶梯 60°** | 2 | 「和 ≈ 0」不成立 ⇒ 它是**值域判别** |
| 3 个等权成分 0/120/240 | 0 | 相量相消、相位无意义 | — | 守门量是**幅度/相干度** |
| 2 个成分 ~97 %/3 %（相差 120°） | 0.970 | 离阶梯 **4.33°** | **1** | 和能看见、簇数看不见 |
| 2 个成分 ~70 %/30 % | 0.615 | 离阶梯 44.4° | 2 | 两者都能看见 |

定量边界：`(1−f, f)` 两成分相距 120° 时，离阶梯距离 ≈ `3f·sin120°`（小 f），**5° 交点 f = 0.0319**。
⇒ **簇数回答"有几个相位"，相干度回答"峰还在不在"，「和 ≈ 0」只回答"加权圆均值是否落在阶梯附近"**。

### 3.2 论文式成对相位差分析（v3 新增）

对**每一对**反射 `(j, k)` 定义三个量（全部逐像素，来自 §3.1 的解调场）：

```
D_jk(r) = wrap( arg ψ_j(r) − arg ψ_k(r) ) = arg( ψ_j(r)·conj(ψ_k(r)) )      ∈ (−π, π]
a_jk(r) = ( |ψ_j(r)| − |ψ_k(r)| ) / ( |ψ_j(r)| + |ψ_k(r)| )                 ∈ [−1, 1]
有效像素 = 两场有效掩码的交集       权重 = |ψ_j(r)·ψ_k(r)|
```

2D 直方图：**x = `|D| mod π` ∈ [0, π]（默认 180 bins）**，**y = `a_jk` ∈ [−1, 1]（默认 100 bins）**，
按权重 `|ψ_j ψ_k|` 计数（`hist_counts[i, j]` = x 落在第 i 箱、y 落在第 j 箱的加权计数，形状
`[bins_x, bins_y]`；`--pair-bins-x/--pair-bins-y` 可改）。相位差的统计量用圆统计
（mean/median/R/FWHM/簇数），幅度差用加权线性中位数。

| 组（`pairwise.groups`） | 对 | 说明 |
| --- | --- | --- |
| `within_1x1` | (p0,p1)、(p2,p3)、(p4,p5) | `ring_1x1` 内 3 对。峰号是**12 点钟起顺时针**，故这 3 对都是相距 60° 的相邻峰，**刻意避开 Friedel 对** (p0,p3)/(p1,p4)/(p2,p5) |
| `within_r3` | 同上 | `ring_r3` 内 3 对 |
| `cross` | 每个 `ring_1x1` 峰配**方位角最近**的 `ring_r3` 峰 | 共 6 对。两环相差 30° 时会出现**等距并列**，此时取 `ring_r3` **下标最小**者（`phasepipe.ANGLE_TIE`，脚本与 self-test 用同一规则） |

**为什么避开 Friedel 对**：实图像恒等式给 `ψ_{−q} = conj(ψ_q)`，于是
`D_{j,−j} = 2 arg ψ_j`（一个场的平凡函数）、`a_{j,−j} ≡ 0`（精确为 0）、`θ_j + θ_{−j} ≡ 0`——
这一对不携带新的独立信息，所以组内对不用它（self-test 逐条实测，阈值 1e-12/1e-15）。

### 3.3 v2 → v3 的「变与不变」

| 项 | 变化 |
| --- | --- |
| 引擎 | 硬圆 mask + 单反射 iFFT（v2）→ **local-q-map Gaussian 窗解调**（v3，`--lambda-nm`，默认 3.0 nm）；`--pct` 与全部 `mask_radius` 相关产物（log 行、JSON 字段、`mask_in/mask_out/qspace_mask/mask_all_in/mask_all_out/case_phase_histograms` 图）**删除** |
| 相位口径 | 废除 `φ(r) = angle(ψ) − (2π/N) q·(r−c)`；**`θ = arg ψ ≈ +φ`，无斜坡、无每峰常数** |
| raw 单峰相位 | 相对 v2 **每峰平移 `−π(q_x + q_y)`**（v2 的相位值 = `arg X(q) + (2π/N) q·c`，`c = (N/2,N/2)`）。该常数**恰是原点平移 `(N/2,N/2)`**，被 gauge 拟合吸收：`r0_v3 = r0_v2 − (N/2,N/2)` ⇒ **gauge 后相位逐位不变**（self-test 实测 ≤ 1e-9°） |
| 不变 | 峰检测 / 环聚类 / 1/√3 配对 / `choose_rings` / 三重积 / `--project-q` / 120° 折叠 / gauge 拟合与分支表 / 系统带 / Friedel 与三独立和 / 每峰分布形状 / 幅度与相干度 —— **只换了输入相位场**，代码逻辑不动 |

## 4 输出与**图集清单（交付契约）**

非图产物：`phase_stats.json`（所有数字的唯一来源，含 `pairwise` 段）、`phase_stats.csv`（逐峰表）、
`phase_relations.csv`（Friedel 对/三独立和/三重积）、`phase_ring_comparison.csv`（两族对照表）、
`ring_candidates.csv`、`phase_stats.log`、`atlas_manifest.json`。
（`phase_stats.json` 的整数计数数组按行内联写出，避免逐元素换行把文件撑成百万行。）

**图集：每族 31 张（`ring_1x1` 31 + `ring_r3` 31）+ 跨环 19 张 = 81 张**（`--no-figures` 关闭）。
文件名与张数写死在 `stm_phase_analysis.py` 顶部
（`PER_PEAK_FIGURES` / `SUMMARY_FIGURES` / `PAIR_FIGURES` / `FIGURES_PER_RING` / `CROSS_FIGURES` /
`TOTAL_FIGURES`），manifest 与 `phase_stats.json` 的 `atlas` 段同时声明 31/31/19/81。

| # | 文件名模式（`RING` = `ring_1x1` 或 `ring_r3`，`i` = 0…5；`i_k` = 该对的峰号） | 面板（manifest 的 `panels` 字段逐字声明） |
| --- | --- | --- |
| 1 | `RING_p{i}_amplitude.png` | `["amplitude \|psi(r)\| (log scale)"]` |
| 2 | `RING_p{i}_theta_map.png` | `["theta(r) map", "amplitude gate mask"]` |
| 3 | `RING_p{i}_theta_dist.png` | `["theta distribution (gated)", "theta distribution folded mod 120 deg"]` |
| 4 | `RING_theta_hist_summary.png` | 六峰 θ 分布 2×3 汇总（每格标题带中位数，参考线同下） |
| 5 | `RING_theta_map_summary.png` | 六峰 θ(r) 图 2×3 汇总 |
| 6 | `RING_theta_field.png` | 三重积 θ(r) 图 + θ 分布（参考线同下） |
| 7 | `RING_ring_members_qspace.png` | FFT2 log 幅度上标出该环六个谱峰位置，`p0…p5`（编号 = `phase_stats.json` 峰号）标在**径向外侧**，不遮住谱峰本身 |
| 8–10 | `RING_pair_{j}_{k}_phase_diff.png` / `..._amp_diff.png` / `..._2dhist.png` | 组内每对 3 张（`(j,k)` ∈ {(0,1),(2,3),(4,5)}）：`["D(r) = wrap(arg psi_j - arg psi_k)"]` / `["a(r) = (\|psi_j\| - \|psi_k\|) / (\|psi_j\| + \|psi_k\|)"]` / `["2D histogram (x = \|D\| mod pi, y = amplitude difference)"]` |
| 跨环 | `cross_pair_{i}_phase_diff.png` / `..._amp_diff.png` / `..._2dhist.png`（`i` = 0…5，即 `ring_1x1` 峰号） | 每对 3 张，面板同上 |
| 跨环汇总 | `cross_pair_phase_diff_grid.png` | `["six cross-pair D(r) maps (2x3)"]` |

计数核对：每族 `3×6 + 4 + 3×3 = 31`，跨环 `6×3 + 1 = 19`，合计 **81**。

**相位直方图的参考线约定**：所有 θ 分布面板（含 120° 折叠面板）的 x 轴标签写明
`2 pi k / 3, k = 0,1,2 = 0/120/240 deg`（虚线即这三条线，另加 `pi` 刻度仅作读图参考）——
不出现任何 k 空间/物理称呼。

**manifest 条目模式（逐字段，`atlas.py --check` 逐项核对）**

| 字段 | 取值 / 含义 |
| --- | --- |
| `file` | PNG 文件名（见上表模式） |
| `group` | `ring_1x1` / `ring_r3` / `cross` |
| `kind` | `per_peak`（`peak` = 0…5）、`summary`（`peak = null`）、`pairwise`（`pair` = `"<j>+<k>"`） |
| `peak` | 峰号 0…5（`kind=per_peak`），其余为 `null` |
| `pair` | 该图的反射对键（`kind=pairwise`），其余为 `null` |
| `panels` | 该图面板的自左向右（网格则逐行）名称表，见上表 |
| `size_px` | `[width, height]`，与 PNG 实际尺寸一致 |
| `label` / `annotation` / `title` | 人类可读前缀 / 取整后的注解数字 / `label + " \| " + annotate(annotation)` |
| `paths` | 每个注解字段在 `phase_stats.json` 里的点分路径；**成对图的路径全部指向 `pairwise.groups.<组>.pairs.<序号>.*`** |

**图-数字契约（可程序化核对，不需要 OCR）**：每张图的标题由**一份注解字典**渲染
（`atlas.annotate`，字段顺序由 `atlas.FIELD_ORDER` 规范固定），同一份字典写入
① 图标题、② `atlas_manifest.json` 的 `figures[]`、③ PNG 的 `tEXt` 块 `stm-atlas`（JSON）。
`paths` 指向 `phase_stats.json` 的点分路径，注解值是该 JSON 数字按打印位数取整的结果（核对容差 = 半个末位）。
第三方可用**一条命令**独立复核：

```bash
.venv/bin/python skills/phase-analysis/scripts/atlas.py \
  --check OUT/phase/atlas_manifest.json --stats OUT/phase/phase_stats.json \
  --expected-figures 81 --expected-per-ring 31 --expected-cross 19
```

命令逐张检查：存在、非零、PIL 可开且尺寸与 manifest 一致、PNG 内嵌注解（含 `panels`）= manifest、
标题 = 注解字典的渲染、注解数字 = `phase_stats.json` 对应路径的数字（末位半单位容差）、
`group`/`kind`/`peak`/`pair` 自洽、`panels` 非空、总数与每族/跨环张数自洽。
**self-test 与端到端冒烟都真的以命令行方式调用它**（不是"有能力但没跑"）。

**命名规则**：文件名、图标题、脚本 title/label 只用 `ring_1x1` / `ring_r3` 与数学量名；
相位直方图的参考线一律是数学阶梯 `2πk/3`（0/120/240°），不使用任何 k 空间/物理称呼。

## 5 一键 self-test

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/phase-analysis/scripts/selftest.py \
  --stm-lib /path/to/STM_DataProcessing/src
# 可选：--quick（跳过端到端阶段）、--workdir DIR、--size N、--keep、--lambda-nm、--nm-per-px
```

覆盖与验收阈值（实测 **80/80** 通过）：

| 类 | 量 | 验收阈值 |
| --- | --- | --- |
| **引擎** | `Σψ = ΣT e^(−i q·r)`（3 个 λ × 3 个 q） | 相对偏差 ≤ 1e-9 |
| | 平移律（3 个平面波） | ≤ 1e-9 rad |
| | `θ = arg ψ = +φ`（9 个平面波） | ≤ 0.01° |
| | 无斜坡（相邻像素相位差） | ≤ 1e-9 rad |
| | Friedel `ψ_{−q} = conj(ψ_q)` | 相对 ≤ 1e-12 |
| | v2→v3 平移 `−π(q_x+q_y)` | ≤ 1e-6° |
| | 单区三独立和 / 逐像素三重积 = 3Φ | ≤ 0.5° |
| **gauge 层** | 平移律残差、`r0 − δ` 模型、`(N/2,N/2)` 吸收 | ≤ 1e-9° |
| | 周期画布：原点随图像走时 gauge 后相位不变 | ≤ 1e-9° |
| | θ / R / FWHM / 簇数不变 | ≤ 0.5° / ≤ 0.05 / 精确 |
| **成对** | 注入 Δφ = 40° 的 D 回收 | ≤ 0.5° |
| | 幅度差回收（3 组注入比值） | ≤ 1e-3 |
| | Friedel 对：和 ≡ 0、`a ≡ 0`、`D = 2θ_j` | 1e-12 / 1e-15 / 1e-12 |
| | 组内 3 对（60°、非 Friedel）+ 跨环 6 对（最近方位角） | 结构断言 |
| | 2D 直方图 x∈[0,π]、y∈[−1,1]、形状 180×100 | 结构断言 |
| **图集** | 81 张、命名模式、PIL、内嵌注解 = manifest、数字 = JSON | 0 失败 |
| | `atlas.py --check --expected-figures 81 --expected-per-ring 31 --expected-cross 19` | 退出码 0 |
| | `--pct` 被 CLI 拒绝（已废除） | 非零退出 |
| **确定性** | 两次运行的 `atlas_manifest.json` **逐字节相同**、`phase_stats.json` 掩掉各自输出目录字段后**逐字节相同**、图逐像素相同（PNG 字节也实测相同） | 相同 |

检查清单：

1. **引擎源码契约**：交付脚本中不出现已删除的 mask 引擎 token；`phasepipe` 只经
   `gaussian_field` → `localqmap.demodulate` 取场；CLI 有 `--lambda-nm`、无 `--pct`。
2. **圆统计量**：加权圆均值；圆中位数 = L1 目标精确极小；退化样本极小集被标记；
   高斯样本去卷积 FWHM = 2.3548σ（1 %）；双成分簇心/权重。
3. **引擎恒等式**：见上表。
4. **gauge 层**：精确律、吸收律、周期画布不变性、非周期画布的漂移率与残差（如实报告）、
   拟合分支数与本数据下的诱导位移表。
5. **成对契约**：组的构成、Friedel 对为何平凡、交换 (j,k) 让 D/a 反号（`|D| mod π` 不变）、
   直方图边界。
6. **成对回收**：注入相位差与幅度比。
7. **环定位分支**：`auto`/`radius` 的 1/√3 定位、参考半径过远、无配对环、最内环即 r3；
   峰编号从 12 点钟起顺时针。
8. **多成分判别边界**：见 §3.1 表；另经引擎实测全画布相位值 = 面积加权相量预言（残差 < 1°，
   残余来自另一族环对解调窗的谱泄漏）。
9. **端到端**：真跑两次完整管线（**先断言第二次运行的返回码**）→ 图集清单断言（81 张、命名模式、
   PIL、注解 = JSON）、**确定性**（`atlas_manifest.json` 逐字节相同、`phase_stats.json` 掩掉各自
   输出目录字段后逐字节相同、图逐像素且 PNG 逐字节相同）、log 数字 = JSON 数字、
   缺 r3 分支（退出码 2、有明确信息、0 张图）。
10. **`--size-nm-from-log`**（真跑三次）：矫正后画布行优先；只有一行时回退取该行；
    一处都没有 → 非零退出码 + 明确信息。

## 6 变与不变（引用任何一个数字前请读）

| 类 | 量 | 对图像原点 / 对 r0 分支 |
| --- | --- | --- |
| **稳健量（可以直接引用、跨约定比较）** | Friedel 对和 | 不变（恒等式） |
| | 三独立相位之和、逐像素三重积 θ | **精确不变**（Σq = 0 时） |
| | **成对 D / a 场与它们的 2D 直方图** | **不随 r0 分支变**（D、a 由**原始解调场**定义，不进入 gauge 层）；只随图像原点整体平移 `−(q_j − q_k)·δ`；交换 (j,k) 让 D、a 反号，故 `\|D\| mod π` 与直方图的 x 轴不变、y 轴镜像 |
| | 每峰分布形状（R / FWHM / 簇数 / 簇宽） | 不变（整体平移） |
| | 幅度与相干度 | 不变（守门量：相消时相位无意义） |
| **参考量（必须附条件）** | 单峰绝对相位 | **随分支变**：点阵平移给 `(2π/N)q·L` |
| | 三峰 spread、跨峰合并 / 120° 折叠分布的**位置** | 随分支变（诱导位移一般不是 120° 的整数倍） |

引用单峰绝对相位时必须写明：`r0` 用的是解集里的**哪一个分支** + 由参考环残差反推的**系统带
`σ_peak/3 ≈ RMS_res/2.12`**。

* **随图像原点漂移**：单峰原始相位按 `θ → θ − (2π/N) q·δ` 线性漂移（大 |q| 时每像素可达几十度）；
  闭合和、分布形状与成对 D（同分支下）不受影响。
* **Δλ 与 Δ(视场)**：λ 只改变解调窗宽，**不改变**逐峰相位值（`k = 0` 权重恒为 1 的恒等式）；
  但改 λ 会改变 R/FWHM 与场图的外观。`nm/px` 只通过 `λ_px = λ_nm/(L/N)` 进入引擎。
* **依赖口径**：FWHM（平滑/分箱口径）、簇宽（窗口口径）、幅度统计（是否加门）、
  成分权重（**面积加权**，因为不加窗；v2 的"窗加权"口径随 Hann 窗一起废除）。
* **门是一种隐式空间选择**：`p50` 门偏向画布中心，门后的相位和最多可与全区差十几度。
  相位**值**用未加门的加权圆均值，门后的分布只用来描述形状。

## 7 不随本 skill 落盘的方法学附件

`origin_space.py`（r0 解集与不确定度传播的纯数学脚本）**不属于本 skill 的落盘内容**——它是方法学
研究附件，源头在 `data_processing/phase_reanalysis/_methods/`。它的四个结论已被本 skill 吸收进
本文档：解集 = {最佳 r0} ⊕ 参考环相位点阵、原点误差传播 `std(Δφ) = σ_peak·|q|/(√3 R1)`
（对 1/√3 对退化为 `σ_peak/3`）、Σq = 0 的闭合和与原点无关、分布形状与平移无关（§6 的稳健量清单）。

## 8 修改记录

2026-09（v1）：初版（topo0009 实测流程），六峰相位分析。
2026-09（v1 入库）：脚本改为调用 `stm_data_processing.utils.bragg_peak` 包。
2026-09（v2）：锚定环显式化、两族同口径双环分析、r3 由 1/√3 定位与「未检出」分支、
唯一的相位估计量定义 + 分布描述量分离、Friedel/三独立/三重积/折叠分布/R 全套、
图集清单与图-数字契约、一键 self-test、命名中性化。
2026-09（v2.1）：`--size-nm-from-log` 改读**矫正后画布**视场。
2026-09（拆分）：几何矫正拆分为独立 skill `topo-correction`（后改名 `affine-correction`）；
本 skill 改名 `phase-analysis`。
2026-09（**v3，本版**）：见 `CHANGES.md` 顶部——
① **引擎替换**：废弃硬圆 mask 引擎（`reflection_field` / `demod_phase` / `circle_mask` 全部删除），
改用 `local-q-map` 的 Gaussian 窗解调（`gaussian_field` → `localqmap.demodulate`），
相位口径统一为 **`θ = arg ψ ≈ +φ`（无斜坡、无每峰常数）**，废除 `--pct` 与全部 `mask_radius` 产物，
新增 `--lambda-nm`（默认 3.0 nm）；
② **论文式成对相位差分析**：组内 3 对/环 + 跨环 6 对，每对 D 场 / 归一化幅度差场 / 2D 直方图，
`phase_stats.json` 新增 `pairwise` 段；
③ **图集契约**：31/环×2 + 跨环 19 = **81 张**，manifest 增 `group`/`kind`/`pair`，
`atlas.py --check` 支持 `--expected-per-ring 31 --expected-cross 19`；
④ self-test 重写（引擎恒等式、gauge 层、成对注入回收、Friedel 对、81 张图集、字节确定性）。
