---
name: phase-analysis
description: STM 拓扑图（矫正后或任意正方形 CSV）的双环相位数学分析 v4：Gaussian 窗局域引擎（local-q-map，λ=3.0 nm，θ=arg ψ≈+φ 无斜坡）给出 1x1 与 r3 两族逐反射相位统计（圆均值/中位数/R/半峰宽/簇/幅度/三独立相位和/逐像素三重积，含 Friedel 恒等式自检与 r0 规范固定），并做成对相位差分析（每环组内 3 对：D(r) 空间图、D 的幅度加权圆直方图、归一化幅度差图）；同一规则取样——矫正画布上全部有效（非 NaN）像素、权重 |ψ(r)|，无任何阈值；完整图集 30/环 = 60 张（含全局统一色标与画在数据区之外的色条）与一键数学 self-test。只做几何与相位数学，不做物理解释。当用户给出拓扑图 CSV 与图像边长（nm），要求两族环的相位/相位统计/成对相位差时使用。
---

# phase-analysis（STM 拓扑图：双环相位数学，v4）

**范围**：本 skill 只做**相位数学**——局域解调场、圆统计量、规范固定、成对相位差、图集与自检。
（几何矫正已拆分为独立 skill `affine-correction`。）**物理解释由使用者进行**：本 skill 的
脚本、图标题、log 与文档都不给出物理结论，也不使用任何 k 空间/高对称点类称呼。

两个环只有两个名字，归属只看**半径比**：

| 名字 | 定义 |
| --- | --- |
| `ring_1x1` | 参考环：它的六个峰做最小二乘定出晶格原点 r0，相位以它为规范 |
| `ring_r3` | 半径是 `ring_1x1` 半径的 **1/√3** 的环（容差可配，默认 3 %） |

**v4 的取样只有一条规则**：矫正画布上**全部有效（非 NaN）像素**，逐峰权重 `|ψ(r)|`、
成对权重 `|ψ_j ψ_k|`。任何阈值/百分位/门限概念都不存在：CLI 没有阈值选项（`--gate` 会被**显式拒绝**），
代码里没有阈值掩码，JSON 里没有阈值字段，图上没有掩码面板，而且**图的可见像素与该图统计所用样本
是同一个集合**。

---

## 1 输入与执行环境

| 项 | 说明 |
| --- | --- |
| 输入 | 正方形拓扑数值矩阵 CSV/txt（可含 NaN）。**第一行 = 扫描起始行 = 图像顶部**。典型输入 = `affine-correction` 的矫正产物（见 §2）；也接受任意方形 CSV（不经矫正，配 `-L` 或 `--size-nm-from-log`） |
| 视场 L (nm) | 正方形扫描边长，**必须显式给出**（文件名里的标注不可信） |
| 分隔符 | 自动识别；显式用 `--delimiter ','` / `--delimiter '\t'` |
| 解释器 | 仓库 venv：`cd /path/to/STM_DataProcessing && MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 .venv/bin/python <脚本>`。**不要用 `uv run`**（会写仓库 `.venv`） |
| 解调引擎 | **`skills/local-q-map/scripts/localqmap.py` 的 `demodulate`**，本 skill 经 `phasepipe.gaussian_field` 调用（不复制、不修改引擎代码）。引擎目录按两处查找：本 skill 目录的同级（仓库 checkout 与 `~/.agents/skills` 都是这个布局），以及 `~/.agents/skills`（脚本从别处的目录运行时用） |
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
  --lambda-nm 3.0 --anchor auto
```

**`--size-nm-from-log` 读的是矫正后画布的视场**：`affine-correction` 的 `stm_topo_correct.py` 产出的
log 里有两处 `field of view <value> nm`——第一处是**输入画布**（`# canvas ... field of view 50 nm`），
第二处是**矫正后画布**（`# corrected canvas: ... field of view 51.7090 nm`）。本脚本分析的是
**矫正后的 CSV**，所以解析按以下优先级取值：
① 优先取带 `corrected canvas` 标签的那一行；
② 没有该行时取 log 里**最后一处**匹配（单行 log 即那一行）；
③ 一处都没有 → 报错退出。
取值后 log 打印一行来源说明（如
`# field of view from log: 51.7090 nm (corrected canvas line; .../correction.log)`），
`phase_stats.json` 的 `field_of_view_source` 同文。手工指定 `-L` 时以命令行值为准（同一行也打印来源）。

**已废除的选项**：`--gate`（v4 删除）与 `--pct`（v3 删除）**显式报错拒绝**——非零退出并打印
`error: <选项> is not an option of this script: <原因>`。名单是脚本顶部的 `REMOVED_OPTIONS` 字典，
唯一改它的地方就是它的唯一定义处，所以不存在"选项被静默忽略"的路径。

### 3.1 解调引擎与相位约定（全脚本唯一一套定义）

```
ψ_q(r) = FFT⁻¹{ FFT[ T(r)·e^(−i q·r) ] · e^(−λ²|k|²/2) }        λ = --lambda-nm（默认 3.0 nm）
θ_q(r) = arg ψ_q(r)  ≈  +φ_q(r)                                    （无 q·r 斜坡、无每峰常数）
```

引擎来自同仓库的 `local-q-map` skill（`localqmap.demodulate`），`λ_px = λ_nm / (L/N)`；
本 skill **不含**任何自己的场提取实现，也不含阈值或掩码半径相关的东西。

三条恒等式（self-test 逐条实测，全部机器精度）：

| 恒等式 | 含义 / 用途 |
| --- | --- |
| `Σ_r ψ_q(r) = Σ_r T(r) e^(−i q·r)` | 窗的 `k = 0` 权重**恰为 1** ⇒ 等式与 λ 无关。**逐峰相位值 = 全画布幅度加权圆均值 = `arg Σ_r T(r) e^(−i q·r)`** |
| `ψ_{−q}(r) = conj(ψ_q(r))` | 实图像的 Friedel 关系（引擎级） |
| `ψ'_q(r) = e^(−i q·δ) ψ_q(r − δ)` | 平移律（`T'(r) = T(r − δ)`）：整幅平移 δ 后**场本身跟着平移**，且每个像素的相位再加常数 `−q·δ`；当场为常数（单个恰好落在 q 的平面波）时退化为“相位整体加 `−q·δ`”，self-test 正是在平面波上实测这一退化形式 |

### 3.2 取样与估计量（唯一一套口径）

| 量 | 定义 |
| --- | --- |
| **逐峰样本** | 矫正画布的**全部有效（非 NaN）像素**，权重 `|ψ(r)|`。唯一被去掉的是权重非有限或恰为 0 的像素——这是"权重正定"的算术要求，不是选择规则（真实数据上通常一个像素都不掉，脚本逐峰报告 `n_zero_weight_px`） |
| **成对样本** | 两个场有效掩码的**交集**，权重 `|ψ_j ψ_k|` |
| **图的可见像素** | 与上面的样本**同一个集合**（同一份掩码数组）；逐峰图标题里的 `pixels` 就是该样本大小，manifest 的路径指向 JSON 的 `n_samples`，`atlas.py --check` 逐个核对 |
| 逐峰相位值 | `θ_val = arg Σ_r ψ_q(r)`（在样本上做幅度加权圆均值）= `arg Σ_r T(r) e^(−i q·r)`，与 λ 无关、无斜坡、无每峰常数 |
| 分布描述量 | 中位数、IQR、FWHM（`fwhm_deg` = 平滑直方图半高全宽，默认 2° 圆高斯平滑、0.1° 分箱）、FWHM 去卷积（`fwhm_deconv_deg` = `√(FWHM² − (2.3548σ)²)`，不满足时为 NaN）、圆标准差、von Mises `kappa`、集中度 R —— **全部在同一个样本上**，引用时不需要附带任何条件 |
| 簇数 / 簇心 / 簇宽 | `phasemath.cluster_list`（平滑直方图严格局部极大 ≥ `min_frac`=0.25 倍峰值、抛物线亚分箱、中心 30° 内贪心合并、±15° 窗口内幅度加权圆均值迭代）；**簇宽** = 该簇 ±15° 内样本的幅度加权圆标准差 |
| 幅度统计 | 样本上的幅度中位数与 FWHM（与相位统计同一个样本） |
| Friedel 对和 | `θ(q) + θ(−q)`（模 360°）：**实图像恒等式**，只作自检；管线按**检出的**波矢量配对，故残余受 `|Σq|` 限制（每对都输出 `|Σq|`） |
| 三独立相位和 | 波矢和为零的三元组（另一三元组是它的对径镜像）：给出三值、和（mod 360°）与镜像和 |
| 逐像素三重积 θ | 三独立反射的逐像素相位之和（默认把 Σq 投影到零；`--no-project-q` 保留检出值），给 θ 的均值/中位数/R/圆标准差/FWHM/簇数/镜像均值，并附 `|q_sum|`。**只报数字，不画图**（v4 删除三重积图）。**注意**：Σq ≠ 0 时三独立场各自带残余载波，θ 会出现跨画布斜坡 `(2π/N)(Σq)·r`（未投影时跨画布跨度 = `360·|Σq|` 度，脚本逐环打印 `theta_ramp_span_deg_if_unprojected`） |
| 集中度 R | 幅度加权圆均值的模长（保存为 `phase_R`） |
| 周期性边界 | 引擎乘上载波后再做 FFT，**没有加窗（无 Hann 切趾）**：画布不是周期图时，边界约 `λ_px` 宽的带进入结果（这是"逐峰相位 = 全画布精确求和"的代价）。真实数据上该带占比 ≈ 4–5 %（topo0009: λ_px≈63 px / 2117 px） |

**参考环（ring_1x1）的选择**：`--anchor auto`（默认）→ `strongest` → `outer` / `inner` →
`radius`（配 `--reference-radius-px`），方法学在 `phasepipe.choose_rings` 里，log 打印实际走的路径。
选定参考环后，**r3 环由参考半径的 1/√3 定位**（`--r3-tol`，默认 3 %）。

**「未检出 r3 圈」分支**：定位不到就**如实输出**——log 写 `ring pair: NOT FOUND`、
`# the r3 ring is NOT reported ...`，写 `phase_stats.json`（`status="r3_not_found"`）与
`ring_candidates.csv`，**退出码 2，不画任何图**（不猜、不硬套）。

**规范固定（gauge fix）**：用 `ring_1x1` 六个峰的相位做
`θ_j = (2π/N) q_j·r0 + c (mod 2π)` 的加权最小二乘（多起点 Gauss–Newton，返回全部局部极小）。
约定：`θ̃ = θ − (2π/N) q·r0 − c`（**另一符号约定只差一个常旋转**，簇数/FWHM/R 完全相同）。

**原点拟合口径（写进 `phase_stats.json` 的 `gauge.fit_convention` / `triple_fit`）**：本 skill 用
**六峰**（三对 Friedel，三峰携带 +Φ、镜像携带 −Φ）对**同一个公共偏移 + 一个原点**做最小二乘。
六个方程只有三个未知量 ⇒ 只有当六条相位能被单一原点表示时残差才为 0；因此**六峰 rms 是模型一致性
诊断量**（参考环不均匀、畸变、或合成图案的镜像反号都会把它抬起来）。作为对照同时报
**q-sum-zero 三峰拟合**：它 3 方程 3 未知、**rms 恒为 0**，因此**没有**诊断信息，只用于给出
"另一个合法分支"。
六峰解可以落在离解调中心任意多个**直接点阵**矢量处（解集 = {最佳解} ⊕ 点阵平移）；
遗留分支把 `ring_r3` 相位移动 `(2π/N) q·L`。

**报告 r0 必须附全四项**（写进 `phase_stats.json` 的 `gauge`）：

1. **残差 RMS**（`rms_deg`）与参考环六峰 gauge 后偏离 0 的 rms（`reference_rms_deg`）；
2. **局部极小个数**（`n_minima`）与**分支表**（每个分支的 `r0`、`c`、rms、是否点阵平移、
   对另一族每峰的**诱导位移** `induced_ring_r3_phase_shifts_deg`）；
3. **解集说明**：解集 = {最佳 `r0`} ⊕ {参考环的**相位点阵**（`q·L ∈ Nℤ`）}——点阵平移给六条模型
   相位各加 2π 的整数倍，**残差逐位不变**，拟合永远分不开（结构性退化，不是数值误差）；
4. **诱导位移律**：点阵平移对另一族每峰的位移是 `(2π/N) q·L`，脚本对当前数据**实测并逐分支列出**。

⇒ **跨峰差（spread）与跨峰合并的位置都随分支变**；随分支**严格不变**的只有闭合和
（三独立相位和、逐像素 θ）、成对 D/a 场与每峰分布形状（R/FWHM/簇数/簇宽）。

**随原点漂移的量**：单峰原始相位按 `θ → θ − (2π/N) q·δ` 漂移。self-test 在**周期画布**（波矢全为
FFT 整数 bin）上实测该律残差 ≤ 1e-9°、并实测"原点跟着图像走时 gauge 后相位不变"（≤ 1e-9°）；
在**非周期合成画布**上如实报告漂移率（可达 ~20 °/px）与对精确律的残差（~0.8°，来源是周期性边界带）
——此时 **严格不变的量**：逐像素三重积 θ、集中度 R、簇数；逐峰分布形状（含 FWHM）不变到
"边界带量级"（本机实测 FWHM 变化 ≈ 0.5°，因为样本是整张有效画布、无切趾）。

**单峰绝对相位的系统带**：由参考环残差反推逐峰散布 `σ_peak ≈ RMS_res/0.707`，
再除以 3 得 √3 配对反射的系统带（精确关系 `std(Δφ_r3) = σ_peak/3`）；该带**不随像素数下降**。

**峰位不确定度**：混合相位会让检出的峰位偏移；脚本输出每族的**六边形残差**
（`hexagon_residual_rms_px` / `..._max_px`）与 `|Σq|`。默认 `--project-q` 把三独立波矢之和
**投影到零**，去掉 θ 的净斜坡；但**不清除逐峰偏置**。

**三重积的镜像二义**：另一闭合三元组是当前三元组的对径，故**有符号 θ 会镜像**——脚本对每条 θ
同时给出镜像值（`theta_mirror_mean_deg`、`scalar_sum_mirror_deg`）。

**多成分判别边界（纯数学结论，self-test 实测）**：

| 配置 | 相干度 | 三分相位和的偏离 | 簇数 | 说明 |
| --- | --- | --- | --- | --- |
| 1 个成分 | 1.000 | 0° | 1 | 恒等 |
| 2 个**等权**成分、相距 1/3 圈 | 0.500 | **60°** | 2 | 「和 ≈ 0」不成立 ⇒ 它是**值域判别** |
| 3 个等权成分、相距 1/3 圈且均匀 | 0 | 相量相消、相位无意义 | — | 守门量是**幅度/相干度** |
| 2 个成分 ~97 %/3 % | 0.970 | **4.33°** | **1** | 和能看见、簇数看不见 |
| 2 个成分 ~70 %/30 % | 0.615 | 44.4° | 2 | 两者都能看见 |

定量边界：`(1−f, f)` 两成分相距 1/3 圈时，三分相位和的偏离 ≈ `3f·sin(1/3 圈)`（小 f），
**5° 交点 f = 0.0319**（self-test 用三分相位和到整倍数的距离实测）。
⇒ **簇数回答"有几个相位"，相干度回答"峰还在不在"，三分相位和只回答"加权圆均值落在哪里"**。

### 3.3 成对相位差分析

对**每一对**反射 `(j, k)`（每环三对，见下）定义三个量（全部逐像素，来自 §3.1 的解调场）：

```
D_jk(r) = wrap( arg ψ_j(r) − arg ψ_k(r) ) = arg( ψ_j(r)·conj(ψ_k(r)) )      ∈ (−π, π]
a_jk(r) = ( |ψ_j(r)| − |ψ_k(r)| ) / ( |ψ_j(r)| + |ψ_k(r)| )                 ∈ [−1, 1]
样本 = 两场有效掩码的交集          权重 = |ψ_j(r)·ψ_k(r)|
```

每对出三张图：

| 图 | 内容 |
| --- | --- |
| `RING_pair_{j}_{k}_phase_diff.png` | `D_jk(r)` 空间图（`seismic`，`±π`） |
| `RING_pair_{j}_{k}_phase_diff_dist.png` | `D_jk` 的**幅度加权圆直方图**（x = `D ∈ (−π, π]`，权重 `|ψ_j ψ_k|`，361 bins） |
| `RING_pair_{j}_{k}_amp_diff.png` | `a_jk(r)` 空间图（`coolwarm`，`±1`） |

相位差的统计量用圆统计（mean/median/R/FWHM/簇数），幅度差用加权线性中位数；
`phase_stats.json` 的 `pairwise.groups.<组>.pairs.<序号>` 逐对给这些数与它们的点分路径。

| 组（`pairwise.groups`） | 对 | 说明 |
| --- | --- | --- |
| `within_1x1` | (p0,p1)、(p2,p3)、(p4,p5) | `ring_1x1` 内 3 对。峰号是**12 点钟起顺时针**，故这 3 对都是相距 60° 的相邻峰，**刻意避开 Friedel 对** (p0,p3)/(p1,p4)/(p2,p5) |
| `within_r3` | 同上 | `ring_r3` 内 3 对 |

**没有跨环组**：图集里不存在连接两族的图，`pairwise.groups` 里也不存在跨环键。

**为什么避开 Friedel 对**：实图像恒等式给 `ψ_{−q} = conj(ψ_q)`，于是
`D_{j,−j} = 2 arg ψ_j`（一个场的平凡函数）、`a_{j,−j} ≡ 0`（精确为 0）、`θ_j + θ_{−j} ≡ 0`——
这一对不携带新的独立信息，所以组内对不用它（self-test 逐条实测，阈值 1e-12/1e-15）。

### 3.4 v3 → v4 的「变与不变」

| 项 | 变化 |
| --- | --- |
| 取样 | 分布描述量（中位数/IQR/FWHM/簇数/簇宽）从"某一幅度阈值下的样本"改为**全部有效像素的样本**；CLI 的 `--gate` 删除（现在显式报错拒绝），掩码面板、两套字段、阈值类字段全部删除 |
| 相位轴 | 逐峰 θ 分布图由双面板改为**单面板**，只画 `θ ∈ [0, 2π)`；重排统计与字段删除 |
| 参考阶梯 | 直方图上的三条等距虚线、轴标签说明、manifest 参考线字段与断言、以及由其派生的全部标量字段删除（三独立相位和与逐像素三重积本身**保留**，只是不再换算成阶梯距离、不再画图） |
| 图集 | 31/环 → **30/环**；成对类别 `2dhist` → **`phase_diff_dist`**；跨环图（18+1）全部删除；合计 81 → **60** |
| 色条与色标 | 所有 map/密度图（40 张）都有色条且画在数据区之外（写入前断言 + 核对器复查）；色标全局统一（theta `[0,2π]`、12 张 amplitude 共用一套 SymLogNorm、D `±π`、a `±1`） |
| 引擎 / 相位约定 / gauge / 环定位 / 系统带 | **不变** |

## 4 输出与**图集清单（交付契约）**

非图产物：`phase_stats.json`（所有数字的唯一来源，含 `pairwise` 段与 `atlas.norms`）、
`phase_stats.csv`（逐峰表）、`phase_relations.csv`（Friedel 对/三独立和/三重积）、
`phase_ring_comparison.csv`（两族对照表）、`ring_candidates.csv`、`phase_stats.log`、
`atlas_manifest.json`。

**图集：每族 30 张（`ring_1x1` 30 + `ring_r3` 30）= 60 张，没有任何跨环图**（`--no-figures` 关闭）。
文件名与张数写死在 `stm_phase_analysis.py` 顶部
（`PER_PEAK_FIGURES` / `SUMMARY_FIGURES` / `PAIR_FIGURES` / `FIGURES_PER_RING` /
`TOTAL_FIGURES`），manifest 与 `phase_stats.json` 的 `atlas` 段同时声明 30/30/60。

| # | 文件名模式（`RING` = `ring_1x1` 或 `ring_r3`，`i` = 0…5；`j_k` = 该对的峰号） | 面板（manifest 的 `panels` 字段逐字声明） |
| --- | --- | --- |
| 1 | `RING_p{i}_amplitude.png` | `["amplitude \|psi(r)\| (global symlog scale)"]`，色标 = 全局 amplitude 色标 |
| 2 | `RING_p{i}_theta_map.png` | `["theta(r) map over the sample of the statistics"]`，色标 = 全局 theta 色标 |
| 3 | `RING_p{i}_theta_dist.png` | `["theta distribution over the sample of the statistics"]`（单面板，x 刻度 `0/π/2/π/3π/2/2π`） |
| 4 | `RING_theta_hist_summary.png` | 六峰 θ 分布 2×3 汇总（每格标题带中位数） |
| 5 | `RING_theta_map_summary.png` | 六峰 θ(r) 图 2×3 汇总（同一份样本掩码、同一全局色标、右侧留条带放色条） |
| 6 | `RING_ring_members_qspace.png` | FFT2 log 幅度上标出该环六个谱峰位置，`p0…p5`（编号 = `phase_stats.json` 峰号）标在**径向外侧**，不遮住谱峰本身；带色条 |
| 7–9 | `RING_pair_{j}_{k}_phase_diff.png` / `..._phase_diff_dist.png` / `..._amp_diff.png` | 环内每对 3 张（`(j,k)` ∈ {(0,1),(2,3),(4,5)}）：`["D(r) = wrap(arg psi_j - arg psi_k)"]` / `["D distribution over the pair sample, weight \|psi_j psi_k\|"]` / `["a(r) = (\|psi_j\| - \|psi_k\|) / (\|psi_j\| + \|psi_k\|)"]` |

计数核对：每族 `3×6 + 3 + 3×3 = 30`，合计 **60**。

**色条契约**：每张 map/密度图（本机实测 40 张：每环 6 amplitude + 6 theta_map + 1 theta 网格 +
1 q 空间 + 3 phase_diff + 3 amp_diff）都带**自己的色条轴**，且该轴与**任何**数据轴的包围盒交集面积为
0；`atlas.Atlas.add` 在写文件前逐图断言（map 图没有色条同样报错），manifest 逐图记录
`mappable` / `colorbars` / `colorbar_outside` / `colorbar_overlap`，`atlas.py --check` 再核一遍，
self-test 另有"把色条挪进数据区必须变红"的**反面检查**。1-D 分布面板（`theta_dist`、
`phase_diff_dist`、`theta_hist_summary`）没有 mappable，因此不带色条。

**色标契约（全局统一）**：所有 theta(r) map `vmin=0, vmax=2π`；12 张 amplitude map 共用**同一套**
`SymLogNorm`（`vmin=0`、`vmax` = 12 个解调场有效 `|ψ|` 的**全局最大值**、`linthresh` = **12 个场全部
正幅度的合并中位数**（全局统计量）、`linscale=0.5`）；所有 `D(r)` map `±π`；所有 `a(r)` map `±1`；
q 空间图为归一化 log 幅度 `[0, 1]`。每张图的色标记录写进 manifest 的 `norm`
（`{scale, kind, vmin, vmax[, linthresh, linscale]}`），并与 `phase_stats.json` 的 `atlas.norms`
全局声明逐字段比对 —— 因此"跨峰跨环可以直接目视比较"是一句被程序核对的话。

linear 区取**全局中位数**而不是最小正值的理由（实测）：2117² 画布上最小正 `|ψ|` 比主体数据低好几个
数量级，linear 区因此只有 ~1e-17 宽，整幅图被挤到色条顶端的 t≈0.87–0.96，弱峰与强峰看上去几乎同色；
把 linear 区放到合并中位数后 `vmin`/`vmax` 仍全局统一，而弱峰（t≈0.22）与强峰（t≈0.72–0.75）在色条上
相差 0.5，一眼可辨。

**被掩像素的颜色**：theta(r) map（逐峰与六联汇总）用 `hsv` 但先 `set_bad(#b0b0b0)`，被掩像素
（矫正画布的 NaN 填充、样本外的像素）画成与 amplitude map 相同的灰，读者能看到分析在哪里停止，
而不是一片透明的空白；色标本身（`[0, 2π]`、hsv）不变。

**画布标题契约**：每张图都必须把 manifest 里的 `title` **画在画布上**（figure 级、`wrap=True` 自动折行、
长注解不会撑宽画布）。`atlas.Atlas.add` 在保存前把画布上真实渲染的标题文本（`fig._suptitle` /
各 axes 的 title / `fig.texts`）与 `title_for(label, annotation)` 逐字比对，缺标题或文本不一致即抛错；
写完再量一次"数据轴之上那条标题带"的墨迹像素数，空带同样抛错。manifest 逐图记录
`canvas_title`（画布上已核对过的那个字符串）、`title_strip_px`（数据轴之上的带高）与 `title_band_ink`，
`atlas.py --check` 用 `title_strip_px` 在 PNG 上**重新数一遍**墨迹（`canvas_title == title` 也逐图核对）。

**标题只有一行（v4.2）**：图的顶部不重复写同一件事。

| 图的类型 | 画布上的标题 |
| --- | --- |
| 单数据轴图（逐峰 `amplitude`/`theta_map`/`theta_dist`、成对 `phase_diff`/`phase_diff_dist`/`amp_diff`、`ring_members_qspace`） | **只有一行**图级注解标题（= manifest 的 `title`）；该数据轴**不再**画自己的短标题 |
| 多面板网格（`theta_map_summary`、`theta_hist_summary`） | 一行图级注解标题 + **每个面板自己的标识标题**（`p0..p5`；直方图网格的面板标题还带该峰的中位数） |

这条设计也是程序化的：`Atlas.add` 会在写文件前数各轴的标题——单轴图带任何轴标题即抛错，
多面板图的轴标题个数必须等于面板数（每个面板都必须有自己的标识）；manifest 逐图记录
`panel_axes`（数据轴个数）与 `axes_titles`（各轴标题文本），`atlas.py --check` 按同一条规则复核。

**manifest 条目模式（逐字段，`atlas.py --check` 逐项核对）**

| 字段 | 取值 / 含义 |
| --- | --- |
| `file` | PNG 文件名（见上表模式） |
| `group` | `ring_1x1` / `ring_r3`（仅此两种） |
| `kind` | `per_peak`（`peak` = 0…5）、`summary`（`peak = null`）、`pairwise`（`pair` = `"<j>+<k>"`） |
| `peak` | 峰号 0…5（`kind=per_peak`），其余为 `null` |
| `pair` | 该图的反射对键（`kind=pairwise`），其余为 `null` |
| `panels` | 该图面板的自左向右（网格则逐行）名称表，见上表 |
| `mappable` | 该图是否画 2D 地图/密度（map/密度类图必须为真且必须有色条） |
| `canvas_title` | 保存前在画布上核对过的标题文本，必须逐字等于 `title` |
| `panel_axes` / `axes_titles` | 该图数据轴的个数 / 各轴自己的标题文本；单轴图必须是 `1` / `[]`，多面板图必须面板数 = 标题数 |
| `title_strip_px` / `title_band_ink` | 数据轴之上标题带的高度（px）与其中的墨迹像素数；核对器按该高度在 PNG 上重新计数 |
| `colorbars` / `colorbar_outside` / `colorbar_overlap` | 色条个数 / 是否与所有数据轴不相交 / 最坏交集面积 |
| `norm` | 色标记录；与 `phase_stats.json` 的 `atlas.norms[scale]` 必须逐字段相等 |
| `size_px` | `[width, height]`，与 PNG 实际尺寸一致 |
| `label` / `annotation` / `title` | 人类可读前缀 / 取整后的注解数字 / `label + " \| " + annotate(annotation)` |
| `paths` | 每个注解字段在 `phase_stats.json` 里的点分路径；**成对图的路径全部指向 `pairwise.groups.<组>.pairs.<序号>.*`** |

**图-数字契约（可程序化核对，不需要 OCR）**：每张图的标题由**一份注解字典**渲染
（`atlas.annotate`，字段顺序由 `atlas.FIELD_ORDER` 规范固定），同一份字典写入
① 图标题、② `atlas_manifest.json` 的 `figures[]`、③ PNG 的 `tEXt` 块 `stm-atlas`（JSON，另含
`panels`/`mappable`/`colorbars`/`norm`）。`paths` 指向 `phase_stats.json` 的点分路径，注解值是该 JSON
数字按打印位数取整的结果（核对容差 = 半个末位）。第三方可用**一条命令**独立复核：

```bash
.venv/bin/python skills/phase-analysis/scripts/atlas.py \
  --check OUT/phase/atlas_manifest.json --stats OUT/phase/phase_stats.json \
  --expected-figures 60 --expected-per-ring 30
```

命令逐张检查：存在、非零、PIL 可开且尺寸与 manifest 一致、PNG 内嵌注解（含 `panels`/`mappable`/
`colorbars`/`norm`）= manifest、标题 = 注解字典的渲染、注解数字 = `phase_stats.json` 对应路径的
数字（末位半单位容差）、`group`/`kind`/`peak`/`pair` 自洽、`panels` 非空、map 图必有色条且色条与
所有数据轴不相交、每张图的色标 = JSON 的全局声明、总数与每族张数自洽。
**self-test 与端到端冒烟都真的以命令行方式调用它**，并另外跑两次负面对照：
声明 59 张、声明每族 29 张时它必须变红。

**命名规则**：文件名、图标题、脚本 title/label 只用 `ring_1x1` / `ring_r3` 与数学量名；
不使用任何 k 空间/物理称呼，也不使用任何参考线/阶梯类名词。

## 5 一键 self-test

```bash
cd /path/to/STM_DataProcessing
MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python skills/phase-analysis/scripts/selftest.py \
  --stm-lib /path/to/STM_DataProcessing/src
# 可选：--quick（跳过端到端阶段）、--workdir DIR、--size N、--keep、--lambda-nm、--nm-per-px
```

覆盖与验收阈值（本机实测 **124/124** 通过；最后一行打印的"通过数/检查数"与 `check()` 自己
计数的行数必须一致，且检查数不少于 95，所以某个 section 不会悄悄消失）：

| 类 | 量 | 验收阈值 |
| --- | --- | --- |
| **交付源码契约** | 4 个交付脚本不出现已删除层（旧掩码引擎、阈值取样、重排轴、参考阶梯）的任何 token | 0 命中 |
| | 扫描器自证非空转（注入一个 token 必须被报出） | 注入即报 |
| | CLI 有 `--lambda-nm`、有已废除选项名单与拒绝函数；已废除选项非零退出 | 结构断言 + 非零退出 |
| | `atlas.py` 里没有任何竖线/横线参考线调用，分布面板只有显式相位刻度 | 无参考线 |
| **取样规则** | 掩码 = 有效像素减去"权重非有限或恰为 0"的像素；幅度整体缩放到 1e-9 掩码不变 | 逐条精确 |
| **引擎** | `Σψ = ΣT e^(−i q·r)`（3 个 λ × 3 个 q） | 相对偏差 ≤ 1e-9 |
| | 平移律（3 个平面波） | ≤ 1e-9 rad |
| | `θ = arg ψ = +φ`（9 个平面波） | ≤ 0.01° |
| | 无斜坡（相邻像素相位差） | ≤ 1e-9 rad |
| | Friedel `ψ_{−q} = conj(ψ_q)` | 相对 ≤ 1e-12 |
| | v2→v3 平移 `−π(q_x+q_y)` | ≤ 1e-6° |
| | 单区三独立和 / 逐像素三重积 = 3Φ | ≤ 0.5° |
| **gauge 层** | 平移律残差、`r0 − δ` 模型、`(N/2,N/2)` 吸收 | ≤ 1e-9° |
| | 周期画布：原点随图像走时 gauge 后相位不变 | ≤ 1e-9° |
| | θ / R / 簇数不变；非周期画布的 FWHM 变化（边界带量级，如实报告） | ≤ 0.5° / ≤ 0.05 / 精确 / ≤ 1° |
| **成对** | 注入 Δφ = 40° 的 D 回收 | ≤ 0.5° |
| | 幅度差回收（3 组注入比值） | ≤ 1e-3 |
| | Friedel 对：和 ≡ 0、`a ≡ 0`、`D = 2θ_j` | 1e-12 / 1e-15 / 1e-12 |
| | 组内 3 对（60°、非 Friedel）+ 每环各自三对、不存在跨环配对函数 | 结构断言 |
| | D 分布：一维、`x ∈ (−π, π]`、密度积分 = 1 | 结构断言 |
| | 成对样本 = 两掩码交集（半数有效掩码给出半数像素） | 精确 |
| **标题与掩码** | 60 张图的 `canvas_title == title`；6 张环级图（qspace/两网格）标题带墨迹 | 逐图相等；≥ 20 墨迹像素 |
| | 56 张单轴图的 `axes_titles == []`（只有一行图级标题）；4 张网格各保留 6 个面板标识 | 逐图结构断言 |
| | 写入前给单轴图加轴标题必须报错；多面板图每个面板必须有标识 | 前抛错、后通过 |
| | 60 张图的标题带（数据轴之上）都不为空（A4 缺陷类） | 0 空带 |
| | 画布读回标题 == manifest 标题；无标题 / 改一字符 / 写入前丢标题 | 前者通过、后三者抛错 |
| | theta map 被掩像素画成 #b0b0b0（纯 hsv 的 NaN 是透明） | 逐像素命中 |
| | amplitude linear 区 = 12 个场正幅度的合并中位数；vmin=0、vmax 全局 | 中位数、非最小值 |
| **图集** | 60 张、命名模式、PIL、内嵌注解 = manifest、数字 = JSON | 0 失败 |
| | `atlas.py --check --expected-figures 60 --expected-per-ring 30` | 退出码 0 |
| | 同上但声明 59 张 / 每族 29 张 | 必须非零退出 |
| | 40 张 map/密度图每张都有色条且色条与所有数据轴不相交 | 交集面积 = 0 |
| | 色条挪进数据区 / map 图无色条 | 断言变红 |
| | 12 张 amplitude map 色标完全相同、theta/D/a 色标 = 全局声明 | 逐字段相等 |
| | 每张逐峰图的 `pixels` 注解 = 该峰 `n_samples` | 相等 |
| | 已废除选项被 CLI 拒绝 | 非零退出 |
| **确定性** | 两次运行的 `atlas_manifest.json` **逐字节相同**、`phase_stats.json` 掩掉各自输出目录字段后**逐字节相同**、60 张 PNG **逐字节相同** | 相同 |
| **验收覆盖** | R1–R11 每项都有跑过的具名检查；检查计数 = 打印的通过数 | 0 缺失 |

检查清单：

1. **交付源码契约**：见上表前两行；另检查 `phasepipe` 只经 `gaussian_field` → `localqmap.demodulate`
   取场。
2. **取样规则**：单元检查 + 幅度尺度不变性（反面检查：阈值规则会去掉一半像素，本规则不会）。
3. **圆统计量**：加权圆均值；圆中位数 = L1 目标精确极小；退化样本极小集被标记；
   高斯样本去卷积 FWHM = 2.3548σ（1 %）；双成分簇心/权重。
4. **引擎恒等式**：见上表。
5. **gauge 层**：精确律、吸收律、周期画布不变性、非周期画布的漂移率与残差（如实报告）、
   拟合分支数与诱导位移表。
6. **成对契约**：组的构成、Friedel 对为何平凡、交换 (j,k) 让 D/a 反号、D 直方图边界与归一化、
   成对样本 = 掩码交集。
7. **成对回收**：注入相位差与幅度比。
8. **环定位分支**：`auto`/`radius` 的 1/√3 定位、参考半径过远、无配对环、最内环即 r3；
   峰编号从 12 点钟起顺时针。
9. **多成分判别边界**：见 §3.2 表；另经引擎实测全画布相位值 = 面积加权相量预言（残差 < 1°，
   残余来自另一族环对解调窗的谱泄漏）。
10. **色条契约**：参考图（色条在数据区外）→ 移到数据区后同一审计必须变红 → map 图无色条时
    写入必须报错。
11. **画布标题契约**：读回画布标题（不从 manifest 读）→ 没有标题 / 改一个字符 / 写入前丢标题
    三种反面情况都必须抛错 → "有标题的图 vs 无标题的图"的标题带墨迹像素必须分开 → theta map
    被掩像素的灰必须真的画出来（含"纯 hsv 的 NaN 是透明的"这一反面事实）→ 单轴图带短轴标题必须
    被拒、多面板图保留面板标识必须通过。
12. **端到端**：真跑两次完整管线（**先断言第二次运行的返回码**）→ 图集清单断言（60 张、命名模式、
    PIL、注解 = JSON、色标、色条）、**确定性**（manifest 与 PNG 逐字节相同、`phase_stats.json`
    掩掉各自输出目录字段后逐字节相同）、log 数字 = JSON 数字、已废除选项非零退出、
    缺 r3 分支（退出码 2、有明确信息、0 张图）。
13. **`--size-nm-from-log`**（真跑三次）：矫正后画布行优先；只有一行时回退取该行；
    一处都没有 → 非零退出码 + 明确信息。
14. **验收覆盖**：R1–R11 与本次修复的 F1–F5/F8 → 具名检查的映射表；任一映射的检查没跑就是红行。

## 6 变与不变（引用任何一个数字前请读）

| 类 | 量 | 对图像原点 / 对 r0 分支 |
| --- | --- | --- |
| **稳健量（可以直接引用、跨约定比较）** | Friedel 对和 | 不变（恒等式） |
| | 三独立相位和、逐像素三重积 θ | **精确不变**（Σq = 0 时） |
| | **成对 D / a 场与其 D 分布** | **不随 r0 分支变**（D、a 由**原始解调场**定义，不进入 gauge 层）；只随图像原点整体平移 `−(q_j − q_k)·δ`；交换 (j,k) 让 D、a 反号，故直方图关于 `D = 0` 镜像 |
| | 每峰分布形状（R / FWHM / 簇数 / 簇宽） | 不变（整体平移；非周期画布上到边界带量级） |
| | 幅度 | 不变（守门量：相消时相位无意义） |
| **参考量（必须附条件）** | 单峰绝对相位 | **随分支变**：点阵平移给 `(2π/N)q·L` |
| | 跨峰 spread | 随分支变（诱导位移一般不是整齐的分数圈） |

引用单峰绝对相位时必须写明：`r0` 用的是解集里的**哪一个分支** + 由参考环残差反推的**系统带
`σ_peak/3 ≈ RMS_res/2.12`**。

* **随图像原点漂移**：单峰原始相位按 `θ → θ − (2π/N) q·δ` 线性漂移（大 |q| 时每像素可达几十度）；
  闭合和、分布形状与成对 D（同分支下）不受影响。
* **Δλ 与 Δ(视场)**：λ 只改变解调窗宽，**不改变**逐峰相位值（`k = 0` 权重恒为 1 的恒等式）；
  但改 λ 会改变 R/FWHM 与场图的外观。`nm/px` 只通过 `λ_px = λ_nm/(L/N)` 进入引擎。
* **依赖口径**：FWHM（平滑/分箱口径）、簇宽（窗口口径）、成对样本（两掩码交集）——这些都在文档里
  写明；**取样本身不再是口径**：只有"全部有效像素 + 权重 `|ψ|`"一条规则。
* **样本的空间含义**：样本是整张有效画布（含周期性边界带），不偏向画布任何一侧；
  这正是"逐峰相位值 = 全画布精确求和"的代价与含义。

## 7 不随本 skill 落盘的方法学附件

`origin_space.py`（r0 解集与不确定度传播的纯数学脚本）**不属于本 skill 的落盘内容**——它是方法学
研究附件，源头在 `data_processing/phase_reanalysis/_methods/`。它的四个结论已被本 skill 吸收进
本文档：解集 = {最佳 r0} ⊕ 参考环相位点阵、原点误差传播 `std(Δφ) = σ_peak·|q|/(√3 R1)`
（对 1/√3 对退化为 `σ_peak/3`）、Σq = 0 的闭合和与原点无关、分布形状与平移无关（§6 的稳健量清单）。

## 8 修改记录

2026-09（v1）：初版（topo0009 实测流程），六峰相位分析。
2026-09（v1 入库）：脚本改为调用 `stm_data_processing.utils.bragg_peak` 包。
2026-09（v2）：锚定环显式化、两族同口径双环分析、r3 由 1/√3 定位与「未检出」分支、
唯一的相位估计量定义 + 分布描述量分离、Friedel/三独立/三重积/R 全套、
图集清单与图-数字契约、一键 self-test、命名中性化。
2026-09（v2.1）：`--size-nm-from-log` 改读**矫正后画布**视场。
2026-09（拆分）：几何矫正拆分为独立 skill `topo-correction`（后改名 `affine-correction`）；
本 skill 改名 `phase-analysis`。
2026-09（v3）：解调引擎换成 `local-q-map` 的 Gaussian 窗（`θ = arg ψ ≈ +φ`），
新增成对相位差分析（D/a 场 + 二维直方图）与跨环成对组，图集 31/环 + 跨环 19 = 81 张。
2026-09（**v4，本版**）：见 `CHANGES.md` 顶部——
① **取样只有一条规则**（全部有效非 NaN 像素、权重 `|ψ|`），CLI 阈值选项删除且显式拒绝、
阈值掩码与两套字段、掩码面板全部删除；
② **相位轴不再重排**（θ 分布单面板）；
③ **相位参考阶梯删除**（虚线、轴标签说明、manifest 参考线字段与断言、以及由其派生的标量字段），
三独立相位和与逐像素三重积 θ 的均值/中位数/R/FWHM/簇数**保留在 JSON 里**；
④ **图集 81 → 60**（每环 30 = 逐峰 3 类 ×6 + 汇总 3 + 环内 3 对 ×3 类；新增 D 分布图，
删除三重积图、成对二维直方图与全部跨环图）；
⑤ **色条与色标契约**（色条画在数据区之外并有程序化断言与反面检查；theta/amplitude/D/a 全局统一色标）。
