# 相对仓库现有版本（v1）的改动说明

> **最新一版是 v3.0（2026-09-20）**，改动清单见下面第一节；它相对的是 **v2**（不是 v1）。
> 下面其余各节是历史条目（v2 与 v2.1），保留原样。

对照对象：仓库 `skills/stm-topo-phase-analysis/`（`SKILL.md` + `README.md` +
`scripts/stm_topo_correct.py` + `scripts/stm_phase_analysis.py`）。
下面逐条列出**改了什么**、**为什么**。所有数字都可在 self-test 输出里查到。

---

## v3.0（2026-09-20：换引擎 + 论文式成对相位差 + 81 张图集）

用户批准的计划（图集粒度 B）分三部分，全部在 staging 内实现、复核后落盘。

### A 引擎替换：废弃硬圆 mask 引擎，改用 local-q-map 的 Gaussian 窗

| # | 改动 | 理由 |
| --- | --- | --- |
| A1 | `phasepipe.py` **删除** `reflection_field`（单反射圆 mask + iFFT）、`demod_phase`（`φ = angle(ψ) − (2π/N) q·(r−c)`）、`circle_mask`、`complex_ifft`；新增 `gaussian_field(topo, q_px, lambda_nm, nm_per_px)`，把 FFT 像素波矢换成 `q_rad_px = 2π q_px / N` 后**薄封装调用 `localqmap.demodulate`**（按路径 import sibling skill `skills/local-q-map/scripts/localqmap.py`，不复制、不修改引擎源码） | 论文口径要求 Gaussian 窗局域解调；硬圆 mask 的"减法解调"会引入与窗/掩膜几何有关的伪影，且解调相位自带斜坡 |
| A2 | 相位口径统一为 **`θ_q(r) = arg ψ_q(r) ≈ +φ_q(r)`**：无 `q·(r−c)` 斜坡、无每峰常数；`phasepipe.theta_field(psi)` 是唯一入口 | 与 `local-q-map` skill 同一约定，跨 skill 可直接比较 |
| A3 | `stm_phase_analysis.py`：删除 `--pct`、`mask_radius` 及全部相关产物（log 行、JSON 的 `mask_pct`/`mask_radius_px`、per-peak `mask_in/mask_out` 图、`qspace_mask`/`mask_all_in`/`mask_all_out`/`case_phase_histograms` 图）；新增 `--lambda-nm`（默认 **3.0 nm**）与 `--pair-bins-x/--pair-bins-y`（默认 180/100） | 引擎换代后 mask 半径不再是参数；λ 是引擎唯一的新参数 |
| A4 | `analyse_ring` 签名改为 `(topo, valid, ring, lambda_nm, nm_per_px, …, peaks_override=None)`，返回 `(records, fields, psi)` | 逐峰场现在来自实空间图像的窗解调，不再来自 FFT2 的 mask |
| A5 | **不动的部分**：峰检测、环聚类、1/√3 配对、`choose_rings`、三重积（含 `--project-q`）、`fold_to_120`、gauge 拟合与分支表、系统带、Friedel/三独立和、每峰分布形状、幅度/相干度 | 只换输入相位场；这些量的定义与代码逻辑逐字未改，v2 的方法学结论继续有效 |
| A6 | 三条恒等式写进 SKILL 与 self-test：`Σ_r ψ = Σ_r T e^(−i q·r)`（窗的 `k=0` 权重恰为 1 ⇒ 与 λ 无关）、`ψ_{−q} = conj(ψ_q)`、`ψ'(r) = e^(−i q·δ) ψ(r−δ)` | 逐峰相位值 = 全画布幅度加权圆均值 = `arg Σ T e^(−i q·r)`，这是**换引擎后唯一改变的定义**（旧口径是 `mask` 内相干和） |

**raw 单峰相位的平移**：v2 的相位值（单 bin mask 恒等式）= `arg X(q) + (2π/N) q·c`，`c = (N/2, N/2)`；
换成 v3 后同一反射的相位值为 `φ`，即**每峰平移 `−π(q_x + q_y)`**。该常数恰是**原点平移
`(N/2, N/2)`**（`(2π/N) q·(N/2, N/2) = π(q_x+q_y)`），被 gauge 拟合完全吸收
（`r0_v3 = r0_v2 − (N/2, N/2)`）⇒ **gauge 后相位、闭合和、分布形状、成对差全部逐位不变**。
self-test 对该平移与其吸收各有一条断言（阈值 1e-6° 与 1e-9°）。

### B 论文式成对相位差分析（新增）

| # | 改动 | 理由 |
| --- | --- | --- |
| B1 | 每对 `(j,k)` 定义 `D_jk(r) = wrap(arg ψ_j − arg ψ_k) = arg(ψ_j·conj(ψ_k))`、`a_jk(r) = (\|ψ_j\|−\|ψ_k\|)/(\|ψ_j\|+\|ψ_k\|)`、权重 `\|ψ_j ψ_k\|`、有效像素 = 两场有效掩码**交集** | 论文式成对量：相位差场 + 归一化幅度差场 |
| B2 | 2D 直方图：x = `\|D\| mod π` ∈ [0,π]（默认 180 bins），y = `a` ∈ [−1,1]（默认 100 bins） | D 与其 Friedel 对应共享同一分布 ⇒ 折叠到 [0,π] |
| B3 | 组内对 `(p0,p1),(p2,p3),(p4,p5)`（12 点钟起顺时针编号）**刻意避开 Friedel 对** | Friedel 对 `(q,−q)`：`D = 2θ_j`（一个场的平凡函数）、`a ≡ 0`、`θ_j+θ_{−j} ≡ 0`，不携带独立信息（self-test 逐条实测） |
| B4 | 跨环对 = 每个 `ring_1x1` 峰配**方位角最近**的 `ring_r3` 峰，共 6 对；两环相差 30° 时的等距并列取 `ring_r3` **下标最小**者（`phasepipe.ANGLE_TIE`） | 跨环配对只看方位角，不看强度/倍率 |
| B5 | `phase_stats.json` 新增 `pairwise` 段：`definition` / `bins` / `groups`（`within_1x1`、`within_r3`、`cross`），每对含 `j`、`k`、`q_j`、`q_k`、`phase_diff_mean/median/R/fwhm_deg/n_clusters`、`amp_diff_median/fwhm`、`n_valid`、`hist_counts`（含 `hist_x_edges`/`hist_y_edges`）、`paths` | 成对量的唯一数字来源；图-数字注解指向这些路径 |
| B6 | log 新增成对段：每组每对的 D 均值/中位数/R/FWHM/簇数与 a 中位数、有效像素数 | 便于日志核查 |

### C 图集契约：31/环×2 + 跨环 19 = 81 张

| # | 改动 | 理由 |
| --- | --- | --- |
| C1 | 逐峰图改 3 张：`RING_p{i}_amplitude`（\|ψ\| log 图）、`RING_p{i}_theta_map`（θ(r) 图 + 幅度门掩膜）、`RING_p{i}_theta_dist`（θ 分布 + `θ mod 120°` 折叠分布，虚线 = `2πk/3`） | mask 图随引擎废除；θ 是新的唯一相位场 |
| C2 | 每族汇总 4 张：`RING_theta_hist_summary`（六峰 2×3）、`RING_theta_map_summary`（六峰 2×3）、`RING_theta_field`（三重积 θ 图 + 分布）、`RING_ring_members_qspace`（FFT2 log 幅度上标出六个谱峰位置，`p0…p5` 标在径向外侧） | 用 θ 场的对应图替换 φ 场的旧图；不再画 mask 圆 |
| C3 | 组内成对 3 张 ×3 对：`RING_pair_{j}_{k}_phase_diff / _amp_diff / _2dhist` | 每对三个交付图 |
| C4 | 跨环 6 对 ×3 张 + `cross_pair_phase_diff_grid`（六对 D 场 2×3 汇总）= 19 张 | 跨环契约 |
| C5 | 张数：每族 `3×6 + 4 + 3×3 = 31`，跨环 `6×3 + 1 = 19`，合计 **81** | 粒度 B |
| C6 | `atlas.py`：条目新增 `group`（`ring_1x1`/`ring_r3`/`cross`）、`kind`（`per_peak`/`summary`/`pairwise`）、`pair`（成对键 `"<j>+<k>"`）；manifest 的 `per_ring` 换成 `per_group`（含 `per_peak`/`summary`/`pairwise`/`peaks`/`pairs` 计数）；`check_manifest` 增 `expected_cross`，并核对 `group`/`kind`/`peak`/`pair` 自洽；`--check` 支持 `--expected-per-ring 31 --expected-cross 19`（`--expected-figures 81`） | 图集契约必须机器可核 |
| C7 | 删除 `atlas.py` 的 mask 类绘图方法（`mask_pair` / `qspace_mask` / `case_histograms` / `phi_distribution` / `grid_histograms` / `grid_maps`），换成 `amplitude_map` / `theta_map` / `theta_distribution` / `grid_theta_histograms` / `grid_theta_maps` / `theta_field` / `ring_members` / `pair_field` / `pair_2dhist` / `pair_grid` | 与 C1–C4 一一对应 |
| C8 | `phase_stats.json` 的整数计数数组（`hist_counts`）按行内联写出（`dump_json`），避免逐元素换行把 JSON 撑成百万行 | 81 张图集 + 12 对的直方图后，默认缩进写法体积/行数都不可用 |

### D 文档与自检

| # | 改动 | 理由 |
| --- | --- | --- |
| D1 | `SKILL.md` 重写：§3.1 引擎与相位约定（`θ=+φ`、无斜坡、恒等式三件套、周期性边界如实说明）、§3.2 成对分析、§3.3 v2→v3 变与不变、§4 新图集清单与 manifest 字段、§5 self-test 表、§6 变与不变更新；`SKILL_VERSION = "3.0"` | 文档必须与实现逐条对应 |
| D2 | `README.md` 更新：文件表、依赖（显式声明 sibling skill `local-q-map`）、命令（81/31/19）、关键点 | 同上 |
| D3 | `selftest.py` 重写（80 项）：① 引擎源码契约（无旧引擎 token、只经 `gaussian_field`、CLI 有 `--lambda-nm` 无 `--pct`）；② 引擎恒等式（Σψ 与 λ 无关 3×3、平移律、θ=+φ 9 个平面波、无斜坡、Friedel、v2→v3 平移）；③ gauge 层（精确律、`(N/2,N/2)` 吸收、周期画布上 gauge 不变、非周期画布漂移率如实报告、分支表）；④ 成对契约（组内 3 对 60°、跨环 6 对最近方位角、Friedel 对平凡性、交换反号、直方图边界）；⑤ 成对注入回收（Δφ ≤ 0.5°、幅度比 ≤ 1e-3）；⑥ 环定位分支；⑦ 多成分边界（面积权重）；⑧ 端到端（81 张图集、命名模式、命名纪律扫描（图名/图标题 + SKILL.md/README.md）、PIL、注解 = JSON、`--pct` 被拒、两个 JSON **逐字节相同**（`phase_stats.json` 掩掉各自输出目录字段 `atlas.dir` 后）、图逐像素相同、缺 r3 分支）；⑨ `--size-nm-from-log` 三项 | 覆盖用户列出的全部验收项 |

**验证**：`selftest.py --stm-lib <repo>/src` 退出码 **0**、**80/80** 通过；
`atlas.py --check … --expected-figures 81 --expected-per-ring 31 --expected-cross 19` 打印
`ATLAS CHECK PASSED`（0 failure）；`py_compile` 五个脚本全部通过；
`cmp` 确认 `skills/local-q-map/scripts/localqmap.py` 与仓库原件逐字节一致（本 skill 只读引用）。

**未改动**：`data/`、`data_processing/` 的既有产物，以及任何本 skill 目录之外的文件。

---

## 拆分（最新，2026-09-19：拆为 topo-correction + phase-analysis 两个 skill）

按用户要求把几何矫正拆成独立 skill（矫正产物可接任意后处理）：

- 本目录改名 `skills/phase-analysis/`（skill 名 `phase-analysis`，原 `stm-topo-phase-analysis`）；
  几何矫正移入新 skill `skills/topo-correction/`（见其 CHANGES.md）。
- `scripts/stm_topo_correct.py` 移出；`scripts/selftest.py` 移除矫正阶段检查
  （`test_correction_stage` + `_ratio_text`），检查总数 72 → **68**，其余检查与阈值不变。
- `stm_phase_analysis.py` 的 `phase_stats.json` 里 `skill` 字段值由 `stm-topo-phase-analysis`
  更正为 `phase-analysis`（纯字符串值，无消费者解析它）。
- SKILL.md：删除原「第一步：矫正」章，新增「§2 输入来源：矫正产物（topo-correction）」；
  其余相位口径、图集契约、稳健量表一字未改。

## v2.1 修复（2026-09-19；三处缺陷，未重新处理任何数据）

修复对象是 **v2 自身**的三个缺陷（不是相对 v1 的改动）；机制与验证证据见
`FIX_SUMMARY.md`，验证产物保留在 `_work/`（`selftest/`、`fov_check/`、
`anchor_wrong/`、`anchor_right/`）。**未改动**：`data/` 与
`data_processing/topo4_50nm_analysis_mask10px/` 的既有产物、任何相位口径与估计量定义。

| # | 缺陷（现象） | 改动 |
| --- | --- | --- |
| F1 | `stm_phase_analysis.py --size-nm-from-log` 用 `re.search` 取 log **第一处** `field of view`，即**输入画布**行（topo4: 50 nm），而脚本分析的是**矫正后** CSV（矫正后画布 51.7090 nm ⇒ nm/px、环半径、物理 |q| 全错） | 改为按优先级解析：① 带 `corrected canvas` 标签的行；② 无该行时取**最后一处**匹配；③ 都没有则报错。取值后 log 打印来源说明（`# field of view from log: 51.7090 nm (corrected canvas line; ...)`），JSON 增 `field_of_view_source`（纯新增字段，既有字段与数字不变） |
| F2 | 锚定自检只看**矫正后两个最强环的半径比**。当 raw 数据本身已有精确 1 : √3 环对时，错锚定把整个拟合按 1/√3 全局拉伸，矫正后比值**仍是** √3 ⇒ 误报 `verdict=consistent`（topo4 错锚实测 `\|det M\|^(1/2) = 0.57843`、反演 1x1 晶格常数偏 +73.9 %，却仍 consistent） | 新增**全局拉伸尺度 tell-tale**：`\|det M\|^(1/2)` 偏离 1 超过 `STRETCH_SCALE_TOL = 5 %` 即打 WARNING 并把 `anchor_self_check.verdict` 判为 `inconsistent`；`verdict_reason` 说明是"锚定环标错"还是"视场 L 标错"（两者都会整体缩放拟合）。`anchor_self_check` 新增 `stretch_scale_sqrt_det` / `stretch_scale_deviation` / `stretch_scale_consistent` / `ratio_consistent` / `verdict_reason`，**既有字段（`ratio`/`deviation`/`consistent`/`verdict`/`outer_radius_px`/`inner_radius_px`/`tolerance`/`n_rings`）保持不变**；`unverifiable` 分支与 ratio 判据逻辑不变。容差理由：包自身口径是环聚类 3 % / 标签匹配 2 %，正确实测偏 ≤2 %，错锚实测偏 42 %/73 %，5 % 两侧都有余量 |
| F3 | `atlas.py` 的 `qspace_mask` 图只画六个 mask 圆、**不标峰号**，看图无法把圆与 `p0…p5`（per-peak 图与 `phase_stats.json` 的峰号）对应 | 每个圆在**圆外沿径向外侧**（背离 FFT 中心方向、半径 `mask_radius + 8` px）标 `p{i}`，颜色同圆框 `#39ff14`、字号 10、粗体、`ha/va=center`；该图 `panels` 改为 `["FFT2 log amplitude with the ring masks (p0-p5 labelled at the circle rims)"]`，`SKILL.md` 图集表第 4 行同步 |

**验证结果（实测，命令与输出见 `FIX_SUMMARY.md`）**：

| 验证 | 结果 |
| --- | --- |
| 全量 `selftest.py`（含图集契约、锚定自检、端到端两轮管线、新增的 `--size-nm-from-log` 三项） | 退出码 **0**，全部检查通过（**72/72**，其中 v2.1 新增 3 项；v2 原为 69 项） |
| `stm_phase_analysis.py ... --size-nm-from-log <correction.log>`（矫正后 CSV，`--no-figures`） | log 头部 `# field of view from log: 51.7090 nm (corrected canvas line; ...)`、`# canvas: 1059 x 1059 px, field of view 51.709 nm (0.048828 nm/px)` ⇒ **不再是 50 nm** |
| `stm_topo_correct.py ... --anchor-ring 1x1`（错锚） | `\|det M\|^(1/2) = 0.57843`、WARNING、`anchor verdict = inconsistent`（`verdict_reason` 指全局拉伸） |
| `stm_topo_correct.py ... --anchor-ring r3`（正确锚） | `\|det M\|^(1/2) = 1.00187`、`anchor verdict = consistent`（`stretch scale ... consistent` + `ring-pair ... consistent`） |
| `atlas.py --check`（selftest 内两轮图集） | 非零 checks 全部通过、`ATLAS CHECK PASSED` |
| `RING_qspace_mask.png` 峰号标注（selftest 图集两族 + 文本调用探针） | 每张图 6 个 `p0…p5` 文本、位置 = 圆心 + (mask 半径+8)·径向外单位矢量（实测与契约逐位一致，d = 0）、颜色 `#39ff14`、字号 10、`ha/va=center`；12/12 个标签（两族 ×6）落在圆外径向带上 |

**保留的已知行为**（本次未改，如实记录）：`stm_topo_correct.py` 的 `correction_report.json`
顶层 `stretch_scale_sqrt_det` 字段（v2 既有字段）与 `anchor_self_check.stretch_scale_sqrt_det`
现在**同值双份**——顶层那份是 v2 的既有契约不动，`anchor_self_check` 里那份让自检结论与数字就地可读。

---

## v2（原版条目，下同）

## 0 结构

| # | 改动 | 理由 |
| --- | --- | --- |
| 0.1 | 新增 `scripts/phasepipe.py`（管线层）与 `scripts/phasemath.py`（纯数学：加权圆统计 + 规范固定），脚本从它们 import | 相位统计量原本只存在于一次性分析脚本里，口径散落；把估计量集中到一个模块才能"唯一定义"，也才能被 self-test 直接调用 |
| 0.2 | 新增 `scripts/atlas.py`（图集 + manifest + PNG 内嵌注解 + 独立核对器） | 图与数字的一致性原来是人工眼看；现在由同一份注解字典渲染并可用一条命令第三方复核 |
| 0.3 | 新增 `scripts/selftest.py`（一键自检；**该条目写入时为 49 项，最终版本为 69 项**） | v1 没有任何随 skill 交付的自检；恒等式与判别边界只在一次性研究目录里验证过 |
| 0.4 | 新增 `CHANGES.md`（本文件） | 便于复核与回滚 |
| 0.5 | **落盘清单（captain 裁定 2026-09-18）**：落盘 = `SKILL.md` + `README.md` + `CHANGES.md` + `scripts/` 的**六个主流程脚本**（`stm_topo_correct.py`、`stm_phase_analysis.py`、`phasepipe.py`、`phasemath.py`、`atlas.py`、`selftest.py`）；**`scripts/origin_space.py` 仍在 `scripts/` 中**（不存在 `aux/` 目录；它是方法学附件，源头在 `_methods/`），**拷入仓库时必须把它排除**；`e2e_scratch/`、`selftest_output.txt`、`DELIVERY.md` 亦不入库 | 如实描述 staging 的真实布局，并让落盘时该排除哪个文件无需猜测 |

## 1 矫正：锚定环显式化

| # | 改动 | 理由 |
| --- | --- | --- |
| 1.1 | 新增 `--anchor-ring {1x1,r3}`：`a_ref = a` 或 `√3·a` | 包内数据锚定检测认定的"第一个环"未必是 1x1 环。当 r3 环在内侧（半径比 1/√3）时，按 1x1 给参考晶格会把 r3 环拉到 1x1 的理想半径上——实测：真实案例（50 nm 视场）里 135.78 px 的内环是 r3、1x1 在 235.2 px；用 `--a 0.4261` 这种"手工换算"在 v1 里只能靠用户自己算，v2 直接说清 |
| 1.2 | 新增检测环表（`--list-rings` + 默认打印半径 px / nm⁻¹ / 成员数 / 总幅度）与"最内环位于最强环的 1/√3 处"几何提示 | 让用户**有依据地**选择 `--anchor-ring`，而不是猜 |
| 1.3 | 新增**非循环**锚定自检：矫正后两个最强环的半径比是否 1 : √3（`anchor verdict` = consistent / inconsistent / unverifiable + WARNING） | "矫正后锚定环 = 理想半径"是拟合的必然结果、不能自证；只有**另一族**环的关系能判别锚定对错。实测：正确锚定 ratio 1.732311（0.015 %）→ consistent；错误锚定 ratio 1.957 → inconsistent |
| 1.4 | 新增反演晶格常数（锚定环 `a_ref` 与 1x1 的 `a`，矫正前后各一次）与 `correction_report.json` | v1 只打印 `|b1|` 与偏差；把"这个环对应的晶格常数是多少"直接给出，配合 1.3 一起看 |
| 1.5 | 排版：`metadata`/报告写入 `correction.log` 与 JSON 双份 | 每个数字要能被第三方用一条命令复现与比对 |

## 2 相位分析：两族同口径 + 唯一估计量

| # | 改动 | 理由 |
| --- | --- | --- |
| 2.1 | **两族都分析**：`ring_1x1` 与 `ring_r3` 走**同一个** `analyse_ring`（同 mask 半径、同门、同分箱/平滑、同统计量），并给 `phase_ring_comparison.csv` 对照表 | v1 只分析 r3 族（`r3_peaks()` 取"标号 h²+k²+hk==1 的第一环"，实测在另一份数据上取到的其实是另一族），且两族口径无法直接对比 |
| 2.2 | r3 环**由参考半径的 1/√3 自动定位**（`--r3-tol`，默认 3 %），不再依赖包内标号；定位不到 → `ring pair: NOT FOUND`、退出码 2、**不画图**、写 `ring_candidates.csv` | v1 的 `r3_peaks()` 依赖包内 `index_hk` 标号；标号失败时会退化成"最内六峰"，静默分析错环 |
| 2.3 | 参考环**可显式选择**：`--anchor auto|strongest|outer|inner|radius`（+ `--reference-radius-px`），log 打印实际路径 | 同 1.1：环的身份是用户的选择，不该由脚本默默假设 |
| 2.4 | **相位估计量唯一定义**：相位值 = mask 内**全部有效像素**的幅度加权圆均值（未加门）；中位数/FWHM/IQR/簇/簇宽 = **加门**样本的形状描述量，门随每个数字写明 | v1 把"相位"等同于门后直方图的量，且没有写清门的空间偏置（Hanning 窗 + p50 门使居中区域被放大，实测可使相位和偏离 12°） |
| 2.5 | 新增 Friedel 对和（含每对的 `|Σq|`）、三独立值（mod 360° 与 mod 120°，含镜像值）、到 0/120/240 阶梯的距离、逐像素三重积 θ（均值/中位数/R/FWHM/簇数/镜像/`3θ mod 360°`/`|q_sum|`）、120° 折叠分布、簇宽 | v1 只有直方图与"某角处倍率"；倍率依赖画布原点约定、不可跨画布比较（实测同一峰在不同画布上 14.9 vs 6.38），v2 换成规范稳健量 |
| 2.6 | 新增 **r0 规范固定**：`ring_1x1` 六峰最小二乘（多起点，返回全部局部极小）+ 分支表 + 参考环偏离 0 的 rms | 绝对相位与三独立值必须先在统一规范下才可比；分支表把"120° 重标号"这一内在自由度量化 |
| 2.7 | 相位值改用**未加门**加权圆均值；`--no-project-q` 可关闭 Σq 投影（默认投影到零） | 未加门均值复现窗加权真值（实测偏差 0.0014°），而门后值最多偏离 12°；Σq ≠ 0 会把 θ 变成跨画布斜坡 |
| 2.8 | 峰编号改为**从 12 点钟起顺时针**（原来按 qy 升序，同 qy 的两峰顺序不稳定） | 编号必须无歧义且写进文档；Friedel 配对由波矢量计算、不靠编号假设 |

## 3 图集与命名

| # | 改动 | 理由 |
| --- | --- | --- |
| 3.1 | 图集写成**交付契约**：每族 25 张（每峰 3 张 × 6 + 汇总 7 张），两族 50 张，文件名与张数列入 `SKILL.md`，并由 `atlas_manifest.json` + `phase_stats.json` 的 `atlas` 段同时声明 | v1 的输出清单散在 SKILL.md 里、与脚本实际产出无法自动对账 |
| 3.2 | 每张图的标题 = 注解字典的渲染；同一份字典写入 manifest 与 PNG `tEXt: stm-atlas`；注解数字带其在 `phase_stats.json` 中的点分路径 | 实现"图注记数字 = JSON 数字"的**程序化**核对（不需要 OCR），第三方可自写脚本比对 |
| 3.3 | 新增 `atlas.py --check`：存在/非零/PIL 可开/尺寸/内嵌注解/标题重渲染/数字对齐/张数自洽 | 一条命令独立复核图集 |
| 3.4 | **命名中性化**：文件名、图标题、脚本 title/label、文档一律只用 `ring_1x1` / `ring_r3` 与数学量名；直方图参考线是数学阶梯 `2πk/3`（0/120/240°）；不再出现任何 k 空间/物理称呼 | 用户要求：skill 只做几何与相位数学，物理解释由使用者进行 |
| 3.5 | 每张图的注解里同时写清 mask 半径/gate/峰号/半径/SNR/中位数/FWHM/簇数/R | 图离开 log 也能自证来源 |
| 3.6 | manifest 每张图新增 `panels`（自左向右/逐行的面板名表）并在 `--check` 里核对；`--check` 同时校验 `kind`/`peak` 自洽与 `panels` 非空；SKILL.md §4 增补「manifest 条目逐字段模式表」 | 让「逐峰 φ(r) 相位图在哪」这类问题**可程序化回答**（它是 `*_phi_dist.png` 的左面板），也让 `ring`/`kind`/`peak` 命名规则有出处 |
| 3.7 | self-test 新增：以**命令行**调用 `atlas.py --check ACTUAL_MANIFEST.json`（断言退出码 0 + `ATLAS CHECK PASSED`），并断言 18 张 per-peak 图与每族 7 张汇总图的 `panels` 与文档一致 | 图集契约不能只「有能力核对」，必须真跑 |
| 3.8 | **命名裁定落实**：SKILL.md 图集清单照实采用代码产出的文件名（不改代码）；所有 φ/θ 直方图面板 x 轴标签写明 `2 pi k / 3, k = 0,1,2 = 0/120/240 deg`；manifest 新增机器可核对的 `reference_lines_deg = [0,120,240]` 与 `reference_lines_note`，`--check` 与 self-test 各断言一次 | 参考线是纯数学阶梯，必须自描述且可核对，不能靠读图猜 |
| 3.9 | 端到端冒烟改为**两个数据集各跑一轮完整分析**（A：512 px + 2 % 各向异性；B：384 px），各出一个 50 张图集、各调用一次 `atlas.py --check`，摘要打印每族/单集/合计张数 | 单数据集一轮不足以证明图集契约可重复 |

## 4 self-test

| # | 改动 | 理由 |
| --- | --- | --- |
| 4.1 | 一键 `selftest.py`（**当时 49 项**；最终版 **69 项**，约 3 min；`--quick` 约 25 s） | 无自检的 skill 无法随着代码演进保持正确 |
| 4.2 | 覆盖：圆统计量（含圆中位数与暴力搜索比对）、Fourier 恒等式（解调约定/Friedel/**掩膜代数**/规范变换律/三重积）、**gauge 漂移律与不变量**（精确律残差 < 1e-9°，加窗画布实测漂移率与残差）、注入相位回收、**多成分判别边界**（1/2/3 成分、等权/不等权，给出 5° 交点 f = 0.0319 与管线实测）、环定位四个分支、**图集清单断言**、**确定性**（两次运行逐像素相同、逐字节相同）、log-数字对齐、缺 r3 分支、矫正阶段（注入 2 % 各向异性 → 回收 M 与反演 a、正确/错误锚定判别） | 逐条对应验收要求；每条都打印验收阈值与实测值 |
| 4.3 | 禁词检查也写进 self-test（文件名/图标题），且该文件用片段拼装禁词，避免自己触发扫描 | 交付物里的自检脚本本身不应包含禁词字面量 |

## 5 吸收 t6（方法学验证）的最终结论（2026-09-18 追加）

| # | 改动 | 理由 |
| --- | --- | --- |
| 6.1 | SKILL.md 明确：逐峰相位 = **未加门**幅度加权圆均值 ≡ `arg X(q)`，实测复现窗加权真值到 0.04°；`p50` 门**只作分布形状展示**，并写入门改变成分比 2 % 即让 `3Φ̄` 偏 12° 的实测对照（128.73 / 141.11 / 128.77） | 原来只说了"未加门是首选"，没有把这组决定性数字写进文档 |
| 6.2 | SKILL.md 新增**稳健量清单表**（Friedel 对和、闭合和、每峰分布形状、幅度/相干度 = 可引用；单峰绝对相位、spread、折叠分布位置 = 参考量）与引用条件（写明 r0 分支 + 系统带 `σ_peak/3 ≈ RMS_res/2.12`，含真实数据示例 10.39°→4.9°、23.75°→11.2°） | 交付物必须能告诉使用者"哪些数字可以跨讨论引用" |
| 6.3 | **更正**原稿"ring_r3 绝对相位只在 mod 120° 意义下可比"：改为「解集 = {最佳 r0} ⊕ 参考环**相位点阵**（`q·L ∈ Nℤ`），点阵平移残差逐位不变；诱导位移 `(2π/N)q·L` **是否**是 120° 的整数倍由脚本对每组数据**实测输出**（理想六方几何下恰为 ±120°，真实数据分析中不同解代表点实测可差 ±180°/±171.7°），故不得假定 mod 120° 可比」 | 旧稿的 mod 120° 说法来自 `_methods` 早期版本，t6 已明确更正 |
| 6.4 | `phase_stats.json` 的 `gauge` 段新增 `sigma_peak_deg`、`single_peak_systematic_band_deg`、`solution_set`、`lattice_vectors`、`lattice_branches`；每个分支新增 `delta_r0_px`、`delta_r0_lattice_coefficients`、`lattice_translation_of_the_best_fit`、`induced_ring_r3_phase_shifts_deg`、`induced_shifts_are_multiples_of_120deg` | 让"r0 报告必附四项"成为脚本的强制输出，而不是文档里的口头要求 |
| 6.5 | 新增**峰位不确定度**输出：每族 `q_position`（六边形残差 rms/max、径向 rms、平均半径）与 `theta_ramp_span_deg_if_unprojected = 360·\|Σq\|`，并在 log/JSON 写明「混合相位使检出峰位偏 0.019 px（单区）→ 0.911 px（两区 ΔΦ=120°）；投影去掉净斜坡但不清除逐峰偏置」 | 原稿只给 `\|Σq\|`，没说它意味着什么、量级多大 |
| 6.6 | log 的 reading guide 增加三条：稳健量 vs 参考量、镜像二义（353.26↔6.74、345.23↔14.77，到阶梯距离不变）、r0 分支依赖 | 每个数字在被引用前都能在 log 里看到它的适用条件 |
| 6.7 | self-test 新增第 5 组 `robust quantities vs single-peak absolute phases`（9 项）：相位点阵闭合、点阵平移残差逐位不变、诱导位移与公式一致、**阈值分档**（稳健量 0.5°/0.01/0.5°/1°；单峰绝对相位只判"落在 σ_peak/3 带内"）、镜像对 | t6 要求把验收阈值**分开定**并写进 SKILL.md；现在两边一致 |

## 6 吸收 verifier 的估计量定义发现（2026-09-18，t2 前置侦察）

| # | 改动 | 理由 |
| --- | --- | --- |
| 6.1 | SKILL.md 把逐峰相位的**像素集合**钉死为「全画布全部有效像素（权重 `\|ψ(r)\|`）」，写明公式 `φ(r) = angle(ψ(r)) − (2π/N)q·(r−c)`，并明确**不是**实空间圆盘子集、**不是**加门样本 | 同一句"mask 内幅度加权圆均值"有两种不等价口径；本 skill 实现的是全画布（无偏）那一支，必须写明 |
| 6.2 | 自测新增单 bin 平面波的定义钉死检查（`= arg X(q) + (2π/N)q·c`，实测 4.8e-12°） | 让"定义"成为可执行断言，而不是文档措辞 |
| 6.3 | 自测新增**有界两区域反例**：全画布口径差 0.0015°，实空间圆盘子集口径差 106.250°（阈值 > 50° 仅记录）；检查文本打印完整几何并附 verifier 独立值 | 让"换个像素集合会差几十度"成为可复算的证据 |
| 6.4 | §5 的回收阈值限定为**单区域 / 全局调制**配置并给出几何参数 | 回收精度声明必须与定义、配置、几何绑定 |
| 6.5 | gauge 段写明**六峰**拟合口径与六峰 rms 的语义（模型一致性诊断；合成注入 Φ_ref = 25° → rms = 25.0000°），并新增 `gauge.triple_fit`（q-sum-zero 三峰，rms 恒 0、无诊断信息）与 `gauge.fit_convention` | verifier 独立发现六峰最小二乘对 ±Φ 图案是折中解；必须给出两个口径的数字与各自含义 |

| 6.6 | 自测补**等窗权重**两区域用例（左右各半、ΔΦ = 120°、窗权重 0.5026/0.4974）：全画布口径差 **0.0002°**，同配置圆盘子集口径 +1.437°（偏差随几何变，与 6.3 的 +106.25° 并列记录）；r₀ 行内与 JSON 增「六峰 rms 是模型一致性诊断、不是相位噪声」的说明（`gauge.rms_deg_note`） | captain/verifier 的补充点 1、2：等权用例要单独有，"六峰 rms 别被下游误读为相位噪声"要在 r₀ 报告里就地写明 |

## 7 t3 修复（2026-09-18，来自 t2 的 F4/F5/F6）

| # | finding | 改动 |
| --- | --- | --- |
| 7.1 | **F4** `selftest.py` 的规范律断言是恒真式 | 旧实现比较 `angle(psi) − (angle(psi) − offset)` 与 `wrap(offset)`，代数上恒等、永不失败。改为**在解调场上实测**：用同一个参考点 `c` 与 `c + k·δ` 分别解调，比较实测相位差与 `−(2π/N)q·δ·k`，实测 **7.1e-12°**；并新增**反面控制**：把预言换成偏移扰动 `δ+(0.5,0.5)` px 后同一比较给出 **127.765°**（阈值 > 30°），证明该断言**可失败**。顺带修掉两处残差留在弧度未转度的单位错误 |
| 7.2 | **F5** 端到端段先读 run b 的 JSON、后才可能因返回码失败 | 在读取 run b 产物**之前**先断言其返回码（新增一条检查），并在 JSON/图集比较前加 guard：run b 未产出时如实判 FAIL 并 print，而不是抛 traceback |
| 7.3 | **F6** `DELIVERY.md` 的检查计数与实际不一致 | 计数统一为**实测值 69**（`selftest_output.txt` 头部、`DELIVERY.md` ×2、`SKILL.md` §5）；并在 §5 清单里补记 F4 的反面控制与 F5 的返回码检查 |
| 7.4 | 交付文件布局 | `origin_space.py` 移回 `scripts/`（使 `FREEZE.txt` 的 12 条与 captain 指定一致），落盘清单在 `DELIVERY.md` 顶部明确标注"拷入仓库时排除 `scripts/origin_space.py`" |

## 8 明确的局限（写进 v2 文档）

1. 矫正的可用范围：环半径聚类容差决定各向异性 ≲ 3 %；超出时包返回
   `method="identity_fallback"`、`fallback=True`，**什么都没矫正**（脚本会 WARNING）。
2. 锚定自检需要矫正后至少有两族环（`unverifiable` 时不能确认锚定）。
3. Friedel 对和是恒等式，但管线按**检出的**波矢量配对，残余受 `|Σq|` 限制（实测
   1e-4–4e-3°，每对都输出 `|Σq|`）。
4. `r0` 只在参考环的直接晶格意义下唯一；绝对相位必须附约定与不确定度。
5. 门的空间选择效应、窗加权 vs 面积权重、FWHM 的平滑/分箱口径——引用时必须一并给出。
6. 合成自检用白噪声与理想平面波，不含真实扫描畸变与噪声关联；判别边界是针对本合成尺寸/窗测得的。
