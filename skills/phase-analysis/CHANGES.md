# phase-analysis 变更记录

本文件逐条记录本 skill 的改动。**v4 段在文件顶部**；更早版本的内容改写成"当时交付了什么"的摘要，
不再复述已经删除的概念（那些概念在 v4 里既不在代码里、也不在 JSON 里、也不在图上）。

---

## v4（本版）

v4 只做一件事的两半：**把"相位怎么取样"和"相位怎么画"收缩到一条规则**，并把由旧规则派生的
图与字段一并撤掉。**没有引入任何新的物理假设**：解调引擎（`local-q-map` 的 Gaussian 窗）、
相位约定（`θ = arg ψ ≈ +φ`，无斜坡、无每峰常数）、规范固定、Friedel 恒等式自检、
环定位与"未检出 r3"分支、系统带报告全部与 v3 相同。

### 1 取样：一条规则取代旧的选择规则

| 项 | v4 |
| --- | --- |
| 逐峰样本 | 矫正画布的**全部有效（非 NaN）像素**，权重 `\|ψ(r)\|`；唯一被去掉的是权重非有限或恰为 0 的像素 |
| 成对样本 | 两个场有效掩码的**交集**，权重 `\|ψ_j ψ_k\|` |
| 图的可见像素 | 与该图统计所用样本**是同一个集合**（同一份掩码数组） |
| 逐峰统计字段 | 一套：`phase_mean_deg` / `phase_median_deg` / `phase_R` / `phase_circ_std_deg` / `phase_fwhm_deg` / `phase_fwhm_deconv_deg` / `phase_iqr_deg` / `phase_median_span_deg` / `phase_kappa` / `n_clusters` / `cluster_centres_deg` / `cluster_weight_fractions` / `cluster_widths_deg` / `n_samples` / `n_zero_weight_px` |

删除：

* CLI 的 `--gate` 选项——现在**显式报错拒绝**（非零退出并说明取样规则），与更早废除的 `--pct`
  同等待遇（`stm_phase_analysis.REMOVED_OPTIONS` 是唯一的"已废除选项"名单）；
* `phasepipe` 里按阈值生成掩码的函数，以及 `reflection_stats` 的阈值形参；
* 逐峰统计的"阈值样本 / 全体样本"两套字段、阈值样本占全画布的比例字段、只由阈值定义的幅度统计
  （阈值下的幅度中位数按构造就是阈值本身）；
* `RING_p{i}_theta_map.png` 的第二个面板（掩码图），该图改为单面板 `θ(r)`；
* JSON 的 `gate` 字段与 `conventions.phase_shape` 里"分布是阈值样本"的措辞。

### 2 相位轴不再重排

`RING_p{i}_theta_dist.png` 由双面板改为**单面板**，只画 `θ ∈ [0, 2π)` 的幅度加权分布；
直方图 x 轴刻度为 `0 / π/2 / π / 3π/2 / 2π`。

删除：`stm_phase_analysis` 里把相位重排到更窄区间的函数、`phasepipe` 里重排用的常数、
逐峰统计的重排字段、与成对二维直方图相关的 `--pair-bins-*` 选项（成对分布改为 D 的一维圆直方图）。

### 3 相位参考阶梯删除（图与字段一起）

删除：所有相位直方图面板上的三条等距虚线及其轴标签说明、`atlas.py` 里参考线相关的常量与
`_hist_panel` 的参考线绘制、`check_manifest` 对 manifest 参考线字段的断言、
manifest 的参考线字段与说明、`phasemath` 里到阶梯距离的辅助函数、
`phasepipe` 里同一个量的副本，以及由参考线派生的全部标量字段：

* 逐峰：模周期值、到阶梯距离、三倍角值一类；
* 三独立相位与三重积：和与逐像素 θ 的同一批派生量（模周期值、到阶梯距离、三倍角值，
  以及它们在镜像侧的两个副本）；
* gauge 分支表：诱导位移"是否为阶梯整数倍"的布尔字段与其日志分支。

**保留**（都是非门限、非重排、非阶梯的数字，只是不再画图、不再换算成阶梯距离）：
三独立相位和及其镜像和、逐像素三重积 θ 的均值/中位数/R/圆标准差/FWHM/簇数/镜像均值
（`rings_analysis.<环>.triple_product`），以及 `theta_hist` 密度（三重积 θ 的分布数组）。

### 4 图集重排 81 → 60

| 项 | v3 | v4 |
| --- | --- | --- |
| 每环 | 31 | **30** = 每峰 3 类 ×6 + 汇总 3 + 环内 3 对 ×3 类 |
| 跨环 | 19 | **0**（跨环成对图、跨环网格、`pairwise.groups.cross`、跨环配对函数全部删除） |
| 成对类别 | `phase_diff` / `amp_diff` / `2dhist` | **`phase_diff` / `phase_diff_dist` / `amp_diff`** |
| 合计 | 81 | **60** |
| 删除的图 | — | `RING_theta_field.png`（三重积）、`RING_pair_*_2dhist.png`、`cross_pair_*`（18 张）、`cross_pair_phase_diff_grid.png` |
| 新增的图 | — | `RING_pair_{j}_{k}_phase_diff_dist.png`：`D = wrap(arg ψ_j − arg ψ_k)` 的幅度加权圆直方图（权重 `\|ψ_j ψ_k\|`，x 轴 `D ∈ (−π, π]`），标题注解含 `mean_deg/median_deg/R/fwhm_deg/n_clusters/n_valid` |

manifest 的 `groups` 只剩 `ring_1x1` / `ring_r3`；`atlas.py --check` 支持
`--expected-figures 60 --expected-per-ring 30`，`--expected-cross` 删除。

### 5 色条与色标契约

* **色条**：所有 map / 密度图（本机实测 40 张）都带自己的色条轴，且色条轴与**任何**数据轴的包围盒
  交集面积为 0；`atlas.Atlas.add` 在写文件前逐图断言（地图无任何色条同样报错），
  manifest 逐图记录 `mappable` / `colorbars` / `colorbar_outside` / `colorbar_overlap`，
  `atlas.py --check` 再核一遍；此前缺色条的 `RING_ring_members_qspace.png` 已补上。
  六联 theta map 网格的色条改为手工布局（放在网格右侧留出的条带里），因为 matplotlib 的自动
  布局会与该列数据轴重叠——这一条正是被断言抓出来的。
* **色标全局统一**：所有 theta(r) map `vmin=0, vmax=2π`；12 张 amplitude map 共用同一套
  `SymLogNorm`（`vmin=0`、`vmax` = 12 个解调场有效 `|ψ|` 的全局最大值、`linthresh` = 全局最小正
  `|ψ|`、`linscale` 固定）；所有 `D(r)` map `±π`；所有 `a(r)` map `±1`。每张图的色标记录写入
  manifest，并与 `phase_stats.json` 的 `atlas.norms` 全局声明逐字段比对。

### 6 产物与文档

* `phase_stats.json`：新增 `sample`（取样规则）、`atlas.norms`（全局色标声明）、
  逐峰 `n_samples` / `n_zero_weight_px`；`pairwise.bins` 改为 D 的一维区间；
  `pairwise.definition.sample` 写明成对样本；删除全部阈值、重排与阶梯字段。
* `phase_relations.csv`：`deviation_deg` 只对 Friedel 对和有定义，其余行留空。
* `phase_ring_comparison.csv`：改为 `theta_mean_deg` / `theta_median_deg` / `theta_R` /
  `theta_fwhm_deg` / `theta_n_clusters`。
* self-test：新增取样规则的单元检查与反面检查（幅度整体缩放到 1e-9 不改样本）、
  CLI 拒绝已废除选项、色条布局断言 + "把色条挪进数据区必须变红"的反面检查、
  "声明 59 张时 `atlas.py --check` 必须变红"、色标全局一致性、删除层 token 扫描
  （并用注入 token 证明扫描不是空转）、R1–R11 验收覆盖表与检查计数一致性；
  端到端阶段真跑两次完整管线（每环 30 张、合计 60 张）并核对逐字节确定性。
* 文档：`SKILL.md` / `README.md` / 本文件与代码一致（图集清单、字段名、CLI 选项、self-test 覆盖表）。

### 7 v4.1 修复：标题 + amplitude 线性区口径 + theta map 掩码底色

t4 的像素级复核把 A4（标题）判为失败：`ring_1x1/r3_ring_members_qspace.png` 整张图**没有任何标题**
（顶部标题带纯白，而 manifest/tEXt 里写着 title），`ring_1x1/r3_theta_map_summary.png` 与
`ring_1x1/r3_theta_hist_summary.png` 只有 p0..p5 面板标题、**没有图级标题**。根因是只有绘制函数自己
`set_title`，两个网格与 q 空间图从没设过图级标题，而 manifest 的 `title` 是按 label 现算、不从画布读回，
所以 `atlas.py --check` 在结构上抓不到这类缺陷。v3 同名图也无标题，不是 v4 引入的回归。

* **标题补全 + 程序化门**：新增 `atlas.set_canvas_title()` / `rendered_title_texts()` /
  `require_canvas_title()`。每张图的绘制函数在 `tight_layout()` **之前**把
  `title_for(label, annotation)` 作为 figure 级标题画到画布上（`wrap=True` 自动折行，长注解不会撑宽画布，
  `Text.get_text()` 仍返回未折行的一行字符串）；`Atlas.add` 在保存前把画布上**真实渲染**的标题文本
  （suptitle / axes title / figure texts）与同一字符串**逐字比对**，缺标题或文本不一致即抛错；写完后按
  "数据轴之上那条带"的高度重新数一遍墨迹像素，空带同样抛错。manifest 增加 `canvas_title`、
  `title_strip_px`（带高，px）与 `title_band_ink`，`atlas.py --check` 按记录的高度在 PNG 上**重新计数**，
  并核对 `canvas_title == title`。注入故障探针 `_check/title_gate_probe.py` 实测：抽掉标题、改一个字符
  都让真实管线非零退出（`GATE FIRED`）。
* **amplitude 色标口径**：linear 区从"12 个场里最小的正幅度"改成"12 个场**全部正幅度的合并中位数**"
  （全局统计量；`vmin=0` 与 `vmax`=全局最大仍统一）。理由（本机 2117² 数据实测）：最小正值 4.53e-17
  比主体数据低约 4 个数量级，linear 区因此只有 ~1e-17 宽，整幅图被挤到色条顶端（t 中位 0.93–0.96），
  弱峰与强峰几乎同色；改成中位数 3.4076e-13（vmax 6.4102e-13）后强峰 t 中位 ≈0.72、最弱峰 ≈0.22，
  色条上相差 0.5，强弱一眼可辨，而同一条色标仍被 12 张图共用。
* **theta map 掩码底色**：`hsv` 色图先 `set_bad("#b0b0b0")`，被掩像素（NaN 填充与样本外像素）画成与
  amplitude map 相同的灰，读者能看到分析在哪里停止；逐峰 theta map 与六联汇总同样处理，色标本身
  （`[0, 2π]`、hsv）不变。实测灰像素数：逐峰 theta map 55 → 69,382；六联汇总 210 → 153,905。
* **未改（不在本次裁定范围内）**：成对 `D(r)`/`a(r)` 图仍用默认的透明 bad 颜色（t4 把它们判为通过）；
  若要统一底色，另开一条即可。

### 8 v4.2 收尾：docstring 口径 + 单轴图只留一行标题

* **`amplitude_norm` 的 docstring 与代码对齐**（t8 评审的 requiredFix）：原文把 `linthresh` 说成
  "整个 run 里最小的正 ``|psi|``"（旧口径），而 t6 已把代码改成"12 个场全部正幅度的合并中位数"。
  现写明 `vmin = 0`、`vmax` = 整个 run 有效 `|psi|` 的全局最大值、`linthresh` = **整个 run 全部正幅度
  像素的合并（pooled）中位数**，并说明每张 amplitude 图共用同一条 norm、因而是可比的；
  `stm_phase_analysis` 里解释为什么不用最小值的那段也一并改措辞（脚本里不再出现旧口径字样）。
* **单数据轴图去掉重复的短轴标题**（用户裁定"去掉短轴标题"）：逐峰 `amplitude`/`theta_map`/`theta_dist`、
  成对 `phase_diff`/`phase_diff_dist`/`amp_diff`、`ring_members_qspace` 现在只画**一行**图级注解标题
  （= manifest 的 `title`），数据轴自己不再画标题。**多面板网格不动**：`theta_map_summary` 与
  `theta_hist_summary` 每个面板继续画自己的标识标题（`p0..p5`，直方图网格还带该峰中位数）。
  配套：`_distribution_panel` 新增 `show_title`（网格 True、单轴 False）；两个 1-D 分布图与直方图网格
  也把数据轴交给 `Atlas.add` 记账。
* **这条设计也是门**：`Atlas.add` 写文件前用 `axes_title_texts()` 数各轴标题——单轴图带任何轴标题即
  抛错、多面板图的轴标题个数必须等于面板数；manifest 逐图记录 `panel_axes` 与 `axes_titles`，
  `atlas.py --check` 按同一规则复核。标题门（`canvas_title == title` + 数据轴之上的墨迹带）不变，
  self-test 104 → 118 → **124** 项（新增"56 张单轴图 `axes_titles == []`"、"4 张网格各 6 个面板标识"、
  "直方图网格面板标题带中位数"、"写入前给单轴图加轴标题必须报错"、"多面板图保留面板标识必须通过"）。
* manifest 的 `panels` 文案本来描述的是**面板内容**（例如 `amplitude |psi(r)| (global symlog scale)`），
  不是被删掉的轴标题，因此无需改动；单轴图的标识信息仍完整地在图级注解标题里（它以 `label` 开头）。

### v4 的"变与不变"

| 类 | 量 | 变化 |
| --- | --- | --- |
| 引擎 | `ψ_q(r)`、`θ = arg ψ ≈ +φ`、三条恒等式、平移律 | **不变**（仍是 `local-q-map` 的 Gaussian 窗） |
| 取样 | 相位值（幅度加权圆均值 = `arg Σ_r T(r)e^(−i q·r)`） | **不变**（v3 就已声明它是全体有效像素上的量） |
| 取样 | 中位数 / IQR / FWHM / 簇数 / 簇宽 | **从旧的阈值样本改为全体有效像素样本**（这就是 v4 的主改动，引用时不再需要附带条件） |
| 图集 | 每环张数与成对分析的图种 | 31 → 30；`2dhist` → `phase_diff_dist`；跨环全删 |
| 稳健量 | Friedel 对和、三独立相位和、逐像素三重积 θ、每峰分布形状、成对 D/a | **不变**（仍可跨规范引用） |
| 参考量 | 单峰绝对相位 | **不变**：仍随原点与 r0 分支变，引用时必须写明分支与系统带 |

---

## v3

* 解调引擎换成 sibling skill `local-q-map` 的 Gaussian 窗（`--lambda-nm`，默认 3.0 nm），相位口径
  统一为 `θ = arg ψ ≈ +φ`（无斜坡、无每峰常数）；v2 的硬圆掩码引擎及其选项、log 行、JSON 字段与
  相关图全部删除。
* 新增成对相位差分析：每环境内 3 对 + 跨环 6 对，每对一个 D 场、一个归一化幅度差场与一个
  二维直方图；`phase_stats.json` 新增 `pairwise` 段。
* 图集契约改为 31/环×2 + 跨环 19 = 81 张，manifest 增 `group` / `kind` / `pair`。
* 相对 v2，每峰原始相位平移 `−π(q_x + q_y)`；该常数恰是原点平移 `(N/2, N/2)`，被 gauge 拟合吸收，
  gauge 后相位逐位不变（self-test 实测 ≤ 1e-9°）。

## v2.1

* `--size-nm-from-log` 改读**矫正后画布**的视场：优先带 `corrected canvas` 标签的行，其次 log 里
  最后一处匹配，一处都没有则报错退出。

## v2

* 锚定环显式化；两族同口径双环分析；r3 环由参考半径的 1/√3 定位，并给出"未检出"分支
  （退出码 2、不画图、不猜）。
* 唯一的相位估计量定义与分布描述量分离；Friedel 对和、三独立相位和、逐像素三重积、集中度 R 全套；
  图集清单与"图-数字契约"（标题 / manifest / PNG 内嵌注解同源）；一键 self-test；命名中性化
  （不用任何 k 空间或高对称点称呼）。

## v1

* 初版：单幅拓扑图的六峰相位分析（topo0009 实测流程）。
* 入库版：峰检测与 FFT2 改为调用 `stm_data_processing.utils.bragg_peak` 包。

## 拆分

* 几何矫正拆分为独立 skill（`topo-correction`，后改名 `affine-correction`），本 skill 改名
  `phase-analysis` 并只负责相位数学与图集。
