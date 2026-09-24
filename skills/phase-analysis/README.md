# phase-analysis（STM 拓扑图：双环相位数学）

仓库内置 skill（**v4**，原 `stm-topo-phase-analysis`）。**只做相位数学**：局域解调场、圆统计量、
规范固定、成对相位差、图集与一键自检。物理解释由使用者进行——脚本、图标题、log 与文档都不给物理结论，
也不使用任何 k 空间/高对称点类称呼；两个环只有 `ring_1x1`（参考环）与 `ring_r3`（半径为其 1/√3）。
**几何矫正见独立 skill `skills/affine-correction/`**，其产物（`<stem>_corrected.csv` + `correction.log`）
是本 skill 的典型输入。

**v4 五件事**：

1. **取样只有一条规则**：矫正画布上**全部有效（非 NaN）像素**，权重 `|ψ(r)|`。阈值/百分位/门限概念从
   CLI、代码、JSON 与图里一并删除——逐峰统计只有一套字段，`theta_map` 图上的掩码面板删除，图的可见像素
   与该图统计所用样本是同一个集合；已废除的 `--gate` 现在**显式报错拒绝**（非零退出 + 明确信息，
   与更早废除的 `--pct` 同等待遇）。
2. **相位轴不再重排**：`RING_p{i}_theta_dist.png` 变成单面板，只画 `θ ∈ [0, 2π)` 的未重排分布；
   重排后的分布统计量与相应 JSON 字段删除。
3. **相位参考阶梯删除**：所有相位直方图上的三条等距虚线、其轴标签说明、manifest 的参考线字段与断言，
   以及由参考线派生的全部标量字段（到阶梯的距离、模周期值、三倍角值）一并删除。
4. **图集重排 81 → 60**：每环 **30** 张 = 逐峰 3 类 × 6（amplitude / theta_map / theta_dist）+
   环汇总 3（theta_hist_summary / theta_map_summary / ring_members_qspace）+
   环内 3 对 × 3 类（phase_diff / **phase_diff_dist** / amp_diff）。**跨环图全部删除**，
   三重积图与成对二维直方图删除；新增的 **D 分布图**是 `D = wrap(arg ψ_j − arg ψ_k)` 的
   幅度加权圆直方图（权重 `|ψ_j ψ_k|`，x 轴 `D ∈ (−π, π]`）。
5. **色条与色标契约**：所有 map/密度图都有色条，且色条画在数据区**之外**（独立色条轴，与任何数据轴的
   包围盒交集面积为 0；写入前逐图断言，`atlas.py --check` 再核对一次，self-test 另有"把色条挪进数据区
   必须变红"的反面检查）；色标全局统一：所有 theta map `[0, 2π]`、12 张 amplitude map 共用同一套
   SymLogNorm（同一 `vmax`，`linthresh` = 12 个场正幅度的**合并中位数**）、所有 D map `±π`、
   所有 a map `±1`。
6. **标题与掩码（v4.1/v4.2 修复）**：每张图都把 manifest 的 `title` 画在画布上（figure 级、自动折行），
   写入前逐图核对"画布上真实渲染的标题 == `title_for(label, annotation)`"，写完再量一次数据轴之上的
   标题带墨迹（空带即报错）；`atlas.py --check` 按 manifest 记录的 `title_strip_px` 在 PNG 上重新
   数一遍。**单数据轴图只有这一行标题**（数据轴不再画自己的短标题），多面板网格额外保留每个面板的
   标识标题（`p0..p5` 及其数值）——写入前会数各轴标题，违反即报错。theta map（含六联汇总）的被掩
   像素画成与 amplitude 相同的灰 `#b0b0b0`，不再是透明空洞。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_phase_analysis.py` | 双环相位分析：两族**同口径**逐反射统计（单一样本）、gauge fix、Friedel 恒等式自检、三独立相位和与逐像素三重积 θ（只报数字、不画图）、**成对相位差（D(r) / D 分布 / a(r)）**、图集（60 张）与 JSON/CSV/log |
| `scripts/phasepipe.py` | 管线层（反射检测/环聚类/1/√3 配对/Gaussian 窗解调 `gaussian_field`/单一样本规则 `sample_mask`/逐对上场与统计/合成地面真值/重采样） |
| `scripts/phasemath.py` | 纯数学：加权圆统计（圆均值/中位数/分位数/FWHM/簇/R/幅度）+ 晶格参考规范固定 |
| `scripts/atlas.py` | 图集：绘图、全局色标、色条布局断言、manifest（`group`/`kind`/`peak`/`pair`/`mappable`/`colorbars`/`norm`）、PNG 内嵌注解（`tEXt: stm-atlas`）与独立核对器 `--check` |
| `scripts/selftest.py` | **一键自我验证**（本机实测 **124/124** 全绿）：引擎恒等式、估计量、取样规则、gauge 层、成对注入回收、Friedel 对、环定位分支、60 张图集与色标/色条契约、反面检查（声明 59 张时核对器必须变红等）、确定性、`--size-nm-from-log` |
| `SKILL.md` | 完整方法学、图集清单（每族 30 张 = 每峰 3 张 ×6 + 汇总 3 张 + 环内成对 3 张 ×3）、口径与限制 |
| `CHANGES.md` | v4 改动逐条记录（顶部）与更早版本的摘要 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（峰检测/亚像素定位/FFT2）、
  `stm.preview_plot.gwyddion`（色标）
- **同仓库 sibling skill `skills/local-q-map/scripts/localqmap.py`**：逐峰解调场由它给出
  （`phasepipe.gaussian_field` 薄封装调用 `localqmap.demodulate`；本 skill 不含自己的场提取实现）。
  引擎目录按两处查找：本 skill 目录的同级（仓库 checkout 与 `~/.agents/skills` 都是这个布局），
  以及 `~/.agents/skills`（脚本从别处的暂存目录运行时用）
- 输入通常来自 `skills/affine-correction/` 的矫正产物（见下）；也可直接分析任意方形 CSV
  （配 `-L` 或 `--size-nm-from-log`；`--detector builtin` 可不用包）
- **执行环境**（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 0) 矫正（affine-correction skill；示例：数据里最内环其实是 r3 环）
.venv/bin/python skills/affine-correction/scripts/stm_topo_correct.py \
    INPUT.csv -L 50 --anchor-ring r3 -o OUT --list-rings

# 1) 双环相位分析 + 成对相位差 + 图集（60 张）
.venv/bin/python skills/phase-analysis/scripts/stm_phase_analysis.py \
    OUT/INPUT_corrected.csv -o OUT/phase --size-nm-from-log OUT/correction.log --lambda-nm 3.0

# 2) 图-数字契约复核（一条命令：张数 + 数字 + 色标 + 色条）
.venv/bin/python skills/phase-analysis/scripts/atlas.py \
    --check OUT/phase/atlas_manifest.json --stats OUT/phase/phase_stats.json \
    --expected-figures 60 --expected-per-ring 30

# 3) 一键自检（引擎恒等式 + 取样规则 + gauge 层 + 成对回收 + 端到端 60 张图）
.venv/bin/python skills/phase-analysis/scripts/selftest.py \
    --stm-lib /path/to/STM_DataProcessing/src
```

不在仓库根目录运行时用 `--stm-lib /path/to/STM_DataProcessing/src`。

## 关键点（详见 `SKILL.md`）

1. **输入来源**：矫正产物 `OUT/INPUT_corrected.csv` + `OUT/correction.log`（`--size-nm-from-log`
   优先解析 `corrected canvas ... field of view X nm` 行取**矫正后画布**视场）。
2. **两族同口径**：`ring_r3` 由 `ring_1x1` 半径的 1/√3 定位（容差可配）；定位不到就如实报
   "r3 ring not found"（退出码 2，不画图，不猜）。
3. **唯一的解调引擎**：`ψ_q(r) = FFT⁻¹{ FFT[T(r)e^(−i q·r)]·e^(−λ²|k|²/2) }`（`local-q-map`，
   `--lambda-nm` 默认 3.0 nm），`θ = arg ψ ≈ +φ`，**无斜坡、无每峰常数**；相位值 = 全画布
   幅度加权圆均值 = `arg Σ_r T(r) e^(−i q·r)`（窗的 k=0 权重恒为 1 ⇒ 与 λ 无关）。
4. **唯一的样本规则**：非 NaN 的有效像素、权重 `|ψ(r)|`（逐峰）与 `|ψ_j ψ_k|`（成对）；
   中位数/IQR/FWHM/簇数与圆均值在**同一个样本**上计算，引用这些数不需要附带任何条件。
5. **成对分析**：每环组内 (p0,p1)/(p2,p3)/(p4,p5)（避开 Friedel 平凡对），每对给
   `D = wrap(arg ψ_j − arg ψ_k)`、`a = (|ψ_j|−|ψ_k|)/(|ψ_j|+|ψ_k|)` 与 **D 的圆直方图**
   （x = `D ∈ (−π, π]`，权重 `|ψ_j ψ_k|`）；`phase_stats.json` 的 `pairwise` 段含逐对统计。
6. **图集契约**：每环 **30** 张、合计 **60** 张（无任何跨环图）；每张图的标题、manifest
   （含 `panels`/`group`/`kind`/`pair`/`mappable`/`colorbars`/`norm`）与 PNG `tEXt` 用**同一份注解
   字典**，成对图的注解数字指向 `phase_stats.json` 的 `pairwise` 路径，可用 `atlas.py --check`
   一条命令第三方复核（self-test 以命令行方式真跑这一步，并另跑"声明 59 张必须变红"）。
7. **gauge 与漂移**：单峰绝对相位随图像原点按 `θ → θ − (2π/N)q·δ` 漂移；**周期画布**上 self-test
   实测该律与"原点随图像走时 gauge 后相位不变"都在 1e-9° 内，非周期画布上如实报告漂移率与残差
   （该残差来自周期性 FFT 边界带，样本是整张有效画布、无切趾）；θ、R、FWHM、簇数不变。
