# phase-analysis（STM 拓扑图：双环相位数学）

仓库内置 skill（**v3**，原 `stm-topo-phase-analysis`）。**只做相位数学**：局域解调场、圆统计量、
规范固定、成对相位差、图集与一键自检。物理解释由使用者进行——脚本、图标题、log 与文档都不给物理结论，
也不使用任何 k 空间/高对称点类称呼；两个环只有 `ring_1x1`（参考环）与 `ring_r3`（半径为其 1/√3）。
**几何矫正见独立 skill `skills/affine-correction/`**，其产物（`<stem>_corrected.csv` + `correction.log`）
是本 skill 的典型输入。

**v3 三件事**：① 解调引擎换成 **sibling skill `local-q-map` 的 Gaussian 窗**（`--lambda-nm`，默认
3.0 nm），相位口径统一为 **`θ = arg ψ ≈ +φ`，无斜坡、无每峰常数**，硬圆 mask 引擎与 `--pct` 全部废除；
② 新增**论文式成对相位差分析**（组内 3 对/环 + 跨环 6 对，每对 D 场、归一化幅度差场、2D 直方图）；
③ 图集契约改为 **31/环×2 + 跨环 19 = 81 张**，manifest 增 `group`/`kind`/`pair`。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_phase_analysis.py` | 双环相位分析：1x1 与 r3 **同口径**逐反射统计、gauge fix、Friedel 恒等式自检、三重积 θ、120° 折叠分布、**成对相位差（D/a/2D 直方图）**、图集（81 张）与 JSON/CSV/log |
| `scripts/phasepipe.py` | 管线层（反射检测/环聚类/1/√3 配对/**Gaussian 窗解调（`gaussian_field`）**/逐对上场与直方图/统计/合成地面真值/重采样） |
| `scripts/phasemath.py` | 纯数学：加权圆统计（圆均值/中位数/分位数/FWHM/簇/R/幅度）+ 晶格参考规范固定 |
| `scripts/atlas.py` | 图集：绘图、manifest（`group`/`kind`/`peak`/`pair`）、PNG 内嵌注解（`tEXt: stm-atlas`）与独立核对器 `--check` |
| `scripts/selftest.py` | **一键自我验证**（**80 项**：引擎恒等式、估计量、gauge 层、成对注入回收、Friedel 对、环定位分支、图集 81 张契约、确定性、`--size-nm-from-log` 解析） |
| `SKILL.md` | 完整方法学、图集清单（每族 31 张 = 每峰 3 张 ×6 + 汇总 4 张 + 组内成对 3 张 ×3）、口径与限制 |
| `CHANGES.md` | 相对 v2 的逐条改动与理由（v3 段在文件顶部） |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（峰检测/亚像素定位/FFT2）、
  `stm.preview_plot.gwyddion`（色标）
- **同仓库 sibling skill `skills/local-q-map/scripts/localqmap.py`**：逐峰解调场由它给出
  （`phasepipe.gaussian_field` 薄封装调用 `localqmap.demodulate`；本 skill 不含自己的场提取实现）
- 输入通常来自 `skills/affine-correction/` 的矫正产物（见下）；也可直接分析任意方形 CSV
  （配 `-L` 或 `--size-nm-from-log`；`--detector builtin` 可不用包）
- **执行环境**（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 0) 矫正（affine-correction skill；示例：数据里最内环其实是 r3 环）
.venv/bin/python skills/affine-correction/scripts/stm_topo_correct.py \
    INPUT.csv -L 50 --anchor-ring r3 -o OUT --list-rings

# 1) 双环相位分析 + 成对相位差 + 图集（81 张）
.venv/bin/python skills/phase-analysis/scripts/stm_phase_analysis.py \
    OUT/INPUT_corrected.csv -o OUT/phase --size-nm-from-log OUT/correction.log --lambda-nm 3.0

# 2) 图-数字契约复核（一条命令）
.venv/bin/python skills/phase-analysis/scripts/atlas.py \
    --check OUT/phase/atlas_manifest.json --stats OUT/phase/phase_stats.json \
    --expected-figures 81 --expected-per-ring 31 --expected-cross 19

# 3) 一键自检（引擎恒等式 + gauge 层 + 成对回收 + 端到端）
.venv/bin/python skills/phase-analysis/scripts/selftest.py \
    --stm-lib /path/to/STM_DataProcessing/src
```

不在仓库根目录运行时用 `--stm-lib /path/to/STM_DataProcessing/src`。

## 关键点（详见 `SKILL.md`）

1. **输入来源**：矫正产物 `OUT/INPUT_corrected.csv` + `OUT/correction.log`（`--size-nm-from-log` 优先解析
   `corrected canvas ... field of view X nm` 行取**矫正后画布**视场）。
2. **两族同口径**：`ring_r3` 由 `ring_1x1` 半径的 1/√3 定位（容差可配）；定位不到就如实报
   "r3 ring not found"（退出码 2，不画图，不猜）。
3. **唯一的解调引擎**：`ψ_q(r) = FFT⁻¹{ FFT[T(r)e^(−i q·r)]·e^(−λ²|k|²/2) }`（`local-q-map`，
   `--lambda-nm` 默认 3.0 nm），`θ = arg ψ ≈ +φ`，**无斜坡、无每峰常数**；相位值 = 全画布
   幅度加权圆均值 = `arg Σ_r T(r) e^(−i q·r)`（窗的 k=0 权重恒为 1 ⇒ 与 λ 无关）；
   中位数/FWHM/簇是**加门**样本的形状描述量——引用时门必须一起给出。
4. **成对分析**：组内 (p0,p1)/(p2,p3)/(p4,p5)（避开 Friedel 平凡对）与跨环 6 对，
   每对给 `D = wrap(arg ψ_j − arg ψ_k)`、`a = (|ψ_j|−|ψ_k|)/(|ψ_j|+|ψ_k|)` 与 2D 直方图
   （x = |D| mod π ∈ [0,π]、y = a ∈ [−1,1]）；`phase_stats.json` 的 `pairwise` 段含逐对统计与 `hist_counts`。
5. **图集契约**：每族 31 张（每峰 3 张 ×6 + 汇总 4 张 + 组内成对 3 张 ×3）+ 跨环 19 张 = **81 张**；
   每张图的标题、manifest（含 `panels`/`group`/`kind`/`pair`）与 PNG `tEXt` 用**同一份注解字典**，
   成对图的注解数字指向 `phase_stats.json` 的 `pairwise` 路径，可用 `atlas.py --check` 一条命令
   第三方复核（self-test 以命令行方式真跑这一步）。
6. **gauge 与漂移**：单峰绝对相位随图像原点按 `θ → θ − (2π/N)q·δ` 漂移；**周期画布**上 self-test
   实测该律与"原点随图像走时 gauge 后相位不变"都在 1e-9° 内，非周期画布上如实报告漂移率与残差；
   θ、R、FWHM、簇数不变。v2→v3 的 `−π(q_x+q_y)` 常数被 `r0 → r0 + (N/2,N/2)` 吸收，gauge 后不变。
