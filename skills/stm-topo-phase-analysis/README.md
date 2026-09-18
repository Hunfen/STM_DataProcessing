# stm-topo-phase-analysis（STM 拓扑图：几何矫正 + 双环相位数学）

仓库内置 skill（**v2 staging 版**）。**只做几何与相位数学**：峰检测、亚像素定位、对称正定拉伸矫正、
圆统计量、规范固定、图集与一键自检。物理解释由使用者进行——脚本、图标题、log 与文档都不给物理结论，
也不使用任何 k 空间/高对称点类称呼；两个环只有 `ring_1x1`（参考环）与 `ring_r3`（半径为其 1/√3）两个名字。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_topo_correct.py` | 几何矫正：包内数据锚定检测 → 显式 `LatticeSpec` → 对称正定拉伸 → 重采样；**锚定环可显式指定**（`--anchor-ring 1x1|r3`） |
| `scripts/stm_phase_analysis.py` | 双环相位分析：1x1 与 r3 **同口径**逐反射统计、gauge fix、Friedel 恒等式自检、三重积 θ、120° 折叠分布、图集与 JSON/CSV/log |
| `scripts/phasepipe.py` | 管线层（反射检测/环聚类/1/√3 配对/单反射场/统计/合成地面真值/重采样） |
| `scripts/phasemath.py` | 纯数学：加权圆统计（圆均值/中位数/分位数/FWHM/簇/R/幅度）+ 晶格参考规范固定 |
| `scripts/atlas.py` | 图集：绘图、manifest、PNG 内嵌注解（`tEXt: stm-atlas`）与独立核对器 `--check` |
| `scripts/selftest.py` | **一键自我验证**（**69 项**：恒等式、估计量、判别边界、环定位分支、图集契约、确定性、矫正阶段） |
| `SKILL.md` | 完整方法学、图集清单（每族 25 张 = 每峰 3 张 ×6 + 汇总 7 张）、口径与限制 |
| `CHANGES.md` | 相对仓库现有版本（v1）的逐条改动与理由 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（峰检测/亚像素定位/对称正定拉伸/重采样/FFT2）、
  `stm.preview_plot.gwyddion`（色标）、`utils.plot_funcs.subtractMeanPlane`
- **执行环境**（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 1) 矫正（示例：数据里最内环其实是 r3 环）
.venv/bin/python skills/stm-topo-phase-analysis/scripts/stm_topo_correct.py \
    INPUT.csv -L 50 --anchor-ring r3 -o OUT --list-rings

# 2) 双环相位分析 + 图集
.venv/bin/python skills/stm-topo-phase-analysis/scripts/stm_phase_analysis.py \
    OUT/INPUT_corrected.csv -o OUT/phase --size-nm-from-log OUT/correction.log

# 3) 图-数字契约复核（一条命令）
.venv/bin/python skills/stm-topo-phase-analysis/scripts/atlas.py \
    --check OUT/phase/atlas_manifest.json --stats OUT/phase/phase_stats.json \
    --expected-figures 50 --expected-per-ring 25

# 4) 一键自检（数学恒等式 + 判别边界 + 端到端 + 矫正阶段）
.venv/bin/python skills/stm-topo-phase-analysis/scripts/selftest.py
```

不在仓库根目录运行时用 `--stm-lib /path/to/STM_DataProcessing/src`。
相位分析也可以用 `--detector builtin` 走 skill 自带的反射检测（不需要包，用于可移植/自检），
但矫正必须用包。

## 关键点（详见 `SKILL.md`）

1. **锚定环显式**：包内数据锚定检测认定的"第一个环"未必是 1x1 环。当 r3 环在内侧（半径比 1/√3）时，
   按 1x1 给参考晶格会把 r3 环拉到 1x1 半径上、破坏几何。`--anchor-ring r3` 把 `a_ref` 取为 `√3·a`。
   报告给出 `method/fallback`、`|b1|` 前后、`n_out`、视场、NaN 比、反演晶格常数，以及**非循环**的
   `anchor verdict`（矫正后两强环是否构成 1 : √3 对）。
2. **两族同口径**：`ring_r3` 由 `ring_1x1` 半径的 1/√3 定位（容差可配）；定位不到就如实报
   "r3 ring not found"（退出码 2，不画图，不猜）。
3. **唯一的相位估计量**：相位值 = mask 内**未加门**的幅度加权圆均值（= 该反射全局相位）；
   中位数/FWHM/簇/簇宽是**加门**样本的形状描述量——引用时门必须一起给出。
4. **图集契约**：每族 25 张（每峰 3 张 ×6 + 汇总 7 张）、两族 50 张；逐峰 φ(r) 相位图是
   `RING_p{i}_phi_dist.png` 的**左面板**；每张图的标题、manifest（含 `panels`）与 PNG `tEXt` 用**同一份
   注解字典**，注解数字指向 `phase_stats.json` 的路径，可用 `atlas.py --check` 一条命令第三方复核
   （self-test 与 `e2e_scratch/run_smoke.sh` 都以命令行方式真跑这一步）。
5. **gauge 与漂移**：单峰绝对相位随图像原点按 `φ → φ − (2π/N)q·δ` 漂移（精确律；加窗画布上
   残差 0.003°，实测漂移率可达 20 °/px）；gauge 后相位（mod 120°，变化 0.0037°）、θ（0.0000°）、
   R、FWHM、簇数不变——self-test 会把这些漂移率与残差逐条打印出来。
