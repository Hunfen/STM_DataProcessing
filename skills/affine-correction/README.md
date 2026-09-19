# affine-correction（STM 拓扑图几何矫正）

仓库内置 skill（**v2.2**）。**只做几何矫正数学**：数据锚定峰检测、亚像素定位、对称正定拉伸
矫正（纯拉伸、零旋转）、锚定自检。物理解释由使用者进行——脚本、图、log 与文档都不给物理结论，
也不用任何 k 空间/高对称点类称呼。矫正产物可接**任意后处理**（双环相位数学见
`skills/phase-analysis/`）。拟合出的变换可导出成独立 JSON，再应用到指定的另一张拓扑图（两段式）。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_topo_correct.py` | 几何矫正 CLI：读入 → 预处理 → 数据锚定检测 → 显式 `LatticeSpec` → 对称正定拉伸 → 重采样；**锚定环可显式指定**（`--anchor-ring 1x1|r3`）；`--save-transform FILE` 额外导出独立变换 JSON（不加开关时既有输出逐字节不变） |
| `scripts/stm_apply_transform.py` | **两段式第二段**：把变换 JSON 应用到另一张拓扑图（`INPUT.csv --transform T.json -L SIZE -o OUT`）；按目标图自己的像素数重算画布与 offset，只做重采样（不重新检测峰、不重跑自检） |
| `scripts/correction_lib.py` | 自包含辅助模块：环半径聚类 `group_rings`、绘图样式 `setup_style`、gwyddion 色图 `load_colormap`（含 builtin 回退）与 `BAD_COLOR`（**不 import phase-analysis skill**） |
| `scripts/selftest.py` | **一键自我验证**（**6 项**：stage 跑通、M 回收、隐含晶格常数、正确/错误锚定自检、两段式变换导出与套用） |
| `SKILL.md` | 完整方法学、锚定自检双 tell-tale、变换 JSON 契约、下游接口契约与限制 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（峰检测/亚像素定位/对称正定拉伸/重采样/FFT2）、
  `stm.preview_plot.gwyddion`（色标，缺省 builtin 锚点回退）、`utils.plot_funcs.subtractMeanPlane`
- 执行环境（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 矫正（示例：数据里最内环其实是 r3 环）
.venv/bin/python skills/affine-correction/scripts/stm_topo_correct.py \
    INPUT.csv -L 50 --anchor-ring r3 -o OUT --list-rings

# 一键 self-test（6 项）
.venv/bin/python skills/affine-correction/scripts/selftest.py --workdir tmp_verify/corr_selftest --keep
```

## 两段式用法（变换导出 + 套用到另一张图）

```bash
# 1) 在参考图上拟合，并导出独立变换 JSON
.venv/bin/python skills/affine-correction/scripts/stm_topo_correct.py \
    REF.csv -L 50 --anchor-ring r3 -o OUT --save-transform OUT/transform.json

# 2) 把同一变换套用到目标图（-L 给目标图自己的视场，可与参考图不同）
.venv/bin/python skills/affine-correction/scripts/stm_apply_transform.py \
    TARGET.csv --transform OUT/transform.json -L 30 -o OUT_TARGET
```

目标图可以换尺寸：`n_out`、offset 与矫正后视场（`L · n_out / n`）都按目标图自己的像素数重算。
apply 段不做峰检测与锚定自检——`apply_report.json` / log 里的 `anchor_verdict` 是**参考图**的结论。
变换 JSON 的字段契约见 `SKILL.md` §3。

## 输出

`<stem>_corrected.csv` / `<stem>_corrected_fft2.npy` / `<stem>_corrected.png` /
`<stem>_corrected_fft.png` / `correction.log` / `correction_report.json`
（apply 段同名，但报告为 `apply_report.json`）。
下游接口契约（`corrected canvas ... field of view X nm` 行、`affine_q` 等）见 `SKILL.md` §3。

## 修改记录

见 `CHANGES.md`。
