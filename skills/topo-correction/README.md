# topo-correction（STM 拓扑图几何矫正）

仓库内置 skill（**v2.1 拆分版**）。**只做几何矫正数学**：数据锚定峰检测、亚像素定位、对称正定拉伸
矫正（纯拉伸、零旋转）、锚定自检。物理解释由使用者进行——脚本、图、log 与文档都不给物理结论，
也不用任何 k 空间/高对称点类称呼。矫正产物可接**任意后处理**（双环相位数学见
`skills/phase-analysis/`）。

## 文件

| 路径 | 作用 |
| --- | --- |
| `scripts/stm_topo_correct.py` | 几何矫正 CLI：读入 → 预处理 → 数据锚定检测 → 显式 `LatticeSpec` → 对称正定拉伸 → 重采样；**锚定环可显式指定**（`--anchor-ring 1x1|r3`） |
| `scripts/correction_lib.py` | 自包含辅助模块：环半径聚类 `group_rings`、绘图样式 `setup_style`、gwyddion 色图 `load_colormap`（含 builtin 回退）与 `BAD_COLOR`（**不 import phase-analysis skill**） |
| `scripts/selftest.py` | **一键自我验证**（**5 项**：stage 跑通、M 回收、隐含晶格常数、正确/错误锚定自检） |
| `SKILL.md` | 完整方法学、锚定自检双 tell-tale、下游接口契约与限制 |

## 依赖

- 仓库 `STM_DataProcessing` 源码（`src/`）：`utils.bragg_peak`（峰检测/亚像素定位/对称正定拉伸/重采样/FFT2）、
  `stm.preview_plot.gwyddion`（色标，缺省 builtin 锚点回退）、`utils.plot_funcs.subtractMeanPlane`
- 执行环境（系统 Python 无 numpy；不要用 `uv run`——它会写仓库 `.venv`）：

```bash
cd /path/to/STM_DataProcessing
export MPLCONFIGDIR=<可写目录> PYTHONDONTWRITEBYTECODE=1

# 矫正（示例：数据里最内环其实是 r3 环）
.venv/bin/python skills/topo-correction/scripts/stm_topo_correct.py \
    INPUT.csv -L 50 --anchor-ring r3 -o OUT --list-rings

# 一键 self-test（5 项）
.venv/bin/python skills/topo-correction/scripts/selftest.py --workdir tmp_verify/corr_selftest --keep
```

## 输出

`<stem>_corrected.csv` / `<stem>_corrected_fft2.npy` / `<stem>_corrected.png` /
`<stem>_corrected_fft.png` / `correction.log` / `correction_report.json`。
下游接口契约（`corrected canvas ... field of view X nm` 行、`affine_q` 等）见 `SKILL.md` §3。

## 修改记录

见 `CHANGES.md`。
