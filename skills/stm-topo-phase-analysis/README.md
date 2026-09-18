# stm-topo-phase-analysis（STM 拓扑图矫正 + 相位分析 skill）

仓库内置版本。两个脚本的峰检测、亚像素定位、对称正定拉伸矩阵与重采样**全部调用**
`stm_data_processing.utils.bragg_peak` 包（`detect_bragg_peaks` / `correct_bragg_peaks` /
`compute_fft2`），本身不再实现任何峰检测或高斯拟合。

## 功能

1. `scripts/stm_topo_correct.py`：读入拓扑图矩阵（第一行 = 扫描起始行）→
   `flipud(subtractMeanPlane(...))` → 包内数据锚定峰检测 → 显式
   `LatticeSpec(a_nm=a, symmetry="hexagonal", orientation_deg=θ)`（a 默认 0.246 nm，θ 取数据
   自身的 (1,0) 方向；**不做点群自动推断**）→ 对称正定拉伸校正 → 输出
   `<stem>_corrected.csv`、`<stem>_corrected_fft2.npy`、`<stem>_corrected.png`、
   `<stem>_corrected_fft.png`
**已知限制**：矫正需要先有带标签的晶格，而标号容差（环半径 3 %、`match_labels` 2 %）决定其可用范围——各向异性约 **≤ 3 %**；超出时包返回 `method="identity_fallback"`、`fallback=True`、`n_labelled=0` 且**未做任何矫正**（返回数组只是重铺到更大的 NaN 画布上，偶数 n 还有半像素偏移，别当成已矫正图像；**不静默**：`meta` 与警告日志都有标记，脚本也打印 fallback 标记），**请先检查 `meta['method']` / 打印的 fallback 标记再使用矫正结果**。更强各向异性与畸变矩形/方晶格需要路线图 L1 的宽容差对应路径（`docs/design/bragg_peak_detection.md` 第 9 节）。

2. `scripts/stm_phase_analysis.py`：对矫正后数据做 nomask / r3all / 6 个 r3 峰分别的 Kekulé
   相位分析（q 由包内 17×17 高斯定位器给出，不做全局平面修正），输出直方图、相位图与 npy

详细流程、输出文件名清单与物理要点见 `SKILL.md`。

## 依赖

- 仓库 `STM_DataProcessing` 的源码（`src/`）：`bragg_peak` 包、`gwyddion` colormap、
  `subtractMeanPlane`
- 在仓库根目录用 uv 环境执行（系统 Python 无 numpy/matplotlib）：

```bash
cd /path/to/STM_DataProcessing
uv run python skills/stm-topo-phase-analysis/scripts/stm_topo_correct.py INPUT.txt -L 100 -o OUT_DIR
uv run python skills/stm-topo-phase-analysis/scripts/stm_phase_analysis.py OUT_DIR/INPUT_corrected.csv \
    -o OUT_DIR/phase --pct 5 --bins 1024
```

不在仓库根目录运行时，用 `--stm-lib /path/to/STM_DataProcessing/src` 指定库源码路径。

## 安装到本机 skills 目录（新 clone 后执行一次）

```bash
cd /path/to/STM_DataProcessing
cp -R skills/stm-topo-phase-analysis ~/.agents/skills/
# 或改为符号链接（改仓库副本即时生效）：
ln -s "$PWD/skills/stm-topo-phase-analysis" ~/.agents/skills/
```

刷新已安装副本（编辑仓库副本后执行）：

```bash
cd /path/to/STM_DataProcessing && rm -rf ~/.agents/skills/stm-topo-phase-analysis && cp -R skills/stm-topo-phase-analysis ~/.agents/skills/
```
