# CHANGES — topo-correction

## 2026-09：从 stm-topo-phase-analysis 拆分为独立 skill

用户要求把几何矫正拆成独立 skill（矫正产物可接任意后处理），与相位分析并列：

- 新目录 `skills/topo-correction/`；原目录 `skills/stm-topo-phase-analysis` 改名
  `skills/phase-analysis`（相位部分，见其 CHANGES.md）。
- `stm_topo_correct.py` 原样移入（除 import 改造：`atlas`/`phasepipe` → 本地
  `correction_lib`）；输出文件名、`correction.log` 行格式、`correction_report.json` 字段
  **逐字节不变**（跨 skill 契约）。`correction_report.json` 的 `skill` 字段值由
  `stm-topo-phase-analysis` 更正为 `topo-correction`。
- 兄弟依赖内聚为 `scripts/correction_lib.py`（自包含拷贝：`group_rings`、`setup_style`、
  `load_colormap`/`_GWYDDION_ANCHORS`、`BAD_COLOR`），不 import phase-analysis 的任何模块。
- 自带一键 self-test（`scripts/selftest.py`，5 项），从原 self-test 的矫正阶段移植
  （合成数据生成器为独立实现，不依赖 phasepipe）。
- 相位 skill 的 self-test 同步移除矫正阶段检查（72 → 68 项）。

## 2026-09（v2.1）：锚定自检新增全局拉伸尺度 tell-tale

原自检只用"矫正后两个最强环的半径比是否 1 : √3"；当原始数据本身就有精确 1 : √3 环对时，
错锚定按全局因子（1/√3 或 √3）缩放整个拟合、比值仍然通过 ⇒ 误报 consistent。
新增 `|det M|^(1/2)` 拉伸尺度判据（容差 5 %），两条 tell-tale 合并判定：
都过 = consistent，任一失败 = inconsistent（比值通过、尺度失败也是 inconsistent），
环对无法形成但尺度在容差内 = unverifiable。实测错锚偏 42 % / 73 %，正确锚定偏 ≤2 %。

## 2026-09（v2）：锚定环显式化（`--anchor-ring`）

包内数据锚定检测把它认定的第一个环标成 (1,0)；r3 环在 1x1 环内侧（半径比 1/√3）时检测可能
锚定在 r3 环上，按 1x1 给参考晶格会把 r3 环拉到 1x1 理想半径、破坏几何。
`--anchor-ring 1x1|r3` 显式声明锚定环，参考晶格 `a_ref = a` 或 `√3·a`。

## 2026-09（v1，原 stm-topo-phase-analysis）

初版矫正流程（topo0009 实测）；v1 入库时改为调用 `stm_data_processing.utils.bragg_peak` 包。
