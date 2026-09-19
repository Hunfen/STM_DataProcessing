# FFT Bragg 峰检测模块使用指南（`stm_data_processing.utils.bragg_peak`）

本模块在一张 STM 拓扑图的 FFT 上做四件事：**检出尽可能多的 Bragg 峰**（石墨烯 1×1 环及更弱的高阶 Bragg 点）、给出每个点的**亚像素 `q` 与逐点不确定度**、用晶格模型做**GLS 基矢精修**并给每个峰整数指数 `(h,k)`、以及**把畸变扫描矫正回理想倒格几何**（`correct_bragg_peaks`，见 §1.5）。**真实数据在默认参数下直接可用**——不存在旧版的"质量门默认拒收"语义。

- 代码（模块化包）：`src/stm_data_processing/utils/bragg_peak/`（10 个模块，合计 1991 行）
- 兼容入口：`src/stm_data_processing/utils/bragg_peak_detection.py`（29 行 shim，旧 import 不变）
- 设计规格：[`design/bragg_peak_detection.md`](design/bragg_peak_detection.md)（含**矫正能力路线图 L1–L4**，§9）
- 回归自检：`scripts/regression/check_bragg_peak_detection.py`（R1 合成真值 + R2 标准数据物理断言 + R3 证据图 + R4 矫正/方向守卫/identity 真 no-op，54/54 检查，约 100 s）
- Skill（随仓库分发）：`skills/affine-correction/`（几何矫正）+ `skills/phase-analysis/`（双环相位分析），内部调用本包

---

## 1. 快速开始

### 1.1 最小示例（合成图，不依赖外部数据）

```python
import numpy as np

from stm_data_processing.utils.bragg_peak import LatticeSpec, detect_bragg_peaks

# 合成演示图：30 nm 场、六方 a = 2 nm 的反射 + 高斯噪声（固定种子）
n, size_nm = 256, 30.0
b_px = 4 * np.pi / (np.sqrt(3) * 2.0) / (2 * np.pi / size_nm)   # |b1|，单位 px
axis = np.arange(n) - n // 2
xg, yg = np.meshgrid(axis, axis)
b1 = np.array([b_px, 0.0])
b2 = b_px * np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
image = np.zeros((n, n))
for h in range(-2, 3):
    for k in range(-2, 3):
        q = h * b1 + k * b2
        if (h, k) == (0, 0) or np.hypot(*q) > 0.6 * (n // 2):
            continue
        image += np.exp(-(np.hypot(*q) / 40.0) ** 2) * np.cos(
            2 * np.pi * (q[0] * xg + q[1] * yg) / n
        )
image += 1.0 * np.random.default_rng(20260917).normal(size=(n, n))

result = detect_bragg_peaks(image, size_nm, lattice=LatticeSpec(a_nm=2.0, symmetry="hexagonal"))
print(len(result.peaks), result.lattice.fit_ok if result.lattice else None)
for peak in result.peaks[:3]:
    print(peak.index_hk, peak.q_px, peak.sigma_q_px)
```

`result.peaks` 是元组，按 `snr` 降序；每个峰给出 `q_px`/`q_nm_inv`、`sigma_q_px`/`sigma_q_nm_inv`、`cov_q_px`/`cov_q_nm_inv`、`amplitude`、`snr`、`method`、`quality`，以及与晶格一致时的 `index_hk`、`q_model_px`、`sigma_q_model_px`、`residual_px`。

### 1.2 标准石墨烯数据（用户钦定的验收数据，只读）

```python
from stm_data_processing.utils.bragg_peak import detect_bragg_peaks, load_image

# 100 nm 场，2048²，tab 分隔
img = load_image("/Users/hunfen/Documents/论文/c6lic6/data_processing/final/topo0009.txt")
result = detect_bragg_peaks(img, 100.0)          # lattice=None，默认参数即可
print(result.lattice.fit_ok)                     # True（harmonic_ladder）
print([p.index_hk for p in result.peaks if p.index_hk is not None][:6])  # 一环 ±(1,0) ±(0,1) ±(1,−1)

# 30 nm 场，1024²，tab 分隔（扩展名 .csv 但内容仍是 tab）
img2 = load_image("/Users/hunfen/Documents/论文/c6lic6/data_processing/20251117_topo4_30nm.csv")
result2 = detect_bragg_peaks(img2, 30.0)
```

实测基线（默认调用、不调参）见 §6；证据图（log FFT + 峰叠加）在 `tmp_verify/bragg_rewrite/`。

### 1.3 真实 Nanonis 数据

```python
import numpy as np

from stm_data_processing.io.nanonis_loader import NanonisFileLoader
from stm_data_processing.utils.bragg_peak import LatticeSpec, detect_bragg_peaks

loader = NanonisFileLoader("path/to/topo0002.sxm")
z_index = loader.channels.index("Z") if "Z" in loader.channels else 0
image = np.asarray(loader.data[2 * z_index], dtype=float)   # 前向/后向两通道交错
size_nm = float(loader.range[0]) * 1e9                     # 场边长（nm）

result = detect_bragg_peaks(image, size_nm, lattice=LatticeSpec(a_nm=0.246, symmetry="hexagonal"))
print(result.lattice.fit_ok if result.lattice else None, result.meta["basis_source"])
```

### 1.4 复用已算好的谱

```python
from stm_data_processing.utils.bragg_peak import BraggPeakDetector, compute_fft2

fft2 = compute_fft2(image, size_nm, window="hann", subtract_plane=True, nan_policy="plane")
detector = BraggPeakDetector(lattice=LatticeSpec(a_nm=0.246, symmetry="hexagonal"))
result = detector.detect_from_fft2(fft2, size_nm)          # 不再做加窗/去平面
```

### 1.5 畸变矫正（把歪掉的扫描拉回理想倒格几何）

```python
from stm_data_processing.utils.bragg_peak import LatticeSpec, correct_bragg_peaks, load_image

image = load_image("topo0009.txt")
spec = LatticeSpec(a_nm=0.246, symmetry="hexagonal")   # 显式点群：不做自动推断
correction = correct_bragg_peaks(image, 100.0, lattice=spec)

print(correction.meta["method"])        # "weighted_lsq" = 成功；"two_vector_fallback" / "identity_fallback" = 回退
print(correction.measured_radius_px, correction.target_radius_px, correction.residual_ratio)
print(correction.size_nm)               # 矫正后视场：换算 q_nm_inv 必须用它
# correction.image 为矫正后图（NaN 填充），correction.fft2 为其复数谱
```

要点：① 几何解为**对称正定拉伸**（纯拉伸、零旋转）并锚定**物理真值**（\|b₁\|=4π/(√3·a)），不用拟合 `affine`（它在 `lattice=None` 下≈单位阵，会漏掉数 % 的真实偏差）；② 标准两图实测（独立复核值）：topo0009 29.8367→**29.5415 nm⁻¹**（0.159 % 偏差）、topo4 28.1891→**29.5791 nm⁻¹**（0.287 %），角距误差 ≤0.25°；经 skill 路径（含 `flipud(subtractMeanPlane)` 预处理）为 29.4695 / 29.5765 nm⁻¹（0.078 % / 0.284 %）；③ **`identity_fallback` 必须提示用户**（见 §8），该分支现在返回**输入原样**（逐位一致、不重铺画布、无半像素偏移、不增 NaN，`n_out = n_px`、`offset = 0`）；④ 方向守卫须用**正向矩阵反向**（转置反向是恒等、无效）。

---

## 2. 输入与输出

### 2.1 主要入口

| 入口 | 说明 |
|---|---|
| `detect_bragg_peaks(image, size_nm, **kwargs)` | 一步完成预处理 → FFT → 检测 → 亚像素定位 → 环聚类 → 晶格精修 |
| `correct_bragg_peaks(image, size_nm, *, lattice=None, result=None, order=3, pad=10, return_fft2=True)` | **畸变矫正**：锚定物理真值解对称正定拉伸并重采样，返回 `CorrectionResult` |
| `compute_fft2(image, size_nm, *, window="hann", subtract_plane=True, nan_policy="plane")` | 只做预处理 + FFT（fftshift 后的复数谱） |
| `BraggPeakDetector(**kwargs)` + `.detect(image, size_nm)` / `.detect_from_fft2(fft2, size_nm, *, return_fft2=False)` | 复用配置；`detect_from_fft2` 接受已 fftshift 的复数谱（不做加窗/去平面） |
| `load_image(path)` | 读 tab/逗号/空白分隔的方阵 ASCII（`.txt`/`.csv`） |
| `LatticeSpec(a_nm=..., symmetry="hexagonal"\|"square"\|"oblique", orientation_deg=..., bvecs_nm_inv=...)` | 参考晶格；`orientation_deg` 是**逆时针**旋转角（约定 `basis @ rot.T`）；`lattice=None` 时从数据推断（六方专用，见 §8） |

### 2.2 `BraggDetectionResult`

| 字段 | 含义 |
|---|---|
| `peaks` | 全部检出峰（含 `independent=False` 的 −q 镜像伙伴，位置由镜像给出） |
| `lattice` | `LatticeFit` 或 `None`（无环/拟合失败） |
| `size_nm` / `n_px` | 场边长（nm）/ 谱边长（px） |
| `dq_nm_inv` / `q_nyquist_nm_inv` | `2π/L` 与 `πN/L`（nm⁻¹） |
| `noise_sigma` / `n_candidates` | 稳健 Rayleigh 噪声尺度 / 预截断候选数 |
| `fft2` | 仅 `return_fft2=True` 时给出 |
| `meta` | 诊断：`basis_source`、`ring_radii_px`/`ring_snr`/`ring_members`、`ladder_rings`、`footprint_px`、`n_candidates_before_cap`、`lattice_error`、`fallback_report` 等 |

### 2.3 `LatticeFit`（关键字段）

| 字段 | 含义 |
|---|---|
| `fit_ok` / `quality` | **只看 `fit_ok`** 判断拟合是否收敛；它**不是**数据质量门——失败也不清峰 |
| `bvecs_nm_inv` / `cov_bvecs_nm_inv` | 拟合倒格基（行 b1, b2，nm⁻¹）与其 4×4 协方差 |
| `affine` / `cov_affine` | `M`（`q_obs = q_ideal @ M`）与其协方差 |
| `rotation_deg` / `rotation_sigma_deg` / `rotation_mod_deg` / `rotation_is_absolute` | 旋转角（度）、其 1σ、点群模数（60/90/180）、是否有绝对取向 |
| `principal_stretches` | 主拉伸（畸变椭圆的两轴比） |
| `reference_radius_px` / `rms_residual_px` / `chi2_reduced` / `n_independent` / `symmetry` | 参考环半径 / 像素残差 RMS / 加权 χ² / 独立点数 / 对称类 |

---

## 3. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `lattice` | `None` | 参考晶格；`None` 时按谐波阶梯规则推断（§5.4） |
| `min_snr` | `4.0` | 候选的径向 SNR 门（相对局部背景的 z 分数） |
| `dc_mask_frac` | `0.03` | DC 掩模半径（占 `n` 的比例），**排除**而非置 0 |
| `q_max_px` | `None` | `None` ⇒ `0.95·(n//2)`；同时约束候选与标号池 |
| `max_candidates` | `2048` | 候选上限（按 SNR 截断）；保证真实数据只报几十个峰 |
| `max_peaks` | `None` | 最终上报峰数上限 |
| `footprint` / `patch_half` | `None` / `3` | NMS footprint（None ⇒ 自适应）/ 定位 patch 半宽 |
| `sigma_model_floor_px` | `0.006` | 逐点位置协方差的模型地板（px） |
| `window` / `subtract_plane` / `nan_policy` | `"hann"` / `True` / `"plane"` | 预处理选项（`nan_policy` 另可 `"raise"`/`"zero"`） |
| `return_fft2` | `False` | 是否在结果里带上复数谱 |

> 旧版 7 个调参 kwarg（`min_snr_seed`、`min_snr_verify`、`lattice_tolerance`、`max_iterations`、`residual_max_px`、`min_pool_spacing_px`、`chi2_red_max`）已随质量门一起删除，仓库内无调用方。

---

## 4. 检测语义（读结果前必读）

1. **SNR 是径向的**：`z(r) = (|F(r)| − level(r)) / sigma(r)`，`level`/`sigma` 为 2 px 径向 bin 的稳健背景。真实 STM 谱背景随 |q| 陡降且带 1/f 脊，只有径向归一化才能同时保住弱高阶峰又不被低 q 脊淹没。
2. **峰按环上报**：`peaks` = 检出环的全部成员 + 拟合晶格标号的峰（含 −q 镜像）；不在任何环里的背景极大值不算 Bragg 峰。
3. **不拒收**：`fit_ok=False`（拟合不收敛/秩亏）只清空模型声明（`index_hk` 等置 `None`、`lattice=None`），峰的位置、σ、SNR 全部保留——拟合失败不剥夺任何测量。
4. **参考环推断**（`lattice=None`）：取**自带谐波阶梯的最大环**为 1×1 参考环。这是刻意的：√3×√3 R30° 超结构环位于 |b|/√3，其反射集合包含基本晶格全部反射，"解释反射数最多"永远偏好更细的胞；只有要求"带自己的谐波阶梯"才能正确选中 1×1（100 nm 标准图的最强环正是 √3R30 环）。
5. **超结构/摩尔纹峰**：保留为未标号峰（`index_hk=None`），不会被强行塞进 1×1 晶格。

---

## 5. 与旧版（2515 行单体）的差异

| 项 | 旧版 | 新版 |
|---|---|---|
| 真实数据 | 三质量门（G-1/2/3）**默认拒收**，"拒绝即设计" | 默认直接可用；`fit_ok`=收敛 |
| 峰数 | 30 nm 图爆出 5042 峰 | 同一张图 48 峰（候选上限 2048 + 环模型） |
| 参考环 | 复杂斜方 gauge / predict-verify 循环 | 谐波阶梯规则 + 三循环旋转搜索（锚定真角） |
| 文件 | 单文件 2515 行 | 10 模块 1945 行 + 29 行 shim |
| 矫正 | 无（旧 skill 脚本自带简化检峰 + `fsolve` 两点解） | 包内 `correct_bragg_peaks`：物理真值锚定 + 全标号峰加权 LSQ + 方向性由回归钉死 |
| 参数 | 15+ 调参项 | 12 项（见 §3），删 7 个旧 kwarg |

旧 import 路径 `from stm_data_processing.utils.bragg_peak_detection import ...` 仍可用（shim），新代码请用 `...utils.bragg_peak`。

---

## 6. 标准数据实测基线（回归脚本 R2 断言依据，默认调用）

| 量 | topo0009.txt（2048²，100 nm） | 20251117_topo4_30nm.csv（1024²，30 nm） |
|---|---|---|
| 一环半径 r₁ | 473.70 px = 29.7635 nm⁻¹（1.009× 理想） | 135.78 px = 28.4378 nm⁻¹（0.964×） |
| 一环峰 | 6 个，±(1,0), ±(0,1), ±(1,−1) | 6 个，同上 |
| 更弱标号峰 | 10 | 14 |
| `fit_ok` / 参考来源 | True / `harmonic_ladder` | True / `harmonic_ladder` |
| 反演 a | 0.24316 nm（−1.15 %） | 0.25738 nm（+4.62 %） |
| 峰数 / 环数 / 运行时间 | 42 / 7 / 14.3 s | 48 / 8 / 14.3 s |

验收断言（每张图）：6 个一环峰、半径 RMS 与最大偏差 ≤2 %、角距 60±4°、r₁∈[0.93,1.07]·29.49 nm⁻¹、≥4 个弱峰、`fit_ok`、a 偏差 ≤±7 %、≤60 s。30 nm 图的一环是真实椭圆（三成员半径 134.58/135.29/137.47 px，全距 2.12 %）——断言按 RMS/最大偏差口径，全距仅打印。

---

## 7. 复现命令

```bash
# 模块回归（R1 合成真值 + R2 标准数据断言 + R3 证据图 + R4 矫正/方向守卫/identity 真 no-op；54/54，约 100 s，预算 ≤150 s）
.venv/bin/python scripts/regression/check_bragg_peak_detection.py

# 全部回归（9 套，必须全 exit 0）
for f in scripts/regression/check_*.py; do .venv/bin/python "$f" || exit 1; done

# 代码规范（项目口径）
.venv/bin/python -m ruff check src/stm_data_processing/utils/ scripts/regression/check_bragg_peak_detection.py
```

> 标准数据文件缺失时 R2/R3 打印 SKIP 并仍 exit 0，保证其他机器回归全绿。证据图与验证报告（`verification_report.md`、`review.md`）在 `tmp_verify/bragg_rewrite/`（已 gitignore，仅本地）。

---

## 8. 已知边界（诚实记录）

- **矫正受限于检测器的标号能力（重要）**：`correct_bragg_peaks` 必须先能标号（环聚类三元组半径差 ≤3 %、`match_labels` 容差 2 %·|q|），因此**各向异性大致 >3 % 时矫正退化为 `identity_fallback`**（结果图 ≈ 输入图）；畸变的矩形/方晶格因天生各向异性会直接撞上这条限制。**务必检查 `result.meta["method"]`**：为 `identity_fallback` 即表示未成功矫正。
- **残余 0.17–0.29 % 属非仿射部分**（扫描蠕变/漂移），单一全局拉伸模型无法再降；升级路线（L1 宽容差自举 → L2 迭代收敛 → L3 一般仿射 → L4 逐行/分块非仿射）见设计规格 §9，含每级设计要点与风险。
- **自动点群推断不做**（刻意）：点群须由 `LatticeSpec` 明示；`lattice=None` 的推断路径仍为六方专用。
- **参考环推断**对带强 2× 谐波阶梯的人造晶格可能取到因子 2 的谐波环（详见设计规格 §8）；真实石墨烯数据不受影响。
- **斜方对称**只支持两点种子，不发布斜方应变口径；需要应变请给 `LatticeSpec`。
- 旧版文档中"真实数据大概率被拒"一节已随三质量门一并废除；若你依赖旧版的拒收语义（例如用 `meta["lattice_quality"]` 判断数据可用性），请改用 `result.lattice.fit_ok` + `meta` 里的拟合诊断自行判定。
