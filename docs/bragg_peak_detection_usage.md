# FFT Bragg 峰检测模块使用指南（`stm_data_processing.utils.bragg_peak_detection`）

本模块在一张 STM 拓扑图的 FFT 上做三件事：**检出尽可能多的 Bragg 点**、给出每个点的**亚像素 `q` 与逐点不确定度**、用晶格模型做**带全局质量门的约束精修**并给每个峰一个整数指数 `(h,k)`。

- 代码：`src/stm_data_processing/utils/bragg_peak_detection.py`
- 设计规格（算法、公式、阈值、推导）：[`design/bragg_peak_detection.md`](design/bragg_peak_detection.md)
- 回归自检：`scripts/regression/check_bragg_peak_detection.py`（R1–R25，合成真值 + 确定性用例，约 18 s）

---

## 1. 快速开始

### 1.1 最小示例（可直接运行，不依赖外部数据）

```python
import numpy as np

from stm_data_processing.utils.bragg_peak_detection import (
    LatticeSpec,
    detect_bragg_peaks,
)

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

result = detect_bragg_peaks(
    image, size_nm, lattice=LatticeSpec(a_nm=2.0, symmetry="hexagonal")
)

print(len(result.peaks))                       # 30（含 ±q 两侧）
print(result.lattice.fit_ok, result.lattice.quality)          # True ok
for peak in result.peaks[:3]:
    print(peak.index_hk, peak.q_px, peak.sigma_q_px, peak.q_model_px)
```

> **为什么演示图要有噪声**：`chi2_reduced` 用逐点不确定度（含 0.006 px 模型地板）衡量残差。一张**无噪声**的合成图里残差只是量化/模型误差，会远超声明 σ ⇒ χ² 门（G-3）正确地把它拒掉。真实数据总有噪声，统计一致性才有意义；§5 的三张真实扫描被拒则是因为拟合本身与数据不一致/退化，属另一回事。

`result.peaks` 是元组，按 `snr` 降序；每个峰给出 `q_px`（亚像素，px）、`q_nm_inv`、`sigma_q_px`/`sigma_q_nm_inv`、`cov_q_px`/`cov_q_nm_inv`、`amplitude`、`snr`、`method`、`quality`，以及与晶格一致时的 `index_hk`、`q_model_px`、`sigma_q_model_px`、`residual_px`。

### 1.2 真实 Nanonis 数据

```python
import numpy as np

from stm_data_processing.io.nanonis_loader import NanonisFileLoader
from stm_data_processing.utils.bragg_peak_detection import (
    LatticeSpec,
    detect_bragg_peaks,
)

loader = NanonisFileLoader("path/to/topo0002.sxm")
z_index = loader.channels.index("Z") if "Z" in loader.channels else 0
image = np.asarray(loader.data[2 * z_index], dtype=float)   # 前向/后向两通道交错
size_nm = float(loader.range[0]) * 1e9                     # 场边长（nm）

result = detect_bragg_peaks(image, size_nm, lattice=LatticeSpec(a_nm=2.0, symmetry="hexagonal"))
print(result.lattice.fit_ok, result.lattice.quality, result.meta["lattice_error"])
```

> 真实数据**很可能被拒**（见 §5），这是设计意图，不是缺陷。

### 1.3 复用已算好的谱

若你已经算过 FFT（例如做过线/平面扣除），可以跳过重复计算：

```python
from stm_data_processing.utils.bragg_peak_detection import compute_fft2, detect_bragg_peaks

fft2 = compute_fft2(image, size_nm, window="hann", subtract_plane=True, nan_policy="plane")
result = detect_bragg_peaks(image, size_nm, return_fft2=True)   # 也可以直接返回谱
# 或者：detector = BraggPeakDetector(lattice=spec); detector.detect_from_fft2(fft2, size_nm)
```

---

## 2. 输入与输出

### 2.1 主要入口

| 入口 | 说明 |
|---|---|
| `detect_bragg_peaks(image, size_nm, **kwargs)` | 一步完成预处理 → 检测 → 亚像素定位 → 晶格精修 |
| `compute_fft2(image, size_nm, *, window="hann", subtract_plane=True, nan_policy="plane")` | 只做预处理 + FFT（fftshift 后的复数谱） |
| `BraggPeakDetector(**kwargs)` + `.detect(image, size_nm)` / `.detect_from_fft2(fft2, size_nm)` | 复用配置；`detect_from_fft2` 接受已 fftshift 的复数谱（不做加窗/去平面） |
| `LatticeSpec(a_nm=..., symmetry="hexagonal"|"square"|"oblique", orientation_deg=...| bvecs_nm_inv=...)` | 参考晶格；`orientation_deg` 是**逆时针**旋转角（约定 `basis @ rot.T`） |

### 2.2 `BraggDetectionResult`

| 字段 | 含义 |
|---|---|
| `peaks` | 全部检出峰（含 `independent=False` 的 −q 伙伴，其位置由镜像索引给出） |
| `lattice` | `LatticeFit` 或 `None`（未做晶格阶段） |
| `size_nm` / `n_px` | 场边长（nm）/ 谱边长（px） |
| `dq_nm_inv` / `q_nyquist_nm_inv` | `2π/L` 与 `πN/L`（nm⁻¹） |
| `noise_sigma` / `n_candidates` | 稳健 Rayleigh 噪声尺度 / 候选数 |
| `fft2` | 仅 `return_fft2=True` 时给出 |
| `meta` | 诊断与决策通道：`fit_ok`、`lattice_quality`、`lattice_error`、`pool_spacing_px`、`residual_max_px`、`min_pool_spacing_px`、`chi2_red_max`、`inferred_symmetry`、`oblique_gauge`、`oblique_gauge_margin`、`reference_gauge`、`nan_filled`、`degenerate_input` 等 |

### 2.3 `LatticeFit`（关键字段）

| 字段 | 含义 |
|---|---|
| `fit_ok` / `quality` | **只看这个布尔量**判断晶格是否可用；`quality` 是原因串（`"ok"` 或失败门按 G-1→G-2→G-3 用 `+` 连接） |
| `bvecs_nm_inv` / `cov_bvecs_nm_inv` | 拟合倒格基（行 b1, b2，nm⁻¹）与其 4×4 协方差 |
| `affine` / `cov_affine` | `M`（`q_obs = q_ideal @ M`）与其协方差；推理路径（无参考基）下 `affine=None` |
| `rotation_deg` / `rotation_sigma_deg` / `rotation_mod_deg` / `rotation_is_absolute` | 旋转角（度）、其 1σ、点群模数（60/90/180/360）、是否有绝对取向 |
| `principal_stretches` / `linearized_strain` / `strain_gauge` | 主拉伸、线性应变 `polar(M)[1] − I`、应变口径（`"absolute"` / `"deviatoric"` / `"undefined"`） |
| `pool_spacing_px` | **实测**最短倒格矢长度（首环半径，px），≡ `meta["pool_spacing_px"]` |
| `rms_residual_px` | **像素**单位的残差 RMS，接受/拒绝两态都等于逐点 `residual_px` 的 RMS |
| `residual_max_px` | G-2 实际采用的上限（显式传入或 `min(1.0, 0.25·pool_spacing_px)`） |
| `consistent_fraction` | `residual_px ≤ lattice_tolerance · hypot(sigma_q_px)` 的比例（**仅测量 σ**；审计量，不参与门；拒绝态仍报告） |
| `chi2_reduced` / `n_independent` / `n_iterations` / `symmetry` | 加权 χ²、独立点数、迭代次数、对称类 |

---

## 3. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `lattice` | `None` | 参考晶格；`None` 时从强峰推断对称类（斜方为两点种子） |
| `min_snr` / `min_snr_seed` / `min_snr_verify` | `5.0` / `8.0` / `3.5` | 候选 / 种子 / 预测验证的 SNR 门 |
| `dc_mask_frac` | `0.03` | DC 掩模半径（占 `n` 的比例）；**排除**而非置 0 |
| `footprint` / `patch_half` | `None` | `None` ⇒ 自适应（footprint ∈ [3, 11]、patch ∈ [3, 7]）；显式整数覆盖（消融用） |
| `max_peaks` / `q_max_px` | `None` | `None` ⇒ 无上限 / `0.95·(n//2)`。`q_max_px` **同时**约束候选与预测池 |
| `lattice_tolerance` | `3.0` | 逐点接受判据与 `consistent_fraction` 的容差（乘组合 σ / 测量 σ） |
| `max_iterations` | `3` | predict → verify → refine 轮数 |
| `sigma_model_floor_px` | `0.006` | 逐点位置协方差的模型地板（px） |
| `chi2_red_max` | `25.0` | **G-3** 上限；`math.inf` 关闭 |
| `residual_max_px` | `None` | **G-2** 上限；`None` ⇒ `min(1.0, 0.25·pool_spacing_px)` |
| `min_pool_spacing_px` | `3.0` | **G-1** 下限；`0.0` 关闭 |
| `return_fft2` | `False` | 是否在结果里带上复数谱 |

---

## 4. 质量门与拒绝语义（读结果前必读）

三个门（设计规格 §4.4(e)）必须**同时**成立，另加 `rank = 4 ∧ isfinite(chi2_reduced) ∧ |det B| > 1e-9`：

| 门 | 判据 | 默认 |
|---|---|---|
| **G-1** 可分辨性 | `pool_spacing_px >= min_pool_spacing_px` | 3.0 px |
| **G-2** 几何一致性 | `rms_residual_px <= residual_max_px` | `min(1.0 px, 0.25·pool_spacing_px)` |
| **G-3** 统计一致性 | `chi2_reduced <= chi2_red_max` | 25.0 |

`fit_ok=False` 时**全部模型声明被收回**：每个峰的 `index_hk`/`q_model_px`/`sigma_q_model_px`/`residual_px` 置 `None`、`affine`/`cov_affine`/`rotation_*`/`principal_stretches`/`linearized_strain` 置 `None`、`strain_gauge="undefined"`；**只保留** `peaks`（它们确实被检出）与诊断量。理由：把错误基矢标成"可用"比拒绝更有害（规格 D13 与 LFA E3/E4 的教训）。

**判决可复算**：`(result.lattice, result.meta)` 两个对象就够——`meta` 里有实测值与两个门限回显，例如

```python
lat, meta = result.lattice, result.meta
assert lat.fit_ok is False or lat.pool_spacing_px == meta["pool_spacing_px"]
print(meta["lattice_quality"], meta["lattice_error"])   # 例如 "residual_above_max+chi2_above_max" 与理由串
```

`consistent_fraction` 只用**逐点测量 σ**、用**调用方**的 `lattice_tolerance`、单位 px，并且在清空标签**之前**计算，所以被拒的拟合照样给出这个审计比例。

---

## 5. 真实数据行为与预期管理（重要）

三张 Si(111)-Pb 岛扫描（`Si111_Pb_islands/raw_data/2025-07-09/`，只读）在**默认门**下的实测基线（设计规格 §6.4）：

| 图像 | n / 场 | χ²_red | rms (px) | `pool_spacing_px` | `consistent_fraction` | 拒因 | `fit_ok` |
|---|---|---|---|---|---|---|---|
| topo0002.sxm | 512 / 30 nm | 6708 | 1.4925 | 9.312 | 0.1389 | `residual_above_max+chi2_above_max` | False |
| topo0011.sxm | 512 / 30 nm | 8.617e4 | 3.1366 | 10.003 | 0.1944 | `residual_above_max+chi2_above_max` | False |
| topo0005.sxm | 128 / 900 nm | 39.26 | 1.1037 | 0.250 | 0.5556 | `spacing_below_min+residual_above_max+chi2_above_max` | False |

**它们被拒是正确行为**：topo0002/0011 的拟合与数据不一致（χ² 远超门），topo0005 的首环半径只有 0.25 px（远小于检测 footprint 下限 3 px ⇒ 退化拟合，`spacing_below_min`）。

被拒时的处置顺序建议：

1. **先提质数据**：更大场 / 更平整的区域、线扣除与平面扣除、降低扫描噪声、确认场标定与扫描畸变（漂移、蠕变）已处理；
2. **显式给出参考**：`LatticeSpec(a_nm=..., symmetry=..., orientation_deg=...)`（已知取向走全池最近邻，标号是绝对的）；只给 `bvecs_nm_inv` 而不给取向时，斜方按**最小畸变约定**标号并只在 `meta["oblique_gauge_margin"]` 健康（≫1）时才可信；
3. **检查是否是别的真特征**：`index_hk=None` 的点可能是超结构/摩尔纹/杂质态，不必强行归入本晶格；
4. **不要**为了让它"通过"而放宽 `chi2_red_max`/`residual_max_px`/`min_pool_spacing_px`——这三个参数是给消融实验与已知场景用的，不是默认调参项。真实数据断言不进回归脚本（R16 只做不变量 smoke）。

---

## 6. 精度与不确定度（数字与来源）

**合成真值基准**（`scripts/regression/check_bragg_peak_detection.py`，256²、L = 30 nm、六方 a = 2 nm、M_true = [[1.02, 0.03], [0.01, 0.98]]、固定种子 20260917；来源：回归脚本 R1–R25 的打印值）：

| 指标 | 阈值 | 实测 |
|---|---|---|
| 逐环召回（SNR ≥ 50 / ≥ 15 / ≥ 5） | ≥0.95 / ≥0.90 / ≥0.75（汇总 SNR ≥ 8：≥0.90） | 1.000 / 1.000 / 1.000（汇总 1.000） |
| 假阳性率 FPR | ≤0.05（低 SNR ≤0.10） | 0.0200 / 0.0220 / 0.0267 |
| 定位 rms₂D（SNR ≥ 50 / 15–50 / <15） | ≤0.05 / ≤0.12 / ≤0.40 px | 0.0197 / 0.0602 / 0.1421 px |
| 逐轴偏置（SNR ≥ 50） | ≤0.010 px | < 0.001 px |
| σ 标定（中位 pull） | ∈ [0.5, 2.0] | 0.834 / 0.844 |
| 模型预测 rms₂D(q_model) | ≤0.05 px 且 < 逐点 rms | 0.0078–0.0364 vs 0.0179–0.0602 px |
| 主拉伸绝对误差 / 旋转误差（模点群） | ≤2e-3 / ≤0.05° | 8e-5–1.08e-3 / 0.0012–0.0218° |
| 运行时间 | ≤180 s | **25/25 PASS，约 18 s** |

**质量门与真实数据**：合成四场景 `fit_ok=True`（χ² 1.34–1.84、rms 0.0594/0.1191/0.2013/0.0931 px、`pool_spacing_px` 16.84–16.85、`consistent_fraction` 1.000）；失配规格场景 `fit_ok=False` 且无标签。真实三图见表 §5。

**来源与可追溯**：

- 设计规格 §6.4（真实数据 A-6 基线）与本指南 §6 的合成数值 = 回归脚本 R1–R25 的实测打印；
- 独立验证报告：`tmp_verify/bragg_audit/verification_report.md`（t6）、`tmp_verify/bragg_review2/t10_review_report.md`（t10）、`tmp_verify/bragg_review3/t13_review_report.md`（t13，verdict = PASS）；
- 裁决与勘误（含单位修复、`consistent_fraction` 口径、A-6 表换算脚注）：`tmp_verify/bragg_spec/rulings.md` 的 A-15.11 / A-15.13 / A-15.21 / A-15.23。

> 注：`tmp_verify/` 是本地验证产物目录（已 gitignore），不随仓库分发；上面的数字都能用 §7 的命令复现。

---

## 7. 复现命令

```bash
# 模块自检（R1–R25，含合成真值基准与确定性用例）
.venv/bin/python scripts/regression/check_bragg_peak_detection.py

# 全部回归（9 套，必须全 exit 0）
for f in scripts/regression/check_*.py; do .venv/bin/python "$f" || exit 1; done

# 代码规范（repo root + pyproject 口径）
.venv/bin/python -m ruff check src/stm_data_processing/utils/ scripts/regression/check_bragg_peak_detection.py

# 真实数据门行为（只读探针；需要本地 Nanonis 数据）
.venv/bin/python tmp_verify/bragg_impl/realdata_gates.py
```

---

## 8. 已知边界与 backlog（诚实记录）

- **真实数据默认大概率被拒**（§5）——这是设计意图；请按 §5 的处置顺序提质，而不是放宽门限。
- **推理斜方（`lattice=None`）不发布形变量**（`strain_gauge="undefined"`）：其参考基由数据派生，形变量恒为无畸变而无信息；需要应变就给 `LatticeSpec`。
- **门拒绝理由串的子串断言尚不完整**：当前回归断言了 `chi2_reduced` 的格式化子串；`rms_residual_px` / `residual_max_px` / `pool_spacing_px` 三个子串断言、以及"第二 dq 基准上门两态 `rms_residual_px` 恒等"各一行，属**已识别、非阻塞、延后**的加固项（实现侧理由串已全部由字段值格式化、无字面量，干跑已验证这三条断言必过）。任何后续改动若触及门的判定/理由串构造、残差计算或 R25 的第二 dq 段，应把这些断言一并并入。
- **性能与可见性（非阻塞）**：斜方候选集的 expand+refit 可进一步截断（当前 16–18 s ≪ 180 s 预算）；`meta["oblique_gauge_margin"] < 1.5` 时可加 warning（当前三例 margin ≈3.1，属健康区间）。
- **ruff 口径**：以 repo root + `pyproject.toml`（`.venv/bin/python -m ruff check`）为权威；系统 `ruff --isolated` 的默认规则集更宽（含 `BLE001`），**不是**本项目口径，也**不要**为其添加 `# noqa`（会触发 `RUF100`）。
