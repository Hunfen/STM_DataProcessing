# STM Data Processing

STM / DFT / Wannier90 数据处理工具包：从 **Nanonis 扫描隧道显微镜实验数据**到 **Wannier90 紧束缚模型**的理论计算，覆盖准粒子干涉（QPI）、Lindhard 响应函数虚部、能带展开、BTK 超导谱等凝聚态物理常用分析流程。

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## 功能特性

- **实验端**：Nanonis 仪器文件（`.sxm` / `.dat` / `.3ds`）统一懒加载解析，自动扫描方向校正、结构化 header 解析，支持形貌图 / dI/dV 谱 / 网格谱的可视化与 PowerPoint 自动报告生成
- **理论端**：基于 Wannier90 MLWF 紧束缚哈密顿量的完整计算链
  - 任意 k 点的 `H(k)` 批量计算（扁平化 GEMM 收缩，GPU 友好）
  - 二维能带 `E(k)`、Green 函数
  - QPI 准粒子干涉：JDOS（FFT 自相关）、Born 近似（含散射势）、T-matrix 形式
  - Wang-2012 方法 Lindhard 响应函数虚部（Im χ0）、轨道选择性裸 Lindhard 函数
- **双后端计算**：模块级 CPU（NumPy）/ GPU（CuPy）自动探测与手动切换，GPU 模式下自动按显存容量分批调度
- **高精度晶格运算**：基于 mpmath 任意精度（默认 50 位）的晶体学约定，支持正/倒格矢互求、超胞/子胞变换、旋转与一致性校验
- **OpenMX 工具**：能带 / DOS 解析、谱函数展开（unfolding）、Gaussian cube 差分
- **统一 IO**：计算结果 HDF5 存取，`calculate()` 返回结构与 `load_*_from_h5()` 完全对齐

## 安装

需要 **CPython ≥ 3.14**（本仓库在 3.14.7 上开发与验证；`.python-version` 锁定 3.14）。

```bash
# 使用 uv（推荐）
git clone https://github.com/Hunfen/STM_DataProcessing.git
cd STM_DataProcessing
uv sync

# 或使用 pip
pip install -e .
```

核心安装只含轻量科学计算依赖，重且按功能选装的依赖拆成 extras：

| extra | 内容 | 何时需要 |
| --- | --- | --- |
| `analysis` | `pymc` / `pytensor` / `ramanspy` | 贝叶斯建模与 Raman 谱分析 |
| `ppt` | `opencv-python` | PPT 自动报告与绘图助手（`utils/plot_funcs.py`、`utils/AutoPPt_winnew_modified.py` 在模块级 `import cv2`） |
| `gpu` | `cupy-cuda13x` | CUDA GPU 后端 |
| `dev` | `ruff` / `pytest` | 开发与回归门 |

```bash
uv sync --extra ppt --extra dev
# 或
pip install -e ".[ppt,dev]"
```

核心计算路径（`import stm_data_processing` 及其子模块）不需要任何 extra；只有用到 PPT/绘图助手时才需要 `ppt`。

可选 GPU 后端（需 CUDA 环境）：

```bash
uv sync --extra gpu
# 或
pip install -e ".[gpu]"
```

## 快速开始

### 后端配置

`config` 模块在导入时自动探测可用后端，也支持手动指定（需在导入计算模块之前调用）：

```python
from stm_data_processing.config import set_backend, get_backend, get_xp

set_backend("gpu")  # 尝试 GPU（不可用时自动回退 CPU）
set_backend("cpu")  # 强制 CPU
set_backend("auto")  # 自动探测（默认）

xp = get_xp()  # 返回 cupy 或 numpy，二者接口一致
```

### 1. 加载 Nanonis 实验数据

```python
from stm_data_processing.io.nanonis_loader import NanonisFileLoader

loader = NanonisFileLoader("topography.sxm")  # 支持 .sxm / .dat / .3ds
img = loader.data  # 懒加载，首次访问时解析，自动校正扫描方向
header = loader.header
bias = loader.bias  # 常用 SPM 参数便捷属性
```

### 2. 加载 Wannier90 紧束缚哈密顿量

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian

ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")
hk = ham.hk(k_frac)  # H(k) = Σ_R e^{2πi R·k} H(R) / ndegen(R)，支持批量 k 点
```

### 3. QPI 准粒子干涉

```python
import numpy as np
from stm_data_processing.stm.qpi_jdos import JDOSQPI
from stm_data_processing.stm.qpi_born import BornQPI

# JDOS QPI：QPI(q,E) = FFT⁻¹[ |FFT[A(k,E)]|² ]
qpi = JDOSQPI(ham, nk=256, eta=0.001)
result = qpi.calculate(
    energy_range=np.linspace(-1.0, 1.0, 50),
    q_range=(-0.3, 0.3),
    output_path="./qpi.h5",  # 可选：保存为 HDF5
)
qpi_maps = result["qpi_layers"]  # shape: (n_energies, nq, nq)
metadata = result["metadata"]

# Born 近似（可指定散射势 V 与实空间掩码）
born = BornQPI(ham, nk=256, eta=0.005)
result = born.calculate(energy_range=0.5, V=V)
```

### 4. Lindhard 响应函数虚部（Wang 2012）

```python
from stm_data_processing.dft.wannier90.mlwf_im_susceptibility import (
    SusceptibilityCalculator_wang2012,
)

sus = SusceptibilityCalculator_wang2012(ham, nk=256, eta=0.005)
sus.set_orbital_selection(minit=m_i, mfin=m_f)  # 可选：轨道选择矩阵
result = sus.calculate(omega_limit=1.0, resolution=0.01, q_range=(-0.5, 0.5))
```

### 5. 高精度晶格运算

```python
from stm_data_processing.utils.lattice import LATTICE

lat = LATTICE(bvecs=bvecs)  # 或 avecs=...，内部用 mpmath 高精度计算
lat.set_precision(100)  # 调整计算精度（十进制有效位数）
print(lat.a1, lat.b1, lat.volume)
lat.rotate(30)  # 旋转晶格
lat.verify_consistency()  # 一致性校验
```

### 6. BTK 超导隧道谱

```python
from stm_data_processing.utils.btk import BTK

btk = BTK(Delta=1.5, Z=0.3)  # 超导能隙 Δ、势垒强度 Z
dI_dV = btk.spectrum(E_min=-5, E_max=5, n_points=500, T=0)  # 零温
dI_dV_T = btk.spectrum(T=4.2)  # 有限温度
```

### 7. OpenMX 输出解析

```python
from stm_data_processing.dft.openmx.parser import OpenMX
from stm_data_processing.dft.openmx.diff_gcube import diff_cube_files

mx = OpenMX()
bvecs = mx.read_bvecs_from_out("system.out")
diff_cube_files("before.cube", "after.cube", "diff.cube")  # 势差分析
```

### 8. FFT Bragg 峰检测与晶格精修

在一张 FFT 图上尽可能多地检出 Bragg 点，给出亚像素 `q`、逐点不确定度与整数指数 `(h,k)`，并对晶格做带全局质量门的约束精修：

```python
import numpy as np

from stm_data_processing.utils.bragg_peak_detection import (
    LatticeSpec,
    detect_bragg_peaks,
)

# 合成演示：30 nm 场、六方 a = 2 nm 的反射 + 高斯噪声
n, size_nm = 256, 30.0
b_px = 4 * np.pi / (np.sqrt(3) * 2.0) / (2 * np.pi / size_nm)  # |b1|，单位 px
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
        image += np.exp(-((np.hypot(*q) / 40.0) ** 2)) * np.cos(
            2 * np.pi * (q[0] * xg + q[1] * yg) / n
        )
image += 1.0 * np.random.default_rng(20260917).normal(size=(n, n))

result = detect_bragg_peaks(
    image, size_nm, lattice=LatticeSpec(a_nm=2.0, symmetry="hexagonal")
)
print(len(result.peaks), result.lattice.fit_ok, result.lattice.quality)
for peak in result.peaks[:3]:  # 亚像素 q 与不确定度（px）
    print(peak.index_hk, peak.q_px, peak.sigma_q_px, peak.q_model_px)
```

要点：`fit_ok=False` 时全部模型声明被收回（`index_hk`/`q_model_px`/affine/形变量为 `None`），只保留诊断量，判决可仅凭 `(result.lattice, result.meta)` 复算；真实数据被拒时先做数据侧提质，**不要**放宽门限。用法、参数与精度数字来源见 [`docs/bragg_peak_detection_usage.md`](docs/bragg_peak_detection_usage.md)，算法与阈值见 [`docs/design/bragg_peak_detection.md`](docs/design/bragg_peak_detection.md)。

## 模块结构

仓库顶层（`intercalation/` 已拆分到独立仓库，不再属于本仓库）：

```
STM_DataProcessing/
├── .github/workflows/ci.yml   # CI：lint（含测量数据守卫）/ 核心导入 / 数据无关回归子集
├── src/stm_data_processing/   # 包本体（轻量核心，见下）
├── tests/regression/          # 回归自检：9 个 check_*.py + pytest 入口
├── scripts/                   # 运维/服务器脚本（run_lindhard_re_chi_parallel.py、server/）
├── docs/                      # 接口文档、设计规格、使用指南
├── app/                       # Tauri 桌面应用（latticeSIM）
├── skills/                    # 自包含的 Agent 技能
├── pyproject.toml             # 依赖分层与 pytest 配置
└── uv.lock                    # 锁定依赖解析结果
```

`src/stm_data_processing/` 内部：

```
src/stm_data_processing/
├── config.py          # CPU/GPU 后端统一管理（BACKEND / get_xp / set_backend）
├── dft/
│   ├── openmx/        # OpenMX：parser / band / dos / unfolding / diff_gcube
│   └── wannier90/     # Wannier90：mlwf_hamiltonian / mlwf_gk / mlwf_ek2d /
│                      #            mlwf_im_susceptibility / lindhard_re_chi
├── stm/               # STM 端：qpi_jdos / qpi_born / qpi_tmat / vortex_num /
│                      #         preview_plot
├── io/                # IO 层：nanonis_loader / w90hr_loader / ek2d_io /
│                      #        lattice_loader / qpi_io / susceptibility_io
└── utils/             # lattice（高精度）/ lattice_operations / bragg_peak_detection /
                       # lindhard1dfree / btk / miscellaneous / monitor / plot_funcs /
                       # nanonis_ppt_generator（自动 PPT 报告）
```

## 桌面应用

[`app/latticeSIM/`](app/latticeSIM/) 是倒格子 / LEED / RHEED 模拟器的 Tauri 2 桌面版（`docs/reciprocal_lattice_simulator.html` 的多 tab 扩展版 + 衍射物理核心 `ui/diffraction.js`）。构建与运行方式见 [`app/latticeSIM/README.md`](app/latticeSIM/README.md)：

```bash
cd app/latticeSIM
npm install
npx tauri build --bundles app    # 产物：src-tauri/target/release/bundle/macos/*.app
```

## 文档

详细的模块接口文档见 [`docs/stm_data_processing/`](docs/stm_data_processing/)（接口表、数学公式、示例与接口对齐检查清单）。

Bragg 峰检测模块另有两份专门文档：

- [`docs/bragg_peak_detection_usage.md`](docs/bragg_peak_detection_usage.md)：使用指南（API 示例、参数、输出字段、真实数据被拒时的处置、精度数字来源）；
- [`docs/design/bragg_peak_detection.md`](docs/design/bragg_peak_detection.md)：设计规格（算法栈、公式、质量门推导、基准协议与全部阈值）。

## 开发

回归门只有一条命令：`tests/regression/` 下的 9 个 `check_*.py` 由 `tests/regression/test_regression_suite.py` 逐个以子进程运行（`cwd` = 仓库根），任一脚本非 0 退出即整轮失败。

```bash
.venv/bin/python -m pytest -q      # 全量回归（约 4–5 分钟，脚本自身打印 PASS/FAIL 汇总）
.venv/bin/python -m ruff check .   # 代码规范
```

开发依赖（含 `pytest`）与按需 extras：

```bash
uv sync --extra dev --extra ppt
# 或
pip install -e ".[dev,ppt]"
```

PPT 自动报告与绘图助手（`utils/plot_funcs.py`、`utils/AutoPPt_winnew_modified.py`）需要 `ppt` extra（`opencv-python`）；核心计算路径不需要它。

### 本地数据与 CI 的分工

9 个回归脚本里有 5 个（`check_3ds_real_data.py`、`check_bragg_peak_detection.py`、`check_lindhard_re_chi.py`、`check_nanonis_3ds.py`、`check_nanonis_sxm.py`）要读本机 `/Users/hunfen/Documents/...` 下的真实测量数据（论文 Nanonis 文件、Wannier 模型）；这些数据**永不入库**（见 `.gitignore` 的测量数据段与 CI 守卫）。它们带 `localdata` 标记：

```bash
.venv/bin/python -m pytest -q                     # 本机一把梭：全部 9 个脚本（唯一的完整门）
.venv/bin/python -m pytest -q -m "not localdata"  # CI 口径：只跑与本地数据无关的 4 个，摘要打印 deselect 数量
```

CI（`.github/workflows/ci.yml`）的 `full-test` 作业跑的就是 `pytest -q -m "not localdata"`；另两个作业是 `lint`（ruff，含"禁止提交测量数据"的守卫）与 `core-import`（不带 extras 装包并导入核心子模块）。标记规则是机械的：脚本里出现 `/Users/` 绝对路径即判为 `localdata`，因此新加的数据相关脚本会自动被 CI 排除，而不是让 CI 变红，也不会靠"内部 SKIP 侥幸变绿"。

## 许可证

MIT License
