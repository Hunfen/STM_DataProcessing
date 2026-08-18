# STM Data Processing

STM / DFT / Wannier90 数据处理工具包：从 **Nanonis 扫描隧道显微镜实验数据**到 **Wannier90 紧束缚模型**的理论计算，覆盖准粒子干涉（QPI）、磁化率、能带展开、BTK 超导谱等凝聚态物理常用分析流程。

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

## 功能特性

- **实验端**：Nanonis 仪器文件（`.sxm` / `.dat` / `.3ds`）统一懒加载解析，自动扫描方向校正、结构化 header 解析，支持形貌图 / dI/dV 谱 / 网格谱的可视化与 PowerPoint 自动报告生成
- **理论端**：基于 Wannier90 MLWF 紧束缚哈密顿量的完整计算链
  - 任意 k 点的 `H(k)` 批量计算（扁平化 GEMM 收缩，GPU 友好）
  - 二维能带 `E(k)`、Green 函数
  - QPI 准粒子干涉：JDOS（FFT 自相关）、Born 近似（含散射势）、T-matrix 形式
  - Wang-2012 方法自旋磁化率、轨道选择性裸 Lindhard 函数
- **双后端计算**：模块级 CPU（NumPy）/ GPU（CuPy）自动探测与手动切换，GPU 模式下自动按显存容量分批调度
- **高精度晶格运算**：基于 mpmath 任意精度（默认 50 位）的晶体学约定，支持正/倒格矢互求、超胞/子胞变换、旋转与一致性校验
- **OpenMX 工具**：能带 / DOS 解析、谱函数展开（unfolding）、Gaussian cube 差分
- **统一 IO**：计算结果 HDF5 存取，`calculate()` 返回结构与 `load_*_from_h5()` 完全对齐

## 安装

```bash
# 使用 uv（推荐）
git clone https://github.com/Hunfen/STM_DataProcessing.git
cd STM_DataProcessing
uv sync

# 或使用 pip
pip install -e .
```

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

set_backend("gpu")   # 尝试 GPU（不可用时自动回退 CPU）
set_backend("cpu")   # 强制 CPU
set_backend("auto")  # 自动探测（默认）

xp = get_xp()        # 返回 cupy 或 numpy，二者接口一致
```

### 1. 加载 Nanonis 实验数据

```python
from stm_data_processing.io.nanonis_loader import NanonisFileLoader

loader = NanonisFileLoader("topography.sxm")  # 支持 .sxm / .dat / .3ds
img = loader.data     # 懒加载，首次访问时解析，自动校正扫描方向
header = loader.header
bias = loader.bias    # 常用 SPM 参数便捷属性
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
    output_path="./qpi.h5",     # 可选：保存为 HDF5
)
qpi_maps = result["qpi_layers"]  # shape: (n_energies, nq, nq)
metadata = result["metadata"]

# Born 近似（可指定散射势 V 与实空间掩码）
born = BornQPI(ham, nk=256, eta=0.005)
result = born.calculate(energy_range=0.5, V=V)
```

### 4. 自旋磁化率（Wang 2012）

```python
from stm_data_processing.dft.wannier90.mlwf_susceptibility import (
    SusceptibilityCalculator_wang2012,
)

sus = SusceptibilityCalculator_wang2012(ham, nk=256, eta=0.005)
sus.set_orbital_selection(minit=m_i, mfin=m_f)  # 可选：轨道选择矩阵
result = sus.calculate(omega_limit=1.0, resolution=0.01, q_range=(-0.5, 0.5))
```

### 5. 高精度晶格运算

```python
from stm_data_processing.utils.lattice import LATTICE

lat = LATTICE(bvecs=bvecs)            # 或 avecs=...，内部用 mpmath 高精度计算
lat.set_precision(100)                # 调整计算精度（十进制有效位数）
print(lat.a1, lat.b1, lat.volume)
lat.rotate(30)                        # 旋转晶格
lat.verify_consistency()              # 一致性校验
```

### 6. BTK 超导隧道谱

```python
from stm_data_processing.utils.btk import BTK

btk = BTK(Delta=1.5, Z=0.3)           # 超导能隙 Δ、势垒强度 Z
dI_dV = btk.spectrum(E_min=-5, E_max=5, n_points=500, T=0)  # 零温
dI_dV_T = btk.spectrum(T=4.2)         # 有限温度
```

### 7. OpenMX 输出解析

```python
from stm_data_processing.dft.openmx.parser import OpenMX
from stm_data_processing.dft.openmx.diff_gcube import diff_cube_files

mx = OpenMX()
bvecs = mx.read_bvecs_from_out("system.out")
diff_cube_files("before.cube", "after.cube", "diff.cube")  # 势差分析
```

## 模块结构

```
src/stm_data_processing/
├── config.py          # CPU/GPU 后端统一管理（BACKEND / get_xp / set_backend）
├── dft/
│   ├── openmx/        # OpenMX：parser / band / dos / unfolding / diff_gcube
│   └── wannier90/     # Wannier90：mlwf_hamiltonian / mlwf_gk / mlwf_ek2d /
│                      #            mlwf_susceptibility / bare_lindhard
├── stm/               # STM 端：qpi_jdos / qpi_born / qpi_tmat / vortex_num /
│                      #         preview_plot
├── io/                # IO 层：nanonis_loader / w90hr_loader / ek2d_io /
│                      #        lattice_loader / qpi_io / susceptibility_io
└── utils/             # lattice（高精度）/ lattice_operations / lindhard1dfree /
                       # btk / miscellaneous / monitor / plot_funcs /
                       # nanonis_ppt_generator（自动 PPT 报告）
```

## 文档

详细的模块接口文档见 [`docs/stm_data_processing/`](docs/stm_data_processing/)（接口表、数学公式、示例与接口对齐检查清单）。

## 开发

```bash
uv sync --extra dev
uv run ruff check .
```

## 许可证

MIT License
