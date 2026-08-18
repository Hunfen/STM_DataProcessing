# stm_data_processing 模块文档索引

本目录收录 `stm_data_processing` 各模块的接口文档。文档统一采用以下结构：模块概述、核心类/函数、参数与返回结构表、数学公式（适用时）、使用示例、依赖项、错误处理、接口对齐检查清单。

## 顶层基础设施

| 模块 | 文档 |
|------|------|
| `config.py`（CPU/GPU 后端管理） | [config.md](config.md) |
| `logger.py`（简化日志库） | [logger.md](logger.md) |

## dft / openmx

| 模块 | 文档 |
|------|------|
| `band.py`（能带解析） | [dft/openmx/band.md](dft/openmx/band.md) |
| `dos.py`（DOS/PDOS 加载） | [dft/openmx/dos.md](dft/openmx/dos.md) |
| `parser.py`（OpenMX 输出解析） | [dft/openmx/parser.md](dft/openmx/parser.md) |
| `unfolding.py`（能带展开/谱函数） | [dft/openmx/unfolding.md](dft/openmx/unfolding.md) |
| `diff_gcube.py`（Gaussian cube 差分） | [dft/openmx/diff_gcube.md](dft/openmx/diff_gcube.md) |

## dft / wannier90

| 模块 | 文档 |
|------|------|
| `mlwf_hamiltonian.py` | [dft/wannier90/mlwf_hamiltonian.md](dft/wannier90/mlwf_hamiltonian.md) |
| `mlwf_gk.py` | [dft/wannier90/mlwf_gk.md](dft/wannier90/mlwf_gk.md) |
| `mlwf_ek2d.py` | [dft/wannier90/mlwf_ek2d.md](dft/wannier90/mlwf_ek2d.md) |
| `mlwf_susceptibility.py` | [dft/wannier90/mlwf_susceptibility.md](dft/wannier90/mlwf_susceptibility.md) |
| `bare_lindhard.py`（算法说明） | [dft/wannier90/static_lindhard_overlap.md](dft/wannier90/static_lindhard_overlap.md) |

## io

| 模块 | 文档 |
|------|------|
| `nanonis_loader.py` | [../STM_README.rst](../STM_README.rst) |
| `w90hr_loader.py` | [io/w90hr_loader.md](io/w90hr_loader.md) |
| `ek2d_io.py` | [io/ek2d_io.md](io/ek2d_io.md) |
| `lattice_loader.py` | [io/lattice_loader.md](io/lattice_loader.md) |
| `qpi_io.py` | [io/qpi_io.md](io/qpi_io.md) |
| `susceptibility_io.py` | [io/susceptibility_io.md](io/susceptibility_io.md) |

## stm

| 模块 | 文档 |
|------|------|
| `qpi_jdos.py` | [stm/qpi_jdos.md](stm/qpi_jdos.md) |
| `qpi_born.py` | [stm/qpi_born.md](stm/qpi_born.md) |
| `qpi_tmat.py` | [stm/qpi_tmat.md](stm/qpi_tmat.md) |
| `vortex_num.py` | [stm/vortex_num.md](stm/vortex_num.md) |
| `preview_plot.py` | [stm/preview_plot.md](stm/preview_plot.md) |
| AutoPPT 脚本（`utils/AutoPPt_winnew_modified.py`） | [stm/auto_powerpoint.md](stm/auto_powerpoint.md) |

## utils

| 模块 | 文档 |
|------|------|
| `lattice.py` | [utils/lattice.md](utils/lattice.md) |
| `lattice_operations.py` | [utils/lattice_operations.md](utils/lattice_operations.md) |
| `lindhard1dfree.py` | [utils/lindhard1dfree.md](utils/lindhard1dfree.md) |
| `btk.py` | [utils/btk.md](utils/btk.md) |
| `miscellaneous.py` | [utils/miscellaneous.md](utils/miscellaneous.md) |
| `monitor.py` | [utils/monitor.md](utils/monitor.md) |
| `plot_funcs.py` | [utils/plot_funcs.md](utils/plot_funcs.md) |
| `extract_k_type.py` | [utils/extract_k_type.md](utils/extract_k_type.md) |
| `nanonis_ppt_generator.py` | [utils/nanonis_ppt_generator.md](utils/nanonis_ppt_generator.md) |
