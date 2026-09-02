# src 代码 Bug 审查报告（2026-08-18）

> 审查范围：`src/stm_data_processing/` 全部 36 个 Python 文件（约 12,000 行）。
> 审查方式：7 组并行静态审查 + 关键发现逐一运行时/数值验证（`.venv` Python 3.12 实测）。
> 验证标记：
> - ✅ = 已通过运行时或数值实验复现
> - 🧪 = 通过代码推导 + 局部数值实验确认
> - ⚠️ = 存疑（suspicious），需真实数据进一步确认

## 修复状态（2026-09-01）

> 本报告 **10 个 High 级 bug（H1–H10）已全部修复**，并完成 CPU 计算路径性能优化；改动经 AgentTeams（`deepseek-v4-flash` 编码 + `deepseek-v4-pro` 审查，6/6 审查 PASS）验证，已提交并推送到 `master`（commit `d5abd11`）。
>
> - 涉及 10 个源文件：`qpi_born.py`、`qpi_jdos.py`、`mlwf_ek2d.py`、`bare_lindhard.py`、`mlwf_susceptibility.py`、`lattice_operations.py`、`plot_funcs.py`、`AutoPPt_winnew_modified.py`、`nanonis_ppt_generator.py`、`diff_gcube.py`。
> - CPU 性能：BornQPI 的 CPU QPI 由 `np.roll` 双循环改为 FFT 相关定理（nk=32 提速 58×、nk=64 提速 207×）；`bare_lindhard` 向量化（nk=16 提速 158×）；`mlwf_susceptibility` 逆 FFT 方向修正并复用计划；`mlwf_ek2d` 本征分解加缓存。
> - Medium（19）项已于同日全部处理完毕（见下节）；Low（25）项已于 2026-09-02 全部处理完毕（见下节），并顺带清零 `extract_k_type.py` 的 22 条 ruff cosmetic 基线（`ruff check src` 全树 0 错误）。

### Medium 全部修复（2026-09-01）

> 本报告 **19 个 Medium 级 bug（M1–M19）已全部处理完毕**（M15 经真实数据验证确认无需改变行为，仅做健壮性加固）。改动经 AgentTeams `stm-medium-bugfix`（3 名 `deepseek-v4-flash` 编码成员 + 1 名 `deepseek-v4-pro` 评审成员）执行：6 轮评审 + 2 轮 repair 复审全部 PASS，集成验收通过。
>
> - 涉及 12 个源文件：`nanonis_loader.py`、`bare_lindhard.py`、`mlwf_susceptibility.py`、`dos.py`、`band.py`、`parser.py`、`AutoPPt_winnew_modified.py`、`plot_funcs.py`、`qpi_tmat.py`、`monitor.py`、`config.py`、`lattice_loader.py`。
> - 新增 8 个回归脚本（`scripts/regression/`），全部 exit 0：`check_nanonis_3ds.py`(8)、`check_nanonis_sxm.py`(6)、`check_3ds_real_data.py`(2)、`check_bare_lindhard.py`(6)、`check_mlwf_susceptibility.py`(4)、`check_openmx_parsers.py`(7)、`check_ppt_helpers.py`(8)、`check_core_misc.py`(7)。
> - 集成验收：`ruff check src` 仍为 22 条既有 cosmetic 基线（全部在 `extract_k_type.py`）无新增；39 个模块导入冒烟 38 通过，唯一失败项为既有 L20（`AutoPPt_winnew_modified` 模块级 `input()`）。

#### 评审抓出的两个回归（已闭环）

评审由 `deepseek-v4-pro` 独立执行（读代码 + 重跑脚本 + 真实文件复现），两轮 `needs_revision`：

1. **M3 首版修复引入回归**：守卫写成 `if "Bias Spectroscopy" in self._raw_header:`，而 raw 键恒为 `"Bias Spectroscopy>..."`，判断恒为 False，使 `MultiLine Settings → DataFrame` 转换变成死代码（真实多段 `.dat` 实测由 `(6,5)` DataFrame 退化为 `str`）。已改为检查重构后的字典 `if "Bias Spectroscopy" in header:`；**`_reform_3ds_header` 中同款守卫本来就是错的（本报告 M3 原文误将其当作正确范例），一并修正**。
2. **M11 首版未真正修复**：`od_data[1:]` 仍在 `saw_initial_state` 分支中执行，但 `O_D=` 行只出现在各 Cycle、Initial State 块无该行，真实 `.wout` 会触发新加的一致性检查而崩溃；且其回归 fixture 人为给 Initial State 补了 O_D 行掩盖了缺陷。已改为三标志解耦（`saw_initial_state` / `in_initial_state_block` / `saw_initial_state_od`），同时兼容"旧版 W90 确实打印 Initial State O_D"的变体。

#### 行为变更（调用方需注意）

| 变更 | 影响 |
|---|---|
| `bare_lindhard` 分母统一为标准约定 `ε_m(k) − ε_n(k+q) + iη` | **Re χ₀ 相对旧版本整体变号**；与旧结果对比时需注意 |
| `mlwf_susceptibility` 能量权重改为 `dε=\|ω\|/(n_eps−1)` | `\|ω\|/resolution` 为整数时**逐位无变化**；非整数时幅度校正（示例 8.11%） |
| `dos['pdos'][atom]['s']` 由（本已失效的）DataFrame 改为 dict | 仓库内无消费者；外部脚本若按旧形状取 PDOS 需同步 |
| `.Band` 返回值新增 `nspin` 键；nspin=2 时 `bands` 为 `(nspin, nk, nband)` | nspin=1 保持 `(nk, nband)` 向后兼容 |
| `mlwf_susceptibility` 在 `n_eps == 1` 时抛 `ValueError` | 旧行为为静默产出无意义结果 |
| `TmatQPI` 由空壳 stub 变为可用（CPU 单杂质 s 波 T 矩阵） | GPU 分支抛显式 `NotImplementedError`，接口已就位 |

#### 真实数据验证结论（用户提供文件）

- **M15 无需改变行为**：6 个真实 `.sxm`（topo0002/0007/0011/0013/0019/0020）的布局均为 header 结束后 `\n\n\n\x1a\x04` + 数据，旧启发式（跳 2 行 + seek 2 字节）与按 `\x1a\x04` 标记定位**落在同一字节（4539）**；新旧实现输出逐元素完全一致（2,621,440 floats，`equal_nan=True`），与 nanonispy 5 通道对比 max abs diff = `0.000e+00`。改动仅为：标记定位 + 找不到时回退旧逻辑并告警 + 数据量下限告警。
- **3DS loader**：`2025-07-09/Grid Spectroscopy002.3ds` 为 520×1 一维 line（`grid (520,14,401)`，无 NaN）；`2025-10-24/Grid Spectroscopy008.3ds` 声明 50×50 但**原始文件仅记录 1940/2500 像素**（10,910,560 floats = 1940×5624，测量中断），修复后按 NaN 正确补齐为 `(2500,14,401)`（NaN 比例 0.2240）并告警，而非越界崩溃。
- **带 MultiLine Settings 的真实文件**是 `2025-06-25/Grid Spectroscopy002.3ds`（2025-07-09 的同名文件无此键），持久断言已按实际含键文件固化。

#### 后续轮次（2026-09-01 晚，用户复查图像后追加）

用户复查预览图时发现 `topo0002.sxm` 的形貌图整幅空白，由此定位到一个**不在原报告 54 条内的新缺陷**，并顺带修掉两个此前记录的既有隐患。

##### N1. `subtractMeanPlane` 被 NaN 污染，中断扫描图像整幅变 NaN 🔴（已修复 2026-09-01）

- **位置**：`utils/plot_funcs.py` `subtractMeanPlane`（M9 修复后 `AutoPPt_winnew_modified.py` 已复用该实现，两处同时受影响）
- **现象**：Nanonis 对**未扫完**的 SXM 区域写入 NaN（真实文件 `2025-07-09/topo0002.sxm`：SCAN_PIXELS 512×512、字节完整 2,621,440 floats，但每通道 finite 仅 **0.623**，约 319/512 行）。`subtractMeanPlane` 把 NaN 一起喂进 `lstsq`，得到 NaN 平面系数 → **输出 finite 比例 0.000**，整幅图变 NaN、绘图全空白。loader 行为正确（按 header 建数组 + 顺序填值），缺陷完全在绘图侧。影响 PPT 批处理的形貌总览盒与所有 map 显示路径。
- **修复**：平面拟合只用 finite 像素（`np.isfinite` 掩码筛选设计矩阵与目标向量），平面对全图求值后相减，**NaN 位置原样保留**；finite 点 < 3 时返回原数组副本并 warning。新增 `finite_range()`（nan 安全 vmin/vmax）与 `topo_colormap()`（NaN 显示为不透明 `#d9d9d9`），接入 `plot_sxm_topo`/`plot_map_bias`/`plot_qpi_bias`/`plot_map_current_bias`/`plot_sts`/`plot_linecut` 及 AutoPPt 的 ShowMap/ShowMapI/SXM/PlotSXM/STS/Linecut/inmap。
- **验证**：全 finite 输入与修复前**位级一致**（评审员独立复现，含 512×512）；30% NaN 平面的 finite 区域残差 8.9e-16、NaN 掩码逐点不变；真实中断扫描探针 finite 0.381→0.381（修复前 0.0）。`check_ppt_helpers.py` 12/12。

##### N2 / N3. loader 两个既有隐患（已修复 2026-09-01）

- **N2 `.data` 原地改写 `._raw_data`**：`_reform_sxm_data` 以 view reshape 后对 backward 行原地 `fliplr`，取过 `.data` 之后 `.raw_data` 返回的已是被翻转的数据。修复：reshape 后显式 `copy()` 再翻转（一次 payload 大小拷贝，docstring 已注明）。验证：6 个真实文件访问前后快照逐元素相等、二次访问幂等、flip 语义不变（nanonispy diff 仍为 0）。
- **N3 `.channels` 引号残留**：`header` 未解析时走 raw 回退分支，通道名带首尾双引号。修复：属性先 `_ensure_header_parsed()`，raw 分支再 `strip('"')` 兜底。验证：两种访问顺序得到相同且无引号的 14 个通道名。

##### 3DS 验证按用户口径重做

- dI/dV 通道由 `LI Demod 1 X (A)` 改为用户指定的 **`DSP 7280 Y (%)`**（forward，动态 `channels.index` 取索引，非硬编码）。
- 一维：`2025-07-09/Grid Spectroscopy002.3ds`（520×1、401 点）→ dI/dV 对 bias 的 colormap。
- 二维：改用 `2025-07-14/Grid Spectroscopy001.3ds`（**90×400、Points 7**、14 通道，全部像素已采集）→ 7 个 bias 切片（10.0/6.7/3.3/0.0/−3.3/−6.7/−10.0 mV）各出一张 90×400 的 dI/dV map + 多子图总览。基于 `2025-10-24/Grid Spectroscopy008.3ds`（未扫完）的旧图已删除。
- 预览图统一改为 nan 安全色标（`nanpercentile(2,98)`）+ NaN 区中性灰 `#808080` + 标题标注完整度（topo0002 标 `finite 62.3% (319/512 rows)`）。

### Low 全部修复（2026-09-02）

> 本报告 **25 个 Low 级 bug（L1–L25）已全部处理完毕**，并顺带清零 `extract_k_type.py` 的 22 条 ruff cosmetic 基线（`ruff check src` 全树 **0 错误**）。改动经 AgentTeams `stm-low-risk-cleanup`（4 名 `deepseek-v4-flash` 编码成员 + 2 名 `deepseek-v4-pro` 评审成员 + 1 名集成成员）执行：10 个实现任务 + 10 轮独立评审**全部 PASS**（0 轮 needs_revision/reject），集成验收通过。
>
> - 涉及 18 个文件：`qpi_born.py`、`qpi_io.py`、`w90hr_loader.py`、`btk.py`、`miscellaneous.py`、`vortex_num.py`、`mlwf_susceptibility.py`、`mlwf_ek2d.py`、`bare_lindhard.py`、`lattice_operations.py`、`AutoPPt_winnew_modified.py`、`nanonis_ppt_generator.py`、`diff_gcube.py`、`parser.py`、`dos.py`、`unfolding.py`、`extract_k_type.py`、`scripts/regression/check_nanonis_sxm.py`。
> - 剩余风险项同步处理：② `bare_lindhard` 奇数 nk 半格偏差已修复（q 网格改 `fftshift(fftfreq(nk))`，偶 nk 不变、奇 nk 修正）；④ M15 marker 回退路径补真实数据回归（6 个真实 `.sxm` 剥离标记副本，回退与 marker 路径逐元素一致）；⑤ `check_nanonis_sxm` 像素自检判据加固（`std > 阈值` 等鲁棒判据，去掉 afmhot 无中性灰假设）。
> - 集成验收：8 套回归脚本全部 exit 0；`ruff check src` "All checks passed!"（22 条基线清零）；39/39 模块导入冒烟通过（含 qpi_tmat、pyfftw 回退、L20 `__main__` 守卫无阻塞）。
> - **TmatQPI GPU 分支按用户要求本轮明确不做**（`qpi_tmat.py` 未改动，接口保留）。

#### 剩余风险 / 未处理项

1. `TmatQPI` GPU 分支未实现（显式 `NotImplementedError`，接口保留；**用户明确本轮不做**）。
2. ~~`bare_lindhard` 奇数 nk 半格偏差~~ ✅ 已修复（2026-09-02，q 网格改 `fftshift(fftfreq(nk))`）。
3. ~~L20 模块级 `input()` 与 22 条 ruff cosmetic 基线~~ ✅ 已处理（2026-09-02：L20 移入 `__main__` 守卫；基线清零，`ruff check src` 0 错误）。
4. ~~M15 marker 回退路径仅合成测试覆盖~~ ✅ 已补真实数据回归（2026-09-02，6 真实 `.sxm` 剥离标记副本验证）。
5. ~~`_check_png_content` 像素自检偏脆弱~~ ✅ 已加固（2026-09-02，`std > 阈值` 等鲁棒判据）。

已记录的非阻塞备注（不影响交付，建议后续真实数据校验）：① OpenMX f 轨道列名顺序与规范 m 序有出入（仓库无真实 f 数据，消费方按 element/atom_index 求和、顺序无关）；② AutoPPt/nanonis_ppt_generator 的 L17 旋转符号是物理约定依赖，与现有 origin 逻辑自洽；③ `check_nanonis_sxm` 的 `finite_frac` 判据对 near-gray 像素为保守近似（最差 topo0019 diff 0.227，距 0.3 阈值余量 ~0.073），主判据（非白/std）裕量充足；④ `extract_k_type.py` `__main__` 保留 3 处中文 CLI 输出（既有、不触发 ruff，语言策略的面向用户输出合理边界）。

## 一、统计总览

| 严重程度 | 数量 | 说明 |
|---------|------|------|
| 🔴 High | 10 | 崩溃 / 输出结果错误 / 模块完全不可用（✅ 已修复 2026-09-01） |
| 🟡 Medium | 19 | 特定输入或路径下结果错误、数据错配（✅ 已全部修复 2026-09-01） |
| 🟢 Low | 25 | 隐患、资源泄漏、文档与实现不一致（✅ 已全部修复 2026-09-02） |
| **合计** | **54** | |

**最严重的结论**：README 快速开始的三个核心 QPI 类中，`JDOSQPI` 与 `BornQPI` 在当前代码状态下**构造/调用即崩溃**；`bare_lindhard.py` 的 CPU 路径多重错误叠加后默认参数下结果**恒为零**；`mlwf_susceptibility.py` 默认 CPU 路径把逆 FFT 做成了正 FFT，**结果量级差 nk² 倍**。

---

## 二、High（10）

### H1. `BornQPI` 引用不存在的属性 `hk_grid_cpu` / `hk_grid_gpu` ✅（已修复 2026-09-01）
- **位置**：`stm/qpi_born.py:148`、`qpi_born.py:189`（定义在 `:68`、`:76` 的 `self.hk_grid`）
- **现象**：`__init__` 只赋值 `self.hk_grid`，CPU/GPU 计算函数却分别读取 `self.hk_grid_cpu` / `self.hk_grid_gpu`，全仓库无任何地方定义这两个属性。`calculate()` 在 CPU、GPU 后端均抛 `AttributeError`（已用最小 mock 哈密顿量实测复现）。
- **修复**：统一属性名（如 CPU 分支 `self.hk_grid_cpu`、GPU 分支 `self.hk_grid_gpu`），或让两个计算函数直接使用 `self.hk_grid`。

### H2. `JDOSQPI` 调用 `EK2DCalculator` 中不存在的方法 ✅（已修复 2026-09-01）
- **位置**：`stm/qpi_jdos.py:103`（`_compute_ek2d_cuda`）、`:108`（`_compute_ek2d`）；实际方法名为 `_compute_eigen` / `_compute_eigen_cuda`（`dft/wannier90/mlwf_ek2d.py:70`、`:109`）
- **现象**：`JDOSQPI.__init__` 在对角化阶段即抛 `AttributeError`（已实测复现）。即使改名还有第二层不匹配：`_compute_eigen*` 返回 `(evals, evecs)` 二元组且 evals 形状为 `(nk², nw)`，而调用方期待 `(nw, nk, nk)` 再 `transpose((1,2,0))`；且 `_compute_eigen` 依赖的 `self.out_eigvec_flag` 只在 `calculate()/calculate_eigh()` 里设置、`__init__` 未初始化。
- **修复**：改回正确的调用（并 reshape/transpose），或在 `EK2DCalculator` 补回 `_compute_ek2d/_compute_ek2d_cuda` 接口；在 `__init__` 初始化 `out_eigvec_flag`。

### H3. pyFFTW "IFFT" 计划实际是正向 FFT ✅（已修复 2026-09-01）
- **位置**：`dft/wannier90/mlwf_susceptibility.py:400-406`（计划创建）、`:296-298`（"ifft" 计划）、`:347`（使用处）
- **现象**：`_init_fftw_plan` 未传 `direction`，`pyfftw.FFTW` 默认 `FFTW_FORWARD`，因此名为 `ifft_conv` 的计划做的是正向 FFT。数值实验：默认计划与 `np.fft.ifftn` 相差因子 64（nk²，nk=8 时），而 `direction="FFTW_BACKWARD"` 与 `ifftn` 一致到机器精度。pyfftw 是硬依赖且无条件导入，**默认 CPU 路径必然命中此 bug**（numpy 回退分支是死代码，见 L8）。对 nk=256 结果差 65536 倍量级。
- **修复**：`_init_fftw_plan` 增加 `direction` 参数；逆变换计划传 `direction="FFTW_BACKWARD"`。

### H4. `bare_lindhard` CPU 路径多重错误，结果恒为零 🧪（已修复 2026-09-01）
- **位置**：`dft/wannier90/bare_lindhard.py`
  1. **权重用同 k 本征矢**（`:152-158` on-the-fly、`:116-120` 预计算）：`overlap = Σ_a u_{a,m}(k) conj(u_{a,n}(k)) = δ_mn`（本征矢正交），而物理上应为 `⟨u_{m,k}|u_{n,k+q}⟩`（GPU 路径 `:351-356` 是对的）；
  2. **对角项被跳过**（`:144-145`）：`f_m - f_n` 在同 k 下对 m=n 恒为 0，`continue` 丢弃全部带内项（带内贡献应为 `f(ε_m(k)) − f(ε_m(k+q))`，非零）；
  3. **k+q 索引广播塌缩**（`:166-171`）：`eps_n[k1q_idx, k2q_idx[:, None]]` 的高级索引把 `(nk, n_q1)` 与 `(nk2, 1)` 按元素配对，输出是 2D `(nk, n_q1)`，k2 维被行号 i 取代（数值实验确认 `eps_n_shift[i,q1] = ε[(i+q1)%nk, i]`）。
- **现象**：错误 1+2 叠加使 m≠n 权重为 0（全轨道选择下本征矢正交）、m=n 被跳过 → **默认参数下 CPU 路径返回的 χ(q) 恒为 0**（轨道子集选择时权重不全为零，但仍是同 k 的错误量）；错误 3 使即便修复 1+2 仍会算错。GPU 路径无这些问题 → 双后端结果严重不一致。
- **修复**：权重改为对 (k, k+q) 本征矢求重叠；删除/修正对角项跳过逻辑；用 3D 索引 `eps_n[k1q_idx[:,:,None], (k2_idx+ q2)%nk2]` 构造 `(nk, nk, n_q1)`。

### H5. 超胞倒格基变换转置错误 🧪（已修复 2026-09-01）
- **位置**：`utils/lattice_operations.py:126` `b_p = b @ np.linalg.inv(t).T`
- **现象**：`LATTICE` 类文档与 `supercell()` 采用的约定是 `A_super = Mᵀ @ A_old`，对应的倒格变换应为 `B_super = inv(M) @ B`。代码写的是 `B @ inv(M)ᵀ`，隐含约定 `A_super = A @ M`，与 `lattice.py` 冲突。数值实验：`lat.supercell(M).bvecs == inv(M) @ B` 为 True、`== B @ inv(M).T` 为 False；对非对称变换矩阵（如 `[[2,1],[0,1]]`）两式给出的 Bragg 点完全不同。
- **修复**：`b_p = np.linalg.inv(t) @ b`。

### H6. `get_1stbz_vertices` 在常见取向下直接抛异常 ✅（已修复 2026-09-01）
- **位置**：`utils/lattice_operations.py:302`
- **现象**：`extend_vecs_c3(include_neg=False)` 在 b1、b2 方位角相差 120° 的取向（石墨烯惯例，实测 b1=-30°、b2=90°）下只返回 3 个唯一矢量，随后 `gs.shape[1] != 6` 抛 `ValueError`。同一物理晶格的另一种等价取向（差 60°）却正常。
- **修复**：改为 `include_neg=True`（两种取向下都恰好得到 6 个最近邻倒格矢）。

### H7. `get_divider` 子串匹配顺序错误，d10/d100 全部按 d1 处理 ✅（已修复 2026-09-01）
- **位置**：`utils/plot_funcs.py:13-22`
- **现象**：`"d1" in name` 是 `"d10"`、`"d100"` 的子串，永远先命中 `return 1`，后两个分支为死代码。所有依赖它的绘图函数（`plot_map_bias`、`plot_qpi_bias`、`plot_map_current_bias`、`plot_sts`、`plot_linecut`，及 `nanonis_ppt_generator.py` 的 `:230/:316/:367`）对 d10/d100 高增益数据偏压换算错误 10/100 倍。对照：`AutoPPt_winnew_modified.py:1079-1084` 的内联版本用三条独立 `if` 覆盖赋值，恰好正确。
- **修复**：从长到短匹配：`if "d100" in name: return 100; if "d10" in name: return 10; return 1`。

### H8. 无效 SXM 清理逻辑死代码，坏文件导致整个脚本崩溃 ⚠️（已修复 2026-09-01）
- **位置**：`utils/AutoPPt_winnew_modified.py:862-869`（清理循环）+ `:406`（`SXM()` 用 `except Exception: return` 吞掉一切异常）
- **现象**：清理循环期望 `SXM()` 对坏文件抛 `ValueError` 以便 `del SxmFiles[i]`，但 `SXM()` 永不抛异常，坏文件残留，随后 `:912` 的 `nap.read.Scan(topopath)`（无保护）直接抛异常，`prs.save()` 前整个脚本崩溃、已生成内容全部丢失。
- **修复**：`SXM()` 读失败时重新抛出（或返回状态），并把 `:912` 的读取包进 try。

### H9. `nanonis_ppt_generator.py` 无法作为包模块导入 ✅（已修复 2026-09-01）
- **位置**：`utils/nanonis_ppt_generator.py:15` `from plot_funcs import (...)`
- **现象**：裸导入依赖 `utils` 目录在 `sys.path` 上；`import stm_data_processing.utils.nanonis_ppt_generator` 直接抛 `ModuleNotFoundError: No module named 'plot_funcs'`（导入测试复现）。此外该文件无 `__main__` 保护，模块级 `input()` 交互提示（`:38-69`）使其即便修好导入也无法作为库使用。
- **修复**：改为相对导入 `from .plot_funcs import ...`（或 `from stm_data_processing.utils.plot_funcs import ...`），并将交互逻辑移入 `if __name__ == "__main__":`。

### H10. `diff_gcube` 写出的原子坐标整体错位、z 坐标丢失 ✅（已修复 2026-09-01）
- **位置**：`dft/openmx/diff_gcube.py:107-111`（与 `:60` 的读取解析对照）
- **现象**：`read` 把原子行解析为 5 列 `[Z, charge, x, y, z]`；`write` 却输出 `f"{atom[0]} {0.0} {atom[1]} {atom[2]} {atom[3]}"`，即 `Z, 0.0, charge, x, y`：电荷被硬编码为 0，随后 charge→x、x→y、y→z 整体错位一位，真正的 z（`atom[4]`）被丢弃。任何读取输出 cube 原子块的工具（VESTA/ASE/VMD）都会得到错误的原子位置（网格差分数据本身正确）。
- **修复**：`f"{atom[0]:5.0f} {atom[1]:12.6f} {atom[2]:12.6f} {atom[3]:12.6f} {atom[4]:12.6f}\n"`。

---

## 三、Medium（19）

### M1. `TmatQPI` 方法缺少 `self` ✅（已修复 2026-09-01，并实现 CPU 单杂质 T 矩阵 QPI）
- **位置**：`stm/qpi_tmat.py:26`、`:32`
- **现象**：`def _compute_tmat():` / `def calculate():` 定义在类体却无 `self`，实例调用抛 `TypeError`（已实测复现）。类本身是占位 stub。
- **修复**：补 `self` 参数并实现逻辑。

### M2. OpenMX PDOS 加载对两种目录布局均失效 ✅（已修复 2026-09-01）
- **位置**：`dft/openmx/dos.py:66-103`
- **现象**：两处不一致叠加，任何布局下 `pdos` 都是空或错误的：
  1. 代码 `sorted(dos_dir.glob("atom*"))` 假设 `atomN/` 子目录布局，但 OpenMX 4.0 手册（§23）显示 DosMain 产出的是**扁平带点文件名**（如 `*.PDOS.Tetrahedron.atom1.s1`）→ 找不到任何目录，`pdos` 静默为空；
  2. 即使按模块自身 docstring 的子目录布局（文件名 `s1`/`p1`/`d1`），分类正则 `re.search(r"\.s\d$", name)` 要求点前缀，无点文件名全部落入 `else` 被误标 `total`，且 `name[-2:]` 对 `p10` 等两位数索引生成错误键。实测：`s1/p1/d1/p10` 全部归类为 `total`。
- **修复**：按真实扁平文件名 `*.atomN.orbital` 归类（原子序号与轨道从文件名提取），并同步修正 docstring；轨道键用完整序号（`name[1:]` 等）。

### M3. `.dat` header 重组对非 Bias Spectroscopy 文件 KeyError ⚠️（已修复 2026-09-01，含 3ds 同款错误守卫）
- **位置**：`io/nanonis_loader.py:414-431`
- **现象**：`_reform_dat_header` 无条件遍历 `header["Bias Spectroscopy"]`；`.dat` 不仅用于 Bias Spectroscopy（也有 Z 谱等），无该模块的文件在首次访问 `header` 时抛 `KeyError`。同文件 `_reform_3ds_header:542` 有 `if "Bias Spectroscopy" in self._raw_header` 保护，`.dat` 版本缺失。
- **修复**：补上与 3ds 版本相同的保护。

### M4. `bare_lindhard` q 坐标网格与 fftshift 后数据错位 ✅（已修复 2026-09-01）
- **位置**：`dft/wannier90/bare_lindhard.py:452-453`
- **现象**：`linspace(-0.5, 0.5, nk, endpoint=False)` 本身已以 q=0 为中心（q=0 在 `nk//2`），再 `fftshift` 一次把 q=0 移到索引 0；而数据 `chi_q` 在 `:448` 已 fftshift（q=-0.5 在索引 0）。数值实验（nk=4）确认两者错位半个 BZ，返回的 `q1_grid/q2_grid/qx_grid/qy_grid` 与数据逐像素错标。
- **修复**：删除对 `q_vals` 的 fftshift（linspace 已居中）。

### M5. `bare_lindhard` 分母符号约定翻转 Re χ0 ⚠️（已修复 2026-09-01，统一标准约定）
- **位置**：`dft/wannier90/bare_lindhard.py:180`（CPU）、`:368-370`（GPU）
- **现象**：分母为 `ε_n(k+q) − ε_m(k) + iη`，是标准 Lindhard 分母 `ε_m(k) − ε_n(k+q) + iη` 的相反数（且无 ω）。虚部不变、实部符号翻转。若意图是标准约定，Re χ0 符号错误；若是作者自选约定，应在文档中注明。
- **修复**：确认约定后统一（或改分母为 `eps_m − eps_n_shift + iη` 并注明）。

### M6. 磁化率 CUDA 路径忽略轨道选择矩阵 ⚠️（已修复 2026-09-01）
- **位置**：`dft/wannier90/mlwf_susceptibility.py:182-213`（CPU 路径 `:315`、`:323` 有 `minit/mfin` 的 einsum 投影）
- **现象**：CPU 路径应用 `minit`/`mfin` 投影，CUDA 路径 reshape 后直接 FFT，从不使用 `self._minit/_mfin` → 非单位选择矩阵时 GPU 结果静默错误（等价于 identity）。
- **修复**：CUDA 路径 FFT 前做同样的 einsum 投影。

### M7. 磁化率能量积分权重与实际网格间距不一致 ⚠️（已修复 2026-09-01）
- **位置**：`dft/wannier90/mlwf_susceptibility.py:167/231/252/368`
- **现象**：`n_eps = round(|ω|/resolution)+1` 且 `eps = linspace(-|ω|, 0, n_eps)`，实际 dε = |ω|/(n_eps−1)；但最终归一化乘的是 `-|resolution|/(2π)`。当 |ω|/resolution 非整数时两者差几个百分点，积分幅度失真。
- **修复**：统一用 `dε = np.abs(omega_limit)/(n_eps-1)` 做权重。

### M8. STS / Linecut 段落形貌偏压用了残留变量 ✅（已修复 2026-09-01）
- **位置**：`utils/AutoPPt_winnew_modified.py:1085`、`:1252`
- **现象**：两处位于 `if topopath != "Topography Not Found"` 分支，本应使用刚读出的 `raw_data_topo`（`:1062`/`:1229`），却写成 `raw_data.header["bias"]` —— `raw_data` 是形貌循环（`:912`）残留的**最后一个 SXM** 对象 → STS/Linecut 页标注的形貌 `Vs=` 错误。
- **修复**：改为 `raw_data_topo.header["bias"]`。

### M9. `subtractMeanPlane` 扁平索引步长错误 ✅（已修复 2026-09-01，改为复用 plot_funcs 实现）
- **位置**：`utils/AutoPPt_winnew_modified.py:769-786`（`coordMatrix[i*xdim+j]`、`zVector[i*xdim+j]`）
- **现象**：`matrix.shape = (xdim, ydim)` 行主序扁平下标应为 `i*ydim + j`，代码用 `i*xdim + j`；非方形图像（如 10×20）出现下标碰撞与空洞，拟合平面错误，且未命中的元素保持 0。该函数用于形貌总览盒。`plot_funcs.py:64-72` 的重写版已修正。
- **修复**：改为 `i*ydim + j`，或直接复用 `plot_funcs.subtractMeanPlane`。

### M10. `load_wout_file` Initial State 剔除条件用错变量 ⚠️（已修复 2026-09-01）
- **位置**：`utils/monitor.py:233-236`
- **现象**：`if len(spreads_data) > 1 and current_cycle == -1` —— `current_cycle` 保存的是**最后一个** cycle 编号（正常文件为 N≥0），条件恒为 False → Initial State 行从不剔除，`wannierise_spreads` 比文档声称的 `(num_cycle, num_wann, 1)` 多一行。
- **修复**：维护独立布尔 `saw_initial_state` 标志，用它决定是否剔首行。

### M11. `load_wout_file` od_data 无条件丢首项，与展宽数组错位 ⚠️（已修复 2026-09-01，二轮修复）
- **位置**：`utils/monitor.py:239-240`
- **现象**：`od_data` 无条件 `[1:]`，注释假设首项属 Initial State；但 Wannier90 的 `O_D=` 行只出现在各 Cycle（`:216` 要求 `<-- DLTA`），首项实为 Cycle 1。与 M10 叠加：spreads 多 1 行、od 少 1 行，二者长度错位 2，按行配对的下游代码全部错配。
- **修复**：与 spreads 共用 `saw_initial_state` 标志；仅在确认首项属 Initial State 时剔除。

### M12. `BackendArray` 忽略自身 backend 参数 ⚠️（已修复 2026-09-01）
- **位置**：`config.py:188-193`
- **现象**：`self.xp = get_xp() if self.backend == "gpu" else _numpy_module` —— `get_xp()` 读全局 `BACKEND`。构造 `BackendArray(backend="gpu")` 且全局为 cpu（CuPy 实际可用）时，`self.backend == "gpu"` 但 `self.xp` 是 NumPy，`to_gpu()` 却用 `_cupy_module.asarray`，实例内部自相矛盾，"gpu" 被静默忽略。
- **修复**：`self.xp = _cupy_module if (self.backend == "gpu" and _cupy_usable()) else _numpy_module`。

### M13. `LatticeLoader` 的 (2,2) bvecs 路径必然崩溃 ⚠️（已修复 2026-09-01）
- **位置**：`io/lattice_loader.py:97-101`（配合 `utils/lattice.py:173-174`、`:362-363`）
- **现象**：`create_lattice(bvecs_array=(2,2))` 补零成 (3,3) 后 `b3=(0,0,0)` → `LATTICE._validate_matrix` 报 "bvecs is singular"；即使绕过，`_reciprocal_to_real` 也会因 `volume_rec = 0` 报 "linearly dependent"。docstring 明确声称支持 (2,2) 输入，实际该路径完全不可用。
- **修复**：2D 输入给有限的离面分量（如 `b3=(0,0,1)`），或在 `_init_vectors` 对 2D 特判。

### M14. 多段偏压扫描时 bias 标签列表短于数据轴 ⚠️（已修复 2026-09-01，抽出共享 build_bias_labels）
- **位置**：`utils/AutoPPt_winnew_modified.py:183-192`（ShowMap）、`:243-258`（QPI）、`:317-332`（ShowMapI）；同样模式经 `plot_funcs.py:119-134/175-190/247-262` 影响 `nanonis_ppt_generator.py`
- **现象**：标签用 `temp[1:]` 对段间重复点去重，长度 = Σsteps−(段数−1)；循环帧数用 `len(sweep_signal)`（= Points）。若 3ds 数据轴保留段间重复点，循环末尾 `f"{bias[n]:.2f}"` 抛 `IndexError`。单段扫描（最常见）不触发，多段才暴露。
- **修复**：使标签长度与数据轴严格一致并加长度断言。

### M15. SXM 二进制数据偏移启发式 ⚠️（已验证 + 加固 2026-09-01：6 个真实文件新旧输出逐元素一致）
- **位置**：`io/nanonis_loader.py:118-121`
- **现象**：`:SCANIT_END:` 后固定"跳过 2 行 + 再 seek 2 字节"才开始读 `>f` 数据。SXM 规范中 header 与二进制之间只应有换行，该偏移与实际文件字节布局匹配与否未经任何校验，布局变化时整张图像静默错位（此加载器同时被 nanonispy 之外的地方复用）。
- **修复**：参照 nanonispy 的实现逐字节定位（读到空行后开始 `fromfile`），并至少断言 `data.size ≥ 通道数×像素数×2`。

### M16. 3DS 分块循环的边界情况 ⚠️（已修复 2026-09-01）
- **位置**：`io/nanonis_loader.py:570-575`、`:619-626`
- **现象**：`block_size = param_length + data_length` 由 header 推导；① header 缺字段导致 `block_size == 0` → `i // block_size` 除零且 `range(0, len, 0)` 抛错；② `raw_data` 长度超过 `total_pts` 时不截断，`n` 超出 `range(total_pixels)`，`params.loc[n]` 抛 `KeyError`、`grid[n]` 越界 `IndexError`；③ 尾部多余字节不是 `block_size` 整数倍时 `reshape((len(channels), pts_per_chan))` 尺寸不符抛错。
- **修复**：`block_size <= 0` 时抛明确异常；循环上限用 `total_pixels * block_size` 并忽略/警告尾部多余字节。

### M17. `.Band` 自旋极化体系被当作单一自旋，路径/能带翻倍 ⚠️（已修复 2026-09-01）
- **位置**：`dft/openmx/band.py:75-81`、`:135-171`
- **现象**：`.Band` 首行为 `nband, 自旋自由度, μ(Hartree)`，数据段按 **spin 外循环、k-point 内循环**排列；代码忽略 `header[1]`，自旋极化（磁性体系）时每个 k 点出现两次，被串成一条路径 → `kpts_frac`/`bands` 变成 `(2·nkpts, ...)`，`dist` 在路径中点出现虚假的"跳回起点"大跳变，累计距离错误。参考：ASE 的 `read_band_file` 与 OpenMX 论坛对格式的说明。
- **修复**：读取 `header[1]` 判断 `nspin`；自旋极化时按自旋拆分返回（如 `bands` 为 `(nspin, nk, nband)`），或至少检测到 `nspin>1` 时显式告警/抛错。

### M18. ANG（埃）坐标被静默丢弃，无法回退 ⚠️（已修复 2026-09-01）
- **位置**：`dft/openmx/parser.py:636-648`、`:786-797`
- **现象**：`_parse_species_and_coordinates` 检测到 `Unit Ang` 后分支只写注释不做事，返回 `positions_frac=None`（仅 `positions_ang`）；`read_atomic_positions` 优先级 2/3 只判断 `positions_frac is not None`。Ang 是 OpenMX 输入最常见的坐标单位；当 `.out` 无 "final structure" 段（只解析 `.dat`、或计算未收敛）时直接 `RuntimeError("No atomic positions found")`——而 `self.avecs` 已可用，本可完成 `frac = cart @ inv(avecs)` 转换。
- **修复**：在 `read_atomic_positions` 对 `positions_ang` 做 `positions_frac = positions_ang @ inv(avecs)` 转换后进入正常流程。

### M19. 3DS 参数列的空串分裂与 param_length 不一致 ⚠️（已修复 2026-09-01）
- **位置**：`io/nanonis_loader.py:600-607`、`:622`
- **现象**：`"".split(";")` 返回 `['']` 而非 `[]`——当 "Fixed parameters"/"Experiment parameters" 字段为空或缺失时，`param_columns` 混入空列，与 `# Parameters (4 byte)`（param_length）数量不一致。实测：param_length=1、columns=`['', 'Sweep (V)']` 时 `params.loc[n] = block[:1]` 被 pandas 广播到两列，参数写入错误列；param_length 更大时直接抛错。
- **修复**：`[p for p in s.split(";") if p]` 过滤空串，并加 `len(param_columns) == param_length` 断言。

---

## 四、Low（25）✅ 已全部修复（2026-09-02，见修复状态节）

| # | 位置 | 问题 | 修复建议 |
|---|------|------|----------|
| L1 | `stm/qpi_born.py:274,296` | `save_qpi_to_h5`/metadata 写死 `normalize=True`，但 Born 结果从未归一化（JDOS 是真归一化） | 要么真归一化，要么标志改 False |
| L2 | `io/qpi_io.py:81` | `bands == "all"` 对 ndarray 产生歧义真值（潜伏） | 先 `isinstance(bands, str)` |
| L3 | `utils/btk.py:92` | `sigma_finite_T(E, T=0)` 直接除零 `ZeroDivisionError`（`spectrum` 内部有规避，公开方法无） | `T<=0` 时返回 `sigma_zero_T(E)` |
| L4 | `utils/miscellaneous.py:246-250,273` | `fermi()` 标量输入 T=0 时 `bool.astype` 崩溃；且与 `fermi_cuda` 在 `e==mu` 处 `<` vs `<=` 不一致 | 用 `np.where(e<mu,1.0,0.0)`，统一边界约定 |
| L5 | `stm/vortex_num.py:11` | 注释 "1 Wb <=> 1 T*m**2" 错误；field/area 单位未声明，nm² 输入会差 1e-18 | 修正注释并写明单位（T、m²） |
| L6 | `mlwf_susceptibility.py:161` | CUDA 显存池硬编码 24GB，与真实显存不符 | 用 `cp.cuda.Device().mem_info` 查询 |
| L7 | `mlwf_susceptibility.py:268-273` | 线程数按亲和性算完又被 `num_threads = 10` 覆盖 | 删除硬编码或 `min(...)` |
| L8 | `mlwf_susceptibility.py:8,245-247` | pyfftw 顶层无条件导入 → numpy 回退分支永不生效（缺失时模块直接导入失败） | 顶层 try/except 导入 |
| L9 | `dft/wannier90/mlwf_ek2d.py:224` | `raw_output` 参数文档承诺返回裸数组，实际被忽略 | 实现分支或删除参数 |
| L10 | `bare_lindhard.py:414,417` | `calculate()` 接受 `output_path`/`q_range` 但从不保存/裁剪 | 实现或移除参数 |
| L11 | `lattice_operations.py:151-154` | 折叠窗边界条件不对称（`[0.5-tol, 0.5)` 的点被静默丢弃而 -0.5 侧保留） | 用对称边界条件 |
| L12 | `lattice_operations.py:417-432,264-291` | 多边形顶点按质心角排序仅对凸多边形有效，凹多边形破坏射线法（实际用例为凸 BZ，无碍） | 文档注明限制或改用不动点算法 |
| L13 | `AutoPPt_winnew_modified.py:1724` | 清理 `glob("temp*.tif")` 漏掉 `boxtemp.tif`，每次运行残留 | 改 `"*temp*.tif"` |
| L14 | `nanonis_ppt_generator.py:461/537/589` | 三个 `animation_*.gif` 从不清理（AutoPPt 有对应删除） | 清理段补 unlink |
| L15 | `AutoPPt_winnew_modified.py:1414/1556/1638` | 动画 `fig` 无 `plt.close(fig)`，每个 map 泄漏 3 个 figure | `animation.save` 后 close |
| L16 | `nanonis_ppt_generator.py:96-98` / `AutoPPt:115-117` | Linux 上 `st_ctime` 是 inode 变更时间而非创建时间 → 按创建时间排序错误 | Linux 用 `st_birthtime`/`statx`，回退时注明 |
| L17 | `nanonis_ppt_generator.py:238-240` / `AutoPPt:799` | 总览盒旋转不区分扫描方向（标点图 down 时取 `-angle`），两者不一致 | 按 `direction` 决定旋转角符号 |
| L18 | `dft/openmx/diff_gcube.py:69-78` | 数据值不足 `ngrid` 乘积时 `values[idx]` IndexError；多出的值被静默丢弃 | 校验值数量并给出明确错误 |
| L19 | `dft/openmx/parser.py:513-518` | "final structure" 取**第一次**出现（与 `_parse_fermi_level` 取最后一次不一致）；若优化过程多次打印该段，取到的是初始结构而非收敛结构；起始偏移 `i+3` 硬编码脆弱 | 记录最后一次匹配；按行内容（首个 5 列数字行）定位数据起始 |
| L20 | `nanonis_ppt_generator.py:38-69` / `AutoPPt:27-65` | 模块级 `input()` 交互、无 `__main__` 保护（脚本定位，但被包内 import 时会阻塞/崩溃） | 移入 `if __name__ == "__main__"` |
| L21 | `dft/openmx/dos.py:68-70,93` | 自旋极化（LSDA）DOS 为 5 列（E, up, down, up-IDOS, down-IDOS），代码固定按 3 列命名 → down-DOS 被误当 IDOS | 先探测列数，5 列时用 `E/DOS_up/DOS_dn/IDOS_up/IDOS_dn` |
| L22 | `dft/openmx/parser.py:450-463` + `unfolding.py:291-295` | 只解析 s/p/d 轨道，含 f 轨道基组（镧系等）时展开权重列数校验 `ValueError` 或列名错位 | 补充 f 轨道（每壳层 7 分量），或检测到 f 时明确报错 |
| L23 | `dft/openmx/unfolding.py:142-145` | `atom_index` 文档写 "typically 1-based"，实现为 0-based（`enumerate`）→ 按文档传入 1 会选中第二个原子 | 统一文档为 0-based，或实现中做 1→0 换算 |
| L24 | `utils/extract_k_type.py:22-33` | 电压列不足 10 个的整行被静默丢弃 → -270°C（靠 `fix_first_row` 硬编码修补）与表尾高温段数据缺口；行尾温度正则 `(-?\d+)\s*$` 在"行尾是带小数电压"的布局下会把小数部分（如 `41.276` → `276`）误当基准温度 | 按实际可用列数写入 `base_temp+i`，不整行丢弃；尾部温度解析加合理性区间校验 |
| L25 | `io/w90hr_loader.py:171-177` | `.wout`/`.out` 存在但倒格矢解析不出时，`LatticeLoader.create_lattice` 抛 `ValueError` 穿透 `load()` → HR 数据本身可用时整次加载崩溃（如旧版 `.wout`、b_3 缺失）；与 `_load_bvecs` 声明的 "未找到返回 None" 语义矛盾 | `_load_bvecs` 内捕获 `ValueError` 返回 None，或先判断 `_load_reciprocal_vectors` 返回 None 再构造 `LATTICE` |

---

## 五、修复优先级建议

1. **P0 — 立即**（核心功能完全不可用）：H1、H2（QPI 主链）、H3、H4（磁化率/裸 Lindhard 结果错误）。
2. **P1 — 尽快**（常用功能静默错值）：H5、H6（晶格几何）、H7、H8（实验端偏压/崩溃）、H10（cube 原子坐标损坏）、M2、M8、M9（PDOS/PPT 标注）。
3. **P2 — 常规**：其余 M 级（M3-M7、M10-M19）。✅ 已于 2026-09-01 全部完成。
4. **P3 — 清理**：L 级（L1-L25），其中 L15 内存泄漏在批量跑图时值得优先。

**补充建议**：
- 为 H1-H4 建立最小回归测试（nk=8、1-2 轨道 mock 哈密顿量即可复现，本次审查已用该方式实测）。
- `bare_lindhard` 与 `mlwf_susceptibility` 的 CPU/GPU 双路径应做数值对拍（同一小模型下两后端结果一致才算通过）。
- `nanonis_ppt_generator.py` 与 `AutoPPt_winnew_modified.py` 大量逻辑重复（且一处修了另一处没修，如 `get_divider`、`subtractMeanPlane`、GIF 清理），建议合并到 `plot_funcs.py` 单一实现。
- `dos.py`/`band.py`/`parser.py` 的修复（M2、M17、M18、L21、L22）建议各用一份真实 OpenMX 输出（含自旋极化、ANG 坐标、PDOS）做回归样例。

## 六、审查方法与验证记录

- **静态审查**：7 组并行覆盖 `utils`（lattice 系、杂项）、`io`、`dft/wannier90`、`dft/openmx`、`stm`、PPT 生成；`ruff check`（30 条均为风格级，无未定义名）。
- **运行时验证**（`.venv` Python 3.12，本机实测）：
  - `JDOSQPI(ham)` → `AttributeError: 'EK2DCalculator' object has no attribute '_compute_ek2d'`；
  - `BornQPI(ham).calculate()` → `AttributeError: ... 'hk_grid_cpu'`；
  - `TmatQPI(ham).calculate()` → `TypeError: takes 0 positional arguments`；
  - `get_1stbz_vertices()`（石墨烯取向）→ `ValueError: Expected 6 reciprocal vectors`；
  - `lat.supercell(M).bvecs == inv(M) @ B`（≠ `B @ inv(M)ᵀ`）；
  - pyFFTW 默认方向计划与 `np.fft.ifftn` 相差 64 倍（nk=8）、`FFTW_BACKWARD` 一致到 1e-16；
  - `eps_n_shift` 高级索引输出 2D `(nk, n_q1)`、k2 维塌缩；`q_vals` 双重 fftshift 后 q=0 索引 0 而数据 q=-0.5 在索引 2；
  - `get_divider`：含 `d10`/`d100` 路径返回 1；`dos.py` 分类：`s1/p1/d1/p10` 全部落入 `total`；
  - 全模块导入测试：仅 `utils.nanonis_ppt_generator` 失败（`No module named 'plot_funcs'`）。

> 本报告为静态+动态审查结论；H8、M3、M14-M16 等涉及真实 Nanonis 文件格式的条目建议用真实数据各验证一次后再修。
