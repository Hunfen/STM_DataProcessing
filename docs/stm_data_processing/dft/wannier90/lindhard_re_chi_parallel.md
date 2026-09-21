# Lindhard Re χ₀ 的 q 行分片多进程并行（CPU）

本文件是 `lindhard_re_chi` 系列的**并行化与运维文档**：讲清楚为什么按 q 行分片、怎么跑、数值上保证什么、日志里能看到什么、崩了怎么续。

物理与公式见 [`Lindhard_Re_chi_from_Wannier90_hr.md`](Lindhard_Re_chi_from_Wannier90_hr.md)（本文引用其节号）。数值语义（符号约定、η、网格、h5 契约）以该文档与模块 docstring 为准，本次改造**没有改动其中任何一条**。

---

## 1. 文件与入口

| 路径 | 作用 |
|---|---|
| `src/stm_data_processing/dft/wannier90/lindhard_re_chi.py` | 引擎。`calculate()` 新增 `q_index_range`（q 行切片）与 `progress_interval_s`；构造函数新增 `band_block` / `block_entries`（替代服务器驱动里的 `_MAX_BAND_BLOCK` monkeypatch）；新增日志/进度/摘要输出 |
| `src/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.py` | 并行 runner。`plan_row_slices` / `run_parallel` / `assemble_slices` / `assemble_from_checkpoints` / `scan_checkpoints` / `write_result_h5` / `configure_logging` / `estimate_worker_rss_bytes` / `available_memory_bytes` / `read_model_shape` / `model_identity` |
| `scripts/run_lindhard_re_chi_parallel.py` | 通用命令行入口（**导入 NumPy 之前**把四个 BLAS 线程变量钉为 1） |
| `scripts/server/run_lindhard_rechi_cwf53.py` | cwf53（C6LiC6，53 Wannier）服务器驱动薄封装，`--proj full|li` |
| `tests/regression/check_lindhard_re_chi.py` | 回归（a）–（u），含分片/拼图/镜像/并行/日志合约 |
| `scripts/server/handover-addendum.md` | 服务器交付包的交接书增补（部署与运行纪律） |

## 2. 为什么是 q 行分片

现状文档（`lindhard-rechi-cpu-parallel-status.md`，2026-09-21）在 10 vCPU Xeon 上的实测结论：

* 热循环并行度上限 ≈1 核：19 个线程里 15 个 0 tick，4 个各约 0.7 核；
* `OPENBLAS_NUM_THREADS=10` 把 **4.1× 的 CPU 烧在 OpenBLAS 池自旋**上，只换到 **1.42× 墙钟**；改成 1 线程结果**逐位一致**（nk=32：wall 137.87 s / CPU 603.7 s → wall 196.14 s / CPU 146.7 s）；
* 主力算子是 NumPy 元素级/归约，天生单线程；唯一走 BLAS 的批量小矩阵乘又太小，够不到并行阈值。

`calculate()` 的主体是 `for iq1 … for iq2 …`，每个 `(iq1, iq2)` 像素只写自己那一格，像素之间零耦合，因此：

> **按 `iq1`（q 的第一维）切行 → N 个单线程进程并行，是唯一无损且有效的杠杆。**

## 3. 用法

### 3.1 通用 CLI

```bash
.venv/bin/python scripts/run_lindhard_re_chi_parallel.py \
  --model-dir /path/to/model --seedname C6LiC6_0 \
  --nk 256 --orbitals 48,49,50,51,52 --projection-label li \
  --workers 8 --no-mirror \
  --output out/c6lic6_cwf53_li_rechi0_256pts.h5 \
  --checkpoint-dir out/ckpt_cwf53_li_256pts
```

| 参数 | 说明 |
|---|---|
| `--model-dir` / `--seedname` | 模型目录与 seedname（loader 优先读 `<seed>_hr.h5`，否则解析 `<seed>_hr.dat`） |
| `--nk` / `--eta` / `--temperature` / `--mu` | 冻结参数，默认 256 / 5e-3 eV / 4.2 K / 0.0 |
| `--orbitals all\|48,49,50,51,52` | `full` 用 `all`；li 用索引列表 |
| `--projection-label` | 写入 h5 的 `projection` attr（默认 `full`/`custom`） |
| `--workers` | worker 进程数，默认 `min(cpu_count − 1, 8)` |
| `--mirror` / `--no-mirror` | 是否启用 χ₀(q)=χ₀(−q) 镜像（**默认关闭**，见 §5.2） |
| `--output` / `--checkpoint-dir` / `--resume` | 产物与分片；`--resume` 只补缺失行段 |
| `--band-block` / `--block-entries` | 引擎阻塞参数（默认 `_MAX_BAND_BLOCK=16` / `_MAX_BLOCK_ENTRIES=8e6`） |
| `--progress-interval` | 进度行节流秒数（0 = 每行都打；<0 = 关闭） |
| `--start-method` | `spawn`（默认）/ `fork` / `forkserver` |
| `--log-file` | 父进程文件日志（worker 明细各写自己的 `worker_<id>.log`） |
| `--max-mem-gb` | 内存预算，默认 MemAvailable 的 80% |
| `--dry-run` | 只打印切片计划与内存预估，**不加载模型、不启动进程**，exit 0 |

### 3.2 服务器驱动

```bash
cd ~/run/lindhard-rechi-server
PY=/home/n04hunfen/src/STM_DataProcessing/.venv/bin/python
"$PY" -u scripts/run_lindhard_rechi_cwf53.py --proj li --nk 256 --workers 6 \
      --checkpoint-dir h5_data/ckpt_cwf53_li_256pts
```

命令行接口与旧版兼容（`--proj full|li --nk`），新增 `--workers/--mirror/--resume/--checkpoint-dir`；产物文件名、attrs、datasets 与旧版完全一致。详细部署步骤与纪律见 `scripts/server/handover-addendum.md`。

### 3.3 两条纪律

1. **每个 worker 必须单线程 BLAS。** CLI 在导入 NumPy **之前**用 `os.environ.setdefault` 把 `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` 设为 `"1"` 并在日志里记录该决定（BLAS 线程池在 NumPy 加载时就已创建，之后再改无效）。**不要**沿用旧的 `OPENBLAS_NUM_THREADS=10`：那会让每个 worker 再开一个 10 线程池，把机器吃满并卡住 SSH。
2. **整体 nk=256 生产运行必须先取得用户明确许可**再启动（这也是一条部署纪律）。

## 4. 工作机制

1. **规划**：父进程按像素数把 `iq1 ∈ [0, nk)` 均衡切成 N 段连续行段（行数差 ≤1；`--mirror` 时只切 `[0, nk//2]`），并打印每段的行段/像素/占比与内存预估。
2. **内存守卫**：`估算 × 段数` 与 MemAvailable 的 80%（可调 `--max-mem-gb`）比较，超限拒绝启动（exit 3）——此时应**降低 `--workers`**，而不是放宽预算。
3. **派发**：每段一个 `spawn` 子进程（可用 `--start-method fork`）。每个 worker 自建 `MLWFHamiltonian`（不加载父进程状态，spawn 安全），用 `calculate(q_index_range=…, progress_interval_s=…)` 算自己那一段。
4. **回流**：worker 把结果写检查点（原子落地），把 digest/耗时/峰值 RSS 与进度事件送回父进程；父进程按同一节流规则打**单行聚合进度**。
5. **拼图**：父进程做覆盖性检查（每行恰好一次）→ 必要时镜像补齐 → **只做一次** `np.fft.fftshift(axes=(0,1))` → 逐 worker digest 交叉核对 → 组装 `(nk, nk)` 结果 → 原子写 h5（临时文件 + `os.replace`）。父进程不参与计算。
6. **失败处理**：任一 worker 非零退出（含 BLAS 段错误这类没有 traceback 的死法）→ 记录 exit code 与该 worker 日志尾部，中止剩余派发，**不写最终 h5**，返回非零。

## 5. 数值契约

### 5.1 分片无损（逐位一致）

单像素的算术只依赖 `(iq1, iq2)`、`nk`、`nw` 与阻塞参数（`m_block` / `row_block` 由 `nk`/`nw` 决定，与 `iq1` 无关），切片只改变被迭代的 `iq1` 集合，因此：

> **行分片 + 拼图的结果与单进程 `calculate()` 逐位相同**（`np.array_equal` 为 True，data/intraband/interband/网格全同）。

已实测覆盖：`nk ∈ {1,2,3,4,5,7,8,16}` × `workers ∈ {1,2,3,nk,nk+5}` × `mirror` 全组合，以及真实 75 轨道模型 nk=8/16 的并行 vs 单进程 h5 字节一致。

### 5.2 镜像（χ₀(q)=χ₀(−q)）：数学正确，但**不再逐位一致**

偶数性不需要晶体对称性：只需 `g(x)=x/(x²+η²)` 是奇函数、`k → k+q` 是网格双射、以及局域近似下 `M_mn(k+q,−q) = M_nm(k,q)*`。因此 `χ₀(q)=χ₀(−q)` 严格成立（回归检查 (c) 实测 max|χ(q)−χ(−q)| = 2.8e-16）。

> ⚠️ **网格上的正确置换容易写错**：在冻结网格 `q = fftshift(fftfreq(nk))`（即 `q_i = ((i − nk//2) mod nk)/nk`）上，`q → −q` 的索引置换是
> `j = (2·(nk//2) − i) mod nk`，等价的写法是**偶数 nk 用 `np.roll(m[::-1,::-1], 1, axis=(0,1))`、奇数 nk 用 `m[::-1,::-1]`**。
> 字面 `m[::-1,::-1]` 在偶数 nk 上偏一格（把 q=−1/2 映到 +1/2−1/nk），不成立。

实现上：`--mirror` 只计算 `iq1 ∈ [0, nk//2]`，其余行按 `row[iq1][iq2] = row[(nk−iq1)%nk][(nk−iq2)%nk]` **精确拷贝**补齐。注意 `iq1 = 0`（以及偶数 nk 的 `iq1 = nk//2`）是**自镜像行**，由引擎直接算出、不经过拷贝，其行内列对称只到浮点精度——这是镜像图残差的唯一来源。

实测与单进程参考的 `max|Δ|`：

| nk | 4 | 5 | 6 | 8 | 16 |
|---|---|---|---|---|---|
| mock 模型 | 1.11e-16 | 1.57e-14 | 1.94e-16 | 2.78e-16 | 5.55e-16 |
| 真实 75 轨道模型 | — | — | — | 5.55e-17 | 4.16e-17 |

**验收界 ≤1e-12，实测典型 ~1e-15**（最差 1.57e-14，出现在奇数 nk 的小网格）；框架评审的独立对抗探针最差 4.441e-16。另有一条自检：在正确置换下镜像图的对称残差 ≤1e-12。

**默认关闭**。收益是把 q 像素数减半（≈2×）；代价是与单进程结果不再逐位一致（差 ~1e-15）。分片（§5.1）无论如何都是逐位一致的。

### 5.3 冻结不变的语义

文档 §7.1 的符号约定（`Re χ₀ = −(1/N_k)Σ F(f_nk − f_m,k+q)Δε/(Δε²+η²)`，`Re χ₀ ≥ 0`）；`degeneracy_tolerance = 1e-12` 及其判据；η 的用法；k 网格 `linspace(-0.5,0.5,nk,endpoint=False)` 与索引折回；`fftshift(fftfreq(nk))` 网格与 h5 结构（`module_type='real_Lindhard'`、attr 名、文件名）；`_HK_ROW_BLOCK = 512`；`_MAX_BAND_BLOCK` / `_MAX_BLOCK_ENTRIES` 默认值。**文档 §8.2 的判据本次未改**（见 §11）。

## 6. 日志契约

引擎 logger 名 `stm_data_processing.dft.wannier90.lindhard_re_chi`；并行 runner 名 `…lindhard_re_chi_parallel`；两者都不在模块里 `basicConfig`，由入口（CLI / `configure_logging`）配置 handler。

| record | 内容 |
|---|---|
| 配置回显 | nk、nw、轨道数、η、T、μ、`include_matrix_elements`、`degeneracy_tolerance`、`band_block`、`row_chunk`、切片范围或 full、本次 q 像素数、pid、`cpu_count` |
| 线程顾问行 | 四个 BLAS 线程变量现值；未设或 >1 时 WARNING（说明导入后无法再改、多进程会叠加抢核） |
| 阶段计时 | `stage=diagonalize rows=… blocks=… s=…` → `stage=occupations` → `stage=q_sum pixels=… s=… px_per_s=…` → `stage=fftshift` →（`q_range` 时）`stage=extend` → `stage=h5_write path=… bytes=… s=…` |
| 进度行 | `progress rows=12/129 (9.3%) rate=0.42 row/s px/s=108 elapsed=28.6s eta=00:08:40 rss_peak=3.62GB`（按 `progress_interval_s` 节流，仅跨行边界时输出） |
| 收尾摘要 | 总耗时、像素数、平均 px/s、峰值 RSS、data/intra/inter 的 `sum/max/min` digest、`max|data-(intra+inter)|` |

峰值 RSS 只用标准库取得（Linux `/proc/self/statm`，macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss`），**不引入 psutil**。

## 7. 检查点与断点续跑

* 每段三个文件：`rows_<start>_<stop>.npz`（data/intraband/interband，float64）、同名 `.json`（元数据 + digest + sha256）、`.done` 标记；三者齐全才算完成，写盘统一「临时文件 + rename」原子落地。worker 明细日志为 `worker_<id>.log`。
* **`--resume` 只派发缺失行段**；被复用的分片必须与本次运行在签名上一致，否则判不兼容并**重算**（带 WARNING）。签名键：`nk`、`eta`、`chemical_potential`、`temperature`、`include_matrix_elements`、`degeneracy_tolerance`、`orbital_select`、**`num_wann`、`bvecs`**（最后两项防止把别的模型的分片静默拼进来）。
* **坏分片自愈**：任何读取失败（0 字节、半截断、非 npz）都按"缺失"处理——丢弃、重算该段，最终 h5 与没有坏分片时**逐位相同**；`assemble_from_checkpoints` 遇到坏分片抛清晰的 `ValueError`。
* **锁**：同一 `--checkpoint-dir` 并发运行被 `.lock` 拒绝（exit 2）；死进程遗留的锁自动回收。
* **产物原子性**：h5 只在全部拼图与 digest 校验通过后原子写入，失败/中断不留下半成品。

## 8. 内存与 worker 数

`estimate_worker_rss_bytes(nk, num_wann, n_orb, nrpts=None)` 估算单个 worker 常驻内存：
`evals + evecs + H(k) 构建临时 + q-sum 工作张量 + nrpts×num_wann²×16（常驻 h_list_flat）+ 1.1 GB（解释器/NumPy/HDF5 缓冲）`。
`num_wann`/`nrpts` 由 `read_model_shape()` 从 `<seed>_hr.h5` 属性或 `hr.dat` 头部（第一行两个整数）**零成本读出**，不加载 236 MB 哈密顿量。

实测对照（真实 75 轨道模型，实测 `rss_peak` 为两次运行）：

| 运行 | 估算 | 实测 |
|---|---|---|
| nk=8 `full` | 1212.8 MB | 320 / 326 MB |
| nk=8 `li` | 1207.6 MB | 315 / 316 MB |
| nk=16 `full` | 1271.1 MB | 425 / 425 MB |
| nk=16 `li` | 1250.6 MB | 408 / 405 MB |
| nk=32 `full` | 1460.5 MB | 1147 / 823 MB |
| nk=32 `li` | 1378.5 MB | 865 / 942 MB |

估算在以上全部场景都是**上界**（最紧余量 ≈27%；小 nk 时因固定开销而偏保守，nk=8 约 3.8×）。

生产规模的精确口径（直接调用 `estimate_worker_rss_bytes`，`nrpts=1681`）：

| 模型 / 配置 | 每 worker 估算 | 8 worker 合计 |
|---|---|---|
| cwf53（53 轨道，服务器）nk=256 `full` | 4.74 GB | **37.9 GB** |
| cwf53 nk=256 `li`（5 轨道） | 2.26 GB | 18.1 GB |
| cwf53 nk=32 `full` | 1.25 GB | 10.0 GB |
| 本地 75 轨道模型 nk=256 `full` | 7.59 GB | 60.8 GB |

服务器（94 GiB，80% MemAvailable ≈70 GB）跑 cwf53 nk=256 `full` × 8 worker：估算 37.9 GB，而**实测**同一配置的单进程 RSS 仅 3.64 GB（现状文档）——估算是实测的 ~1.3×，属刻意保守，不会挡住正常生产运行。

⚠️ **但估算是上界，可能保守拒绝**：轨道数多的模型（如本地 75 轨道）nk=256 `full` × 8 worker 估算 60.8 GB，在 64 GB 机器（预算 ≈51 GB）会被拒，而按实测比例真实占用只有 ~38–43 GB。此时的正解是：**先看清数字再决定**——(i) 降低 `--workers`；或 (ii) 用实测 `rss_peak` 为依据显式给出 `--max-mem-gb`（守卫会尊重该值）。不要盲目强推。

**worker 数建议**：这台 10 vCPU 机器同时是你 SSH 交互使用的机器，建议 `--workers 6`（最多 8），留出余量；worker 数超过行数时自动降为"一行一段"并在日志说明。

## 9. 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | `--checkpoint-dir` 被别的进程占用（锁未释放） |
| 3 | 内存守卫拒绝启动（估算是上界，降 `--workers`） |
| 4 | worker 非零退出 |
| 5 | 有 worker 没回传结果 |
| 6 | 拼图后与某个 worker 自报 digest 不一致 |
| 7 | h5 写入失败 |
| 8 | 组装结果自校验（digest）失败 |
| 130 | 父进程收到 SIGINT：结束子进程、保留检查点、可 `--resume` 续跑 |

## 10. 实测性能（Mac 侧标定）

真实 75 轨道 C6LiC6_CWF 模型，nk=32（单进程为同机旧路径基线）：

| 运行 | 单进程 | 6 worker | 加速 |
|---|---|---|---|
| `full` | 94.1 s | 33.6 s | 2.80× |
| `li` | 44.2 s | 19.1 s | 2.31× |
| `full`（2 worker） | 93.7 s | 50.3 s | 1.86× |
| `li`（2 worker） | 42.8 s | 24.0 s | 1.79× |

nk=32 时效率偏低来自**不可分片的固定开销**：每个 worker 都要加载模型（≈1.3 s）并对全 k 网格做一次对角化（≈2.2 s，O(nk²)），另有内存带宽竞争（单进程 q_sum 23 px/s，6 进程并发时每进程 ≈13.5 px/s）。**nk=256 时固定开销可忽略**（对角化 ≈5–16 min/worker，占 ~21 h 的 ~1%）。

服务器外推方法：先跑一次 `--nk 32 --workers W` 记录墙钟 `w`，则并行效率 ≈ `196.1 / (w × W)`（196.1 s 是现状文档 §8.1 的服务器 nk=32 `full` 单线程基准【实测】），再按 `nk` 翻倍 ×16.3【实测】外推到 nk=256。现状文档的估算：8 个单线程 worker 时 `full` ≈ 21 h、`li` ≈ 8 h【估算】；**以服务器实测为准**。

## 11. 已知限制与本次未做的事

1. **镜像非逐位一致**（§5.2）：差 ~1e-15，默认关闭；分片本身恒为逐位一致。
2. **文档 §8.2 的 df/dE 判据未改**：实现仍只在 `|Δε| ≤ degeneracy_tolerance = 1e-12` 时用 `df/dE` 替换，而文档还要求 `|Δε| ≲ η` 时也替换（§8.3 的 `(v·q)²/η²` 压制）。实测（本地 75 轨道模型、η=5e-3 eV）：把判据放宽到 η 会改变 nk=32 中 112/1024 个像素（max|Δ| = 2.9e-2，约为峰值的 2.2%），nk=256 抽样 400 个 q 像素中落在 `(1e-12, η]` 窗口的带内对角对占 1.31%、399/400 个像素都含这样的对。**本次刻意不动**：服务器上正在跑的 `full` 作业与已产出的 h5 都是 1e-12 口径，改判据会让不同投影之间口径不一致。若要修，应另开任务并同步决定是否重跑 full。
3. **每 worker 重复对角化**：固定开销 O(nk²)，nk=256 时可忽略；若将来 nk 很大可考虑共享本征量。
4. **未做**：GPU/CUDA 路径；真正的 batched GEMM（MKL `cblas_zgemm_batch`，可替掉逐 batch 的 `np.matmul`）；文档 §6.2 的中心相位中间层次；有限 ω。
5. **交付绘图脚本 `plot_lindhard_rechi_cwf53.py` 把 `NK` 写死为 256**：用 nk=32 冒烟产物出图时需直接调用其 `plot_one()`（本次集成即如此），或给它加一个 `--nk` 透传参数。
6. ⚠️ **产物同名冲突**：新驱动的 `full` 产物与服务器上正在跑的旧单进程作业（PID 101468）**同名同路径**。`li` 不冲突可先跑；新 `full` 必须等旧作业结束、并把旧产物挪到 `results/` 留档之后再启动。
7. **两处已知小缺口**（行为正确，仅文档/卫生问题，暂未修）：`assemble_from_checkpoints` 的 docstring 未点名"坏分片会抛 `ValueError`"；`_ensure_file_log` 给 root logger 装的 handler 在返回时不卸载，对 CLI 无影响，但长期常驻进程里反复调用 `run_parallel(log_file=...)` 会持续写同一文件。

## 12. 回归与验证证据

```bash
.venv/bin/python tests/regression/check_lindhard_re_chi.py    # (a)–(u)，全绿即通过
.venv/bin/ruff check src/stm_data_processing/dft/wannier90/lindhard_re_chi.py \
                    src/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.py \
                    scripts/run_lindhard_re_chi_parallel.py \
                    scripts/server/run_lindhard_rechi_cwf53.py
```

回归检查（(p)–(u) 为本次新增）：

* (p) 切片：合法区间逐位等于单进程对应原始行段；非法区间与"切片 + output_path/q_range"报 `ValueError`；
* (q) 拼图：单段/多段/乱序切片拼出的 `(nk,nk)` 与 `calculate()` 逐位相等；
* (r) 镜像：与单进程参考 `max|Δ| ≤ 1e-12`（实测值随测试打印）、正确置换下对称残差 ≤1e-12、镜像+分片 ≡ 镜像不分片；
* (s) 并行执行：spawn 多 worker 与单进程逐位一致、检查点齐全、坏分片/删分片后 `--resume` 只重派缺失段、worker 失败不写 h5；
* (t) 日志契约：上述全部 record 真实存在（含进度行与摘要 digest），模块内无 psutil/cupy/get_backend，`_HK_ROW_BLOCK ≤ 512`；
* (u) 性能数据（只打印不断言）、内存估算上界性、检查点签名拒绝异模型分片。

证据报告：`var/lindhard_verify/report.md`、`var/lindhard_verify/report_r1r3.md`（独立验证）、`var/lindhard_review/report.md`（对抗式评审）、`var/lindhard_deploy/report.md`（集成）、`var/lindhard_repair/report.md`（findings 修复）。
