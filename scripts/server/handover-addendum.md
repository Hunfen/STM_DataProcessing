# 交接书增补：Re χ₀ 的 q 行分片多进程运行

> 本文是 `lindhard-rechi-server-handover.md` 的**增补**，建议整段附加到交接书末尾（包内也可保留为独立文件）。
> 适用范围：`scripts/run_lindhard_rechi_cwf53.py`（新薄封装）+ `src/stm_data_processing/dft/wannier90/lindhard_re_chi.py`、`lindhard_re_chi_parallel.py`。
> 技术细节见仓库文档 `docs/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.md`。
> 本文 Mac 侧数字为集成实测（2026-09-21，M1 Pro / 8 核 / 16 GB）；服务器侧数字引自 `lindhard-rechi-cpu-parallel-status.md`，并标注【实测】/【估算】。

---

## 0. 一句话

驱动脚本的**命令行接口没变**（`--proj full|li --nk 256`），新增 `--workers/--mirror/--resume/--checkpoint-dir`；
**默认行为是与单进程逐位一致的并行**，产物文件名、attrs、datasets 与旧版完全相同，可以直接替换旧驱动。

```bash
cd ~/run/lindhard-rechi-server
PY=/home/n04hunfen/src/STM_DataProcessing/.venv/bin/python
"$PY" -u scripts/run_lindhard_rechi_cwf53.py --proj li --nk 256 --workers 6
```

---

## 1. 并行调用方式

### 1.1 驱动脚本（推荐）

```
--proj full|li          投影（同旧版；不写则两个投影都跑）
--nk 256                q 网格（同旧版）
--workers N             worker 进程数，默认 min(cpu_count − 1, 8)（本机 10 vCPU → 默认 8）
--mirror                只算 iq1 ∈ [0, nk//2]，其余按 χ₀(q)=χ₀(−q) 精确补齐（默认关闭）
--resume                只补缺失的分片（同一 --checkpoint-dir）
--checkpoint-dir PATH   分片检查点目录（默认 h5_data/ckpt_cwf53[_li]_<nk>pts）
```

工作机制：驱动把 q1 行按像素数均衡切成 N 段（连续行段、行数差 ≤1），每个 worker 用 `spawn` 起一个**单线程 BLAS** 进程算自己那一段，结果各自写检查点；父进程收齐后做覆盖性检查 → 镜像补齐（若开）→ 一次性 `fftshift` → digest 校验 → 原子写 h5。父进程只做拼图与汇总，不参与计算。

### 1.2 可以直接用 CLI（可选）

同一实现也提供独立命令行入口，参数更全（例如 `--band-block`、`--dry-run`、`--start-method`、`--log-file`）：

```bash
"$PY" /home/n04hunfen/src/STM_DataProcessing/scripts/run_lindhard_re_chi_parallel.py \
  --model-dir ~/run/lindhard-rechi-server/data --seedname C6LiC6_0 \
  --nk 256 --orbitals 48,49,50,51,52 --projection-label li --workers 6 --no-mirror \
  --output ~/run/lindhard-rechi-server/h5_data/c6lic6_cwf53_li_rechi0_256pts.h5 \
  --checkpoint-dir ~/run/lindhard-rechi-server/h5_data/ckpt_cwf53_li_256pts
```

`--dry-run` 只打印切片计划与内存预估，不加载模型、不启动进程（用于先看一眼规模），exit 0。

### 1.3 两条纪律（务必遵守）

1. **每个 worker 必须单线程 BLAS。** CLI 已在导入 numpy 之前用 `os.environ.setdefault` 把 `OPENBLAS_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS / VECLIB_MAXIMUM_THREADS` 设为 `"1"` 并在日志里记录。**不要**沿用旧命令里的 `export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10`——那会让每个 worker 再开一个 10 线程池，把 10 vCPU 吃满并让 SSH 卡顿（现状文档 §4.1/§8.4 实测：10 线程把 4.1× 的 CPU 烧在 OpenBLAS 池自旋上，只换到 1.4× 墙钟）。
2. **nk=256 的整个生产运行必须先取得用户明确许可**才能启动（见 §6）。

---

## 2. worker 数 / 内存 / SSH 响应建议

### 2.1 worker 数

* 本机 10 vCPU，且**是用户 SSH 交互使用的机器**：建议 `--workers 6`，最多 8；留 2 核给 SSH 与正在跑的旧作业。
* 旧作业（PID 101468）在跑时它约占 2.5 核，新运行请先按 `--workers 6` 起跑，边跑边看 `uptime`（load 长期 >10 就会卡 SSH）。
* worker 数超过行数时驱动自动降为「一行一段」并在日志里说明（不会报错）。

### 2.2 内存

每 worker 的估算由 `estimate_worker_rss_bytes()` 给出（口径：`evals + evecs + H(k) 构建临时 + q-sum 工作张量 + nrpts×num_wann²×16 常驻模型数组 + 1.1 GB 解释器/NumPy/HDF5 开销`），`num_wann`/`nrpts` 由 `read_model_shape()` 从 `hr.h5`/`hr.dat` 头部**零成本**读出。

* **先看确切数字**：在服务器包根目录跑
  `"$PY" scripts/run_lindhard_rechi_cwf53.py --proj li --nk 256 --workers 6 --dry-run`，
  它会打印每段的行数/像素/占比与「每 worker 估算 + 合计」——**规划请以此输出为准**。
* 参考口径（cwf53：`num_wann=53`、`nrpts=1681`，直接调用 `estimate_worker_rss_bytes`）：

  | 配置 | 每 worker 估算 | 8 worker 合计 |
  |---|---|---|
  | nk=256 `full` | 4.74 GB | **37.9 GB** |
  | nk=256 `li` | 2.26 GB | 18.1 GB |
  | nk=32 `full` | 1.25 GB | 10.0 GB |

  本机 94 GiB（80% MemAvailable ≈70 GB）跑 nk=256 `full` × 8 worker 完全放得下；实测同一配置的单进程 RSS 只有 3.64 GB（现状文档），所以估算是刻意保守的上界（≈1.3× 实测）。
* 估算是**上界**：本地 75 轨道模型实测（estimate vs 实测 `rss_peak`）：nk=8 `full` 1212.8 vs 320/326 MB、nk=16 `full` 1271.1 vs 425/425 MB、nk=32 `full` 1460.5 vs 1147/823 MB、`li` 同量级——全部 estimate ≥ 实测（小 nk 因固定开销偏保守，最紧余量约 27%）。
* 启动前驱动做内存守卫：估算总量 vs **80% MemAvailable**，超限直接拒绝启动（exit=3）。⚠️ 因为估算是上界，**机器内存不大或轨道数较多时可能保守拒绝**（例如 75 轨道模型 nk=256 `full` × 8 worker 估算 60.8 GB，在 64 GB 机器会被拒，而按实测比例真实占用仅 38–43 GB）。此时先看清数字再决定：降低 `--workers`，或以实测 `rss_peak` 为依据**显式**给出 `--max-mem-gb`（守卫尊重该值）；不要盲目强推。
* 真实峰值可在日志核对：每个 worker 的 `summary … rss_peak=`，父进程的 `summary … rss_peak=`。

### 2.3 SSH 响应

* 并发进程数 = worker 数；父进程几乎不占 CPU（只做拼图与汇总）。
* 单线程 BLAS 下每个 worker 稳定占 1 核；6 worker + 旧作业 2.5 核 ≈ 8.5 核，仍留有余量。
* 若 SSH 明显卡顿：`kill -INT <父进程 pid>`，驱动会结束子进程、保留检查点、以 130 退出，之后用 `--resume` 续跑即可（**不要** `kill -9` 父进程，那样子进程可能残留）。

---

## 3. 检查点与 resume

* 目录：`--checkpoint-dir`（默认 `h5_data/ckpt_cwf53[_li]_<nk>pts`）。
* 每段三个文件：`rows_<start>_<stop>.npz`（`data/intraband/interband`，float64）+ 同名 `.json`（元数据与 digest）+ `.done` 标记；三者齐全才算一段完成，写盘都是「临时文件 + rename」原子落地。
* `--resume` 先扫描已有完整分片，**只派发缺失行段**；被复用分片必须与本次运行在签名上一致（`nk`、`eta`、`T`、`μ`、`include_matrix_elements`、`degeneracy_tolerance`、`orbital_select`、**`num_wann`、`bvecs`**），否则判不兼容并**重算**——这可以防止把别的模型/别的投影的分片静默拼进来。
* **坏分片自愈**：0 字节、半截断或任何读取失败的分片都按"缺失"处理（丢弃 + 重算该段），最终 h5 与没有坏分片时逐位相同；不会因为一个坏文件让整次 `--resume` 崩掉。
* 产出的 h5 在全部拼图与 digest 校验完成后原子写入，**失败/中断不会留下半成品 h5**。
* 每个 worker 另写 `worker_<i>.log`；父进程日志里会打印每个 worker 的 exit code、失败时的日志尾部与 traceback。
* 同一 `--checkpoint-dir` 并发运行会被 `.lock` 拒绝（exit=2）；死进程遗留的锁会自动回收。

退出码：0 成功 / 2 锁占用 / 3 内存守卫 / 4 worker 失败 / 5 有 worker 未回传结果 / 6 与 worker digest 不一致 / 7 h5 写失败 / 8 组装自校验失败 / 130 SIGINT。

---

## 4. 镜像口径的严格性说明

* `--mirror` 只计算 `iq1 ∈ [0, nk//2]`（`iq1=0` 自镜像；偶数 nk 还有 `iq1=nk//2` 自镜像），其余行按 **`row[iq1][iq2] = row[(nk−iq1)%nk][(nk−iq2)%nk]` 精确拷贝**补齐，然后整图做一次 `fftshift`。
* ⚠️ 在冻结网格 `q = fftshift(fftfreq(nk))` 上，`q → −q` 的索引置换是 `j = (2·(nk//2) − i) mod nk`，等价写法 **偶数 nk 用 `np.roll(m[::-1,::-1], 1, axis=(0,1))`、奇数 nk 用 `m[::-1,::-1]`**；字面反转对偶数 nk 偏一格，不要用它做对称性断言。
* 残差只来自**自镜像行**（由引擎直接算出、不经过拷贝，其行内列对称只到浮点精度）与两次独立浮点求和顺序。
* 实测（对真实 75 轨道模型与本机 mock 模型）：
  * 小模型 nk=4/5/6/8/16：max|Δ| = 1.11e-16 / 1.57e-14 / 1.94e-16 / 2.78e-16 / 5.55e-16；
  * 真实模型 nk=8：5.55e-17；nk=16：4.16e-17。
  * 验收界 **≤1e-12**，实测典型 **~1e-15**（最差 1.57e-14 出现在奇数 nk 小网格）。
* **默认关闭**；只有当你接受「镜像行与单进程参考差 ~1e-15」时才加 `--mirror`（收益是像素数约减半）。注意：不镜像时的分片结果是**逐位一致**的。

---

## 5. 实测加速数据（Mac 侧，nk=32，作为标定参照）

| 运行 | 单进程（旧路径） | 6 worker（新驱动） | 加速比 |
|---|---|---|---|
| `full`（75 轨道全选） | 94.1 s【实测】 | 33.6 s【实测】 | 2.80× |
| `li`（48–52） | 44.2 s【实测】 | 19.1 s【实测】 | 2.31× |
| `full`（2 worker） | 93.7 s | 50.3 s | 1.86× |
| `li`（2 worker） | 42.8 s | 24.0 s | 1.79× |

* 两条路径的 h5 **逐位相同**（镜像关闭；`li` 残差 0、`full` 残差 2.22e-16）。
* nk=32 时并行效率偏低是**固定开销**造成的（每 worker 都要加载模型 ≈1.3 s + 对角化全 k 网格 ≈2.2 s，后者 O(nk²)、**不可分片**）；q_sum 单进程吞吐 23 px/s，6 进程并发时每进程降到 ≈13.5 px/s（内存带宽竞争）。nk=256 时固定开销可忽略（≈5–16 min/worker，占 ~21 h 的 ~1%）。
* **加速比不能直接外推到 10 vCPU 的服务器。** 建议标定：在服务器上跑一次 `--nk 32 --workers 4` 冒烟记录墙钟 w，则 `并行效率 ≈ 196.1 / (w × workers)`（196.1 s = 现状文档 §8.1 的 nk=32 full 单线程基准【实测】），再用成本标度 `16.3³ ≈ 4330`（每次 nk 翻倍 ×16.3【实测】）外推 nk=256【估算】。
* 现状文档对 nk=256 的估算：8 个单线程 worker 时 `full` ≈ 21 h、`li` ≈ 8 h【估算】；请以服务器实测为准（本增补的作者没有在服务器上跑过任何东西）。

---

## 6. 服务器上正在跑的 full 单进程作业（PID 101468）——不要触碰

* 现状：`full` 单进程作业自 2026-09-19T18:42:56 起在跑，**PID 101468**（约 2.5 核，RSS ≈3.6 GB），产物路径 `h5_data/c6lic6_cwf53_rechi0_256pts.h5`。
* **绝不要** kill / 挂起 / 重启 / 挪动它的中间文件，也不要在它的工作目录里删东西。
* ⚠️ **产物同名冲突**：新驱动的 `full` 产物与它**同名同路径**。因此：
  1. 现在可以安全跑的是 **`li`**（产物 `h5_data/c6lic6_cwf53_li_rechi0_256pts.h5`，不冲突）；
  2. 新驱动的 **`full`** 必须等旧作业结束/被用户停掉之后，并且先把旧产物挪到 `results/` 留档，再启动；
  3. 也不要为了「省事」把新 `full` 的输出改到别的路径——交付契约固定了文件名。
* 只读查看用：`ps -o pid,pcpu,rss,etime,cmd -p 101468`。
* 新运行启动前请确认没有残留的旧 `--checkpoint-dir` 锁（`.lock` 里的死 pid 会自动回收）。

---

## 7. 产物契约（与旧版一致，未变）

* 文件名：`h5_data/c6lic6_cwf53_rechi0_<nk>pts.h5`（full）、`h5_data/c6lic6_cwf53_li_rechi0_<nk>pts.h5`（li）。
* attrs：`module_type='real_Lindhard'`、`eta=0.005`、`nq=<nk>`、`chemical_potential=0.0`、`temperature=4.2`、`orbital_select`（数组）、`projection`（`full`/`li`）。
* datasets：`susceptibility` `(nk, nk)`、`bvecs` `(3, 3)`。
* 数据为 `fftshift` 后的原胞网格（q=0 在 `(nk//2, nk//2)`），`q_range` 在绘图/读取时再扩到 ±3 Å⁻¹。
* 绘图脚本 `scripts/plot_lindhard_rechi_cwf53.py` 未改动；其 `NK=256` 是按生产运行写死的。若要用 nk=32 冒烟产物出图，两种做法：把 h5 直接喂给它的 `plot_one()`，或给脚本加一个 `--nk` 透传参数（一行改动，本次未改交付脚本本体）。
