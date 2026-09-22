# 分片-合并并行驱动（`stm_data_processing.parallel.shard_driver`）

本模块把「按分片键切分工作 → 每片一个 worker 写自己的分片文件 → 父进程按序合并」这套
机制从 Lindhard 并行计算器里抽了出来，任何计算器都能复用。模型选择的理由见
`docs/hdf5_checkpoint_shards.md`：本机 `h5py.get_config().mpi` 为 `False`，
并行写只能「每 worker 一个文件 + 合并」。

* 代码：`src/stm_data_processing/parallel/shard_driver.py`
* 包入口：`src/stm_data_processing/parallel/__init__.py`（`__all__` 导出下面全部名字）
* 第一个使用者：`src/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.py`
  （`run_parallel` 现在只负责声明物理，循环交给 `run_shards`）
* 第二个使用者：`src/stm_data_processing/stm/qpi_jdos.py`（`JDOSQPI.calculate` 的
  opt-in 并行路径，见 §5）

## 1. 公共接口（名字 + 签名）

```python
def plan_slices(work_size: int, workers: int) -> list[tuple[int, int]]
def scan_shards(checkpoint_dir: str | Path) -> list[tuple[int, int]]
def available_memory_bytes() -> int | None
def memory_guard(per_worker_bytes: int, n_workers: int, max_mem_gb: float | None,
                 *, label: str = "shard-driver") -> None
def log_plan(slices, nk, mirror, per_worker_bytes, available, max_mem_gb, *, label=...) -> None
def log_aggregate_progress(rows_done, rows_total, pixels_done, pixels_total,
                           started, active_workers, rss_peak, *, label=...) -> None
def absorb_progress(message, pending, worker_rows, worker_pixels) -> None
def tail(path: Path, lines: int = 12) -> str
def terminate_all(processes: Mapping[int, Any]) -> None
def remove_tree(path: Path) -> None
def run_shards(spec: ShardSpec, *, n_workers: int,
               checkpoint_dir: str | Path | None = None, resume: bool = False,
               start_method: str = "spawn", progress_interval_s: float = 30.0,
               dry_run: bool = False) -> int

class ShardStore:
    def __init__(self, directory, *, array_keys, identity,
                 generator="shard_driver", label="shard-driver", logger=None) -> None
    def paths(self, key) -> tuple[Path, Path]      # (shard .h5, .done marker)
    def write(self, key, arrays, meta) -> Path     # atomic, h5_convention
    def load(self, key) -> dict | None             # {"arrays", "meta"} or None
    def scan(self) -> list[tuple[int, int]]
    def compatible(self, meta) -> bool
    def bind(self, directory) -> None              # driver points it at the real dir

@dataclass(frozen=True)
class ShardSpec:  # 见 §3 的字段说明
@dataclass(frozen=True)
class RunStats:   # label, slices, workers, rows_done, pixels_done, rows_total,
                  # pixels_total, live_workers, started, elapsed_s, rss_peak_bytes
```

## 2. 驱动负责什么

| 职责 | 实现 |
| --- | --- |
| 分片文件生命周期 | `ShardStore`：临时文件 + `replace()` 原子落盘、`.done` 最后写、读失败/行数不符只 WARNING 不抛、成功合并后删除（自动 checkpoint 目录）|
| 分片键规划 | `plan_slices()`：均分、连续、升序、半开区间；调用方决定 `work_size`（Lindhard 用行数，mirror 时减半）|
| 恢复过滤 | `run_shards(resume=True)` 只 dispatch 缺失或身份不符的分片，其余记 `resume: rows=[a, b) already complete` |
| 身份校验 | `ShardStore.compatible(meta)`：把分片元数据与 `identity` 逐键比较；已知键不符 → WARNING「does not match the requested parameters and is recomputed」|
| worker 派发 | `spec.worker_process(...)` 造进程、父进程 `start()`；进度队列聚合、失败即终止其余 worker 并返回 4 |
| SIGINT | 收到信号终止活着的 worker、保留已完成分片、返回 130 |
| 内存预算 | 启动前 `memory_guard(spec.per_worker_bytes, len(pending), spec.max_mem_gb)`；不通过返回 3 |
| 合并顺序 | 全部完成后**按 `spec.plan()` 的顺序** `store.load()`，把 `loaded` 交给 `spec.finalize(loaded, reports, stats)` |
| 日志 | 所有驱动记录经 `spec.logger`（默认本模块 logger）发出，标签取自 `spec.label` |

返回码：`0` 成功；`2` checkpoint 目录被锁；`3` 内存预算拒绝；`4` 有 worker 失败；
`5` 分片或 worker 报告缺失；`130` SIGINT；`spec.finalize` 可返回自己的码（Lindhard 用 6/7/8）。

## 3. `ShardSpec`：调用方要声明什么

```python
ShardSpec(
    label="LindhardParallel",  # 日志标签
    store=ShardStore(...),  # 分片目录 + array_keys + identity
    plan=lambda: plan_row_slices(nk, n_workers, mirror=mirror),  # 分片键
    progress_units=lambda key: (key[1] - key[0], (key[1] - key[0]) * nk),
    worker_process=...,  # (context, index, key, payload, rq, pq) -> Process
    worker_payload=lambda: {...},  # 可 pickle 的 worker 参数
    log_plan=...,
    log_progress=...,  # 调用方自己的日志措辞
    describe_plan=...,  # --dry-run 打印的那一行（可 None）
    finalize=...,  # (loaded, reports, stats) -> int
    worker_log_dir=None,
    logger=logger,
    per_worker_bytes=per_worker,
    max_mem_gb=max_mem_gb,
)
```

* **分片键**：任意 `tuple`，但驱动按「连续区间」处理进度与合并顺序；Lindhard 用
  `(start, stop)` 行区间。
* **每片计算函数**：必须是模块级可 pickle 的函数（`spawn` 会重新导入模块），
  通过 `worker_payload()` 拿参数，通过 `store.write(key, arrays, meta)` 落盘。
* **合并**：`finalize` 拿到已按 `plan()` 顺序加载好的 `loaded`，负责校验、合并、
  发布产物与收尾日志，返回退出码；`RunStats` 提供行/像素总计与实测值。
* **内存预算**：调用方给出 `per_worker_bytes`（一次运行一份的估算），驱动负责比较
  与拒绝；`available_memory_bytes()`/`memory_guard()` 也可单独调用。

## 4. 接入清单（通用步骤）

1. **定分片键**：找一个「片与片之间互不影响」的轴，键是它的半开区间
   `(start, stop)`；`plan=lambda: plan_slices(work_size, n_workers)`。
2. **写 worker 函数**（必须模块级，`spawn` 会重新导入模块）：
   `def _worker(index, key, store, payload, rq, pq)` —— 算完本片后
   `store.write(key, {<array_key>: arr}, meta)`，再用 `rq.put({...})`（结果）与
   `pq.put((index, (rows_done, rows_total, pixels_done)))`（进度）回报父进程。
3. **identity**：把决定结果的那组参数写进字典（`nk`、`eta`、`normalize`、上游数组的
   SHA 等），`ShardStore(..., identity=signature)`；换模型/换参数时旧分片自动重算。
4. **finalize**：分片已按 `plan()` 顺序加载好，按需 `concatenate`，校验总数，然后走
   既有落盘函数发布产物 —— 产物写入仍然只走 `io.h5_convention`。
5. **调用**：`run_shards(spec, n_workers=..., checkpoint_dir=..., resume=...)`，
   把返回码映射成调用方的语义（CLI 退出码或异常）；`--dry-run` 直接给 `dry_run=True`。
6. **回归**：像 `tests/regression/check_lindhard_re_chi.py`、`check_qpi_parallel.py`
   那样加检查：产物字节幂等、resume 复用、外来签名拒绝、中断后只重算缺失分片。

## 5. 已接入的第二个计算器：QPI JDOS

`JDOSQPI.calculate()`（`src/stm_data_processing/stm/qpi_jdos.py`）是 §4 清单的第一个
真实落地案例；**串行仍是默认**，并行是显式 opt-in：

```python
calc.calculate(energies, q_range=None, output_path=...)  # 串行（默认，行为不变）
calc.calculate(
    energies, q_range=None, n_workers=4, checkpoint_dir="var/qpi_ckpt"
)  # 并行，临时目录
calc.calculate(
    energies, q_range=None, n_workers=4, checkpoint_dir="var/qpi_ckpt", resume=True
)  # 只补缺失分片
```

* **为什么分片键是能量轴**：JDOS 的每一层 `A(k, E)` 由
  `_jdos_layer(eigenvalues, eta, energy, normalize)` 算出，层与层之间唯一的耦合是
  「按各自的 `max` 归一化」——每层除以自己的最大值，所以切片后逐层重算得到的结果与串行
  逐位相同（不是容差相同）。键就是能量下标区间 `(start, stop)`：同一 `plan_slices`
  策略给出的 4 个 worker 分片是 `[0,16) [16,32) [32,48) [48,64)`。
* **调用方供给**：模块级 `_qpi_jdos_worker(worker_id, key, store, payload, rq, pq)`
  与可 pickle 的 `worker_payload()`（`energies`、`eigenvalues`、`eta`、`normalize`、
  分片元数据）；`array_keys=("qpi_layers",)`、`label="JDOSQPI"`。串行循环与 worker
  调用**同一个** `_jdos_layer`，两条路径不可能算岔。
* **合并顺序**：`finalize(loaded, reports, stats)` 拿到的是驱动按 `spec.plan()` 顺序
  加载的分片，`np.concatenate(layers, axis=0)` 后即串行结果；层数总和与 `n_energies`
  不符时返回 6，父进程抛 `RuntimeError` 而不是发出一个缺层的产物。
* **内存上界**：`per_worker_bytes = eigenvalues.nbytes + max_shard_layers * 3 * nk * nk * 8`
  （每个 worker 同时持有 `A(k)`、`|FFT|²` 和输出层三张 `nk×nk` float64），交给
  `memory_guard` 在 dispatch 前裁决；QPI 不给 `max_mem_gb`（只依赖实测可用内存）。
* **identity**：`{nk, eta, normalize, num_wann, n_energies, eigenvalues_sha256}`。
  换模型（特征值不同）、换 `eta`、换归一化开关都会让旧分片被判为外来并重算并打
  WARNING「does not match the requested parameters and is recomputed」。
* **GPU 后端**：`n_workers>1` 且 `BACKEND == "gpu"` 时**不并行**——worker 会继承 CUDA
  上下文，不安全。此时打 WARNING（含「CUDA context」字样）并退回串行 CUDA 路径；
  单能量请求同样退回串行（并行无意义）。
* **中断语义**：驱动捕获 SIGINT 后终止 worker、保留已完成分片并返回 130，适配层把 130
  翻成 `KeyboardInterrupt`（串行路径被 Ctrl-C 也是它），所以调用方看到的仍是「被打断」
  而不是「跑失败」；下一次 `resume=True` 只 dispatch 缺失分片。

## 6. 实测

**驱动抽取（上一轮）**

* `check_lindhard_re_chi.py`：`RESULT: ALL CHECKS PASSED`（exit 0），
  含 (s) resume/reuse、(v) 空/截断分片、(z) 外来模型拒绝、(t) 父日志、(o) Accelerate 宽模型。
* 产物字节幂等：同一 mock 输入、同一输出路径，抽取前后 sha256 完全一致
  （`1c7085d1…10b2`，13872 字节）。
* 中断→恢复：SIGINT 退出 130 并保留已完成分片，`--resume` 只 dispatch 缺失的 3 片、
  复用 1 片。

**QPI JDOS 接入（本轮）** —— `tests/regression/check_qpi_parallel.py`（合成 mock，
不需要外部数据，因此在 CI 里被选中）：

* (a) 串行 vs 并行：`max|Δ| = 0`、uint8 视图逐位相同、冻结时钟后**写出文件的 sha256
  完全相同**（`506027844a0369ba…`，17075 字节）。重复测量 3 种规模 × 2 次共 6 次，
  6 次全部 sha256 相同（最大的 `nk=64`、33 个能量、4 worker：`ec3d2a2ad9a7c071…`，
  683936 字节）。
* (b) 分片是 `io.h5_convention` 写的 HDF5（分块 + gzip/4 + 必需属性）；显式 checkpoint
  目录合并后保留供 resume，自动临时目录合并成功后删除。
* (c) 真 SIGINT：被杀的子进程退出码 `-2`（shell 里即 130，与 Ctrl-C 串行一致），
  4 片里保住已完成的 1 片；`--resume` 只 dispatch 缺失的 3 片、复用 1 片。
* (d) 换 `eta` 的分片被判外来、重算并打 WARNING。
* (e) GPU 后端 + `n_workers`：打 WARNING 后走串行 CUDA 分支，不写任何分片。
* (f) 共享的 `_jdos_layer` 仍与重构前的串行公式逐位相同（6 组 E × normalize）。
  这条不可省：若 `_jdos_layer` 自己漂移，串行与并行会一起漂，(a) 仍会全绿——实测把
  `1/π` 写成 `1/3.14159265` 后 (a)–(e) 全过、只有 (f) 报红。
* 另有一次「只做一次」的证据（不属于门）：用 `git show HEAD:…/qpi_jdos.py` 载入重构前的
  模块，与当前模块在同一 mock 上比串行 `_compute_jdos_cpu` 的返回值，`nk=8/21` 与
  `nk=32/64` 两组都 `max|Δ| = 0`、层数组 sha256 相同 —— 默认路径没有被这次改造推动。
