# 分片-合并并行驱动（`stm_data_processing.parallel.shard_driver`）

本模块把「按分片键切分工作 → 每片一个 worker 写自己的分片文件 → 父进程按序合并」这套
机制从 Lindhard 并行计算器里抽了出来，任何计算器都能复用。模型选择的理由见
`docs/hdf5_checkpoint_shards.md`：本机 `h5py.get_config().mpi` 为 `False`，
并行写只能「每 worker 一个文件 + 合并」。

* 代码：`src/stm_data_processing/parallel/shard_driver.py`
* 包入口：`src/stm_data_processing/parallel/__init__.py`（`__all__` 导出下面全部名字）
* 第一个使用者：`src/stm_data_processing/dft/wannier90/lindhard_re_chi_parallel.py`
  （`run_parallel` 现在只负责声明物理，循环交给 `run_shards`）

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

## 4. 下一个计算器怎么接入（走查）

以「给 QPI 计算器加并行」为例（QPI 本身的改造是后续任务，这里只列步骤）：

1. **定分片键**：QPI 逐能量层独立，键可以是 `(layer_start, layer_stop)`；
   `plan=lambda: plan_slices(n_energies, n_workers)`。
2. **写 worker 函数**（模块级）：`def _qpi_worker(index, key, store, payload, rq, pq)`
   —— 计算该层的 `qpi_layers`，`store.write(key, {"qpi_layers": arr}, meta)`，
   把进度 `rq.put({"worker_id": index, "status": "ok", "rss_peak_bytes": ...})`、
   进度 `pq.put((index, (rows_done, rows_total, pixels_done)))` 发回父进程。
   `store.array_keys` 设为 `("qpi_layers",)`。
3. **identity**：把决定结果的那组参数写进字典（`nk`、`eta`、`normalize`、`bvecs` 的
   SHA 等），`ShardStore(..., identity=signature)`；换模型/换参数时旧分片自动重算。
4. **finalize**：把所有分片按序 `concatenate`（或你需要的合并规则），
   校验（例如逐片 digest 与 worker 报告比对），然后调用既有的
   `save_qpi_to_h5(...)` 发布产物 —— 产物写入仍然只走 `io.h5_convention`。
5. **调用**：`run_shards(spec, n_workers=..., checkpoint_dir=..., resume=..., max_mem_gb=...)`，
   把返回码当成 CLI 退出码；`--dry-run` 直接交给 `dry_run=True`。
6. **回归**：像 `tests/regression/check_lindhard_re_chi.py` 那样加一组检查：resume 复用、
   空/截断分片自愈、外来签名拒绝、中断后只重算缺失分片、产物字节幂等。

## 5. 实测（本轮抽取）

* `check_lindhard_re_chi.py`：`RESULT: ALL CHECKS PASSED`（exit 0），
  含 (s) resume/reuse、(v) 空/截断分片、(z) 外来模型拒绝、(t) 父日志、(o) Accelerate 宽模型。
* 产物字节幂等：同一 mock 输入、同一输出路径，抽取前后 sha256 完全一致
  （`1c7085d1…10b2`，13872 字节）。
* 中断→恢复：SIGINT 退出 130 并保留已完成分片，`--resume` 只 dispatch 缺失的 3 片、
  复用 1 片。
