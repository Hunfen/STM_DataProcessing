# 分片检查点（shard checkpoint）的 HDF5 格式

`dft/wannier90/lindhard_re_chi_parallel.py` 的并行模式是「按 q 行切片 → 每片一个 worker →
父进程合并」，因为本机 `h5py.get_config().mpi` 为 `False`（serial-only h5py），
并行写只能靠「每 worker 写自己的分片文件 + 合并」。分片文件原先用 `np.savez` 加一个 JSON
边车，现已改为 **HDF5**，并统一走 `src/stm_data_processing/io/h5_convention.py`
（该模块是全包唯一调用 `create_dataset` 的地方，见 `docs/hdf5_convention.md`）。

## 1. 一片 = 一个文件 + 一个完成标记

```
<checkpoint_dir>/
├── rows_0_32.h5      # 分片本体（HDF5）
├── rows_0_32.done    # 完成标记，内容恒为 "ok\n"
├── rows_32_64.h5
├── rows_32_64.done
├── worker_0.log      # 每个 worker 的明细日志
└── .lock             # 同一 checkpoint 目录只允许一个父进程
```

* 文件名由行区间决定：`rows_<start>_<stop>.h5`（`_shard_stem`/`_shard_paths`）。
* `.h5` 先写临时文件再 `replace()`（原子落盘），**`.done` 最后写**：因此
  「`.h5` 存在但 `.done` 缺失」= 未完成分片，`resume` 会重算而不是误用半成品。
* 分片是 worker 私有产物，父进程只读；合并完成后，自动 checkpoint（未显式给
  `--checkpoint-dir` 时用的临时目录）整体删除，显式给出的目录保留供 `--resume` 复用。

## 2. 文件内部结构（dump）

```
$ .venv/bin/python -c "..."   # 见下方字段表
file attrs: schema_version=1  generator='lindhard_re_chi_parallel'  creation_date='...'
            worker_id, row_range, nk, eta, num_wann, orbital_select, bvecs,
            chemical_potential, temperature, include_matrix_elements,
            degeneracy_tolerance, mirror, elapsed_s, rss_peak_bytes, created_unix
            sha256_json, digest_json        # 嵌套字典 -> JSON 文本属性
dset 'data'       : shape=(rows, nq) dtype=float64 chunks=(...) compression=gzip opts=4
dset 'intraband'  : shape=(rows, nq) dtype=float64 chunks=(...) compression=gzip opts=4
dset 'interband'  : shape=(rows, nq) dtype=float64 chunks=(...) compression=gzip opts=4
```

* 三个数据集就是 `_ARRAY_KEYS = ("data", "intraband", "interband")`，一律
  `float64`，由 `h5_convention.create_dataset()` 写出，因此继承显式分块与 gzip/4。
* 文件属性 = `h5_convention.write_file_metadata()` 要求的
  `schema_version` / `generator` / `creation_date` **加上**该分片的检查点元数据。
* HDF5 属性只能存标量、字符串和规则数值数组，所以嵌套字典
  （`sha256`、`digest`）以 `<key>_json` 为名存成 JSON 文本；读回时
  `_shard_meta()` 会还原成与旧 JSON 边车逐键相同的 Python 值，这样
  `_shard_is_compatible()` 的逐键比较语义完全不变。

## 3. 检查点身份（拒绝外来分片）

分片元数据里记录 `nk`、`eta`、`chemical_potential`、`temperature`、
`include_matrix_elements`、`degeneracy_tolerance`、`orbital_select`、`num_wann`、
`bvecs`。`resume` 时父进程按同一组值构造签名并逐键比较
（`_shard_is_compatible`），不匹配的分片**不删不改**，只在父日志里报 WARNING
（`resume: checkpoint rows=[a, b) does not match the requested parameters and is
recomputed`），然后重算该片。`orbital_select=None`（全部轨道）在分片里展开为
`range(num_wann)`，因此与任意「全轨道」请求相容。

## 4. 生命周期与恢复语义

| 情况 | 行为 |
| --- | --- |
| 正常运行 + 合并成功 | 自动 checkpoint 目录整体删除；显式目录保留 |
| 分片缺失（`.h5` 或 `.done` 不在） | 视为缺失，重新 dispatch |
| 分片不可读（空文件、被截断、非 HDF5） | 父进程 WARNING `unreadable checkpoint ... (OSError): ...` 并重算该片 |
| 分片行数与区间不符 | 父进程 WARNING 并重算该片 |
| 分片来自别的模型 / 不同 `num_wann` / 不同参数 | WARNING「不匹配请求参数，重算」，不参与合并 |
| 中断（SIGINT） | 活着的 worker 被终止，**已完成的分片保留**，退出码 130 |
| 合并 | 按行区间（分片键）升序拼接，先拼 primitive-BZ 结果，再用同一批分片做 `q_range` 裁剪 |

实测的中断-恢复（`var/t10/interrupt_resume.py`，mock 模型，4 worker / nk=128）：

```
(1) 收到 SIGINT 时：退出码 130，磁盘上已完成 1 片（rows_32_64.h5 + .done），其余 3 片缺失
(2) --resume：dispatched 3（rows=[0,32)、[64,96)、[96,128)），reused 1（rows=[32,64) 已完整），
    合并后写出产物 81211 字节
```

## 5. worker / 内存预算

* worker 数默认 `min(cpu_count - 1, 8)`；每个 worker 需要 `estimate_worker_rss_bytes()`
  估算的常驻内存（含 H(k)/本征向量/结果数组）。
* 启动前 `_memory_guard()` 用可用内存（`available_memory_bytes()`）与
  `--max-mem-gb` 判定：估算总量超出预算就拒绝启动（退出码 3），
  `--dry-run` 只打印计划与估算值。

## 6. 复用

分片-合并机制（分片键规划、worker 派发、恢复过滤、内存预算、按序合并与分片文件生命周期）
已抽成独立模块 `src/stm_data_processing/parallel/shard_driver.py`，接入方式见
[`docs/parallel_shard_driver.md`](parallel_shard_driver.md)；`lindhard_re_chi_parallel.run_parallel`
现在只是**声明物理**的薄适配层（构造一个 `ShardSpec` 并调用 `run_shards`），本文件描述的
格式与生命周期语义由驱动保证。
