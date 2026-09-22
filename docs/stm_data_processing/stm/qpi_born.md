# Born QPI 模块接口文档

## 模块概述

`qpi_born.py` 提供基于 Wannier90 MLWF Hamiltonian 的 Born 近似 QPI (Quasiparticle Interference) 计算功能，支持 CPU/GPU 双后端。

**后端选择**: 继承自 `mlwf_hamiltonian` 模块，通过 `BACKEND` 常量自动确定，无需额外配置。

**k 网格**: 固定为 `[-0.5, 0.5)` 范围，`nkx = nky = nk`。

**并行（可选）**: `calculate(..., n_workers=N)` 把能量轴切成 N 个分片并行计算（默认串行），
结果与串行逐位一致，支持断点续算，详见 §并行计算与断点续算。

**核心公式**:
```
δρ(q,ω) = -(1/π) Im Σ_k Tr[G0(k,ω) V G0(k+q,ω)]
```

---

## 核心类：`BornQPI`

### 类定义

```python
class BornQPI:
    """QPI calculator using Born approximation."""
```

### 构造函数

```python
def __init__(
    self,
    hamiltonian: MLWFHamiltonian,
    nk: int = 256,
    eta: float = 0.005,
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `hamiltonian` | `MLWFHamiltonian` | 已初始化的哈密顿量实例 |
| `nk` | `int` | k 点网格数量 (nkx = nky = nk)，默认 256 |
| `eta` | `float` | 谱展宽参数，默认 0.005 |

**内部属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `num_wann` | `int` | Wannier 函数数量 |
| `nk` | `int` | k 网格大小 |
| `eta` | `float` | 展宽参数 |
| `gf` | `GreenFunction` | 格林函数计算器 |
| `V` | `np.ndarray` | 散射势矩阵 `(nw, nw)`，默认单位矩阵 |
| `k1_grid`, `k2_grid` | `np.ndarray` | k 点网格 `(nk, nk)` |
| `q1_grid`, `q2_grid` | `np.ndarray` | q 点网格 `(nk, nk)` |
| `hk_grid` | `np.ndarray \| cp.ndarray` | 预计算的哈密顿量网格 `(nk, nk, nw, nw)` |
| `V_gpu` | `cp.ndarray \| None` | GPU 模式下的散射势 |
| `V_cpu` | `np.ndarray` | CPU/GPU 模式下的散射势 (CPU 副本) |

**⚠️ 初始化检查**:
- `hamiltonian.num_wann` 必须已初始化且为正数
- 否则抛出 `ValueError`

---

### 实例方法

#### `calculate(energy_range, q_range, V, output_path, n_workers, checkpoint_dir, resume)`

统一 QPI 计算器，自动选择 CPU/GPU 后端；`n_workers` 可选开启分片并行。

```python
def calculate(
    self,
    energy_range: float | np.ndarray | list[float],
    q_range: tuple[float, float] | None = (-0.5, 0.5),
    V: np.ndarray | None = None,
    output_path: str | None = None,
    n_workers: int | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `energy_range` | `float`, `list[float]`, 或 `np.ndarray` | 目标能量值 (eV) |
| `q_range` | `tuple[float, float]` 或 `None` | q 范围 `[q_min, q_max]`，默认 `(-0.5, 0.5)` |
| `V` | `np.ndarray` 或 `None` | 散射势矩阵 `(nw, nw)`，默认单位矩阵 |
| `output_path` | `str` 或 `None` | HDF5 输出路径 (可选) |
| `n_workers` | `int` 或 `None` | **可选并行**：把能量轴切成该数量的分片并行计算，默认 `None`=串行（行为与结果不变） |
| `checkpoint_dir` | `str`, `Path` 或 `None` | 并行分片目录；`None`=临时目录（合并成功后删除），显式目录保留供 `resume` 复用 |
| `resume` | `bool` | 只补「缺失」或「参数不符」的分片，默认 `False` |

**返回**: `dict[str, Any]` 包含以下键：
| 键 | 类型 | 形状 | 说明 |
|----|------|------|------|
| `qpi_layers` | `np.ndarray` | `(n_energies, nq, nq)` | QPI 强度图 |
| `qx_grid`, `qy_grid` | `np.ndarray` | `(nq, nq)` | 实空间 q 坐标 |
| `q1_grid`, `q2_grid` | `np.ndarray` | `(nq, nq)` | 分数坐标 q 网格 |
| `metadata` | `dict` | - | 元数据字典 (见下表) |

**metadata 字典内容**:
| 键 | 类型 | 说明 |
|----|------|------|
| `module_type` | `str` | 模块类型，固定为 `"born"` |
| `eta` | `float` | 展宽参数 |
| `normalize` | `bool` | 是否归一化 |
| `nq` | `int` | q 点数量 |
| `energy_range` | `np.ndarray` | 能量数组 |
| `bands` | `None` | 能带信息 (当前未使用) |
| `bvecs` | `np.ndarray` | 倒格矢 |
| `V` | `np.ndarray` | 散射势矩阵 |
| `mask` | `None` | 掩码 (当前未使用) |

**示例**:
```python
# 单能量
result = qpi.calculate(energy_range=0.5)

# 多能量
result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 50))

# 自定义散射势
V = np.eye(num_wann) * 0.1
result = qpi.calculate(energy_range=0.5, V=V)

# 保存结果
result = qpi.calculate(energy_range=0.5, output_path="./qpi.h5")

# 可选并行（分片目录见 §并行计算与断点续算）
result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 50), n_workers=4)
```

---

## 并行计算与断点续算（opt-in）

`calculate()` 的**默认路径仍然是串行**，`n_workers=None`（以及 `n_workers<=1`）时逐位沿用原有
串行循环、原样结果；并行只在显式给出 `n_workers > 1` 时启用，复用
`stm_data_processing.parallel.shard_driver`（与 `qpi_jdos.py`、`lindhard_re_chi_parallel.py`
同一个分片-合并驱动，接入清单见 `docs/parallel_shard_driver.md` §4）。

```python
calc = BornQPI(ham, nk=256, eta=0.005)

calc.calculate(energies, q_range=None)                     # 串行（默认，行为不变）
calc.calculate(energies, q_range=None, n_workers=4)         # 并行，自动临时分片目录
calc.calculate(energies, q_range=None, n_workers=4,
               checkpoint_dir="var/born_ckpt")              # 并行，显式目录（保留）
calc.calculate(energies, q_range=None, n_workers=4,
               checkpoint_dir="var/born_ckpt", resume=True) # 只补缺失分片
```

### 为什么能量轴是精确的分片键

每一层 `δρ(q, ω)` 只由 `G0(k, ω)`、`V`、`η` 决定，层与层之间**没有任何耦合**（既不跨能量归一化，
也不做能量间的差分），`1/π` 因子也是逐层相乘。因此把能量下标区间 `(start, stop)` 作为分片键，
逐片重算再按分片顺序拼接，得到的数组与串行逐位相同（不是容差相同）。`plan_slices(n_energies, n_workers)`
给出均分、连续、升序、半开的分片区间，例如 64 个能量 / 4 个 worker 得到
`[0,16) [16,32) [32,48) [48,64)`。

### 共享的层函数（两条路径不可能算岔）

模块级 `_born_layer(hk_grid, V, eta, omega)` 是**唯一**的逐能量层算术实现（含 `1/π` 与
`fftshift`）：串行 CPU 路径的 `_compute_Gkq(omega)` 只是它的薄包装（加日志），并行 worker
`_qpi_born_worker(worker_id, key, store, payload, result_queue, progress_queue)` 也调用它。
worker 是模块级函数、参数只含普通数组/dict，因此可以在 `spawn` 下重新导入执行。

| 组件 | 作用 |
|------|------|
| `_born_layer(hk_grid, V, eta, omega)` | 一层 Born QPI：`G0 = (ω+iη−H(k))⁻¹` → `GV = G0·V` → FFT 互相关 → `fftshift` → `/π` |
| `_qpi_born_worker(...)` | 算一片能量区间，用 `store.write(key, {"qpi_layers": layers}, meta)` 落盘并回报进度 |
| `BornQPI._compute_born_parallel(...)` | 组装签名、内存估算、`ShardStore`/`ShardSpec`，调用 `run_shards` 并在 `finalize` 里按序合并 |

### 分片身份（外来分片拒绝合并）

分片 `identity` 覆盖「一层结果依赖的全部量」：

| 签名键 | 说明 |
|--------|------|
| `nk` | k 网格大小 |
| `eta` | 谱展宽 |
| `num_wann` | Wannier 函数个数 |
| `n_energies` | 能量点数（分片键的总长度） |
| `hk_grid_sha256` | `(nk, nk, nw, nw)` complex128 哈密顿量网格的 sha256（换模型或换 k 网格都会变） |
| `V_sha256` | 散射势矩阵的 sha256（同一次 `calculate` 内的 `V` 是逐调用的） |

`resume=True` 时，磁盘上身份不符的分片会被判为**外来**并重算，且打 WARNING
`resume: checkpoint rows=[a, b) does not match the requested parameters and is recomputed`；
缺失或读坏（截断/空文件/行数不符）的分片同样重算，绝不与本次结果混拼。

### 内存估算与合并顺序

* `per_worker_bytes` 一次性估算「一个 worker 同时持有的东西」：载荷 `hk_grid`、求解器物化的
  单位矩阵批、`z*I − H` 中间量、`G0`、`GV`（共 6 个 `(nk, nk, nw, nw)` complex128）、
  4 个 `(nk, nk, nw)` complex128 的 FFT 临时面、以及该片自己累积的 `max_share` 张
  `nk×nk` float64 层；驱动在 dispatch 前用 `memory_guard` 拿它与实测可用内存比较。
* 合并严格按 `spec.plan()` 的顺序（即串行顺序）`np.concatenate(layers, axis=0)`；若层数总和
  不等于 `len(energy_range)`，`finalize` 返回非零码，父进程抛 `RuntimeError`，而不是写出一个
  缺层的产物。
* 分片文件走 `io.h5_convention`（分块 + gzip/4 + 必需属性），每个 worker 原子地写自己的
  `.h5` 与 `.done`；自动临时目录在合并成功后删除，显式 `checkpoint_dir` 保留。

### GPU 后端回退（绝不 fork worker）

`BACKEND == "gpu"` 且 `n_workers > 1` 时**不并行**：spawn 出来的 worker 若继承 CUDA 上下文、
CuPy 内存池是不安全的，因此只打 WARNING（含「CUDA context」字样）并回到**串行 CUDA 路径**；
单能量请求同样回退串行（并行无意义）。CPU 后端下驱动使用默认的 `start_method="spawn"`：
worker 是全新解释器、重新导入本模块，天然不会继承父进程的 CUDA/CuPy 状态，`fork` 做不到这点。

`spawn` 还带来一条**调用方约束**：`n_workers > 1` 时，发起调用的脚本（或 notebook 单元）
必须把自己放在 `if __name__ == "__main__":` 保护之下，否则子解释器重新导入该脚本会再次执行
顶层的 `calculate`，worker 以错误收场——驱动打印 worker traceback 并返回 4，父进程抛
`RuntimeError`。串行路径（不传 `n_workers`）没有这个约束。

### 中断与续算语义

并行运行被 Ctrl-C（SIGINT）打断时，驱动会终止存活 worker、保留已完成分片并返回 130，
本模块把 130 翻成 `KeyboardInterrupt`（与串行路径被 Ctrl-C 一致，消息里给出保留的分片目录），
所以调用方看到的是「被打断」而不是「跑失败」。随后带同一 `checkpoint_dir` 的
`resume=True` 运行只 dispatch 缺失分片（日志 `resume: rows=[a, b) already complete`），
最终产物与一次性串行跑出来的逐位一致。

### 规则速查

| 场景 | 行为 |
|------|------|
| `n_workers=None` / `n_workers<=1` | 串行，结果与原实现字节级一致 |
| `n_workers>1`，CPU 后端，能量点 ≥2 | 分片并行，按 `plan_slices` 均分能量轴 |
| `n_workers>1`，GPU 后端 | WARNING + 串行 CUDA |
| `n_workers>1`，单能量 | WARNING + 串行 |
| `checkpoint_dir=None` | 临时目录，合并成功后自动删除 |
| 显式 `checkpoint_dir` | 保留分片；`resume=True` 只补缺失/不符分片 |
| SIGINT | 130 → `KeyboardInterrupt`，已完成分片保留 |
| 调用脚本缺 `if __name__ == "__main__":` 保护 | worker 报错（spawn 重新导入脚本），退出码 4 → `RuntimeError` |

---

### 内部方法（不建议直接调用）

| 方法 | 说明 |
|------|------|
| `_validate_hamiltonian(hamiltonian)` | 验证 MLWFHamiltonian 对象 |
| `_estimate_q_block_size(g0_gpu, safety_fraction, hard_max_block)` | 估算 CUDA 批量大小，基于 GPU 内存 |
| `_compute_Gkq(omega)` | CPU 串行路径的一层（薄包装，算术全在模块级 `_born_layer`） |
| `_compute_Gkq_cuda(omega)` | GPU 后端 QPI 计算，使用 FFT 卷积定理 |
| `_compute_born_parallel(energy_array, n_workers, checkpoint_dir, resume)` | opt-in 分片并行路径（见 §并行计算与断点续算） |

模块级（供 spawn worker 与串行路径共享）：

| 函数 | 说明 |
|------|------|
| `_born_layer(hk_grid, V, eta, omega)` | 唯一的逐能量层算术（含 `1/π`） |
| `_qpi_born_worker(worker_id, key, store, payload, result_queue, progress_queue)` | 并行 worker：算一片能量区间并落盘 |

---

## 后端检测与检查

### 后端继承逻辑

模块导入时从 `mlwf_hamiltonian` 继承后端状态：

```python
from ..dft.wannier90.mlwf_hamiltonian import BACKEND

if BACKEND == "gpu":
    import cupy as cp
else:
    cp = None
```

### 公开常量：`BACKEND`

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND

print(BACKEND)  # 'cpu' 或 'gpu'
```

| 常量 | 类型 | 说明 |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | 当前激活的计算后端（从 `mlwf_hamiltonian` 导入） |

**⚠️ 注意**: 
- `BACKEND` 在模块导入时确定，运行时不会改变
- GPU 模式下内部使用 `cp.ndarray`，但最终返回 `np.ndarray`
- 无需设置环境变量，后端由 `mlwf_hamiltonian` 自动检测

---

## 内部属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `ham` | `MLWFHamiltonian` | 关联的哈密顿量实例 |
| `num_wann` | `int` | Wannier 函数数量 |
| `nk` | `int` | k 网格大小 |
| `eta` | `float` | 展宽参数 |
| `gf` | `GreenFunction` | 格林函数计算器 |
| `V` | `np.ndarray` | 散射势矩阵 |
| `hk_grid` | `np.ndarray \| cp.ndarray` | 预计算的哈密顿量网格 `(nk, nk, nw, nw)` |
| `V_gpu` | `cp.ndarray \| None` | GPU 模式下的散射势 |
| `V_cpu` | `np.ndarray` | CPU 模式下的散射势 |

---

## 数学公式

### Born 近似 QPI

```
δρ(q,ω) = -(1/π) Im Σ_k Tr[G0(k,ω) V G0(k+q,ω)]
```

其中：
- `G0(k,ω)` : 无杂质格林函数 `(ω + iη - H(k))⁻¹`
- `V` : 散射势矩阵（默认单位矩阵）
- `Tr` : 轨道空间求迹

### GPU FFT 加速实现

对于复数场，使用卷积定理避免共轭问题：

```
Σ_k A[k] · B[k+q] = IFFT( FFT(A*)* · FFT(B) )
```

**推导**:
- 标准互相关：`IFFT(FFT(A)* · FFT(B)) = Σ_k A[k]* · B[k+q]` ← 有共轭
- 目标公式：`Σ_k A[k] · B[k+q]` ← 无共轭
- 解决方案：使用 `A*` 作为输入 → `IFFT(FFT(A*)* · FFT(B)) = Σ_k A[k] · B[k+q]` ✓

实现步骤：
1. 计算 `GV[k,a,c] = Σ_b G[k,a,b] · V[b,c]`
2. 对每个轨道对 `(a,c)`：
   - `fft_gv = FFT(GV[:, :, a, c]*)`
   - `fft_g0 = FFT(G0[:, :, c, a])`
   - `corr = IFFT(fft_gv* · fft_g0)`
   - `qpi += -Im(corr)`
3. `fftshift` 将零频移到中心
4. 除以 `π` 得到最终结果

---

## 使用示例

### 基础用法

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.stm.qpi_born import BornQPI
import numpy as np

# 加载哈密顿量
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# 创建 QPI 计算器
qpi = BornQPI(ham, nk=256, eta=0.005)

# 计算单能量 QPI
result = qpi.calculate(energy_range=0.5)
qpi_data = result["qpi_layers"]  # shape: (1, 256, 256)

# 计算多能量 QPI
energies = np.linspace(-1.0, 1.0, 50)
result = qpi.calculate(energy_range=energies)
qpi_data = result["qpi_layers"]  # shape: (50, 256, 256)
```

### 根据 BACKEND 决定计算策略

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND
from stm_data_processing.stm.qpi_born import BornQPI

if BACKEND == "gpu":
    # GPU 后端：可处理更大网格和更多能量点
    qpi = BornQPI(ham, nk=512, eta=0.005)
    result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 100))
else:
    # CPU 后端：使用较小网格避免内存压力
    qpi = BornQPI(ham, nk=256, eta=0.005)
    result = qpi.calculate(energy_range=np.linspace(-1.0, 1.0, 50))
```

### 自定义散射势

```python
from stm_data_processing.stm.qpi_born import BornQPI
import numpy as np

qpi = BornQPI(ham, nk=256)

# 单位矩阵散射势（默认）
result = qpi.calculate(energy_range=0.5)

# 自定义散射势
V = np.eye(num_wann) * 0.1
result = qpi.calculate(energy_range=0.5, V=V)

# 非对角散射势
V = np.random.rand(num_wann, num_wann) + 1j * np.random.rand(num_wann, num_wann)
result = qpi.calculate(energy_range=0.5, V=V)
```

### 保存结果到 HDF5

```python
from stm_data_processing.stm.qpi_born import BornQPI

qpi = BornQPI(ham, nk=256)

result = qpi.calculate(
    energy_range=np.linspace(-1.0, 1.0, 50),
    output_path="./output/qpi_born.h5",
)
```

### 自定义 q 范围裁剪

```python
from stm_data_processing.stm.qpi_born import BornQPI

qpi = BornQPI(ham, nk=256)

# 裁剪到 [-0.3, 0.3] 范围
result = qpi.calculate(
    energy_range=0.5,
    q_range=(-0.3, 0.3),
)

# 不裁剪（使用完整 [-0.5, 0.5) 范围）
result = qpi.calculate(
    energy_range=0.5,
    q_range=None,
)
```

### 访问返回的 metadata

```python
from stm_data_processing.stm.qpi_born import BornQPI

qpi = BornQPI(ham, nk=256)
result = qpi.calculate(energy_range=0.5)

# 访问元数据
print(result["metadata"]["module_type"])  # "born"
print(result["metadata"]["eta"])  # 0.005
print(result["metadata"]["V"].shape)  # (num_wann, num_wann)
```

---

## GPU 批量优化

GPU 模式下使用 FFT 卷积定理加速，自动进行内存管理：

| 参数 | 说明 |
|------|------|
| `_CUDA_SAFETY_FRACTION` | GPU 内存安全使用比例 (0.25) |
| `_CUDA_HARD_MAX_BLOCK` | 最大批量大小 (64) |

**自动调整逻辑**:
- 根据可用 GPU 内存计算最优批量大小
- 保守估计以容纳额外临时数组
- 每批次后释放 GPU 内存

**CPU vs GPU 实现差异**:
| 特性 | CPU | GPU |
|------|-----|-----|
| 计算方式 | 逐 q 点循环 | FFT 卷积定理 |
| 轨道处理 | 直接求迹 | 轨道对循环 |
| 进度日志 | 每 q 点更新 | 每能量更新 |

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `numpy` | 是 | 核心计算 |
| `cupy` | 否 | GPU 加速（可选，由 `mlwf_hamiltonian` 检测） |
| `GreenFunction` | 是 | 格林函数计算 |
| `MLWFHamiltonian` | 是 | 哈密顿量计算 |
| `qpi_io` | 是 | HDF5 保存、坐标转换 |
| `miscellaneous` | 是 | QPI 扩展裁剪 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `ValueError` | `hamiltonian.num_wann` 未初始化或非正数、`V` 形状不匹配 |
| `RuntimeError` | GPU 后端激活但 CuPy 不可用；并行 worker 失败、分片缺失、checkpoint 目录被占用或内存预算不足 |
| `KeyboardInterrupt` | 并行运行被 Ctrl-C 打断（已完成分片保留，可用 `resume=True` 续算） |

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`hamiltonian` 必须是已初始化的 `MLWFHamiltonian` 实例**
- [ ] **`hamiltonian.num_wann` 必须为正数**
- [ ] **通过 `BACKEND` 常量检查当前后端，决定计算策略**
- [ ] **`nk` 参数为正整数，典型值 256~512**
- [ ] **`eta` 参数为正实数，典型值 0.001~0.01**
- [ ] **`energy_range` 会被转换为 1D `np.float64` 数组**
- [ ] **`V` 形状必须为 `(num_wann, num_wann)`**
- [ ] **返回结果始终为 `np.ndarray`（GPU 内部计算后转换）**
- [ ] **`q_range=None` 时不进行裁剪**
- [ ] **`n_workers=None`（默认）走串行；只有 `n_workers > 1` 才启用分片并行**
- [ ] **并行结果与串行逐位一致；GPU 后端 + `n_workers>1` 会打 WARNING 并回退串行 CUDA**
- [ ] **长任务想续算必须给显式 `checkpoint_dir` 并沿用 `resume=True`**
- [ ] **注意：`BACKEND` 在模块导入时确定，运行时不会改变**
- [ ] **HDF5 保存需要 `output_path` 参数**
- [ ] **CPU 模式计算较慢，建议使用 GPU 处理大网格**
- [ ] **返回字典包含 `metadata` 键，含详细计算参数**

---

## 与 mlwf_hamiltonian 的接口对齐

| 检查项 | 状态 | 说明 |
|-------|------|------|
| 使用 `BACKEND` 常量判断后端 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| `cp` 根据 `BACKEND` 条件导入 | ✅ | 与 `mlwf_hamiltonian` 一致 |
| 无实例后端状态缓存 | ✅ | 直接使用模块级 `BACKEND` |
| 返回类型统一为 `np.ndarray` | ✅ | GPU 内部计算后转换 |
| 使用 `GreenFunction` 计算 G(k,ω) | ✅ | 与 `mlwf_gk` 接口对齐 |
| k 网格范围 `[-0.5, 0.5)` | ✅ | 与 `qpi_jdos` 一致 |

---

## 版本信息

- 模块路径：`src/STM_DataProcessing/src/stm_data_processing/stm/qpi_born.py`
- 后端检测：继承自 `mlwf_hamiltonian`（导入时自动完成）
- 日志级别：`INFO` 用于计算进度通知
- **无环境变量配置，后端自动继承**
- **k 网格范围：固定为 `[-0.5, 0.5)`**
- **输出格式：始终返回 `np.ndarray`**
- **散射势：默认单位矩阵，可自定义**
- **GPU 加速：使用 FFT 卷积定理，避免逐 q 点循环**

---

## CPU vs GPU 性能对比

| 网格大小 | CPU 时间 | GPU 时间 | 加速比 |
|---------|---------|---------|--------|
| 256×256 | ~10 分钟 | ~30 秒 | ~20× |
| 512×512 | ~40 分钟 | ~2 分钟 | ~20× |
| 1024×1024 | ~160 分钟 | ~8 分钟 | ~20× |

**注意**: 实际性能取决于硬件配置和能量点数量。

## 注意事项

### 内存使用

| 模式 | 内存需求 | 说明 |
|------|---------|------|
| CPU | 中等 | 逐 q 点计算，内存占用较低 |
| GPU | 较高 | 预计算 `hk_grid` 占用 `(nk, nk, nw, nw)` 复数数组 |

**GPU 内存估算**:
```
内存 ≈ nk² × nw² × 16 bytes × 2 (G0 + GV)
例如：nk=256, nw=10 → ~2 GB
```

### 计算时间

| 网格大小 | 能量点数 | CPU 预估 | GPU 预估 |
|---------|---------|---------|---------|
| 256×256 | 10 | ~100 分钟 | ~5 分钟 |
| 256×256 | 50 | ~500 分钟 | ~25 分钟 |
| 512×512 | 10 | ~400 分钟 | ~20 分钟 |

**建议**:
- CPU 模式：`nk ≤ 256`，能量点 ≤ 20
- GPU 模式：`nk ≤ 512`，能量点 ≤ 100

### 散射势 V 的选择

| 类型 | 适用场景 | 说明 |
|------|---------|------|
| 单位矩阵 | 点杂质 | 默认值，各向同性散射 |
| 对角矩阵 | 轨道选择性杂质 | 不同轨道散射强度不同 |
| 非对角矩阵 | 轨道混合杂质 | 包含轨道间散射 |

---

## 常见问题 (FAQ)

### Q1: CPU 模式计算太慢怎么办？

**A**: 建议：
1. 减小 `nk` 网格大小（如 256→128）
2. 减少能量点数量
3. 使用 GPU 后端（安装 CuPy 和 CUDA）
4. 用 `n_workers=4`（或更多）开启分片并行：能量轴被切开并行算，结果与串行逐位一致，
   见 §并行计算与断点续算

### Q2: GPU 内存不足怎么办？

**A**: 建议：
1. 减小 `nk` 网格大小
2. 降低 `_CUDA_SAFETY_FRACTION`（默认 0.25）
3. 关闭其他 GPU 占用程序

### Q3: 如何设置非单位矩阵散射势？

**A**: 在 `calculate()` 中传入 `V` 参数：
```python
V = np.eye(num_wann) * 0.1  # 缩放单位矩阵
result = qpi.calculate(energy_range=0.5, V=V)
```

### Q4: QPI 结果出现负值正常吗？

**A**: 正常。Born 近似 QPI 可正可负，表示电子密度的增减。

### Q5: 如何验证计算结果正确性？

**A**: 建议：
1. 对比 JDOS QPI 结果（`qpi_jdos.py`）
2. 检查对称性（如晶体对称性）
3. 验证能量依赖关系是否合理

---

## 与其他模块的比较

### BornQPI vs JDOSQPI

| 特性 | BornQPI | JDOSQPI |
|------|---------|---------|
| 物理模型 | Born 近似（含散射势） | JDOS 近似（无散射势） |
| 计算复杂度 | O(nk⁴ × nw³) | O(nk² × nw) |
| 散射势 | 可自定义 `V` 矩阵 | 无（隐含单位矩阵） |
| GPU 加速 | FFT 卷积定理 | 批量对角化 |
| 适用场景 | 杂质散射研究 | 快速定性分析 |
| 计算时间 | 较慢 | 较快 |

### 选择建议

| 需求 | 推荐模块 |
|------|---------|
| 快速定性分析 | `JDOSQPI` |
| 杂质散射研究 | `BornQPI` |
| 大网格计算 | `JDOSQPI`（GPU） |
| 轨道选择性散射 | `BornQPI`（自定义 V） |

---

## 调试建议

### 日志级别设置

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 或更详细
logging.basicConfig(level=logging.DEBUG)
```

### 内存检查

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND
import cupy as cp

if BACKEND == "gpu":
    mem_info = cp.cuda.Device().mem_info
    print(f"GPU 可用内存：{mem_info[0] / 1e9:.2f} GB")
```

### 中间结果检查

```python
# 检查格林函数
g0 = qpi.gf.compute_green(qpi.hk_grid, omega=0.5)
print(f"G0 形状：{g0.shape}")
print(f"G0 类型：{type(g0)}")

# 检查散射势
print(f"V 形状：{qpi.V.shape}")
print(f"V 对角元：{np.diag(qpi.V)}")
```

---

## 性能优化建议

### CPU 模式

1. 使用 `np.einsum` 的 `optimize=True` 参数
2. 减少能量点数量
3. 使用较小的 `nk` 网格

### GPU 模式

1. 确保 CuPy 使用最新 CUDA 版本
2. 预分配 GPU 内存（已在 `__init__` 中完成）
3. 使用较大的批量大小（自动调整）

### 通用建议

1. 复用 `BornQPI` 实例进行多能量计算
2. 避免重复创建 `GreenFunction` 实例
3. 使用 HDF5 保存中间结果避免重复计算

---

## 更新日志

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-03 | 初始版本，支持 CPU/GPU 双后端 |
| 1.0.1 | 2026-03 | 优化 GPU FFT 实现，修复共轭问题 |
| 1.0.2 | 2026-03 | 添加散射势自定义功能 |
| 1.1.0 | 2026-09 | 新增 opt-in 分片并行能量路径（`n_workers`/`checkpoint_dir`/`resume`）：抽出的模块级 `_born_layer` 被串行路径与并行 worker 共享（结果与串行逐位一致），CSV/HDF5 产物格式与串行相同；GPU 后端回退串行 CUDA；SIGINT 后保留分片并支持 `resume` 续算 |

---

## 相关文档

- [MLWF Hamiltonian 模块](../dft/wannier90/mlwf_hamiltonian.md)
- [MLWF Green's Function 模块](../dft/wannier90/mlwf_gk.md)
- [JDOS QPI 模块](./qpi_jdos.md)
- [QPI IO 模块](../io/qpi_io.md)
- [分片-合并并行驱动](../../parallel_shard_driver.md)