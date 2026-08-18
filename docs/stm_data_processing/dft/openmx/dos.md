# OpenMX DOS 模块接口文档

## 模块概述

`dos.py` 提供 OpenMX 态密度（DOS）与投影态密度（PDOS）目录树的加载功能，将 OpenMX 生成的 DOS 输出目录解析为便于后续分析的 Pandas DataFrame 结构。

**输入格式**: OpenMX 的 DOS 输出目录，默认名称为 `DOS`，期望布局为：

```
DOS/
├── system.DOS.total          # 总态密度文件（可选，必须唯一）
├── atom1/
│   ├── s1                    # s 轨道 PDOS
│   ├── p1, p2, ...           # p 轨道 PDOS
│   ├── d1, d2, ...           # d 轨道 PDOS
│   └── total                 # 该原子的总 PDOS
├── atom2/
│   └── ...
└── ...
```

每个 DOS 文件为空白分隔的三列文本：能量（eV）、态密度（states/eV）、积分态密度 IDOS；以 `#` 开头的行视为注释。

**输出格式**: 一个 `dict`，键为 `total`（总态密度 DataFrame 或 `None`）与 `pdos`（以原子序号为键的嵌套字典）。

---

## 核心函数

### `load_dos_tree`

```python
def load_dos_tree(dos_dir: str | Path = "DOS") -> dict
```

加载目录树中的总态密度与投影态密度。

| 参数 | 类型 | 说明 |
|------|------|------|
| `dos_dir` | `str \| Path` | DOS 目录路径，默认 `"DOS"` |

**返回**: `dict`，结构如下：

| 键 | 类型 | 说明 |
|----|------|------|
| `total` | `pd.DataFrame \| None` | 系统总态密度，列为 `['E', 'DOS', 'IDOS']`；若无匹配文件则为 `None` |
| `pdos` | `dict[int, dict]` | 按原子序号索引的 PDOS 字典 |

每个原子的 `pdos[atom]` 字典键：

| 键 | 类型 | 说明 |
|----|------|------|
| `s` | `pd.DataFrame \| None` | s 轨道 PDOS（若存在 `s1` 等文件） |
| `p` | `dict[str, pd.DataFrame]` | p 轨道 PDOS，如 `'p1'`、`'p2'` |
| `d` | `dict[str, pd.DataFrame]` | d 轨道 PDOS，如 `'d1'`、`'d2'` |
| `total` | `pd.DataFrame` | 该原子的总 PDOS（匹配不到 `.s\d`/`.p\d`/`.d\d` 后缀的文件） |

**文件归类规则**（对 `atomN/` 目录下的每个文件，按文件名后缀正则匹配）：

- `\.s\d$` → 存入 `pdos[atom]["s"]`
- `\.p\d$` → 存入 `pdos[atom]["p"][name[-2:]]`（如 `p1`）
- `\.d\d$` → 存入 `pdos[atom]["d"][name[-2:]]`（如 `d1`）
- 其余 → 存入 `pdos[atom]["total"]`

**原子目录解析**: 目录名形如 `atomN`（`N` 为数字），取 `int(atom_dir.name.replace("atom", ""))` 作为原子序号；非数字目录名被跳过。

---

## 数学公式

DOS 文件三列含义：

```
DOS(E)  = dN/dE         # 态密度（states/eV）
IDOS(E) = ∫_{-∞}^{E} DOS(E') dE'   # 积分态密度（states）
```

本模块仅负责读取，不进行任何计算或单位转换——三列按原样载入，列名依次为 `E`、`DOS`、`IDOS`。

---

## 使用示例

### 加载默认 `DOS` 目录

```python
from stm_data_processing.dft.openmx.dos import load_dos_tree

dos = load_dos_tree("DOS")

# 系统总态密度
total_df = dos["total"]
if total_df is not None:
    print(total_df.columns.tolist())   # ['E', 'DOS', 'IDOS']
    print(total_df["E"].values)        # 能量轴 (eV)

# 原子 1 的投影态密度
atom1 = dos["pdos"][1]
print(atom1.keys())                    # dict_keys(['p', 'd', ...])
print(atom1["p"]["p1"].head())         # p1 轨道 PDOS
print(atom1["total"].head())           # 原子总 PDOS
```

### 指定自定义目录

```python
from stm_data_processing.dft.openmx.dos import load_dos_tree

dos = load_dos_tree("./out/DOS")
```

### 遍历所有原子的总 PDOS

```python
from stm_data_processing.dft.openmx.dos import load_dos_tree

dos = load_dos_tree("DOS")

for atom, proj in sorted(dos["pdos"].items()):
    df = proj.get("total")
    if df is not None:
        print(atom, df.shape)
```

### 绘制总态密度

```python
import matplotlib.pyplot as plt
from stm_data_processing.dft.openmx.dos import load_dos_tree

dos = load_dos_tree("DOS")
if dos["total"] is not None:
    plt.plot(dos["total"]["E"], dos["total"]["DOS"])
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.show()
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `pandas` | 是 | 以空白分隔读取 DOS 文本文件 |
| `pathlib` | 是 | 目录与文件路径处理（标准库） |
| `re` | 是 | 按后缀正则匹配归类轨道文件（标准库） |
| `logging` | 是 | 日志输出（标准库） |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `FileNotFoundError` | `dos_dir` 目录不存在 |
| `RuntimeError` | `dos_dir` 下匹配 `*.DOS.*` 的总态密度文件超过一个（不唯一） |

**注意**: 若目录中没有任何 `atom*` 子目录，仅记录 `WARNING` 日志，不抛出异常。

---

## 接口对齐检查清单

生成调用此模块的代码时，请确保：

- [ ] **`dos_dir` 目录存在，否则抛出 `FileNotFoundError`**
- [ ] **总态密度文件 `*.DOS.*` 在该目录下唯一（0 个或 1 个），多个则抛出 `RuntimeError`**
- [ ] **返回结构为 `{'total': DataFrame|None, 'pdos': {atom: {...}}}`**
- [ ] **`total` 与各原子 DataFrame 列名均为 `['E', 'DOS', 'IDOS']`**
- [ ] **`pdos` 的键为整数原子序号（由 `atomN` 目录名解析）**
- [ ] **p/d 轨道以 `dict[str, DataFrame]` 存储，键为文件名末两位（如 `'p1'`）**

---

## 版本信息

- 模块路径：`src/stm_data_processing/dft/openmx/dos.py`
- 日志级别：`WARNING`（无 `atom*` 目录时）
- **无环境变量配置**
- **输入格式：OpenMX DOS 目录树（空白分隔三列文本）**
- **输出格式：`{'total', 'pdos'}` 字典，DataFrame 列为 `E`/`DOS`/`IDOS`**
