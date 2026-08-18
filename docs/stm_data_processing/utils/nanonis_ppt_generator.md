# Nanonis 数据自动 PPT 生成器模块接口文档

## 模块概述

- **模块路径**：`src/stm_data_processing/utils/nanonis_ppt_generator.py`
- **职责**：将 STM 实验数据（形貌图 `.sxm`、STS 单谱 `.dat`、linecut / map `.3ds`）自动整理并生成 PowerPoint 演示文稿，包含封面页、形貌图页、单谱页、linecut 页、map（含 QPI、电流图）页，并生成偏压序列 GIF 动画。它是 `utils/AutoPPt_winnew_modified.py` 的重构版，行为与 `docs/stm_data_processing/stm/auto_powerpoint.md` 描述一致。
- **⚠️ 运行形态**：本模块是**脚本式模块**——顶层代码（交互输入、主流程、`prs.save`）在 **import 时即会执行**，不满足「可安全 import」的条件；应通过命令行直接运行脚本，而非作为库导入。
- **交互提示**：通过 `logger.info(...)` 输出提示，配合 `input()` 读取用户输入；`logger` 用 `logging.basicConfig(level=logging.INFO, format="%(message)s")` 配置。

---

## 核心函数

> 以下函数为模块内定义的顶层函数；交互输入与主流程为顶层语句，不属于函数。

### `get_creation_time(file_path)`

跨平台获取文件创建时间。

```python
def get_creation_time(file_path):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | `str \| Path` | 文件路径 |

**返回**：`float`，创建时间戳。macOS（`platform.system() == "Darwin"`）用 `st_birthtime`，其余平台用 `st_ctime`。

### `sort_files_by_creation_time(files)`

按创建时间升序排序文件列表。

```python
def sort_files_by_creation_time(files):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `files` | `Iterable` | 文件路径列表 |

**返回**：`list`，按创建时间升序排列的文件路径列表。

### `read_all_files(folder_path)`

递归收集文件夹内所有文件。

```python
def read_all_files(folder_path: Path):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder_path` | `Path` | 数据文件夹路径 |

**返回**：`list[Path]`，`folder_path.rglob("*")` 中所有文件（`file.is_file()`）。

### `sort_files(files)`

按后缀与内容将文件分类为 dat / sxm / linecut / map 四类。

```python
def sort_files(files):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `files` | `Iterable` | 待分类文件列表 |

**返回**：`(dat_files, sxm_files, linecut_files, map_files)` 四元组：

| 分类 | 判定规则 |
|------|----------|
| `dat_files` | 后缀 `.dat` 且 `nap.read.Spec(...).signals["Bias calc (V)"]` 可读 |
| `sxm_files` | 后缀 `.sxm` |
| `linecut_files` | 后缀 `.3ds` 且 `LI Demod 1 X (A)`（缺省回退 `[AVG]`）`data.shape[0] == 1` |
| `map_files` | 后缀 `.3ds` 且 `data.shape[0] != 1` |

### `find_nearest_file(target_path, file_list)`

在 `file_list` 中寻找创建时间早于 `target_path` 且最接近的文件（用于匹配形貌图与谱/map）。

```python
def find_nearest_file(target_path, file_list):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_path` | `str \| Path` | 目标文件路径 |
| `file_list` | `list` | 候选文件列表 |

**返回**：最接近的候选文件；`file_list` 为空或找不到更早文件时返回字符串 `"Topography Not Found"`。匹配条件为 `0 < target_time - f_time < min_diff`（仅取早于目标、且差值最小者）。

### `add_title_slide(slide, folder_name, folder_time)`

在幻灯片上添加封面标题（文件夹名 + 时间 + 蓝色分隔线）。

```python
def add_title_slide(slide, folder_name, folder_time):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `slide` | `pptx.slide.Slide` | 目标幻灯片 |
| `folder_name` | `str` | 文件夹名（显示为标题，字号 44） |
| `folder_time` | `str` | 时间字符串（取 `.` 前部分，字号 18） |

**返回**：`None`。添加文本框、`RGBColor(0, 0, 255)` 蓝色矩形装饰。

### `add_section_header(slide, title)`

在幻灯片上添加小节标题栏。

```python
def add_section_header(slide, title):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `slide` | `pptx.slide.Slide` | 目标幻灯片 |
| `title` | `str` | 小节标题（字号 24） |

**返回**：`None`。添加文本框与蓝色分隔线。

---

## 交互输入流程

模块顶层（第 37–85 行）依次通过 `logger.info` 提示、`input()` 读取，交互顺序与 `auto_powerpoint.md` 一致：

| 顺序 | 提示内容 | 变量 | 默认值 / 处理 |
|------|----------|------|---------------|
| 1 | 输入数据文件夹路径（逗号分隔多个） | `DataFolderpath` | 空输入直接 `sys.exit` 终止 |
| 2 | 输入 PPT 存储目录 | `Storagepath` | 回车默认第一个数据文件夹 |
| 3 | 输入 PPT 文件名 | `PPTname` | 缺省自动补 `.pptx`；回车生成 `auto_YYYYMMDD_HHMMSS.pptx` |
| 4 | 是否输出 QPI 图（on/off） | `QPI_switch` | 回车或 `on` 为 `True`（默认开） |
| 5 | 是否输出电流图（on/off） | `mapI_switch` | 仅 `on` 为 `True`（默认关） |
| 6 | 是否高斯平滑谱/图（on/off） | `smooth_switch` | 仅 `on` 为 `True`（默认关） |

随后创建 `prs = Presentation()`，设置 `matplotlib.use("Agg")` 与字体（Arial、`font.size=22`）。

---

## 主处理流程

对每个 `Folderpath` 依次执行：

1. **封面页**：`name_folder = "/".join(Folderpath.parts[-3:])`，`time_folder` 取文件夹创建时间，调用 `add_title_slide`。
2. **分类排序**：`read_all_files` → `sort_files` → 四类文件各自 `sort_files_by_creation_time`；SXM 文件用 `plot_sxm_topo` 试绘校验，剔除无效项。
3. **形貌图页**：每页最多 8 张，4 列 2 行；同时构建概览框（`subtractMeanPlane` + `img_rotate_for_box`），每页结束保存 `boxtemp.tif` 缩略图；标注时间、文件名、扫描范围、角度、设定点电压/电流。
4. **STS 单谱页**：每页最多 5 组，上排 `plot_sts` 谱图、下排对应形貌标记图；找不到形貌时显示 `"Topography Not Found"`。
5. **linecut 页**：每页 2 组，`plot_linecut` 返回瀑布图 + overlap 图 + 形貌标记图。
6. **map 页**：每个 map 生成各偏压 `temp_map_{n}.tif`（存入 `folder_map`）与 GIF 动画（`FuncAnimation` + `pillow`，`fps=4`）；每页最多 10 张，首页附动画 GIF、原始形貌图、`temp_inmap.tif`（map 内嵌形貌）。
7. **可选 QPI 页**：`QPI_switch` 开启时生成 `temp_QPI_{n}.tif` 与 `animation_QPI.gif`。
8. **可选电流图页**：`mapI_switch` 开启时生成 `temp_mapI_{n}.tif` 与 `animation_mapI.gif`。
9. **清理**：删除 `Storagepath` 下 `*temp*.tif` 与 `folder_map`/`folder_QPI`/`folder_mapI` 目录，清空四类文件列表。
10. **保存**：`prs.save(Storagepath / PPTname)`，并 `logger.info` 打印保存路径。

---

## 运行方式

本模块依赖同目录的 `plot_funcs`（顶层导入 `from plot_funcs import ...`），因此需保证 `plot_funcs.py` 在 `sys.path` 中。推荐做法：

```bash
# 方式一：在 utils 目录下直接运行脚本
cd src/stm_data_processing/utils
python nanonis_ppt_generator.py

# 方式二：指定绝对路径运行（脚本所在目录会自动加入 sys.path[0]）
python src/stm_data_processing/utils/nanonis_ppt_generator.py
```

运行后按终端提示依次输入：

```
Please enter data processing folder path (separate multiple paths with commas):
> /path/to/data1,/path/to/data2

Please enter PPT storage directory path (press Enter to save in data folder path):
>

Please enter PPT file name (without path, press Enter to auto-generate):
>

Output QPI images? Enter 'on' or 'off' (press Enter for default 'on'):
>

Output current maps? Enter 'on' or 'off' (press Enter for default 'off'):
>

Apply Gaussian smoothing to spectra/maps? Enter 'on' or 'off' (press Enter for default 'off'):
>
```

> 说明：脚本顶部 shebang 为 `#!/Users/hunfen/.../.venv/bin/python`（作者本机虚拟环境，可忽略或按需修改）。因模块顶层含 `input()` 与主流程，**不要**用 `import stm_data_processing.utils.nanonis_ppt_generator` 的方式调用。

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| `python-pptx`（`pptx`） | 是 | 生成 PPT |
| `matplotlib` | 是 | 绘图、GIF 动画 |
| `nanonispy` | 是 | Nanonis 数据读取 |
| `plot_funcs`（同目录） | 是 | 复用绘图函数 |
| `pillow`（`matplotlib` 的 `FuncAnimation` writer） | 是 | GIF 动画保存 |
| `numpy` | 间接 | 经 `plot_funcs`/数据数组使用 |
| `shutil` / `pathlib` / `platform` / `datetime` / `logging` / `sys` | 是 | 标准库工具 |

---

## 错误处理

| 异常 | 触发条件 |
|------|----------|
| `SystemExit` | 首次交互输入的数据文件夹路径为空时 `sys.exit("Error: No data folder path entered...")` |
| `Exception`（被捕获） | `sort_files` 中 `.dat`/`.3ds` 读取校验失败时静默跳过；SXM 校验失败时从列表剔除 |
| `FileNotFoundError` | 目标数据文件不存在（由 `nanonispy`/`matplotlib` 抛出） |

> 交互开关的容错逻辑：`QPI_switch = input().strip().lower() in ("", "on")`（默认开），`mapI_switch`/`smooth_switch` 用 `== "on"`（默认关）。

---

## 接口对齐检查清单

生成调用/修改此模块的代码时，请确保：

- [ ] **本模块为脚本式模块，顶层含 `input()` 与主流程，不可安全 `import`，应直接运行脚本**
- [ ] **`from plot_funcs import ...` 依赖同目录 `plot_funcs.py` 在 `sys.path` 中**
- [ ] **`get_creation_time` 在 macOS 用 `st_birthtime`，其余平台用 `st_ctime`**
- [ ] **`find_nearest_file` 找不到时返回字符串 `"Topography Not Found"`（非 `None`），调用方须按字符串比较判断**
- [ ] **`sort_files` 返回四元组 `(dat, sxm, linecut, map)`，顺序固定**
- [ ] **QPI 开关默认开、电流图/平滑默认关（`input` 判定逻辑不同）**
- [ ] **PPT 文件名未指定时自动生成为 `auto_YYYYMMDD_HHMMSS.pptx`**
- [ ] **中间文件（`*temp*.tif`、`folder_map`/`folder_QPI`/`folder_mapI`）处理完会被删除**
- [ ] **GIF 用 `pillow` writer、`fps=4`、帧间隔 `interval=100`**

---

## 版本信息

- 模块路径：`src/stm_data_processing/utils/nanonis_ppt_generator.py`
- 前身：`utils/AutoPPt_winnew_modified.py`（功能背景见 `docs/stm_data_processing/stm/auto_powerpoint.md`）
- 绘图后端：`matplotlib.use("Agg")`（无图形界面环境可用）
- 日志级别：`logging.INFO`（`format="%(message)s"`，交互提示经 `logger.info` 输出）
- **脚本式模块，需命令行交互运行，产出单个 `.pptx` 文件**
