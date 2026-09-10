# latticeSIM — 倒格子 / LEED / RHEED 桌面应用

[`docs/reciprocal_lattice_simulator.html`](../../docs/reciprocal_lattice_simulator.html) 的 Tauri 2 桌面封装：把单页前端（晶格倒格矢 / LEED / RHEED 三个 tab）打包成原生应用。

## 目录结构

```
app/latticeSIM/
├── package.json / package-lock.json   # 只依赖 @tauri-apps/cli（构建工具），版本已锁定
├── ui/
│   ├── index.html                     # 完整前端，三 tab，无外部 CDN 依赖
│   └── diffraction.js                 # 衍射物理核心（window.Diffraction）
└── src-tauri/                         # Tauri 2 Rust 壳
    ├── Cargo.toml / Cargo.lock        # 依赖版本已锁定
    ├── tauri.conf.json                # productName / identifier / frontendDist = ../ui
    ├── capabilities/default.json      # core:default 权限
    ├── icons/                         # 应用图标（构建必需，已入库）
    └── src/{main.rs, lib.rs}
```

## 前置依赖

| 依赖 | 版本 / 说明 |
| --- | --- |
| Node.js + npm | ≥ 18（实测 26.x），仅用于安装 Tauri CLI |
| Rust | stable 工具链（rustup），`Cargo.toml` 要求 `rust-version = 1.77.2` |
| 系统库 | macOS：Xcode Command Line Tools（`xcode-select --install`）；Linux：webkit2gtk 等；Windows：WebView2 |

## 从 clone 重建

```bash
git clone git@github.com:Hunfen/STM_DataProcessing.git
cd STM_DataProcessing/app/latticeSIM

npm install                      # 安装 @tauri-apps/cli（node_modules/ 不入库）
npx tauri build --bundles app    # 只打 .app；去掉 --bundles 则打全部 bundle 格式
```

产物：

```
app/latticeSIM/src-tauri/target/release/bundle/macos/reciprocal-lattice-simulator.app
```

首次构建需要联网拉取 crates.io 依赖（版本由 `Cargo.lock` 锁定），全量 release 编译在本机实测约 2 分钟（1m47s）。之后在已有 `target/` 的工程里增量重建约 20 秒。

## 开发模式

```bash
cd app/latticeSIM
npx tauri dev
```

`tauri.conf.json` 只配置了 `frontendDist: "../ui"`，没有 `beforeDevCommand`：前端是纯静态文件，没有打包步骤，改完 `ui/index.html` / `ui/diffraction.js` 直接刷新窗口即可。

## 说明

- `node_modules/`、`src-tauri/target/`、`src-tauri/gen/schemas/` 都是生成物，已由 `.gitignore` 排除，clone 后由 `npm install` 与构建过程自动生成，无需入库。
- 前端资源在构建时内嵌进二进制，产出的 `.app` 可独立分发，运行时不需要 `ui/` 目录。
- 未做代码签名与公证（`identifier: com.example.reciprocal-lattice-simulator`）。分发到其他 macOS 机器时若被 Gatekeeper 拦截，执行 `xattr -dr com.apple.quarantine <app>` 或自行签名。
- Python 侧的同款物理核心在仓库根的 `src/stm_data_processing/leed_rheed/`（`kinematic.py` / `kikuchi.py`），与此前验证与 `ui/diffraction.js` 数值一致（~1e-12）。

## 构建问题排查

- `npm error Your cache folder contains root-owned files`：npm 缓存目录属主异常，执行 `sudo chown -R "$(id -u):$(id -g)" ~/.npm`，或临时改用 `npm install --cache /tmp/npm-cache`。
- `cargo` 报无法写入 `~/.cargo`：改用可写目录，如 `CARGO_HOME=/tmp/cargo-home npx tauri build --bundles app`。
- 移动过整个工程目录后构建报错：`tauri-build` 会缓存旧的绝对路径，执行 `cargo clean` 后重新构建。
