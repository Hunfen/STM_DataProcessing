# Python 项目 Roadmap

## 🐛 Bug 审查（2026-08-18）

完整报告见 [bug_review_2026-08.md](./bug_review_2026-08.md)，共 54 项（High 10 / Medium 19 / Low 25）。

**P0 立即修复**：
- [ ] `qpi_born.py` 引用未定义的 `hk_grid_cpu/_gpu`（H1）
- [ ] `qpi_jdos.py` 调用不存在的 `_compute_ek2d*`（H2）
- [ ] `mlwf_susceptibility.py` pyFFTW "IFFT" 计划实为正向 FFT（H3）
- [ ] `bare_lindhard.py` CPU 路径多重错误 → 结果恒为零（H4）

**P1 尽快修复**：`lattice_operations.py` 超胞倒格变换/第一 BZ 顶点（H5/H6）、`get_divider` d10/d100（H7）、`diff_gcube.py` 原子坐标错位（H10）、`dos.py` PDOS 失效（M2）、AutoPPT 偏压/减平面错误（M8/M9）等。

## 📋 开发计划

### 1. 自然语言生成轨道选择矩阵类
- [ ] 设计 NL 到矩阵的映射逻辑
- [ ] 实现轨道选择算法
- [ ] 添加输入验证与错误处理
- [ ] 编写单元测试
- [ ] 完善文档与示例

### 2. Logger 简化库
- [x] 封装统一的 logger 接口
- [x] 支持 info/error/warning/debug 级别
- [x] 简化调用方式（减少重复代码）
- [x] 支持日志文件输出
- [x] 添加日志格式化配置

---

## 📅 优先级
| 任务 | 优先级 | 预计工时 |
|------|--------|----------|
| 轨道选择矩阵类 | 🔴 高 | ??? 天 |
| Logger 简化库 | 🟢 低| ??? 天 |


---

```
