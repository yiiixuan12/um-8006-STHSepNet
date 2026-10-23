# Agent log — AI-JINQI-001 (2026-10-21 · Jinqi)

> 补录说明：2026-10-23 根据本次对话记录重建。未找到原日志文件，本文是脱敏摘要，不是恢复的逐字对话。历史验证不代表合并后代码已重新通过验证。

- 阶段: Implementation / testing / debugging / experiment execution support
- 工具/模型/版本: OpenAI Codex；历史具体模型和版本未保留，记为 unknown；采样设置 not exposed。
- 委托任务: 保持 baseline/improved 命令不变，补全数据集适配、自动登记实验并修复加载错误。

## 中文对话/输出摘要

用户要求每次训练结束自动追加实验登记，支持现有全部数据集，并保留原有输出目录结构。后续明确拒绝 started 状态，要求只记录完成、失败和捕获到的中断。排查 BIKE 时间间隔异常时发现直接将时间戳转换为整数依赖底层时间单位，需要显式统一为纳秒。

## 修改代码与具体部分

- `research/run_coursework.py`：扩展数据集选项和配置选择；在训练入口接入登记包装；保留 baseline/improved 及 --output 的使用方式。
- `research/experiment_registry.py`：增加每次调用的 run_id、配置与报告副本和结束记录；登记 commit、源码/数据/邻接矩阵 SHA256、工作区状态、配置、seed、硬件、耗时、状态、测试指标和输出路径等溯源信息。
- `research/coursework_data.py`：补全 PEMS03、METR-LA、BJ500、BIKE-Inflow、BIKE-Outflow 的数据适配。
- `research/traffic_data.py`：时间戳先显式转换为 datetime64[ns] 再转整数，避免时间单位不同导致间隔检查失败。
- `tests/test_experiment_registry.py`、`tests/test_coursework.py`：覆盖登记行为与时间戳/数据处理回归。
- 历史上还编写了 `script/run_bj500_metr_baseline.ps1`，顺序运行 BJ500、METR-LA 的 2021–2023 种子。补录时该脚本不在当前工作树，不能作为当前可执行证据。

## 决定: Modified

接受数据适配和自动登记；按用户要求移除 started，只在调用结束时登记 completed、failed 或 interrupted。续训使用新 run_id；强制结束或断电无法保证写入。权重保留在 --output 指定目录，配置与报告副本保存在 experiments/runs/<run_id>/。费用 not separately metered 表示未单独计量，不能解释为零成本。

BIKE 保持 50 轮上限、学习率 0.0001、有效 batch 16、d_ff 32；其他数据集为 30 轮、0.0003、32、128。保留改进适配器的 2064 步历史和 288 步间隔；BIKE 的 288 小时为 12 天，48 步预测为 48 小时，不声称已完成小时数据的周期设计。smoke=true 仅验证流程，不作精度证据。

## 人工验证

用户于 2026-10-23 确认：已人工审查代码和实验，实验通过。本条按用户确认记录。没有将本次文档补录视为新的训练或验证。

## 证据

- [训练入口](../../research/run_coursework.py)
- [实验登记](../../research/experiment_registry.py)
- [数据适配](../../research/coursework_data.py)
- [时间戳处理](../../research/traffic_data.py)
- [登记测试](../../tests/test_experiment_registry.py)
- [数据与课程实验测试](../../tests/test_coursework.py)
- [Jinqi AI usage ledger](../../AI_USAGE_jinqi.md)

## 脱敏

省略本机绝对路径；未收录凭据、原始对话或受限材料。
