# Agent log — AI-JINQI-003 (2026-10-27 · Jinqi)

- 阶段：Implementation / debugging / experiment execution support
- 工具/模型：OpenAI Codex（gpt-6.1-sol）；采样设置 not exposed。
- 任务：使README中的基线、H1、KAN训练命令能够运行，并在scripts目录提供安装、统一冒烟及各路线的种子运行脚本。

## 咨询与决定

用户要求统一提供一个安装脚本、一个覆盖Baseline/H1/KAN的冒烟脚本，以及三条路线各自的种子运行脚本。接受脚本方案：顺序运行、失败停止、完成结果跳过，支持命令预览；Baseline/H1通过训练入口原有检查点和源码检查续训，KAN未完成运行需检查后使用新输出目录。

Baseline使用2021/2022/2023三个独立训练种子；KAN默认在BIKE-Inflow上运行MLP控制组和KAN的2021/42/7三个种子；H1先训练或复用每个数据集的一个seed-2021冻结基线，再运行2021/2022/2023三个适配器种子。三种设置在说明中区分，不作为同一组实验混合汇总。

## 修改代码与具体部分

- `scripts/setup_training.ps1`、`scripts/requirements-training.txt`：检查现有CUDA版PyTorch，安装三条BERT路线的运行依赖并执行依赖检查。
- `scripts/smoke_training.ps1`、`scripts/smoke_h1_training.py`：真实数据/GPU短训练，覆盖基线、H1数值校准与适配器、MLP及KAN的保存、重载和测试。
- `scripts/run_baseline_seeds.ps1`、`scripts/run_kan_seeds.ps1`、`scripts/run_h1_seeds.ps1`、`scripts/training_runs.py`：按各路线的种子组织运行，检查已有输出并隔离目录。
- `research/traffic_data.py`：合入显式纳秒转换；`research/run_coursework.py`：恢复五数据集配置及BIKE小时提示，使用与jinqi-dev模型一致的独立修正版模型。
- `src/run_main.py`、`models/ST_SepNet.py`：修复模块搜索路径、重复seed参数和KAN包名导入；增加显式冒烟参数和独立输出目录。
- `src/data_provider/data_loader.py`、`src/utils/tools.py`等：修复Pandas时间特征、NumPy常量兼容性及输出保存路径，保留运行检查点。
- `README.md`、`scripts/README.md`：补充运行命令、输出路径、种子与续训边界。默认正式训练预算未因冒烟而缩小。

## 验证与人工审查

AI执行的11项脚本/队列测试通过；统一GPU冒烟成功完成Baseline、H1、MLP和KAN。此前还逐一验证了五数据集基线，以及H1小时和五分钟两类路径。安装脚本由AI完成命令预览；本次会话另实际安装并验证了accelerate。

**人工验证：用户确认人工测试已通过，所有脚本在本机环境下均成功运行。** 此处按用户确认记录，未把AI运行测试替代为人工执行。

冒烟结果标记smoke，仅验证流程，不能替代完整精度实验或证明已重跑全部种子。

## 证据

- [脚本说明](../../scripts/README.md)
- [安装入口](../../scripts/setup_training.ps1)
- [统一冒烟](../../scripts/smoke_training.ps1)
- [队列实现](../../scripts/training_runs.py)
- [队列测试](../../tests/test_training_scripts.py)
- [H1包装脚本测试](../../tests/test_h1_scripts.py)
- [AI_USAGE](../../AI_USAGE.md)

本日志是咨询与修改的脱敏摘要，不是原始逐字对话；省略本机绝对路径和进程标识，不包含凭据。
