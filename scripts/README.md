# 训练脚本

在已激活的训练环境中，从项目根目录用PowerShell运行以下五个入口。默认使用当前环境的 `python`；可通过 `-PythonPath` 指定解释器。数据集和本地BERT需提前准备。

| 脚本 | 默认行为 |
|---|---|
| `setup_training.ps1` | 检查Python 3.10+、已安装的CUDA版PyTorch；安装三条BERT训练路线的运行依赖并执行pip check |
| `smoke_training.ps1` | 基线、H1、MLP控制组和KAN各执行真实GPU短训练；不作为精度证据 |
| `run_baseline_seeds.ps1` | 五个数据集，独立基线种子2021/2022/2023 |
| `run_kan_seeds.ps1` | BIKE-Inflow，MLP与KAN各运行2021/42/7 |
| `run_h1_seeds.ps1` | 五个数据集各训练一个seed-2021冻结基线，再训练2021/2022/2023三个适配器种子 |

```powershell
.\scripts\setup_training.ps1
.\scripts\smoke_training.ps1
.\scripts\run_baseline_seeds.ps1
.\scripts\run_kan_seeds.ps1
.\scripts\run_h1_seeds.ps1
```

安装脚本不安装或替换PyTorch/CUDA，也不下载数据和BERT。`requirements-training.txt`是已修复代码的运行依赖范围，适用于现代Python环境；原历史环境锁定文件仍在 `environment/`。原 `setup_h1.ps1` 仍保留供Python 3.10历史环境使用。

先加 `-PlanOnly` 可预览命令，无安装、训练或输出目录写入。正式队列按顺序运行，任何失败立即停止。不要同时启动多个队列争用同一GPU。

```powershell
.\scripts\run_baseline_seeds.ps1 -Datasets bike_outflow -Seeds 2022,2023 -PlanOnly
.\scripts\run_kan_seeds.ps1 -Heads kan -Seeds 2021,42,7 -PlanOnly
.\scripts\run_h1_seeds.ps1 -Datasets pems03 -Seeds 2021,2022,2023 -PlanOnly
```

Baseline默认输出 `results_coursework/run_jinqi_baseline/<dataset>/seed_<seed>/`；H1默认输出 `runs/h1_queue/<dataset>/baseline/` 和 `seed_<seed>/`；KAN默认输出 `outputs/kan_queue/<mlp或kan>/seed_<seed>/`。三个队列均支持 `-OutputRoot`。完整结果会跳过；Baseline和H1只在存在最后检查点时自动续训，仍由训练入口核查源码快照。旧实验与当前源码不一致时应换新目录，不能强行混用。KAN入口不支持中断续训，未完成目录需人工检查后换新目录。

统一冒烟默认使用BIKE-Outflow检查基线与H1，BIKE-Inflow检查MLP与KAN；每个划分一个批次、一轮训练，覆盖保存、重载和测试。H1还执行训练集数值校准。每次默认新建带时间戳的输出目录，报告标记 `smoke=true`。可用 `-Datasets pems03` 检查五分钟数据的基线与H1，或用 `-OutputRoot` 指定一个尚不存在的目录。Smoke与正式实验输出严格分开。

## 原H1单次运行工具 — yixuan wang

这些 PowerShell 脚本只包装现有 Python 入口，不改变模型或实验协议。请在 **PowerShell** 中运行（不是 CMD）；Linux/macOS 可直接使用主 README 的 Python 命令。

```text
scripts/
├── setup_h1.ps1          # 安装 H1 Python 依赖；PyTorch 需先单独安装
├── smoke_h1.ps1          # 无训练的接口检查；可选检查数据
├── run_h1.ps1            # H1 修正版基线、改进、消融与测评
├── check_h1_results.ps1  # 复算已发布结果；可选重画图
├── h1_common.ps1         # 上述脚本共用的路径与退出码处理
└── run_coursework.ps1    # 保留的历史队列，不是当前 H1 推荐入口
```

## 环境与冒烟

先准备独立的 Python 3.10 环境并激活，按 [主 README](../README.md#2-安装环境) 安装适配显卡的 PyTorch；脚本不会猜测或替换 CUDA/PyTorch 版本，不自动下载数据或 BERT。

```powershell
.\scripts\setup_h1.ps1
.\scripts\smoke_h1.ps1
```

默认冒烟只运行 CLI 帮助和 9 项标准库接口测试，不需要数据、BERT、PyTorch 或 GPU，也**不代表完整模型前后向已经验证**。下载资源并装好全部依赖后，可额外检查数据：

```powershell
.\scripts\smoke_h1.ps1 -CheckData -Dataset bike_inflow
```

此可选检查在 CPU 读取 CSV/邻接与窗口划分，不训练、不创建输出目录，但会导入完整模型依赖。

## 基线、改进与消融

```powershell
.\scripts\run_h1.ps1 -Dataset pems03 -Stage baseline -Seed 2021 -Output runs/h1/pems03/baseline
.\scripts\run_h1.ps1 -Dataset pems03 -Stage improved -Seed 2021 -Baseline runs/h1/pems03/baseline -Output runs/h1/pems03/seed_2021
.\scripts\run_h1.ps1 -Dataset pems03 -Stage improved -Seed 2022 -Baseline runs/h1/pems03/baseline -NumericCalibration runs/h1/pems03/seed_2021/numeric_calibration.pt -Output runs/h1/pems03/seed_2022
```

第三个 adapter seed 将最后一条命令的 `-Seed` 与输出目录改为 2023，仍共享 seed-2021 基线和数值校准。数据集支持 `pems03`、`metr_la`、`bj500`、`bike_inflow`、`bike_outflow`。

BIKE-Inflow 消融分别用 `-Arm global_long` / `-Arm no_utility`，每臂、每种子单独输出；`global_long` 不得借用完整方法的数值校准，首次自行拟合、其余种子只复用同臂校准。

```powershell
.\scripts\run_h1.ps1 -Dataset bike_inflow -Stage improved -Arm global_long -Seed 2021 -Baseline runs/h1/bike_inflow/baseline -Output runs/h1/bike_inflow/global_long_2021
```

新基线最多 20 轮，改进/消融最多 5 轮，patience 均为 5，验证 raw MSE 选最佳权重；5 轮是预算上限，不等于收敛。共享骨干下三个 adapter seed 不是完整流程三次独立训练。脚本**不会**自动启动全部数据集/种子队列。

## 续训、测评与结果复算

续训在原训练命令追加 `-Resume`，需要同一输出目录中的完整 last checkpoint、优化器、随机状态和准确源码；只保留 best 权重不能续训。未指定续训时不会静默覆盖已有实验，底层 Python 会检查。

```powershell
.\scripts\run_h1.ps1 -Dataset pems03 -Stage evaluate -CheckpointDir runs/h1/pems03/seed_2021 -Baseline runs/h1/pems03/baseline -Output runs/h1/pems03/evaluation_2021
.\scripts\check_h1_results.ps1
.\scripts\check_h1_results.ps1 -Redraw
```

权重测评需要自行训练的 checkpoint、原 BERT、数据与配置；完整权重没有随结果包发布。结果复算仅需 NumPy，不需 GPU/原始数据/权重；`-Redraw` 还需 Matplotlib，输出在 `outputs/h1_ablation/` 和 `outputs/h1_cases/`，不覆盖正式证据。

## 路径与安全范围

- 默认使用当前已激活环境的 `python`；可加 `-PythonPath 'C:\path with spaces\python.exe'`。**所有相对文件路径均相对仓库根目录**，即使从其他目录启动脚本也不变。
- 资源可通过 `-DataRoot` / `-BertPath` 重定位。`run_h1.ps1` 还支持 `-EvalModel`、`-Origins`、`-SavePredictions`、`-MaxWindows`、`-ReplayTrainingRng`、`-RngCheckpoint`，与 Python 同名参数语义一致。`-MaxWindows` 仅限测评冒烟，不能用于正式指标或改变训练量。
- 每个脚本均支持 `-PlanOnly`：只打印将执行的 Python 参数 JSON，不安装、不训练、不读取数据或写结果。
- Python 非零退出会停止脚本，不继续后续步骤；不会清理、删除或自动重跑失败实验。
- 这些脚本仅覆盖 H1；KAN 与同学的独立基线运行方式不在本次修改范围。
