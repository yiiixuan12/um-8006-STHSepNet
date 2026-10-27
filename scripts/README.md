# H1 运行脚本 — yixuan wang

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
