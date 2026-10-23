# STH-SepNet 课程项目

基于 [STH-SepNet](https://github.com/jiawenchen10/STHSepNet) 的修正版基线与多周期记忆改进。当前统一入口支持 **PEMS03、METR-LA、BJ500、BIKE-Inflow、BIKE-Outflow**，需要 NVIDIA CUDA 显卡。

## 1. 安装环境

使用 Python 3.10，先安装适配显卡的 CUDA 版 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements-pilot.txt
```

本地使用 PyTorch 2.10.0+cu128、Transformers 4.44.2。依赖清单为 `requirements-pilot.txt`，不包含需单独安装的 PyTorch。

## 2. 下载资源

- [数据集（原作者 Google Drive）](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)：按下面结构放置。
- [BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main)：下载 `config.json`、`pytorch_model.bin`、`tokenizer_config.json`、`tokenizer.json`、`vocab.txt`，放入 `huggingface/google_bert/`。

```text
dataset/
├── PEMS03/
│   ├── PEMS03.csv
│   └── PEMS03_adj.csv
└── METR/
    ├── METR.csv
    └── METR_adj.csv
```

## 3. 运行

在项目根目录执行，先训练基线：

```bash
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage baseline --output results_coursework/run1/pems03/seed_2021
```

基线完成后，在同一结果目录训练改进方法：

```bash
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage improved --output results_coursework/run1/pems03/seed_2021
```

- METR-LA：将 `pems03` 改为 `metr_la`，同时修改输出目录。
- 三个种子：分别使用 `2021`、`2022`、`2023`，每个种子单独保存。
- 中断续训：原命令末尾加 `--resume`，保持配置、源码和数据不变。
- 验证集连续5轮未改善则早停，保留最佳权重；训练结束自动计算测试指标。

## 4. 查看结果

各组结果目录中的 `report.json` 包含 MAE、RMSE、MSE。汇总两数据集、三个种子的结果：

```bash
python research/summarize_coursework.py --results results_coursework/run1
```

基线输入48步、预测48步；改进方法额外使用2064步历史，比较时需注明输入信息量不同。数据及BERT不随仓库上传；保留原作者 MIT 许可。

## 5. 其他数据集及实验登记

原有 PEMS03/METR-LA 命令不变。以下名称同时支持 `--stage baseline` 和 `--stage improved`；每个数据集、种子使用独立输出目录。

| `--dataset` | 数据文件 | 邻接矩阵 | 采样间隔 |
|---|---|---|---|
| `pems03` | `dataset/PEMS03/PEMS03.csv` | `PEMS03_adj.csv` | 5分钟 |
| `metr_la` | `dataset/METR/METR.csv` | `METR_adj.csv` | 5分钟 |
| `bj500` | `dataset/BJ/BJ.csv` | `BJ_adj.csv` | 5分钟 |
| `bike_inflow` | `dataset/Bike/inflow.csv` | `adj.csv` | 1小时 |
| `bike_outflow` | `dataset/Bike/outflow.csv` | `adj.csv` | 1小时 |

邻接矩阵位于对应数据文件同一目录。`Bike.csv` 是 from/to/cost 边列表，不是额外预测任务；下载备份目录是重复资源。

例如：

```bash
python -u research/run_coursework.py --dataset bj500 --seed 2021 --stage baseline --output results_coursework/run1/bj500/seed_2021
python -u research/run_coursework.py --dataset bike_inflow --seed 2021 --stage baseline --output results_coursework/run1/bike_inflow/seed_2021
python -u research/run_coursework.py --dataset bike_outflow --seed 2021 --stage baseline --output results_coursework/run1/bike_outflow/seed_2021
```

BIKE 保留已有配置：50轮上限、学习率0.0001、有效batch16、d_ff32。其他数据集为30轮上限、学习率0.0003、有效batch32、d_ff128。现有改进适配器保留2064步历史和288步记忆间隔；在BIKE上288小时是12天，而非一天，48步预测是48小时。此处仅适配数据，不声称改进模块已针对小时数据完成周期设计。

每次通过训练入口调用，自动追加 `experiments/registry.csv`。每次调用结束时只登记一条记录，状态仅为 completed（完成）、failed（失败）或 interrupted（捕获到中断）；不登记 started。续训产生新 run_id，不覆盖旧记录。强制结束进程或断电无法保证写入记录。记录包含 commit、源码SHA256、工作区状态、配置、数据及邻接矩阵SHA256、seed、硬件、秒数、状态、测试指标和输出路径。

每次调用的配置及结束时报告副本保存在 `experiments/runs/<run_id>/`。权重仍在输出目录，最终交付应保留或提供这些输出的获取方式。费用填写 `not separately metered`：本地GPU未单独计量，不代表零成本。smoke运行同样标为completed，但notes中smoke=true；其指标仅验证流程，不能作精度证据。登记功能不会补造旧实验；修改前已生成的输出仍受原有源码快照检查保护，不应混用续训。

实验汇总默认覆盖论文五个数据集，每个数据集统计2021、2022、2023三个种子。当前只运行基线时使用：

```bash
python research/summarize_coursework.py --results results_coursework/run_jinqi_baseline --stages baseline
```

输出写入结果根目录的 `summary.json`（各数据集均值、样本标准差及缺失项）和 `per_seed.csv`（每个种子的测试指标）。没有结果的任务列为缺失，不能以0代替；仅一个种子时标准差为空。省略 `--stages` 会汇总baseline和improved两个阶段。也可以显式指定数据集范围：

```bash
python research/summarize_coursework.py --results results_coursework/run1 --datasets pems03 metr_la bj500 bike_inflow bike_outflow
```

已有 `scripts/run_coursework.ps1` 的默认批量队列保持原来两数据集，避免自动增加训练任务。
