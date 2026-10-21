# STH-SepNet 课程项目

基于 [STH-SepNet](https://github.com/jiawenchen10/STHSepNet) 的修正版基线，并在此之上完成**两条独立改进线**：

- **多周期记忆（H1，表示阶段）**：为每个路段构建 recent / intra-day / cross-day 周期 token，分离循环历史模式与当前周期偏差。这是本项目的**主改进**，统一入口支持 **PEMS03、METR-LA**。
- **KAN 预测头（H3，预测阶段）**：将门控融合之后的 MLP 预测头替换为 B 样条 KANLinear。试点数据集为 **BIKE-Inflow**，3 个随机种子 `2021/42/7`。

两条改进共享同一修正版基线、同一自适应门控融合模块与同一 LLM 骨干（BERT），因此 H1 与 H3 的对比是干净的：改的只是各自模块，其余完全一致。需要 NVIDIA CUDA 显卡。

> 分工说明：多周期记忆改进（H1）由负责该模块的成员完成；KAN 预测头改进（H3）由我提出、实现并完成试点与 BIKE-Inflow 对比分析（详见第 6 节）。

## 1. 安装环境

使用 Python 3.10，先安装适配显卡的 CUDA 版 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements-pilot.txt
```

本地使用 PyTorch 2.10.0+cu128、Transformers 4.44.2。依赖清单为 `requirements-pilot.txt`，不包含需单独安装的 PyTorch。

## 2. 下载资源

数据与预训练 BERT 不随仓库上传，需按下述来源下载并放置到指定目录。**两个数据集系列都来自原作者（MIT 许可），主线与 KAN 实验共用同一份 BERT 权重。**

### 2.1 主线数据集（H1：PEMS03、METR-LA）

- [数据集（原作者 Google Drive）](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)：按下面结构放置。

### 2.2 KAN 预测头数据集（H3：BIKE-Inflow）

- **数据集**：BIKE-Inflow 是北京共享单车系统的流入流量（295 个站点，5 分钟粒度），由原作者在官方仓库中提供；其时空周期性与高峰模式是验证预测头表达能力切换的合适基准。
- **下载**：从 [STH-SepNet 官方仓库](https://github.com/jiawenchen10/STHSepNet) 的 `dataset/Bike/inflow.csv` 获取（MIT 许可，与主线数据集同源）；如原作者 Google Drive 镜像包含 Bike 目录，亦可从该处下载。
- **放置**：`dataset/Bike/inflow.csv`（无需邻接矩阵，自适应超图按官方配置在训练时构建）。

### 2.3 BERT（共用）

- [BERT](https://huggingface.co/google-bert/bert-base-uncased/tree/main)：下载 `config.json`、`pytorch_model.bin`、`tokenizer_config.json`、`tokenizer.json`、`vocab.txt`，放入 `huggingface/google_bert/`。

最终目录结构：

```text
dataset/
├── PEMS03/                  # H1 主线
│   ├── PEMS03.csv
│   └── PEMS03_adj.csv
├── METR/                    # H1 主线
│   ├── METR.csv
│   └── METR_adj.csv
└── Bike/                    # H3 KAN 预测头
    └── inflow.csv
```

## 3. 运行

所有实验在项目根目录执行。先训练基线，再在**同一结果目录**内训练改进方法，便于按种子配对比较。

### 3.1 基线

```bash
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage baseline --output results_coursework/run1/pems03/seed_2021
```

### 3.2 主改进（H1 多周期记忆）

```bash
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage improved --output results_coursework/run1/pems03/seed_2021
```

### 3.3 KAN 预测头（H3，BIKE-Inflow）

KAN 试点与原版 MLP 基线使用**完全相同的训练脚本、数据划分、LLM 骨干（BERT）、`fusion_gate`（adaptive）、超参数与种子集合**，唯一受控变量为预测头类型。种子沿用早期冻结消融的 `2021/42/7`（与主线 H1 的 `2021/2022/2023` 是不同集合，跨实验比较时需注明）。

- **已验证入口**（sths-repro，`src/run_main.py`）：在基线命令后追加 KAN 开关

  ```bash
  python -u src/run_main.py --task_name long_term_forecast --is_training 1 \
    --root_path ./dataset/Bike/ --data_path inflow.csv --model_id BIKE_48_48 \
    --model pool --data inflow --features M --seq_len 48 --label_len 48 --pred_len 48 \
    --d_model 768 --d_ff 32 --batch_size 16 --learning_rate 0.0001 --train_epochs 50 \
    --fusion_gate adaptive --enc_in 295 --dec_in 295 --c_out 295 --node_num 295 \
    --scale_hyperedges 3 --llm_model BERT --llm_dim 768 --llm_layers 6 \
    --seed 2021 --use_kan --kan_grid_size 5 --kan_spline_order 3
  ```

  完整参数与 `configs/improvement/kan_adaptive_bike_bert.json` 一一对应；另两个种子将 `--seed` 改为 `42`、`7`。

- **统一入口**：若 `run_coursework.py` 已按第 6.1 节的替换点接入等价 `--use_kan` / `--stage kan` 分支，则按该入口参数运行，替换点与结论不变（以代码为准）。

### 3.4 公共说明

- METR-LA：将 `pems03` 改为 `metr_la`，同时修改输出目录。
- 三个种子：主线 H1 分别使用 `2021`、`2022`、`2023`；KAN（H3）使用 `2021`、`42`、`7`，每个种子单独保存。
- 中断续训：原命令末尾加 `--resume`，保持配置、源码和数据不变。
- 验证集连续 5 轮未改善则早停，保留最佳权重；训练结束自动计算测试指标。

## 4. 查看结果

各组结果目录中的 `report.json` 包含 MAE、RMSE、MSE。

- **主线（H1）汇总**——两数据集、三个种子的结果：

  ```bash
  python research/summarize_coursework.py --results results_coursework/run1
  ```

- **KAN 试点（H3）**——BIKE-Inflow、3 seeds `2021/42/7`：MAE `5.1006 ± 0.0240` / RMSE `13.4306 ± 0.0908`，相对 MLP 基线（`5.2570 ± 0.0302` / `14.1383 ± 0.0462`）分别 **−2.97% / −5.01%**，3/3 种子全部优于基线。完整对比表、参数量与收敛分析见第 6.2 节。

- **信息量口径说明**（比较时必须注明）：基线输入 48 步、预测 48 步；主改进 H1 额外使用 2064 步历史；KAN 头（H3）输入输出维度与原 MLP 完全一致（192 → 48），**不引入额外历史输入**，因此 H3 与基线的对比是等信息的。

数据及 BERT 不随仓库上传；保留原作者 MIT 许可。

## 5. 改进定位与假设选择（H1 与 H3 的关系）

两个改进模块依次对应"长期时序信息如何保留"与"融合表示如何映射到未来交通状态"：

| 模块 | 阶段 | 解决什么 | 试点数据集 | 试点增益（MAE） |
|---|---|---|---|---|
| H1 多周期记忆 | 表示阶段 | BERT 前的逐路段周期信息被平均掩盖 | PEMS03 | −16.4% |
| H3 KAN 预测头 | 预测阶段 | 固定激活 MLP 头的非线性容量有限 | BIKE-Inflow | −2.97% |

按冻结的假设选择规则（试点中相对增益更大的假设进入全面评估）：**全面评估选择 H1**（即第 3.2 节 `--stage improved` 的主改进）；H3 作为完整的受控消融报告（第 6 节），结论是：在门控融合已施加强非线性之后，预测头非线性容量并非该数据集误差的主要瓶颈。

## 6. 我负责的部分：KAN 预测头（H3）

本部分由我提出、实现并完成试点与 BIKE-Inflow 基线对比分析。

### 6.1 动机、替换点与替换方式

原模型在门控融合之后使用固定激活的 MLP 预测头（`FlattenHead.linear = nn.Linear(192, 48)`），表达能力有限；KAN 以**可学习的 B 样条基函数**替代固定激活，能够更自适应地捕捉融合特征到预测目标之间的复杂非线性映射。

- **替换点**：仅替换门控融合模块之后、用于输出最终预测的 MLP 预测头；**不修改门控融合机制内部的线性层**，保证消融只反映预测头这一变量。
- **替换方式**：`nn.Linear(192, 48)` → `KANLinear(192, 48; grid_size=5, spline_order=3)`，`in_features`/`out_features` 与原 MLP 完全一致；网格自适应细化（`update_grid`）刻意关闭，保证对比干净。

### 6.2 试点与结果（BIKE-Inflow，3 seeds：2021/42/7）

| 预测头 | MAE（均值±标准差） | RMSE（均值±标准差） | robust MAPE（\|y\|≥1） | 预测头参数量 | 总参数量 |
|---|---|---|---|---|---|
| MLP（基线） | 5.2570 ± 0.0302 | 14.1383 ± 0.0462 | 77.417 ± 0.827 | 9,264 | 109,219,949 |
| **KANLinear** | **5.1006 ± 0.0240** | **13.4306 ± 0.0908** | 80.175 ± 0.191 | 103,680 | 109,314,365 |
| 相对变化 | **−2.97%** | **−5.01%** | +3.6 pts | +94,416 | **+0.09%** |

- **一致性**：3/3 个种子 KAN 均在 MAE、RMSE 上优于全部 3 个基线种子；增益幅度超过种子间标准差，可排除随机性。
- **参数效率**：KAN 头约为 MLP 头的 11 倍，但占约 1.09 亿总参数的模型仅 +0.09%，以极低的总体成本换取精度提升。
- **收敛**：MLP 基线在 36–50 轮早停（平均 42.3 轮），KAN 跑满 50 轮，单轮耗时相近（约 181 s）；KAN 未出现梯度异常。
- **诚实局限**：robust MAPE（\|y\|≥1）KAN 侧略高于基线，改善集中在 MAE/RMSE（绝对误差），相对误差在高幅样本上未同步下降，已如实记录。BIKE 反归一化后有 41.3% 样本 \|y\|<1，naive MAPE 不可报告，故采用 robust MAPE 口径。

### 6.3 复现

KAN 预测头的独立复现（含 `src/kan/` 实现、`--use_kan` 开关、`scripts/eval_metrics.py` / `collect_results.py`、`experiments/registry.csv`、`results/data_pred/` 保存的预测/真值数组）见课程仓库 **`sths-repro`**；报告分析见 `report/final_report.md` 第 7 / 8.2 / 10 节。数据与 BERT 权重按第 2 节说明下载后即可复现。
