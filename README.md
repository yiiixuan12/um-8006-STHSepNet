# STH-SepNet 课程项目

基于 [STH-SepNet](https://github.com/jiawenchen10/STHSepNet) 的复现与两条**独立**改进线：

- **H1：节点级多周期记忆与预测效用监督的历史检索。负责人：yixuan wang。** 五个数据集的主实验、BIKE-Inflow 三种子消融与未来24小时案例已完成，见[结果与图表](experiments/h1/README.md)。
- **H3：KAN预测头。负责人：jinfeng guo。** 保留独立的 BIKE-Inflow 三种子对照，见第6节。

H1与H3使用不同训练配置和输入信息量，分别与各自基线比较，不能作为严格同条件的直接比较；没有声称已经组合成同一个模型。

## 1. 安装环境

Python 3.10；先安装适配显卡的 CUDA 版 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements-pilot.txt
```

H1实验环境为 PyTorch 2.10.0+cu128、Transformers 4.44.2，训练/测评使用支持BF16的 NVIDIA CUDA GPU。仅查看结果或重新绘图不需要GPU。

## 2. 下载资源

### 2.1 H1数据与目录

[原作者数据下载目录](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)。下载后保持时间列、节点列顺序和配套邻接不变：

```text
dataset/
├── PEMS03/
│   ├── PEMS03.csv
│   └── PEMS03_adj.csv
├── METR/
│   ├── METR.csv
│   └── METR_adj.csv
├── BJ/
│   ├── BJ.csv
│   └── BJ_adj.csv
└── Bike/
    ├── inflow.csv
    ├── outflow.csv
    └── adj.csv
```

H1的五个数据集**均需要配套邻接**。BIKE是小时级；PEMS03、METR-LA、BJ500为五分钟级。基线均读取最近48步、预测未来48步；H1在BIKE读取240小时，在五分钟数据读取2064步。改进使用了额外历史，不能把全部收益归因于token编码或BERT预训练。详见[数据准备](dataset/README.md)。

### 2.2 KAN 预测头数据集（H3：BIKE-Inflow）

此处为H3的原运行方式，与H1的邻接要求区分：

- 数据下载：[原作者Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)。
- 放置：`dataset/Bike/inflow.csv`。H3按原配置构建自适应超图。

### 2.3 BERT

从 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main) 下载 `config.json`、`pytorch_model.bin`、`tokenizer_config.json`、`tokenizer.json`、`vocab.txt`，放到 `huggingface/google_bert/`。H1也可通过 `--bert-path` 指定其他目录。

本次发布包括H1代码、报告、逐窗口误差、案例数值与图，不包含新的原始数据、BERT或训练权重；仓库已有的同学文件不删除。下载资源需遵守各自来源许可。

## 3. 运行

### 3.1 H1：训练每城一份冻结基线

在项目根目录执行：

```bash
python research/run_h1.py --dataset pems03 --stage baseline --seed 2021 --output runs/h1/pems03/baseline
```

### 3.2 H1：多周期记忆与历史检索（负责人：yixuan wang）

```bash
python research/run_h1.py --dataset pems03 --stage improved --seed 2021 --baseline runs/h1/pems03/baseline --output runs/h1/pems03/seed_2021
```

- 数据集参数：`pems03`、`metr_la`、`bj500`、`bike_inflow`、`bike_outflow`。
- 将改进分支的种子和输出目录分别改为2022、2023，**复用同一基线目录**；另加 `--numeric-calibration runs/h1/pems03/seed_2021/numeric_calibration.pt` 共用seed2021的数值周期校准，不重新训练基线或数值通路。
- 新基线默认最多20轮；H1最多5轮；验证集raw MSE连续5轮无改善早停并保留最佳权重。达到5轮不等于收敛。已发布的历史基线保留各自原始训练历程，不冒充本次从零训练结果。
- `--resume`可从同一输出目录的完整状态续训；`--data-root`、`--bert-path`可重定位资源。训练前可加`--check-data`仅检查数据。
- H1代码使用 `models/sthsepnet_h1.py` 与 `layers/h1/`，不覆盖KAN的模型与算子。新入口不依赖旧队列、旧pilot或历史门禁文件。

BIKE-Inflow消融：分别在上述改进命令加 `--arm global_long` 或 `--arm no_utility`，为每臂/种子使用独立输出目录；仍共享同一基线。`global_long`的数值周期校准会独立拟合，不借用完整方法的个体历史校准。

已有**自己训练的**最终权重，可独立重测；评估不会覆盖原报告：

```bash
python research/run_h1.py --dataset pems03 --stage evaluate --checkpoint-dir runs/h1/pems03/seed_2021 --baseline runs/h1/pems03/baseline --output runs/h1/pems03/evaluation_2021
```

需要完整的基线最佳权重、改进最佳权重、数值校准、原始BERT和报告配置。默认按固定种子重测；原GCN在eval仍有随机top-k扰动，因此不保证与历史测试逐位相同。完整RNG重放和逐点流量导出选项见 `python research/run_h1.py --help`。

### 3.3 KAN 预测头（H3，BIKE-Inflow）

KAN 试点与原版 MLP 基线使用完全相同的训练脚本、数据划分、LLM 骨干（BERT）、fusion_gate=adaptive、超参数与训练预算。唯一受控变量是预测头类型：--use_kan。门控融合模块内部的线性层不改动；KAN 头的 in_features/out_features 与原 MLP 完全一致（192 → 48），不引入额外历史输入，因此 H3 与基线是等信息化对比。

替换点：仅替换门控融合之后、输出最终预测的 MLP 头 FlattenHead.linear: nn.Linear(192, 48) → KANLinear(192, 48; grid_size=5, spline_order=3)。

KAN 网格自适应细化：刻意关闭 update_grid，保证对比干净。

种子：2021 / 42 / 7（与主线 H1 的 2021 / 2022 / 2023 是不同集合，跨实验比较时不能按种子配对）。

数据：dataset/Bike/inflow.csv；BERT 权重放在 huggingface/google_bert/。

环境修复共用：seed 生效、Windows spawn 保护、checkpoint key 前缀修复、dtype 适配、本地路径修复等由基线与 KAN 共用同一份代码，不属于科学变量。

3.3.1 BIKE-Inflow MLP 基线命令（无 --use_kan）

  ```bash
  python -u src/run_main.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/Bike/ --data_path inflow.csv --model_id BIKE_48_48 \
  --model pool --data inflow --features M --seq_len 48 --label_len 48 --pred_len 48 \
  --factor 3 --des Exp --itr 1 --d_model 768 --d_ff 32 --batch_size 16 \
  --learning_rate 0.0001 --llm_model BERT --llm_dim 768 --llm_layers 6 \
  --train_epochs 50 --model_comment abl --gamma 1.0 --fusion_gate adaptive \
  --enc_in 295 --dec_in 295 --c_out 295 --node_num 295 --scale_hyperedges 3 \
  --seed 2021 --num_workers 0
  ```

3.3.2 KAN 预测头命令（仅追加 KAN 开关）
  ```bash

python -u src/run_main.py --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/Bike/ --data_path inflow.csv --model_id BIKE_48_48 \
  --model pool --data inflow --features M --seq_len 48 --label_len 48 --pred_len 48 \
  --factor 3 --des Exp --itr 1 --d_model 768 --d_ff 32 --batch_size 16 \
  --learning_rate 0.0001 --llm_model BERT --llm_dim 768 --llm_layers 6 \
  --train_epochs 50 --model_comment abl --gamma 1.0 --fusion_gate adaptive \
  --enc_in 295 --dec_in 295 --c_out 295 --node_num 295 --scale_hyperedges 3 \
  --seed 2021 --use_kan --kan_grid_size 5 --kan_spline_order 3 --num_workers 0
  ```

另两个种子将 --seed 分别改为 42、7，其余参数完全一致。每个种子单独保存日志、预测/真值数组和 report.json，不要覆盖已有 run。
3.3.3 冒烟与验证
在跑正式 3 seeds 前，至少完成：

1.sanity.py：随机 batch (2, 48, 295) 下 MLP/KAN 双版本 forward + backward 通过；参数量符合预期（MLP 头 9,264；KAN 头 103,680；总参数差 94,416）。

2.1-epoch 全流程冒烟：训练 → checkpoint 保存/加载 → vali_save → 清理，exit 0。

3.指标复算：用 eval_metrics.py / collect_results.py 从保存的 .npy 重新计算 MAE/RMSE/robust MAPE，不直接信任训练日志中的打印值。

3.3.4 统一入口说明
若 run_coursework.py 已接入等价 --use_kan / --stage kan 分支，可按统一入口运行；替换点、唯一变量和结论不变。以当前仓库代码为准，并在 README 中记录实际使用的入口。

### 3.4 范围说明

H1主实验与消融的三个种子是**同一冻结骨干下的adapter种子**；24小时案例另有三对独立训练的基线/完整方法。H3使用2021/42/7，保持自己的训练与测评协议。

## 4. H1实验结果（负责人：yixuan wang）

完整基线对照、15份报告、三指标均值/样本标准差见[主实验表](experiments/h1/main/table.md)、[逐种子CSV](experiments/h1/main/per_seed.csv)和[单份基线CSV](experiments/h1/main/baselines.csv)。以下均为本地修正版结果，不是原论文官方指标：

| 数据集 | H1 MAE ↓ | H1 RMSE ↓ | H1 MSE ↓ |
|---|---:|---:|---:|
| PEMS03 | 18.2208 ± 0.1031 | 32.0173 ± 0.1739 | 1025.1255 ± 11.1542 |
| METR-LA | 8.9335 ± 0.1238 | 17.9998 ± 0.1236 | 324.0029 ± 4.4506 |
| BJ500 | 5.2374 ± 0.0205 | 9.6827 ± 0.0435 | 93.7553 ± 0.8437 |
| BIKE-Inflow | 4.4859 ± 0.0509 | 12.5628 ± 0.1608 | 157.8417 ± 4.0548 |
| BIKE-Outflow | 4.4453 ± 0.0579 | 11.5271 ± 0.1172 | 132.8834 ± 2.6966 |

**局限：**H1平均MAE在五个数据集均改善，但METR-LA的RMSE/MSE变差；所有adapter均为5轮预算截止，未确认收敛；没有证明语言预训练本身带来收益。

### 4.1 BIKE-Inflow三种子消融

![H1消融柱状图](experiments/h1/ablation/ablation_bars.png)

[消融表与定义](experiments/h1/ablation/table.md)。节点个体长期历史在三个种子上方向一致；效用辅助监督只有小幅平均收益，第三个种子反向，不能声称两项设计均显著且稳定有效。误差棒为样本标准差，不是置信区间。

### 4.2 未来24小时真实流量案例

三对独立种子的原单位预测与真实流量对比：[案例1](experiments/h1/cases/paired_24h_case_1.png)、[案例2](experiments/h1/cases/paired_24h_case_2.png)、[案例3](experiments/h1/cases/paired_24h_case_3.png)，以及[逐点CSV](experiments/h1/cases/paired_24h_curves.csv)。

![未来24小时案例](experiments/h1/cases/paired_24h_case_1.png)

单次直接预测48小时，展示前24小时；训练段均值/标准差还原，无额外平滑或截断，阴影为三个种子的实际范围。日期与站点按时间位置/训练流量分位选择，未挑选预测较好的案例；仍存在高估与过平滑，不能用几个案例替代整体评价。

不用训练即可从已发布的结果重新绘图：

```bash
python research/plot_bike_ablation.py --summary experiments/h1/ablation/summary.json
python research/plot_h1_cases.py
```

只复算已发布窗口误差中的26条记录（78个MAE/MSE/RMSE数值），不需要数据、BERT、权重或GPU：

```bash
python research/check_h1_results.py
```

图输出到 `outputs/h1_ablation/`、`outputs/h1_cases/`，不覆盖已发布结果。H3结果与复现说明保留于第6节。

## 5. 分工与比较边界

- **yixuan wang：H1。** 节点级多周期记忆、历史检索、五数据集主实验、BIKE消融和真实流量案例。
- **jinfeng guo：H3。** KAN预测头、独立BIKE-Inflow对照及其分析。
- **jinqi Liu：论文基线复现。** 完成全部五个数据集的三种子基线实验、数据适配、实验登记和结果汇总。

H3不是H1完整方法的删模块消融；两条路线的数值不能跨数据集、跨预算直接排名。H2节点—时间融合仅为此前提出的方向，不属于本次H1已验证贡献。

### 5.1 论文基线复现结果（负责人：jinqi Liu）

基于本地修正版 STH-SepNet（BERT），输入48步、预测48步；以下为种子2021、2022、2023的均值 ± 样本标准差，仅包含基线，不包含H1/H3改进。

| 数据集 | MAE ↓ | RMSE ↓ |
|---|---:|---:|
| BIKE-Inflow | 5.51 ± 0.05 | 15.27 ± 0.30 |
| BIKE-Outflow | 5.47 ± 0.02 | 13.83 ± 0.05 |
| PEMS03 | 21.34 ± 0.21 | 35.56 ± 0.47 |
| BJ500 | 5.97 ± 0.10 | 10.28 ± 0.33 |
| METR-LA | 10.14 ± 0.53 | 17.28 ± 0.68 |

结果见[汇总JSON](results_coursework/run_jinqi_baseline/summary.json)和[逐种子CSV](results_coursework/run_jinqi_baseline/per_seed.csv)。上述为本地复现值，并非论文官方数值；与H1/H3比较时应核对各自的训练配置、输入信息和种子口径。

## 6. KAN 预测头（H3）— 负责人：jinfeng guo

负责人：**jinfeng guo**。本部分的实现、试点和 BIKE-Inflow 基线对比分析保持原有记录。

### 6.1 动机、替换点与替换方式

原模型在门控融合之后使用固定激活的 MLP 预测头（FlattenHead.linear = nn.Linear(192, 48)），表达能力有限；KAN 以可学习的 B 样条基函数替代固定激活，更自适应地捕捉融合特征到预测目标之间的复杂非线性映射。

替换点：仅替换门控融合之后、用于输出最终预测的 MLP 预测头；不修改门控融合机制内部的线性层，保证消融只反映预测头这一变量。

替换方式：nn.Linear(192, 48) → KANLinear(192, 48; grid_size=5, spline_order=3)，输入输出维度与原 MLP 完全一致。

控制：关闭 update_grid 网格自适应细化；不引入额外历史输入；不改变 BERT、超图分支、门控融合或训练预算。

公平性：基线与 KAN 共用同一修正版基线、同一环境适配修复、同一数据划分、同一 seeds 集合、同一 epoch/patience 规则；唯一科学变量是 --use_kan。

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
