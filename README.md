## 当前交付状态（2026-10-21）

本仓库是基于原始 STH-SepNet 的研究开发版，不是已完成多数据集验证的正式论文实现。
保留原作者 MIT 许可和下方原始说明。下文带日期的旧结果是历史记录，不代表最新结论。

### 快速使用：下载资源与运行当前版本

本节优先于下方原作者示例和开发历史。请在项目根目录执行命令；不需要本机原有的绝对路径。
当前统一入口 `research/run_coursework.py` **只开放 PEMS03、METR-LA**。
数据配置中虽已登记五个数据集，但 BIKE-Inflow、BIKE-Outflow、BJ500 尚未接入此入口，
不能仅替换 `--dataset` 就运行。当前版本也没有独立的 `--evaluate-only` 参数。

#### 1. 环境

建议使用独立 Python 3.10 环境，并先安装适配显卡的 CUDA 版 PyTorch。
本地运行环境为 PyTorch 2.10.0+cu128、Transformers 4.44.2；RTX 50 系列需要支持
Blackwell 的 PyTorch 构建。当前训练入口直接使用 CUDA，不提供 CPU 训练模式。
激活环境后安装当前研究入口使用的依赖：

```console
python -m pip install -r requirements-pilot.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`requirements-pilot.txt` 不包含 PyTorch，需要事先单独安装；当前默认配置不需要 METIS 分组。
根目录旧 `requirements.txt` 是原作者环境记录，不作为本节推荐安装清单。

#### 2. 下载数据

使用[原作者 README](https://github.com/jiawenchen10/STHSepNet#12-data)提供的
[Google Drive 数据下载目录](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)。
这是原作者提供的下载地址，不保证网络可达性或访问权限；请勿用其他预处理版本直接替换。
下载解压后按以下位置放置，避免多套一层 `dataset/dataset/`：

```text
dataset/
├── PEMS03/
│   ├── PEMS03.csv
│   └── PEMS03_adj.csv
├── METR/
│   ├── METR.csv
│   └── METR_adj.csv
├── Bike/                 # 仅数据准备，当前统一入口尚不支持运行
│   ├── inflow.csv
│   ├── outflow.csv
│   └── adj.csv
├── BJ/                   # 仅数据准备，当前统一入口尚不支持运行
│   ├── BJ.csv
│   └── BJ_adj.csv
└── prompt_bank/          # 仓库中的提示文本，保留原文件
```

只运行当前两个数据集时，不需要下载 Bike 和 BJ。
保留原始 CSV 的时间列、节点顺序及对应邻接矩阵，不自行重新划分、归一化或填补数据。
程序按时间顺序划分 7:1:2，仅训练段拟合归一化。续训会核对数据文件信息；开始训练后不要
覆盖、重新保存数据或改动其时间戳。

#### 3. 下载 BERT

从 [google-bert/bert-base-uncased 文件页面](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
下载下面五个文件。当前只需要 BERT，不需要 GPT、Llama 或 TensorFlow/Flax 格式权重：

```text
huggingface/google_bert/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

`pytorch_model.bin` 应是实际权重文件，不是 Git LFS 指针或下载错误的 HTML 页面。
需要记录下载版本以便复现；已有实验续训时继续使用同一份资源，不替换权重。
如放在其他目录，运行时显式传入 `--bert-path "实际BERT目录"`。
该参数在创建实验时确定，续训不得随意更改。

#### 4. 训练修正版基线与我们的方法

以下为新实验示例。`results_coursework/classmate_v1` 是示例结果目录，
**不要指向或复用其他正在运行的实验目录**。运行前确认当前机器没有争用显存的训练任务。

```console
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage baseline --output results_coursework/classmate_v1/pems03/seed_2021
```

同一基线完成后，在相同目录训练周期记忆与效用检索适配器：

```console
python -u research/run_coursework.py --dataset pems03 --seed 2021 --stage improved --output results_coursework/classmate_v1/pems03/seed_2021
```

也可以在全新目录用 `--stage pair` 顺序训练两者。METR-LA 将数据参数改为 `metr_la`，
并相应更改输出目录；三个种子使用 `2021`、`2022`、`2023`。
中断后使用原命令加 `--resume`，保持数据、源码、BERT路径及 batch 等配置不变。
新增实验可以先用全新输出目录加 `--smoke` 检查运行，但冒烟结果不是正式精度证据，
不能使用同一目录继续正式实验。

Windows PowerShell 中可用现有队列运行两个数据集、三个种子的基线和改进方法（共12个阶段）：

```powershell
& ./scripts/run_coursework.ps1 -PythonPath (Get-Command python).Source -ResultsPath ./results_coursework/classmate_v1
```

这是 PowerShell 命令，不要直接粘贴到 CMD。队列先完成六组基线，再训练对应适配器；
失败会停止，不会将失败结果计为完成。单组全量训练可能耗时较长。

#### 5. 查看结果与方法边界

每组目录的 `report.json` 保存状态及完成阶段的测试 MAE/RMSE/MSE；
`baseline_best.pt`、`improved_best.pt` 为对应最佳权重，`*_last.pt` 用于续训。
改进模型还需要对应基线、`numeric_calibration.pt` 和公共 BERT，不能只拷贝一个适配器权重。
每阶段训练完成会自动用验证集选出的最佳权重评价测试集，不需要另跑历史入口。
汇总命令如下，未完成的阶段或种子会明确标为缺失：

```console
python research/summarize_coursework.py --results results_coursework/classmate_v1
```

基线为修正版原始结构的48步输入、48步预测；改进方法额外读取2064步历史，
包含过去7天同期及其当时已经发生的后续走势。两者输入信息量不同，报告中必须披露。
当前两项研究内容是道路级多周期记忆、预测效用监督的历史检索；不声称语言预训练优势已获证明。
`holt_operator.py` 与 `utility_recoupling.py` 为暂未抽离的历史导入依赖，当前分别由
`holt_mode=off`、`interaction_rank=0` 关闭；后者不是当前保留的历史检索效用监督模块。
资源下载链接不代表已完成五数据集复现；原论文数值、开发集 Pilot 和本地正式测试需分开标注。

数据、BERT、训练权重、缓存和日志是本地运行资源，不应直接提交到代码仓库。
已经被 Git 跟踪的资源需要在上传准备阶段单独处理，仅写入 `.gitignore` 不会自动取消跟踪。

### 当前作业范围：两项改进、两个数据集、三个种子

当前只验证道路级多周期记忆与预测效用监督检索。PEMS03、METR-LA 各使用
2021/2022/2023 三个种子，分别训练原始结构的修正基线和改进方法，共12次训练。
两种方法统一以验证集原始单位MSE选择最佳权重，连续5轮无改善则早停，30轮仅为上限。
该停止策略于首组基线第2轮完整保存后经用户授权调整，优化器、学习率、随机状态及已有训练进度保留。
新增邻路模块及第三项创新不进入这轮实验；此前空间隔离队列已受控停止。
BERT 是借用的骨干，不作为创新，也不声称语言预训练已被证明有效。

统一入口 `research/run_coursework.py`，队列 `scripts/run_coursework.ps1`，
结果位于 `results_coursework/main_v1/{dataset}/seed_{seed}`；
`research/summarize_coursework.py` 汇总每次 MAE/RMSE/MSE 和三种子的均值、样本标准差，
缺失结果明确标注，不补零。正式实验完成前不得把冒烟检查结果当成准确率证据。

数据按原作者取整方式7:1:2顺序划分；仅训练段拟合归一化，完整验证集选择checkpoint，
完整测试集只作最终评价。全部训练窗口和全部道路参与，没有Pilot窗口/节点抽样。
原基线48→48，改进分支额外读取2064步历史（过去7天同期窗口及其当时后续走势），
两者输入信息量与参数量不同，必须披露，不能把全部收益归因于检索算法本身。
原作者脚本存在BERT层数超出真实12层的问题，本轮使用12层预训练BERT、正确解冻LoRA，
属于本地修正复现；原论文表格单独列为参考，不冒充本轮实测值。
改进方法冻结各seed对应的基线，再训练既有6层周期BERT适配器，采用`no_deviation`、
关闭新增邻路与硬性精细形状约束；未独立证实的显式偏离分解不写成已验证贡献。

下面均为开发阶段历史记录，不是当前实验队列。

### Q/K–Value 隔离与等更新延长训练（用户缩小范围后已停止）

本轮不增加模块，只把已实现的上下文内容替换与来源 Value 替换拆成独立开关：

| 配置 | 当前/历史 Q/K 融合及内容替换 | 来源 Value 替换 |
|---|---|---|
| `isolation_control` | 关 | 关 |
| `isolation_qk` | 开 | 关 |
| `isolation_value` | 关 | 开 |
| `isolation_both` | 开 | 开 |

全部使用同一近邻表、初始参数、训练窗口、seed2021、`no_deviation` 与优化配置，
从头连续训练36轮，保存共同第12/24/36轮（3072/6144/9216更新）。中途不重置优化器，
Val-B 保存权重仅在该组训练结束后评价；不按各自最优 Val-B 轮次比较。
`scripts/run_spatial_isolation.ps1` 仍只调用同一个训练入口，结果位于
`pilot_results/spatial_isolation_isolation_*_v1`。

这里的 Value 是既有两条读取路径共享的来源记忆：更改它不改直接上下文 Q/K 编码，
但会通过 `utility_source_read` 间接影响最终检索 Query。这是干预位置隔离，不宣称
网络通路在因果上完全独立；自身88个记忆 token 的 Value 不直接替换。
若36轮仍未收敛，不能把持续负收益直接等同于方法理论无效。

此前整段历史 `time_shuffle` 会把历史候选相对后续的邻路样本搬进扰动 Key，
不能用其“打乱后改善”推断真实时间顺序有害。严格候选窗口内交换后，Val-B MAE
从18.0608略升至18.0668–18.0717。正常输入未发现相应时间越界。
诊断依据保存在 `research/evidence/spatial_repair_diagnosis_summary.json`。

### 邻路读取修复与重新验证（已完成，未通过独立增益门槛）

旧的标量邻路偏置已确认没有独立预测收益，以下修复**也未通过独立增益门槛**：

1. PEMS03 边权经核对为距离，新增近邻选择按距离升序；旧降序保留为消融，旧配置默认不变。
2. 当前 Query 与历史候选 Key 使用同一个目标条件空间读取模块；候选道路及邻路都只读取
   当时已观察的48步上下文，候选后续走势仅作 Value，不进入检索 Key。
3. 保留32维观测内容作为受控残差，同时使用 BERT 特征；这不等于语言预训练产生收益。

模型内统一开关，训练仍使用 `research/run_periodic_bert_repair.py`。队列：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_spatial_ablations.ps1 -PythonPath "C:/path/to/python.exe"
```

七组分别为完整修复、旧读取接口、无新增邻路、度数匹配的随机非邻路、无历史空间匹配、
无观测内容残差、旧距离排序；分配相同参数并以相同初值重新训练。其他部分统一采用
此前开发结果较好的 `no_deviation`，不能把与旧 full-policy 实验的差异全部算成空间收益。
每组 seed2021、12轮/3072更新，MAE/RMSE/MSE 以共同最后一轮为主要比较，另报 Val-A
选择的 checkpoint；正式 test 不参与。结果：`pilot_results/spatial_repair_spatial_*_v1`。

预定判据记录在工作区 `outputs/spatial_repair_plan.json`：完整修复相对旧接口及无新增
邻路至少降低1% MAE，相对随机非邻路至少降低0.5%，且 RMSE/MSE 不恶化；否则不能称
第三项创新已获支持。固定权重干预只作补充机制检查，不替代重训练消融。

### 最新已完成证据

以下仅为 PEMS03、seed2021、1024窗口×16道路训练、Val-B 64窗口×358道路的
探索性 Pilot；每组12轮、3072次更新。Val-A 用于 checkpoint 选择，Val-B 已用于开发。
没有使用正式 test，也没有确认充分收敛，不能与原文正式表格直接计算提升比例。

| 方法 | MAE | RMSE | MSE |
|---|---:|---:|---:|
| 上一版效用检索对照 | 17.9783 | 28.4649 | 810.2497 |
| 分组层级 + 邻路变化 + 硬性修正约束 | 18.0936 | 28.3333 | 802.7770 |
| 取消硬性修正约束，仍保留48步修正头 | 17.8817 | 28.3322 | 802.7112 |

- 预测效用监督有重训练消融支持：预训练版本 MAE 18.3404→17.9783；随机版本
  18.0813→17.8587，RMSE/MSE 同向改善。
- 新增邻路变化在**带硬约束版本**下只有约0.22% MAE收益；不能直接作为最终无约束
  版本的独立贡献。层级编码在该实验中没有收益。
- 最终无约束版本的层级/邻路独立消融及随机骨干对照由下列配置补齐。
- 取消约束版本虽在 Val-B 三指标改善，但 Val-A 未胜出，尚不能称为稳定提升。

最终无约束配置的两组重训练消融现已完成（各12轮/3072更新）：

| 无硬约束的最终接口 | MAE | RMSE | MSE |
|---|---:|---:|---:|
| 保留层级与新增邻路变化（候选） | 17.8817 | 28.3322 | 802.7112 |
| 关闭层级 `final_no_hierarchy` | 17.9638 | 28.3606 | 804.3262 |
| 关闭新增邻路变化 `final_no_relation` | 17.8738 | 28.2954 | 800.6309 |

这说明邻路变化在最终配置下**没有独立收益**，不能继续引用带硬约束版本的0.22%
作为最终方法的贡献。层级结构只有小幅条件收益，也不等于整个多周期记忆已验证。
`final_random` 在保存 checkpoint 后受控停止，以优先运行用户要求的第一点消融；
`periodic_final_ablation_v1` 标记为受控停止，不是三组全部完成，已完成两组结果不删除。

### 第一贡献的补充信息消融

同一模型入口新增 `--memory-policy`，不再将“关闭层级”冒充“去掉多周期记忆”：

- `global_long`：新增周期接口的长期部分改为全路网原始流量均值，按目标道路训练统计量
  标准化；目标道路近期48步保持不变。不会对随机抽到的16条目标道路求均值。
- `recent_only`：新增接口只保留近期48步；数值周期特征、长期 token 和历史候选检索同步禁用。
- `day_only`：保留近期与日内288步；同步禁用跨日 token、跨日数值特征及历史候选检索。
- `no_repeat` / `no_deviation`：分别去掉显式重复/偏离 token 和对应数值特征，保留原始历史。
  这是显式表示的消融，不声称模型无法从其他输入重新推断重复或差异。

每组重新拟合对应的训练段数值通路，用 Val-A 选正则化；不能留一个未消融的完整
ridge/检索旁路。冻结原始基线、短期图状态、来源道路、目标、窗口与 seed 不变。
因此结果只归因于**新增周期接口**，不是删除原模型里全部长期信息。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_memory_ablations.ps1 -PythonPath "C:/path/to/python.exe"
```

该脚本只顺序调用同一个训练入口；每组12轮，失败或预算不足会保留现场并停止队列。
结果在 `pilot_results/periodic_memory_<policy>_v1`，尚未完成的结果不得填成有效贡献。
本轮固定使用预训练 BERT 以隔离新增方法；不主张语言预训练优势，随机 BERT 是辅助
归因对照而不是创新点，已有对照及负结果保留。

```powershell
python research/run_periodic_bert_repair.py --repair-pilot --arms final_no_hierarchy final_no_relation final_random --matched-updates --utility-weight 0.1 --loss-mode step_raw_scaled_mae_mse --selection-mode mae_rmse_guard --interface-lr 5e-4 --backbone-lr 1e-4 --warmup-epochs 0 --epochs 12 --min-epochs 12 --budget-hours 1.5 --output-dir pilot_results/periodic_final_ablation_v1
```

`final_*` 仅表示当前候选配置的对照，**不是**正式完整实验。
结果原件保存在本地 `pilot_results/`（不随 Git 上传），可发布的精简证据在
`research/evidence/periodic_pilot_summary.json`，缺失结果必须保留未完成标记。

### 复现与上传边界

- 目前研究 runner 和基线缓存准备仍绑定 PEMS03；PEMS04/07/08 的数据在本地相邻
  FlashST 数据目录，**本仓库尚未完成它们的基线训练、缓存和正式实验适配**。
- 当前启动依赖冻结基线权重、配置、窗口索引、缓存与本地 BERT。默认路径见下方命令；
  仅克隆 Git 源码不足以重现结果。权重/缓存的分发链接和通用重建流程尚待补齐。
- 不将原始数据、大权重、缓存、环境或历史训练日志直接提交 Git。数据许可及权重
  分发许可需分别核对。原始模型和必要对照、测试、MIT 许可必须保留。
- 提交到新仓库前请核对 remote；当前 `origin` 是原始上游，不应直接向其推送研究改动。

## 当前研究主线：多周期关系表征 V4

当前只维护一个研究模型 `models/periodic_bert_v4.py`、一个训练入口
`research/run_periodic_bert_repair.py`。`numeric`、`repaired`、`random` 和机制消融
共用该接口，不是多套历史模型。原始 STH-SepNet 基线与原始说明保留在下方。

### 结构与输入

```text
models/
  periodic_bert_v4.py     当前模型：周期检索、关系注入、记忆读取
  periodic_tokens.py      共享有序 patch 编码与因果周期切片
  bert_backbone.py        本地 BERT / LoRA 构建
  ST_SepNet.py            原始/修正基线兼容实现
research/
  run_periodic_bert_repair.py  唯一研究训练入口
  prepare_baseline_cache.py   冻结基线数据准备，不训练第二套模型
  frozen_baseline.py          基线 checkpoint 加载和短期特征提取
  traffic_data.py             训练段归一化与缓存窗口对齐
  periodic_relation_data.py   排除目标自身的来源道路输入
  traffic_evaluation.py       历史场景划分
  periodic_reference.py       同数据预算的线性周期对照
  common.py                  共享批次、随机种子、精度与报告工具
tests/                       V4、共享数据、基线接口及兼容性定向测试
pilot_results/               历史与本轮证据，未清理
```

输入为每条道路过去2064步：最近48步、过去288步，以及过去7天各自对齐的
96步历史片段；历史片段内的“后48步”仍发生在当前预测起点之前。
最多4个其他道路提供最近48步来源记忆。输出为未来48步相对冻结基线的修正。
不使用真实星期/节假日、道路方向、节点 ID 或未来流量。
邻接矩阵与流量列顺序一致目前是数据来源假设，不把权重称为物理距离。

### 使用

当前联合修复配置为 `utility_fix / utility_random / utility_numeric`，仍共用上述一个模型和训练入口：

- 保留历史候选的原始48步走势；时间对齐通过选择历史偏移实现，不根据近期相关性缩放整段未来。
- 用**训练段**真实预测误差监督候选效用；按道路、按4个预测时段设置独立“不采用历史”选项。
  标签只进入辅助损失，模型前向、验证和推理不接收标签。辅助损失不受预测门控缩小影响。
- 当前与历史上下文统一为4个12步token，共享有序编码；短期状态只用于条件读取，
  回归头直接残差仅保留跨度/相对时间地址。BERT对照统一使用rank16 LoRA。
- `--matched-updates`关闭早停，所有臂完成相同轮次/更新数；共同最后一轮结果是等预算主对照，
  Val-A选出的最佳checkpoint另行报告。时间预算中断时不得声称等更新预算完成。

```powershell
python research/run_periodic_bert_repair.py --repair-pilot --arms utility_fix utility_random utility_numeric --matched-updates --loss-mode step_raw_scaled_mae_mse --selection-mode mae_rmse_guard --interface-lr 5e-4 --backbone-lr 1e-4 --warmup-epochs 0 --epochs 12 --min-epochs 12 --budget-hours 1.75 --output-dir pilot_results/periodic_utility_repair_v1
```

这是联合修复设计，不是已证实的新精度结果；三项修复的独立贡献仍须另做消融。
效用监督的独立训练消融使用相同命令并增加 `--utility-weight 0`，必须使用新输出目录。
它保留相同模型与前向，只用预测损失训练，不计算teacher；日志中的辅助损失为null，
不把“未启用辅助任务”记成辅助损失为零或检索已学会。与weight0.1版本比较共同最后一轮。

在仓库根目录、已安装 `requirements-pilot.txt` 的环境中运行：

```powershell
# 已有共享缓存时，新实验必须使用新目录；不要覆盖已完成的 V4。
python research/run_periodic_bert_repair.py --output-dir pilot_results/v4_new --arms numeric repaired random
# 仅做入口/训练恢复检查，不作为精度证据。
python research/run_periodic_bert_repair.py --output-dir pilot_results/v4_smoke_new --smoke
# 必要时由保留的基线 checkpoint / 配置 / origins 重新生成缓存。
python research/prepare_baseline_cache.py --output-dir pilot_results/cache_new
python -m unittest discover -s tests -q
```

默认使用 batch4、累积1、训练/验证节点分块64/256、关闭梯度检查点。
显存不足时可恢复较小分块。原始基线入口 `run_main.py`、模型、许可和脚本未重置。
重建历史缓存时须保持原窗口顺序和 batch4：原始基线文字 padding 会影响后接数值
token 的 BERT 位置；缓存冒烟的 `--max-windows` 会向上取整到完整 batch，避免拆散同批样本。
`holt_operator.py`、`utility_recoupling.py` 暂作 `ST_SepNet.py` 的兼容依赖保留，
不是当前研究方案；未为这些历史实验保留独立 runner。

### 本轮结论（2026-10-20）

单种子、同一冻结基线、1024×16训练窗口—道路对、Val-B 64窗口×358道路。
Val-B 是探索性开发集，不是原文正式测试集；长期验证历史存在重叠。

| 方法 | MAE | RMSE | MSE | 轮次 / 最佳轮 |
|---|---:|---:|---:|---|
| 冻结基线 | 21.643 | 33.229 | 1104.15 | — |
| numeric | 21.620 | 33.379 | 1114.19 | 20 / 19，达到上限，未确认收敛 |
| repaired（预训练 BERT） | 21.708 | 33.487 | 1121.40 | 19 / 14，早停 |
| random（随机 BERT + LoRA） | 21.855 | 33.692 | 1135.15 | 10 / 5，早停 |
| 同预算线性周期参照 | 18.590 | 29.125 | 848.28 | Val-A 选择正则化，无训练轮次 |

周期参照是 ridge 回归预测四个小时段修正并扩展至48步，并非复杂模型的同架构消融。
证据说明周期内容有预测价值，但当前 V4 未证明有效提取它；不能 claim PLM 已学会
时空规律或语言预训练带来稳定收益。numeric 的微小 MAE 收益也未改善 RMSE/MSE。
完整证据见 `pilot_results/periodic_bert_v4/report.json`。

### 收益丢失诊断（不修改 V4 checkpoint）

`research/diagnose_periodic_pipeline.py` 是定向诊断工具，不是另一套模型训练入口。
它复用 V4 保存的1024×16训练样本和64×358验证样本，只在训练段拟合探针，
用 Val-A 选择正则化，Val-B 仍仅作为探索性开发评价。

```powershell
python research/diagnose_periodic_pipeline.py --output-dir pilot_results/diagnosis_new
python research/diagnose_periodic_pipeline.py --confirm-only --output-dir pilot_results/confirmation_new
python research/diagnose_periodic_pipeline.py --optimization-only --output-dir pilot_results/optimization_new
python -m unittest tests.test_periodic_pipeline_diagnosis -q
```

2026-10-20 的诊断复现了线性周期 MAE18.590，梯度求解相同 ridge 目标得到18.5902。
去掉基线预测输入后为20.609；去掉短期图状态输入为18.667。
相同直接输入与同类探针下，BERT前 token 为18.829，BERT+32维投影后为20.954。
这些是冻结特征上的新读出探针，不是当前完整 V4 的新指标，也不证明信息不可逆丢失。

当前 V4 在 Val-B 的修正绝对值均值为3.292，但同一道路跨48步的修正标准差仅0.00058，
近似只调整整体偏置。16个 Val-A 窗口的 FP32检查也观察到此现象，因此不是仅由 BF16造成。
短期查询投影的平均范数约32.46，跨度嵌入约0.39；缩放短期状态可以增加时刻差异，
但未直接带来预测收益，不能当成已成功的修复。

直接 MAE 的 AdamW探针在当前学习率下未接近 ridge 的训练误差；从ridge暖启动时，
Val-A最佳点仍是第0步。它说明该优化设置不稳健，不代表MAE目标理论上弱于MSE。
证据位于 `pilot_results/periodic_pipeline_diagnosis/`、`periodic_pipeline_confirmation/`
和 `periodic_pipeline_optimization_check/`。主模型、旧结果和缓存均未修改。

### 历史小时级修复 Pilot（保留复现）

修复仍在同一个 `PeriodicBertV4` 与原训练入口中，通过 `--repair-pilot` 启用。
不启用时，原 V4 参数顺序与已保存 checkpoint 的预测保持不变。

```powershell
python research/run_periodic_bert_repair.py --repair-pilot --loss-mode hourly_normalized_mse --interface-lr 1e-4 --backbone-lr 2e-5 --budget-hours 1 --output-dir pilot_results/periodic_forecast_repair_v1
```

本轮固定 seed2021，沿用原1024×16训练预算与64×358的Val-A/Val-B。
各臂共用“归一化四小时段残差 MSE”训练任务，对齐线性参照；这不同于旧 V4 的
raw-unit MAE，不能把新旧差距全部归因于架构修改。checkpoint仍仅按Val-A MAE选择，
包括第0轮。统一报告 MAE/RMSE/MSE 与修正跨时段标准差。

- `forecast_fix`：直接读取基线预测，并增加分小时校准。
- `query_fix`：再平衡短期状态/时间/跨度查询，使用4个独立小时读出，保留查询残差。
- `full_fix`：再保留冻结的、仅训练段拟合的周期数值通路，BERT学习额外修正。
- `full_random`：同一完整接口，换成随机 BERT，用于检验预训练增量。
- `full_numeric`：同一完整接口，换成小型数值 Transformer。

数值通路的系数、归一化与完整 checkpoint 一起保存；不能将其收益计作 BERT贡献。
当前为单种子探索性Pilot，不预设修复成功；达到训练上限时如实标记未确认收敛。
新结果使用独立目录，原 V4、诊断与缓存均不覆盖。

### 历史逐步模式修复（非严格上下文检索）

保留同一模型与训练入口，新增 `fine` / `pattern` 两个 repair 配置，不复制历史模型。
固定数值周期通路仍只在训练段拟合；新增神经分支零初始化，初始精确恢复该数值预测。

- `fine_fix`：48个不同预测跨度查询和读出，可修改小时内部曲线。
- `pattern_fix`：在上述接口上增加显式模式算子，由神经记忆控制历史日/时间偏移选择、
  幅度和水平校准；近期未解释偏离以衰减形式加入历史模板。
- `pattern_random`：相同接口、随机冻结BERT + LoRA/LayerNorm。
- `pattern_numeric`：相同接口、小型全训练数值Transformer；不是FLOPs匹配对照。

模式算子从过去7天构造 -12/-6/0/6/12 步候选偏移。检索只比较历史上下文，
对应后续曲线也完全在当前预测起点之前。第7天缺失的两个负偏移候选被mask，不循环补齐。
已有近期/日内/跨日token继续作为BERT的输入；这里是显式模式读取与变换，不声称自然语言理解。

```powershell
python research/run_periodic_bert_repair.py --repair-pilot --arms fine_fix pattern_fix pattern_random pattern_numeric --loss-mode step_normalized_mae_mse --interface-lr 1e-4 --backbone-lr 2e-5 --epochs 25 --min-epochs 10 --patience 6 --budget-hours 1.25 --output-dir pilot_results/periodic_pattern_repair_v1
```

本轮仍使用seed2021、相同1024×16训练配对与64×358 Val-A/Val-B。
训练目标改为逐步归一化 MAE + 0.1 MSE，不再只监督小时平均。
因此与上一轮小时级修复的差别包括输出与监督，不能把全部差距归因于单一模块。
`pattern_fix` 对 `fine_fix` 检查模式算子的增量；对随机/数值骨干检查预训练增量；
所有方法首先必须超过无新增神经分支的数值周期对照，分别报告MAE/RMSE/MSE。
算子额外增加少量控制参数，`fine_fix`不是严格参数量匹配消融。

定向验证：新增9项测试，旧预测/runner12项、V4与执行17项，共38项通过；
四臂CUDA/BF16两轮冒烟完成，不作为准确率证据；真实旧 `full_fix` checkpoint
在改动前快照与新代码上FP32预测最大差值为0。数据、缓存、旧结果均未改动。

### 当前修复：严格上下文检索与合法评价边界

继续使用同一个 `PeriodicBertV4` 与 `research/run_periodic_bert_repair.py`，当前配置为
`repair="strict"`。旧配置仅用于checkpoint兼容和必要复现；旧报告不覆盖。

1. 所有新pilot先按共享加载器的验证边界过滤旧缓存起点，再抽样；记录被移除的起点。
   Val-A与Val-B目标区间至少间隔48个未使用采样点，非仅跳过48个预测起点。
   当前缓存离散起点筛选后实际间隔50步。缓存预测沿用原始batch packing，不重新计算。
2. 新增检索Q只编码当前48步及冻结图分支的近期状态；每个K只编码自己的历史48步上下文。
   使用共享BERT前3层独立编码，不与历史后续曲线混合激活。历史后续48步仅进入Value。
   主预测记忆仍可使用所有已发生的历史，严格隔离针对选择权重与幅度控制路径。
3. `pattern_mode=off/no_phase/no_amplitude/no_deviation`完整作用于新增算子；数值周期基线保持不变。
   这些是新增神经算子消融，不代表删除全模型中的全部周期信息。strict拒绝容易误解的旧开关。
4. 旧数值接口对缺失历史的负偏移提前抛出明确ValueError；新候选以mask处理，不循环填补。
5. 逐步损失为原始流量MAE/s +0.1MSE/s²，s取仅训练段拟合的节点标准差中位数。
   checkpoint只在Val-A RMSE不超过数值周期基线（仅1e-6相对数值容差）时按MAE更新。
   第0轮数值基线始终保留；没有合格改善则如实返回第0轮，不用Val-B调参。

```powershell
python research/run_periodic_bert_repair.py --repair-pilot --arms strict_no_pattern strict_fix strict_random strict_numeric --loss-mode step_raw_scaled_mae_mse --selection-mode mae_rmse_guard --interface-lr 1e-4 --backbone-lr 2e-5 --epochs 25 --min-epochs 10 --patience 6 --budget-hours 1.75 --output-dir pilot_results/periodic_strict_repair_v1
```

数据预算仍为seed2021、1024×16训练配对及64×358验证窗口。合法筛选发生在子采样前，
因此新的Val-B窗口集不同于旧报告，不能直接把新旧MAE差值当作架构收益。
四组正式对照共享新的窗口、数值通路、损失与选模协议；小Transformer不是FLOPs匹配对照。
`strict_no_phase/strict_no_amplitude/strict_no_deviation`为独立重训消融配置，已完成运行冒烟，
本轮先运行四组主对照，不把未重训的消融写成准确率证据。

修复定向验证共56项通过（复用未变化范围，未重跑整个原仓库）；7臂CUDA/BF16两轮冒烟完成。
真实旧pattern checkpoint对照旧源码快照的FP32预测差值为0；新strict冒烟checkpoint在历史
后续曲线扰动下的检索权重变化为0。新损失与约束只保证执行口径一致，不保证测试指标提高。

### 清理与兼容边界

按用户要求，74个历史文件的本地清理归档、一次性兼容探针及其记录已直接删除，
不再保留这份源码压缩包。不能再通过该归档恢复旧代码。当前源码、既有未提交修改、
原始数据、权重、缓存和实验结果未重置或删除；也未自动提交 Git。

现有 numeric/repaired/random checkpoint 的参数键、优化器顺序和预测函数兼容；
删除无用初始化后，新建模型的接口随机初始化轨迹可能不同，因此新训练不声称与历史
位级重放一致。历史结果目录中的 source_snapshot/execution_snapshots 未改动。
新入口不允许直接在旧源码布局的结果目录内 `--resume`；需要使用新的输出目录。
原实验快照保留作追溯证据，但不承诺它们独立包含旧流程的全部依赖。

---

## 原始项目：STH-SepNet: Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs

Welcome to STH-SepNet's GitHub repository! This repository hosts the code, data and model weight of **STH-SepNet** (KDD'25 Research Track).

**Abstract**: Spatio-temporal prediction is a pivotal task with broad applications in traffic management, climate monitoring, and energy scheduling. However, existing methodologies often struggle to balance model expressiveness and computational efficiency, especially when scaling to large real-world datasets. To tackle these challenges, we propose STH-SepNet (Spatio-Temporal Hypergraph Separation Network), a novel framework that decouples temporal and spatial modeling to enhance both efficiency and precision. Therein, the temporal dimension is modeled using lightweight large language models, which effectively capture low-rank temporal dynamics. Concurrently, the spatial dimension is addressed through an adaptive hypergraph neural network, which dynamically constructs hyperedges to model intricate, higher-order interactions. A carefully designed gating mechanism is integrated to seamlessly fuse temporal and spatial representations. By leveraging the fundamental principles of low-rank temporal dynamics and spatial interactions, STH-SepNet offers a pragmatic and scalable solution for spatio-temporal prediction in real-world applications. Extensive experiments on large-scale real-world datasets across multiple benchmarks demonstrate the effectiveness of STH-SepNet in improving predictive performance while maintaining computational efficiency. This work may provide a promising lightweight framework for spatio-temporal prediction, aiming to reduce computational demands and while enhancing predictive performance.

<div align="center">

**[<a href="https://arxiv.org/abs/2505.19620">Paper Page</a>]**
**[<a href="https://blog.csdn.net/2301_80828873/article/details/149446879">中文解读</a>]**

</div>

<p align="center">

  
<p align="center">
<img src="img/poster.png" height = "800" alt="" align=center />
</p>


## Citation

If you find this repository helpful for your research, please cite our paper. 

```tex
@inproceedings{chen2025decoupling,
  title={Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs},
  author={Chen, Jiawen and Shao, Qi and Chen, Duxin and Yu, Wwenwu},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2025},
  month={August 3rd-7th},
  address={Toronto, Canada},
  publisher={ACM}
}

```



## 1. Preparation
### 1.1 Environment 

The lightweight training requires torch 2.0+, to install all dependencies , update corresponding libraries: 
```shell
pip install -r requirements.txt
```

### 1.2 Data 
The data can be obtained and downloaded from ([Google Drive](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)), and makedir path ```dataset/``` and put dataset in ```dataset/```.




### 1.3 Large Language Models

The pretrained models can be downloaded from the links in the Table as below,  and makedir path ```huggingface/``` and put pretrained models in ```huggingface/```. For example, ```huggingface/BERT```


| Model 🤗 | Parameters | LLM Dimension | Model Description                                                                 |
|------------------------------------------------------------------------|------------|---------------|-----------------------------------------------------------------------------------|
| [BERT](https://huggingface.co/google-bert/bert-base-uncased)           | 110M       | 768           | A Transformer-based pre-trained model for NLP tasks, excelling in sentence classification and question answering. |
| [GPT-2](https://huggingface.co/openai-community/gpt2)                 | 124M       | 768           | A Transformer-based generative model, specialized in text generation and language modeling. |
| [GPT-3](https://huggingface.co/TurkuNLP/gpt3-finnish-large)           | 7580M      | 4096          | A large-scale Transformer-based generative model supporting various language tasks. |
| [LLAMA-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)            | 1230M      | 2048          | A multilingual model developed by Meta, designed for dialogue and knowledge retrieval tasks. |
| [LLAMA-7B](https://huggingface.co/huggyllama/llama-7b)                | 6740M      | 4096          | A multilingual model developed by Meta, suitable for various natural language generation tasks. |
| [LLAMA-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)   | 8000M      | 4096          | A multilingual model developed by Meta, focused on dialogue and instruction-tuning tasks. |
| [DeepSeek-Qwen1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1500M      | 1536          | A reasoning-focused model enhanced through reinforcement learning for improved reasoning capabilities. |
| |



## 2. Main Results
### 2.1 Training Preparation
#### 2.1.1 Download datasets and place them under `./dataset`.
#### 2.1.2 Download pretrained models and place them under `./huggingface`.
#### 2.1.3 Complete list of parameters

| Parameter | Type | Description | Default Value |
| :-------------: |:-------------:| :-------------------------------|:-------------: |
| `model` | string | Name of the model, among:<br> - `pool`: SHT-SepNet model with adaptive hypergraphs module <br> - `Autoformer`: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021) <br>- `TIMELLM`: Time Series Forecasting by Reprogramming Large Language Models (ICLR 2024) | `pool` |
| `dataset` | string | Name of the dataset, among:<br> - `inflow`: Bike traffic flow inflow <br> - `outflow`: Bike traffic flow outflow <br> - `PEMS03`: California Highway network PeMS traffic flow dataset <br> - `BJ`: Traffic dataset of road network in some areas of Beijing <br> - `METR`:Traffic sensor data in the Los Angeles area <br> You can also specify any additional graph dataset, in *edgelist* format, by editing `data_loader.py`| `inflow`|
| `node_num`| int | the node number of the network <br> -`Inflow, Outflow: 295` <br>-`PEMS03`:358 <br> -`BJ`:500 <br>-`METR`： 207| `295` |
| `features` | string |forecasting task, options:[M, S, MS], among: <br> - `M`: multivariate predict multivariate <br> - `S`: univariate predict univariate <br> - `MS`:multivariate predict univariate <br> | `M` |
| `llm_model`| string | LLM model: `BERT，GPT2，GPT3，LLAMA1b，LLAMA7b,LLAMA8b, deepseek2b ` | `BERT` |
| `static` | bool | Whether to use static adjacency matrix module | `False` |
| `gcn_true` | bool | Whether to use GCN module  | `True` |
| `adaptive_hyperhgnn` | string | Hypergraph neuron network: hgcn, hgat, hsage | `'hgcn'` |
| `hgcn_true` | bool | Whether to use HGCN module  | `True` |
| `temporal_true` | bool | Whether to use Temporal convolutional networks Module | `True` |
| `fusion_gate`| string | Style of module fusion: <br>- `adaptive`:dynamically adjusts the weight of time and spatial features，<br> -`attentiongate`:considers the internal relationship between the two features<br>-`lstmgate`:captures the dependence of space on temporal features <br>-`hyperstgnn `:fully integrated adaptive hypergraph spatio-temporal prediction(without LLMs) | `adaptive` |
| `llm_dim`| int | LLM model dimension <br> - `BERT, GPT2`: 768 <br> -`LLAMA7b,LLAMA8b,GPT3`: 4096 <br> - `LLAMA1b`: 2048 <br> - `deepseek2b`:1536 <br> | `768` |
| `seq_len`| int | input sequence length | `48` |
| `label_len`| int | start token length | `48` |
| `pred_len`| int | prediction sequence length | `48` |
| `enc_in`| int | encoder input size (e.g, Node num) | `295` |
| `dec_in`| int | decoder input size (e.g, Node num) | `295` |
| `c_out`| int | output size (e.g, Node num) | `295` |
| `d_model`| int | dimension of model | `32` |
| `n_heads`| int | num of heads | `16` |
| `e_layers`| int | num of encoder layers | `2` |
| `d_layers`| int | num of decoder layers | `1` |
| `d_ff`| int | dimension of fcn | `32` |
| `llm_layers`| int | num of llm layer | `6` |
| `train_epochs` | int | Number of training epochs  | `50` |
| `align_epochs` | int | Number of alignment epochs  | `10` |
| `alpha`  | float | Adjustable parameter to control hyperSTLLM or STLLM | `0.1` |
| `beta`  | float | Adjustable parameter to control hyperSTLLM or STLLM | `0.2` |
| `gamma`  | float | Adjustable parameter to control hyperSTLLM or STLLM | `0.5` |
| `theta`  | float | Adjustable parameter to control hyperSTLLM or STLLM | `0.2` |
| |


### 2.2 Training STH-SepNet
Run scripts for demonstration purpose under the folder `./scripts`. For example, to evaluate on BIKE datasets by:

```shell
sh ./scripts/BIKE/BERT_Bike_order.sh
sh ./scripts/BIKE/GPT2_Bike_order.sh
sh ./scripts/BIKE/GPT3_Bike_order.sh
sh ./scripts/BIKE/LLAMA1B_Bike_order.sh
sh ./scripts/BIKE/LLAMA7B_Bike_order.sh
sh ./scripts/BIKE/LLAMA8B_Bike_order.sh
sh ./scripts/BIKE/Deepseek_Bike_order.sh
```

### 2.3 Training STH-SepNet-GNN
Run scripts for demonstration purpose under the folder `./scripts`. For example, to evaluate on BIKE datasets by:

```shell
sh ./scripts/BIKE/BERT_Bike.sh
sh ./scripts/BIKE/GPT2_Bike.sh
sh ./scripts/BIKE/GPT3_Bike.sh
sh ./scripts/BIKE/LLAMA1B_Bike.sh
sh ./scripts/BIKE/LLAMA7B_Bike.sh
sh ./scripts/BIKE/LLAMA8B_Bike.sh
sh ./scripts/BIKE/Deepseek_Bike.sh
```


### 3. Ablation Study
### 3.1 STH-SepNet-without LLMs (w/o) (Undecoupled Version)
For example, to evaluate on BIKE datasets, Set `--fusion_gate`  as `hyperstgnn`. Note that fully integrated adaptive hypergraph spatio-temporal prediction(without LLMs)

### 3.2 STH-SepNet-Mixorder 
Run scripts for demonstration purpose under the folder `./scripts`. For example, to evaluate on BIKE datasets by:

```shell
sh ./scripts/BIKE/BERT_Bike_mixorder3.sh
sh ./scripts/BIKE/GPT2_Bike_mixorder3.sh
sh ./scripts/BIKE/GPT3_Bike_mixorder3.sh
sh ./scripts/BIKE/LLAMA1B_Bike_mixorder3.sh
sh ./scripts/BIKE/LLAMA7B_Bike_mixorder3.sh
sh ./scripts/BIKE/LLAMA8B_Bike_mixorder3.sh
sh ./scripts/BIKE/Deepseek_Bike_mixorder3.sh
```


### 3.3 STH-SepNet-Effective Order on Adaptive Hypergraph
Run scripts for demonstration purpose under the folder `./scripts`. For example, to evaluate on BIKE datasets by:

```shell
sh ./scripts/BIKE/BERT_Bike_Outflow_flexible_order3.sh
sh ./scripts/PEMS/BERT_PEMS03_flexible_order.sh
```

### 3.4 STH-SepNet-Fusion Mechanism between Spatio and Temporal Features
The fusion mechanism can be specified using the `--fusion_gate` argument. The available options are:

- `adaptive`: Dynamically adjusts the weight of time and spatial features.
- `attentiongate`: Considers the internal relationship between the two features.
- `lstmgate`: Captures the dependence of space on temporal features.
- `hyperstgnn`: Fully integrated adaptive hypergraph spatio-temporal prediction (without LLMs).

## 4. Performance and Visualization

<p align="center">
<img src="img/figure02.png" height = "680" alt="" align=center />
</p>

<p align="center">
<img src="img/tableb2.png" height = "300" alt="" align=center />
</p>


<p align="center">
<img src="img/tableb4.png" height = "300" alt="" align=center />
</p>


<p align="center">
<img src="img/tableb5.png" height = "200" alt="" align=center />
</p>


<p align="center">
<img src="img/tableb6.png" height = "175" alt="" align=center />
</p>





## Further Reading 
### Our research baselines models refer to the following works and their repository code.


STG4Traffic: {A} Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction. [[Paper]](https://arxiv.org/abs/2307.00495)[[Code]](https://github.com/trainingl/STG4Traffic?utm_source=catalyzex.com).
```tex
@article{DBLP:journals/corr/abs-2307-00495,
  author       = {Xunlian Luo and Chunjiang Zhu and Detian Zhang and Qing Li},
  title        = {STG4Traffic: {A} Survey and Benchmark of Spatial-Temporal Graph Neural
                  Networks for Traffic Prediction},
  journal      = {CoRR},
  volume       = {abs/2307.00495},
  year         = {2023}
}
```

Deep Time Series Models: A Comprehensive Survey and Benchmark. [[Paper]](https://arxiv.org/abs/2407.13278)[[Code]](https://github.com/thuml/Time-Series-Library).
```tex
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}

```
