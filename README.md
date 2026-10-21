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
