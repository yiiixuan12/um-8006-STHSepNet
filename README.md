# STH-SepNet 课程项目

基于 [STH-SepNet](https://github.com/jiawenchen10/STHSepNet) 的修正版基线与多周期记忆改进。当前统一入口支持 **PEMS03、METR-LA**，需要 NVIDIA CUDA 显卡。

## 1. 安装环境

使用 Python 3.10，先安装适配显卡的 CUDA 版 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements-pilot.txt
```

本地使用 PyTorch 2.10.0+cu128、Transformers 4.44.2。不要使用旧的 `requirements.txt` 安装。

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
