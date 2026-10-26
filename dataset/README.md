# 数据准备

H1所需完整数据与预训练权重需从来源下载；本次H1更新不上传新的原始数据，也不删除仓库中同学已有的数据文件。

- [原作者数据下载目录（Google Drive）](https://drive.google.com/drive/folders/1uhQqAdrIplhhKCHn0McnB-trve6_rATD?usp=drive_link)
- [原作者说明](https://github.com/jiawenchen10/STHSepNet#12-data)

下载后保留原文件名、时间列、道路列顺序及配套邻接矩阵，放在本目录：

```text
PEMS03/PEMS03.csv
PEMS03/PEMS03_adj.csv
METR/METR.csv
METR/METR_adj.csv
Bike/inflow.csv
Bike/outflow.csv
Bike/adj.csv
BJ/BJ.csv
BJ/BJ_adj.csv
```

H1入口 `research/run_h1.py` 支持五个数据集。所有H1数据集都需要同节点顺序的邻接CSV；流量CSV首列为时间，其余列为节点，邻接使用默认CSV表头读取并应为N×N。

BIKE-Inflow/Outflow是小时级，H1历史240小时、预测48小时；PEMS03、METR-LA、BJ500是五分钟级，H1历史2064步、预测48步。保留原始采样，不重采样，不把BIKE的48步写成4小时。只用训练段计算各节点均值/标准差，按预测目标边界约7:1:2划分。
`prompt_bank/` 为仓库提供的提示文本，不需要重新下载。
下载链接由原作者维护，访问权限、可达性及下载内容未在本次发布中重新验证。
具体环境、命令和方法边界见根目录 README。H3/KAN的原入口和数据处理保持不变。
