# 数据准备

实际数据 CSV 不随代码仓库分发。

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

当前统一训练入口仅支持 PEMS03、METR-LA，运行这两个数据集只需前四个文件。
`prompt_bank/` 为仓库提供的提示文本，不需要重新下载。
下载链接由原作者维护，访问权限、可达性及下载内容未在本次发布中重新验证。
具体环境、命令和方法边界见根目录 README 的“快速使用”。
