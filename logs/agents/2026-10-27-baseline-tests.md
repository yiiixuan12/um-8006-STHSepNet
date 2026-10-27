# Agent log — AI-JINQI-004 (2026-10-27 · Jinqi)

- 阶段：Implementation / testing / evaluation validation
- 工具/模型：OpenAI Codex（gpt-6.1-sol）；采样设置 not exposed。
- 任务：检查tests目录中基线单元、集成及评估测试的覆盖情况，补全缺项并执行测试。

## 咨询与决定

原有测试已覆盖部分窗口边界、早停、检查点和运行脚本。接受补充关键回归测试的方案：用可手算的合成数据验证数据处理与指标，用小模型执行离线训练流程，并显式启用真实BERT/GPU集成测试。测试补充阶段没有修改训练逻辑。

## 修改代码与具体部分

- `tests/test_baseline.py`：补充五数据集节点数及时间间隔、秒/微秒/纳秒转换、训练集归一化和常量节点、缺失时间间隔保留、无效数据及邻接矩阵拒绝、48步窗口边界测试。
- 同一文件直接测试基线评估函数：零目标不掩码、原单位恢复、全部节点与预测步、末批保留、全局RMSE和窗口误差保存；独立手算期望为MAE=12.8、MSE=456、RMSE=sqrt(456)。
- 同一文件补充小模型真实训练/保存/重载/测试流程、禁止基线误建适配器、输出覆盖和源码续训保护；带dropout的续训与连续训练在参数、优化器和随机状态上完全一致。
- 同一文件补充五数据集×三个种子的发布汇总、样本标准差及误差文件篡改检出测试；这些是合成测试值，不是新增精度实验。
- `tests/test_baseline_gpu.py`：显式启用时在BIKE-Outflow和PEMS03上运行一轮短训练，检查有限指标、保存结果及全部12层LoRA；原始BERT参数不应作为可训练参数保存。
- `tests/README.md`及README测试说明：提供常规测试、GPU测试和已发布指标复算命令。

## AI测试结果

常规回归命令：

```powershell
python -m unittest tests.test_baseline tests.test_coursework tests.test_coursework_early_stop tests.test_training_scripts
```

结果：**27项测试全部通过**。

显式启用`RUN_BASELINE_GPU=1`后执行`python -m unittest tests.test_baseline_gpu`：**1项真实BERT/GPU集成测试通过，包含BIKE-Outflow和PEMS03两个数据集子场景**。GPU测试输出使用临时目录；短训练指标不作正式精度证据。

## 人工验证

**用户已人工查询并核对AI执行的测试结果，确认测试结果均已通过。** 记录的是人工审查测试输出，不宣称人工另外重跑了整套测试，也不把局部测试通过等同于整个仓库全部测试通过。

## 证据

- [基线单元及集成测试](../../tests/test_baseline.py)
- [真实BERT/GPU测试](../../tests/test_baseline_gpu.py)
- [早停与恢复测试](../../tests/test_coursework_early_stop.py)
- [脚本测试](../../tests/test_training_scripts.py)
- [测试运行说明](../../tests/README.md)
- [AI_USAGE](../../AI_USAGE.md)

本日志为脱敏咨询与执行摘要，不含原始数据值、本机绝对路径或凭据；人工确认依据本次用户声明。
