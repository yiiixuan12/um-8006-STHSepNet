# Agent log — AI-KAN-004 (2026-09-18 · KAN 负责人)

- 阶段: Experiment execution / debugging
- 工具/模型/版本: Gemini 3 Pro / 2026-09-18 / 默认设置（temperature 等未暴露则标注 not exposed）
- 委托任务: 协助编写 3 seeds `2021/42/7` 的 MLP/KAN 对照运行脚本，并检查 seed 生效、Windows spawn 保护、checkpoint key 前缀和 dtype 适配。

## 中文对话/输出摘要
Agent 建议统一使用同一训练脚本，仅追加 `--use_kan`，每个 seed 单独输出目录，禁止覆盖已有 run；建议从保存的 .npy 用 `scripts/eval_metrics.py` / `collect_results.py` 复算 MAE/RMSE/robust MAPE，不直接信任训练日志打印值。

## 决定: Accepted
符合"同一数据划分、同一 BERT、同一 fusion_gate、同一预算，唯一变量 `--use_kan`"的协议；环境修复由基线与 KAN 共用，不作为科学变量。

## 人工验证
检查运行命令；确认 seed 传入模型初始化与 DataLoader；确认每个 seed 日志、预测/真值数组、report.json 独立保存；从 .npy 复算指标。

## 证据
- 运行入口: `src/run_main.py`（`--seed`、`--use_kan` 生效）
- 复算脚本: `scripts/eval_metrics.py`、`scripts/collect_results.py`
- 运行登记: `experiments/registry.csv`（run R-KAN-2021 / R-KAN-42 / R-KAN-7 等）
- 保存数组: `results/data_pred/*.npy`
- 汇总: `results/comparison_table.csv`、`results/comparison_summary.json`
- 本日志: `logs/agents/2026-09-18_kan_run_debug.md`

## 脱敏
无。
