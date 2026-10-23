# Agent log — AI-KAN-002 (2026-08-28 · KAN 负责人)

- 阶段: Codebase audit / reproduction planning
- 工具/模型/版本: Gemini 3 Pro / 2026-08-28 / 默认设置（temperature 等未暴露则标注 not exposed）
- 委托任务: 审计 STH-SepNet 中 FlattenHead、`fusion_gate=adaptive` 与 `--use_kan` 开关的插入位置，检查是否存在隐式维度变化、参数共享或 checkpoint key 冲突。

## 中文对话/输出摘要
Agent 给出调用链：融合输出 → FlattenHead → 预测；建议在 FlattenHead 内条件分支切换 `nn.Linear(192,48)` 与 `KANLinear(192,48)`，并保持 `fusion_gate` 内部线性层不变；同时提醒 checkpoint key 前缀和 dtype 适配需与基线共用修复。

## 决定: Accepted
方案保持唯一受控变量为预测头类型，符合 H3 的 information-matched 设计；环境修复由基线与 KAN 共用，不作为科学变量。

## 人工验证
检查 `src/models/ST_SepNet.py` 调用链与 config；运行 sanity（随机 batch (2,48,295) 的 MLP/KAN forward+backward）；核对参数量与预期一致；1-epoch 冒烟训练 → checkpoint 保存/加载 → 清理，exit 0。

## 证据
- 代码: `src/models/ST_SepNet.py`（`--use_kan` 分支）、`src/kan/KANLinear.py`
- 冒烟验证: `tests/synthetic_sanity.py`（forward/backward sanity）
- 本日志: `logs/agents/2026-08-28_kan_code_audit.md`

## 脱敏
无；报错栈中的本地绝对路径已替换为 `[REDACTED: local path]`。
