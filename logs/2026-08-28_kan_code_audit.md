# Agent log — AI-KAN-001 (2026-08-22 · KAN 负责人)

- 阶段: Paper understanding / reproduction planning
- 工具/模型/版本: DeepSeek V4.1 Flash / 2026-08-22 / 默认设置（temperature 等未暴露则标注 not exposed）
- 委托任务: 详细解读 KDD'2025 论文《Decoupling Spatio-Temporal Prediction: When Lightweight Large Models Meet Adaptive Hypergraphs》(STH-SepNet)；识别可改进点；判断能否用 KAN 改进。

## 中文对话/输出摘要
1. 论文核心内容：解耦式架构——时间维度用轻量级 LLM（BERT/GPT-2）捕捉低秩时间动力学，空间维度用自适应超图神经网络动态构造超边建模高阶交互，门控机制融合时空表示。空间侧关键设计：MixProp 多层信息聚合、自适应图卷积与超图卷积固定系数融合（X = r·X_GCN + (1-r)·X_HGCN，r∈[0,1] 可调超参）；全局趋势模块对所有节点序列做平均池化后送入 LLM。
2. 识别可改进点：
   - 表示阶段：AvgPool(X) 将所有路段压缩为共享表示，牺牲路段级 intra-day 与 cross-day 周期模式。
   - 融合阶段：GCN–超图融合系数 r 为全局标量，无法按节点或时间步自适应调整。
   - 预测阶段：门控融合后 FlattenHead.linear = nn.Linear(192,48)，固定激活限制最终非线性映射容量。
3. KAN 可行性：Agent 判断 KANLinear 可保持 in=192、out=48 不变，仅把固定激活替换为可学习 B 样条基函数；替换点应严格限定在门控融合之后的 FlattenHead，不改融合层内部线性层，保持等信息化对比。Agent 同时建议"顺便开启 update_grid 并增加历史输入以放大增益"。

## 决定: Modified
接受论文解读与三条改进点定位；接受"KAN 仅替换预测头、保持 192→48 维度"的可行性判断。拒绝开启 update_grid 与增加历史输入：前者引入第二个自适应变量、破坏 H3 唯一受控变量原则；后者破坏等信息化对比、混淆 KAN 头效应与额外信息。

## 人工验证
对照论文公式 (7)(22)(26)–(27) 与官方代码逐行核对 AvgPool、融合系数 r、FlattenHead 调用路径；确认 MLP 头参数量 9,264、KAN 头 103,680、差值 94,416；确认 H3 输入 48 步、输出 48 步，与基线一致。

## 证据
- 代码: `src/models/ST_SepNet.py`（FlattenHead / gate 调用）、`src/layers/STHGNN.py`（融合系数 r）
- 报告: `report/final_report.md` §6.1 / §7
- 本日志: `logs/agents/2026-08-22_kan_paper_review.md`

## 脱敏
无；仅公开论文与仓库代码片段。
