# 1. 项目目标

在实体消歧任务中，LLM 并不是所有样本都可靠，也不是所有样本都值得调用。研究如何在调用 LLM 之前，根据原始元数据预测 LLM 的可靠性，并据此决定是否调用、使用哪些字段、是否升级到更高成本策略。

给定一个低成本 LLM action，能否提前预测它是否可靠？

# 2. 项目定位

项目定位为：

**数据分析 + 轻量方法创新，而不是纯工程 client，也不是纯 LLM prompting。**

核心问题：
- LLM 在 entity disambiguation 中并非对所有 ambiguous blocks 都可靠。
- 我们能否在调用 LLM 前，用低成本元数据特征预测其可靠性边界，并在预算约束下选择性调用？

核心贡献为三点：

- Problem formulation
  
  将 LLM-based AND 从“全量调用 LLM”重新表述为 pre-LLM reliability routing / selective invocation 问题。

- Feature and failure analysis

  分析哪些 metadata structure 与 LLM 成功/失败相关，包括 over-merging、over-splitting、paper_count、coauthor overlap 等。

- Budget-aware router

  训练并评估一个可自动选择特征、模型和 threshold 的 pre-LLM router，用 accepted precision、coverage、hard-block recall、cost saving 等指标衡量。

工程 client 可以作为**Prototype system**，展示如何把 router 集成进 AND workflow，但不是主贡献。

# 3. 主线

先证明：LLM 在姓名消歧中的成功/失败是可以被 pre-LLM 特征部分预测的。

然后再证明：这种预测可以用于分流，减少无效 LLM 调用。

主要研究下面这四个子问题：

- RQ1: 在不调用 LLM 的前提下，能否根据作者 block 的原始元数据预测该 block 的消歧难度？

- RQ2: 能否先在一个较小、代表性的作者集合上进行有限 LLM 调用，再利用这些调用结果训练一个 pre-LLM reliability predictor，从而预测在未调用 LLM 的作者 block 上，哪些 LLM action 可能可靠？

- RQ3: 基于这种预测，能否构建一个自动选择特征、模型和 threshold 的 router，在保证 accepted 作者质量的同时减少不必要的 LLM 调用？

- RQ4: 字段组合是否应该固定？低成本字段扩展是否可能带来 block-dependent 的收益？
