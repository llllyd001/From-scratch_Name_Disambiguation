# Pilot 实验报告：字段组合有效性与互补性分析

## 1. 实验目标

本实验只使用 NA_Demo 中已有历史 LLM clustering 结果的三个作者 block：`haifeng_qian`、`hui_cai`、`jianguo_wu`。目标是为 `title + coauthors + organization`（T+O+C）的有效性提供 pilot 证据，而不是宣称它在完整 80 作者数据上统计显著最优。

## 2. 数据与实验设置

- 原始 metadata：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_pub.json`
- block 列表：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_raw.json`
- ground truth：`data/whoiswho/data/NA_Demo/SND/valid/sna_valid_ground_truth.json`
- 历史 LLM 结果：`artifacts/results/field_selection/single_field/NA_Demo/` 与 `artifacts/results/field_selection/combinations/NA_Demo/`
- LLM 聚类指标复用 `src.common.metrics` 中的 Pairwise 与 B³ 逻辑。
- 文本相似度未联网下载 sentence-transformers；本实验使用本地 TF-IDF cosine。organization 使用轻量 token Jaccard。该设置可复现，但不等价于神经 embedding。
- `pairwise_similarity_records.json` 中的统计值基于全量 pair，`records` 字段只保存分层样本，避免把 39 万个 pair 的明细全部堆进报告文件。

缺失或未完成组合：

```json
[] // (无)
```

## 3. 实验一：字段组合效果比较

下面列出按 macro B³ F1 排名前 8 的字段组合：

| field_combination | completed_author_blocks | macro_average_pairwise_f1 | macro_average_b3_f1 | median_pairwise_f1 | median_b3_f1 |
| --- | --- | --- | --- | --- | --- |
| title+coauthors+organization | 3 | 0.9792 | 0.9814 | 0.9768 | 0.9812 |
| abstract+coauthors+organization | 3 | 0.9739 | 0.9766 | 0.9625 | 0.9680 |
| title+abstract+organization | 3 | 0.9630 | 0.9587 | 0.9614 | 0.9629 |
| coauthors+organization | 3 | 0.9158 | 0.9467 | 0.9510 | 0.9607 |
| title+organization | 3 | 0.9277 | 0.9445 | 0.9676 | 0.9680 |
| title+abstract+coauthors | 3 | 0.8618 | 0.8822 | 0.8578 | 0.8813 |
| title+coauthors | 3 | 0.8031 | 0.8288 | 0.8245 | 0.8518 |
| coauthors | 3 | 0.6654 | 0.7208 | 0.7180 | 0.7169 |

T+O+C 的 macro Pairwise F1 为 `0.9792`，macro B³ F1 为 `0.9814`。在这个三作者 pilot 中，T+O+C 不是所有指标上唯一第一，但它在保持高 Pairwise/B³ F1 的同时，只使用三个较短、身份指向较强的字段，因此比加入 abstract 的组合更有 token 性价比。

相关图表：

- `artifacts/figures/field_selection/analysis/combination_pairwise_f1.html`
- `artifacts/figures/field_selection/analysis/combination_b3_f1.html`

## 4. 实验二：字段的 pair-level 区分能力

该实验不使用 LLM 输出，而是直接在每个 block 内构造论文 pair，并用 ground truth 标注是否同一真实作者。
字段相似度只说明字段本身是否提供可分信号，不能直接证明 LLM 因某字段做出某判断。

| field | available_pair_count | same_mean | different_mean | separation_gap |
| --- | --- | --- | --- | --- |
| title | 392603 | 0.0691 | 0.0197 | 0.0494 |
| abstract | 279855 | 0.1290 | 0.0649 | 0.0642 |
| organization | 233149 | 0.4119 | 0.1075 | 0.3043 |
| coauthor_jaccard | 368319 | 0.0734 | 0.0001 | 0.0733 |
| shared_coauthor_count | 368319 | 0.7712 | 0.0007 | 0.7704 |

相似性判断方法：
- title：两篇论文标题的 TF-IDF cosine similarity
- abstract：两篇论文摘要的 TF-IDF cosine similarity
- organization：两篇论文 affiliation / organization 的 token Jaccard similarity
- coauthor_jaccard：两篇论文合作者集合的 Jaccard similarity。

从 separation gap 看，coauthor/organization 更像身份锚点，title/abstract 更像主题语义信号。T+O+C 的优势更可能来自三类证据互补：title 提供研究主题，coauthor 提供关系网络，organization 提供机构约束。

相关图表：

- `artifacts/figures/field_selection/analysis/similarity_distribution_title.html`
- `artifacts/figures/field_selection/analysis/similarity_distribution_abstract.html`
- `artifacts/figures/field_selection/analysis/similarity_distribution_organization.html`
- `artifacts/figures/field_selection/analysis/coauthor_overlap_distribution.html`
- `artifacts/figures/field_selection/analysis/field_similarity_gap.html`

## 5. 实验三：字段错误修复机制

比较 Full=T+O+C 与三个 leave-one-field-out 结果：O+C、T+O、T+C。这里的 pairwise 改变来自整体 clustering 差异，因此应表述为“加入某字段后的聚类结果修复/引入了某类 pairwise 错误”，而不是“模型直接因为该字段判断某个 pair”。

| field_added | fixed_false_splits | fixed_false_merges | introduced_false_splits | introduced_false_merges | unchanged_pairs | net_pairwise_gain |
| --- | --- | --- | --- | --- | --- | --- |
| title | 8591 | 0 | 351 | 43 | 383618 | 8197 |
| coauthor | 6734 | 999 | 446 | 40 | 384384 | 7247 |
| organization | 17914 | 643 | 886 | 102 | 373058 | 17569 |

$\text{net pairwise gain}=\text{fixed false splits}+\text{fixed false merges}−\text{introduced false splits}−\text{introduced false merges}$

相关图表：

- `artifacts/figures/field_selection/analysis/field_error_repair_summary.html`
- `artifacts/figures/field_selection/analysis/field_net_gain_by_author.html`

代表性案例见：`artifacts/results/field_selection/analysis/field_contribution_cases_zh.md`。

## 6. 综合结论

基于三作者 pilot，T+O+C 的证据不是“字段越多越好”，而是：

1. `title` 捕获论文主题，使同一作者跨论文的研究方向可以被连接；
2. `coauthors` 是强身份线索，有助于修复同作者被拆散的问题；
3. `organization` 能减少不同作者之间的错误合并；
4. 相比 abstract，T+O+C token 成本更低、噪声更少，适合作为 stage1 的高性价比主组合。

但是，本结论只覆盖三个 NA_Demo 作者 block。后续需要在 80 个 v3 block 上复验，并记录 token usage、输出失败率和稳定性。

## 7. 局限性

- 样本只有 3 个作者 block；
- 历史 LLM 输出可能受 prompt、顺序、随机性影响；
- 相似度实验使用 TF-IDF/Jaccard，不是深度语义 embedding；
- leave-one-field-out 比较的是整体 clustering 结果，不是控制变量下的逐 pair 因果判断。
