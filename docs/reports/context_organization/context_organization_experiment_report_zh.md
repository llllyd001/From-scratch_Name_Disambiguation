# Context organization 实验报告

## 1. 实验目的

本实验固定使用 `title + organization + coauthors`（T+O+C）三类输入证据，只改变论文在 prompt 中的排列或中性分组方式，检验信息组织方式是否会影响 LLM 作者姓名消歧结果。

## 2. 实验设置

- 数据集：`data/whoiswho/data/v3/SND/valid`
- 作者数量：6，其中 easy=2，medium=2，hard=2
- 作者选择：基于既有 T+O+C B³ F1 分层，用固定随机种子 `42` 每层抽取 2 个作者
- 选中作者：

| author_id | difficulty_group | paper_count | historical_b3_f1 |
| --- | --- | --- | --- |
| zheng_hu | easy | 456 | 0.9299 |
| hongbin_sun | easy | 521 | 0.9717 |
| guowei_zhang | medium | 380 | 0.8207 |
| bo_yu | medium | 1535 | 0.8646 |
| yi_qian | hard | 651 | 0.0567 |
| haibo_li | hard | 613 | 0.7872 |

- 模型：`deepseek-v4-flash`，temperature=0，输出格式沿用原 T+O+C compact assignment JSON
- 条件：`original_flat`、`random_order`、`chronological_order`、`organization_grouped`、`coauthor_network`、`multifield_similarity`
- `random_order`：每个作者 3 个固定随机排列
- 指标：Pairwise Precision/Recall/F1 与 B³ Precision/Recall/F1

## 3. 主要结果

| condition | mean_b3_f1 | median_b3_f1 | mean_pairwise_f1 | mean_delta_b3_f1 | improved_author_count | degraded_author_count | parse_failure_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_flat | 0.8871 | 0.8848 | 0.8396 | 0.0000 | 0 | 0 | 0 |
| random_order | 0.8576 | 0.8788 | 0.7976 | -0.0295 | 1 | 5 | 1 |
| chronological_order | 0.8507 | 0.7993 | 0.7879 | -0.0194 | 1 | 3 | 1 |
| organization_grouped | 0.9020 | 0.9390 | 0.8436 | 0.0015 | 3 | 2 | 1 |
| coauthor_network | 0.8837 | 0.8873 | 0.8116 | 0.0059 | 2 | 1 | 2 |
| multifield_similarity | 0.8850 | 0.8846 | 0.8613 | -0.0032 | 1 | 2 | 2 |

说明：本表中 mean/median 指标只基于成功解析且通过 paper 覆盖校验的调用；`mean_delta_b3_f1` 只在同一作者的 `original_flat` 与目标条件都成功时计算。解析失败本身是实验结果的一部分，表示该组织方式增加了输出不完整或格式失败风险。

缺失论文处理方法：本实验没有把 LLM 输出中缺失的论文补成 singleton。每次调用都会先检查所有输入 paper ID 是否在输出中恰好出现一次；若有缺失、未知 ID 或跨 cluster 重复，就将该次调用标记为 `parse_success=False`，不计算 Pairwise/B³，也不纳入条件均值。也就是说，公共指标函数虽然有 missing-as-singleton 的兜底能力，但本实验主表采用更严格的 full-coverage 口径，把缺失论文视为该组织方式的输出可靠性失败。

解析失败统计：

| condition | parse_failure_count |
| --- | --- |
| random_order | 1 |
| chronological_order | 1 |
| organization_grouped | 1 |
| coauthor_network | 2 |
| multifield_similarity | 2 |

## 4. 主要发现

- 在成功解析样本上，平均 B³ F1 最高的是 `organization_grouped`，mean B³ F1=0.9020；baseline `original_flat` 为 0.8871。由于部分条件有解析失败，这个均值不能单独解释为稳定优势。
- `random_order` 的作者级平均 delta B³ F1 为 -0.0295，说明顺序扰动在当前样本上存在明显影响。
- `organization_grouped` 的 mean predicted cluster count 为 18.60，baseline 为 24.33；它在 `bo_yu` 上缺失 774 篇论文，说明大 block 分组会显著增加输出不完整风险。
- `coauthor_network` 的 mean predicted cluster count 为 19.50；它在 `yi_qian` 上有最高个体增益，但在 `bo_yu` 上缺失 229 篇论文，因此更像高风险高方差策略。
- random order 最稳定作者为 `hongbin_sun`，B³ F1 range=0.0080。

## 5. 典型案例

| author_id | difficulty_group | condition | b3_f1 | delta_b3_f1 | predicted_cluster_count |
| --- | --- | --- | --- | --- | --- |
| bo_yu | medium | random_order | 0.6718 | -0.1478 | 18.0000 |
| bo_yu | medium | chronological_order | 0.7571 | -0.0625 | 52.0000 |
| guowei_zhang | medium | chronological_order | 0.7887 | -0.0419 | 16.0000 |
| bo_yu | medium | multifield_similarity | 0.7809 | -0.0387 | 30.0000 |

## 6. 结论

在当前 6 作者 pilot 中，信息组织方式会改变部分作者的 LLM 聚类结果，但平均 delta 很小，且分组方法在大 block 上更容易产生缺失论文或截断输出。因此更稳妥的结论是：信息组织方式是一个有影响但不稳定的 prompt-side 因素；若要作为 routing action，需要先加入 block size / prompt token 风险控制，而不能直接替代原始 flat baseline。

结果文件：`artifacts/results/context_organization/`

图表目录：`artifacts/figures/context_organization/`
