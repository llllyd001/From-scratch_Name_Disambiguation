# Prompt/context instruction 实验报告

## 1. 实验目的

固定作者、论文顺序和 T+O+C 数据，只改变 prompt 和 context instruction，观察任务说明、证据指导、错误提醒和分阶段分析的影响。

## 2. 实验设置

- 数据集：`data/whoiswho/data/v3/SND/valid`
- 作者数量：3，easy=1，medium=1，hard=1
- 模型：`deepseek-v4-flash`，DeepSeek API，temperature=0
- 条件：`minimal`、`task_definition`、`evidence_guidance`、`error_aware`、`analyze_then_cluster`

## 3. Prompt 条件

`minimal` 只说明聚类任务；`task_definition` 明确姓名消歧背景；`evidence_guidance` 解释 T/O/C 证据局限；`error_aware` 提醒过拆分和过合并；`analyze_then_cluster` 要求先检查证据再输出最终 JSON。

## 4. 主要结果

| condition | mean_b3_precision | mean_b3_recall | mean_b3_f1 | median_b3_f1 | mean_pairwise_f1 | mean_cluster_count_bias | mean_input_tokens | mean_output_tokens | parse_failure_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal | 0.7050 | 0.8602 | 0.7150 | 0.7320 | 0.6805 | -2.3333 | 32239.0000 | 16478.3333 | 0 |
| task_definition | 0.7889 | 0.8485 | 0.7816 | 0.7382 | 0.7321 | 0.3333 | 32298.0000 | 20433.6667 | 0 |
| evidence_guidance | 0.8580 | 0.7776 | 0.7777 | 0.7777 | 0.7246 | 0.0000 | 29783.5000 | 24716.5000 | 1 |
| error_aware | 0.7064 | 1.0000 | 0.8279 | 0.8279 | 0.8274 | -3.0000 | 35418.0000 | 12135.0000 | 2 |
| analyze_then_cluster | 0.5850 | 0.9843 | 0.7338 | 0.7338 | 0.6837 | 0.0000 | 37566.0000 | 16008.0000 | 2 |

## 5. 主要发现

- 平均 B³ F1 最高的是 `error_aware`，mean B³ F1=0.8279；没有成功解析的条件不参与 best condition 判断。
- `task_definition` 相比 `minimal` 的 mean B³ F1 变化为 0.0666。
- `evidence_guidance` 相比 `task_definition` 的 mean B³ F1 变化为 -0.0039。
- `error_aware` 相比 `task_definition` 的 mean cluster count bias 变化为 -3.3333。
- `analyze_then_cluster` 相比 `evidence_guidance` 的 mean output tokens 变化为 -8708.5000。

## 6. 典型案例

| author_id | difficulty_group | condition | minimal_b3_f1 | condition_b3_f1 | delta_b3_f1 | delta_cluster_count |
| --- | --- | --- | --- | --- | --- | --- |
| jianping_jia | hard | task_definition | 0.5734 | 0.7346 | 0.1611 | 5 |
| jianping_jia | hard | analyze_then_cluster | 0.5734 | 0.7338 | 0.1604 | 6 |
| yingchun_xu | easy | error_aware | 0.7320 | 0.8279 | 0.0960 | -5 |

## 7. 失败案例

本轮共有 5 个失败的 author-condition。这里的失败没有被补 singleton，也没有手工修正；它们不参与 B³ / Pairwise 均值计算。

| author_id | difficulty_group | condition | paper_count | failure_type | error |
| --- | --- | --- | --- | --- | --- |
| yingchun_xu | easy | analyze_then_cluster | 355 | full_coverage_validation_failed | missing 352 paper IDs |
| yingmin_wang | medium | error_aware | 349 | api_incomplete_read | IncompleteRead(2 bytes read) |
| yingmin_wang | medium | analyze_then_cluster | 349 | api_incomplete_read | IncompleteRead(1 bytes read) |
| jianping_jia | hard | evidence_guidance | 503 | full_coverage_validation_failed | missing 1 paper IDs |
| jianping_jia | hard | error_aware | 503 | full_coverage_validation_failed | missing 3 paper IDs |

失败类型说明：`full_coverage_validation_failed` 表示模型返回了 JSON，但没有覆盖所有输入论文 ID；`api_incomplete_read` 表示 DeepSeek HTTP 响应读取不完整，没有可解析的完整 raw response。

## 8. 结论

这是 3 作者小规模实验。本轮 10 个 author-condition 通过解析和 full-coverage 校验，5 个失败；失败原因包括：missing 352 paper IDs: 1; IncompleteRead(2 bytes read): 1; IncompleteRead(1 bytes read): 1; missing 1 paper IDs: 1; missing 3 paper IDs: 1。在成功样本上，`error_aware` 的平均 B³ F1 最高，但该均值需要结合 parse failure count 解读：部分条件只在少数作者上成功，不能直接视为总体最优。当前结果说明 prompt 内容会改变聚类倾向，但还不能证明更详细 prompt 稳定更好；后续需要更稳的请求策略或分批处理来降低输出完整性风险。
