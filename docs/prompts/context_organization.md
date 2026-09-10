请在当前姓名消歧项目中完成一个小规模实验，研究：

> 在输入字段固定为 title + organization + coauthor（T+O+C）的情况下，不同的信息组织方式是否会影响 LLM 的作者姓名消歧效果。

## 1. 目录要求

代码放在：

```text
src/context_orgnization/
```

实验结果放在：

```text
artifacts/results/context_organization/
```

简洁中文实验报告放在：

```text
docs/reports/context_organization/
```

交互图表放在：

```text
artifacts/figures/context_organization/
```

请保留目录名中的 `orgnization`，不要修改拼写。

---

## 2. 优先复用现有代码

开始前检查项目中已有的：

* validation 数据和 ground truth；
* 原 T+O+C prompt；
* LLM 调用代码；
* LLM 输出解析代码；
* Pairwise 和 B³ 评测代码；
* 之前 T+O+C 的作者级评测结果。

如果有的话，可以直接复用而不重新实现已有流程。

模型、temperature、任务 prompt、输出格式和评测方式都必须与原 T+O+C 实验保持一致。

六种实验之间只能改变论文信息的排列或分组方式。

---

## 3. 实验作者

选择 6 个作者作为 pilot：

* 2 个 easy：原 T+O+C 的 B³ F1 ≥ 0.90；
* 2 个 medium：0.80 ≤ B³ F1 < 0.90；
* 2 个 hard：B³ F1 < 0.80。

使用固定随机种子选取作者，并把名单保存到：

```text
artifacts/results/context_organization/selected_authors.json
```

如果已有数据不足 18 个，则使用所有可用作者。

---

## 4. 六种信息组织方式

> 在输入字段固定为 title + organization + coauthor（T+O+C）的情况下，不同的信息组织方式是否会影响 LLM 的作者姓名消歧效果。

所有方法都使用完全相同的 T+O+C 内容和完全相同的 prompt，只改变论文顺序或分组。

### 4.1 original_flat

保持原始论文顺序，作为 baseline。

```text
Paper ID: P1
Title: ...
Organization: ...
Coauthors: ...

Paper ID: P2
...
```

### 4.2 random_order

随机打乱论文顺序。

每个作者运行 3 个固定随机种子的排列，用于观察顺序稳定性。

### 4.3 chronological_order

按照 publication year 从早到晚排列。

* 缺失年份的论文放在最后；
* 同一年按照 paper ID 排序；
* 不额外增加新的字段。

### 4.4 organization_grouped

按照标准化后的 organization 将论文放在一起。

可以使用中性标题：

```text
Group 1
...
Group 2
...
```

不告诉 LLM 这些 group 对应同一个作者。

缺失 organization 的论文放在单独一组。

### 4.5 coauthor_network

建立论文之间的合作者关系：

* 两篇论文共享至少一个非目标作者的 coauthor，就建立连接；
* 按连通分量组织论文；
* 不共享合作者的论文单独成组；
* 必须排除当前待消歧的目标姓名。

只使用中性标题 `Group 1`、`Group 2`，不能暗示真实聚类。

### 4.6 multifield_similarity

根据三个字段计算论文相似度：

* title：TF-IDF cosine similarity；
* organization：token Jaccard similarity；
* coauthor：coauthor set Jaccard similarity。

综合相似度取三个可用字段相似度的平均值。

使用确定性的 greedy 排序：

1. 从 paper ID 最小的论文开始；
2. 找到与当前论文最相似的未排列论文；
3. 将其放在后面；
4. 重复直到所有论文排列完成。

这个条件只改变顺序，不显示相似度，也不告诉 LLM 哪些论文相似。

---

## 5. 必须保证的信息一致性

对同一个作者，六种条件必须包含：

* 完全相同的 paper IDs；
* 完全相同的 title；
* 完全相同的 organization；
* 完全相同的 coauthor。

只能改变：

* 论文顺序；
* 中性的分组结构。

请在调用 LLM 前自动检查。若字段内容不一致，则停止该作者的实验并报错。

---

## 6. 实验执行

建议提供一个主命令：

```bash
python -m src.context_orgnization.run_experiment
```

执行流程：

1. 加载数据；
2. 选择 18 个作者；
3. 为六种条件生成 context；
4. 调用 LLM；
5. 保存原始响应；
6. 解析聚类结果；
7. 计算评测指标；
8. 生成汇总结果；
9. 绘图；
10. 生成实验报告。

必须支持断点续跑。已经成功得到结果的调用不要重复执行。

每个作者、条件和随机重复的原始结果都要保存。

---

## 7. 保存的主要结果

生成：

```text
artifacts/results/context_organization/author_condition_metrics.csv
artifacts/results/context_organization/condition_summary.csv
artifacts/results/context_organization/random_order_stability.csv
```

### author_condition_metrics.csv

每行对应一个：

```text
author × condition × repetition
```

至少包含：

* author_id；
* difficulty_group；
* condition；
* repetition；
* paper_count；
* Pairwise Precision / Recall / F1；
* B³ Precision / Recall / F1；
* predicted cluster count；
* true cluster count；
* input tokens；
* output tokens；
* latency；
* parse success。

### condition_summary.csv

每种条件汇总：

* mean B³ F1；
* median B³ F1；
* mean Pairwise F1；
* mean predicted cluster count；
* mean input tokens；
* parse failure count。

random_order 使用三个排列的作者级平均结果进行汇总。

### random_order_stability.csv

每个作者记录：

* 三次 random B³ F1；
* mean；
* standard deviation；
* min；
* max；
* range。

---

## 8. 结果比较

以 `original_flat` 为 baseline。

对其他条件计算每个作者的：

```text
delta_b3_f1 = 当前条件 B³ F1 - original_flat B³ F1
```

每个条件汇总：

* mean delta B³ F1；
* median delta B³ F1；
* 改善作者数量；
* 下降作者数量；
* 无变化作者数量。

不需要复杂机器学习。

可选地进行 paired Wilcoxon signed-rank test；如果实现方便则加入，否则不强制。

重点分析：

* 哪一种组织方式平均效果最好；
* 是否有方法主要提高 recall 或 precision；
* organization grouping 是否导致过度拆分；
* coauthor network 是否导致过度合并；
* random order 是否表现出明显顺序敏感性；
* easy、medium、hard 作者是否有不同表现。

---

## 9. 交互图表

使用 Plotly 生成以下 4 张 HTML 图，放在：

```text
artifacts/figures/context_organization/
```

### condition_performance.html

展示六种条件的 B³ F1 分布：

* box plot；
* 叠加作者点；
* hover 显示 author_id、difficulty 和具体 F1。

### paired_comparison.html

比较各条件与 baseline：

* x 轴：original_flat B³ F1；
* y 轴：当前条件 B³ F1；
* 添加 y=x 参考线；
* hover 显示 author_id 和 delta。

### author_delta_heatmap.html

* 行：author；
* 列：五个非 baseline 条件；
* 值：delta B³ F1。

### random_order_stability.html

展示每个作者三次 random order 的 B³ F1 及其波动范围。

---

## 10. 简洁实验报告

生成：

```text
docs/reports/context_organization/context_orgnization_experiment_report_zh.md
```

报告保持简洁，建议只包含以下部分。

### 1. 实验目的

说明固定 T+O+C，只改变信息组织方式。

### 2. 实验设置

说明：

* 作者数量；
* easy / medium / hard 数量；
* 模型；
* 六种条件；
* random order 重复 3 次；
* 使用的主要指标。

### 3. 主要结果表

表格至少包含：

* condition；
* mean B³ F1；
* median B³ F1；
* mean Pairwise F1；
* mean delta B³ F1；
* improved / degraded author count。

### 4. 主要发现

根据实际结果回答：

* 信息组织是否影响 LLM 消歧效果；
* 哪种方法平均最好；
* 哪种方法最稳定；
* 哪种方法容易造成 oversplitting 或 overmerging；
* 不同难度作者是否有不同表现。

### 5. 典型案例

选择 2–4 个作者简要分析：

* 明显改善；
* 明显下降；
* random order 波动较大；
* 某种分组产生错误引导。

### 6. 结论

用简洁语言总结，不要预设一定存在显著提升。

如果六种方法差异很小，也要如实说明：

> 在当前数据、模型和 T+O+C 字段下，信息组织方式的影响有限，主要性能瓶颈可能来自输入证据本身。

---

## 11. 代码要求

* 不硬编码 API key；
* 不覆盖以前的实验结果；
* 使用固定随机种子；
* 保存所有原始 LLM 响应；
* 支持断点续跑；
* 使用项目相对路径；
* 关键函数添加简短注释；
* 不使用 ground truth 构造排序或分组；
* ground truth 只能用于选取难度层和最终评测；
* 不伪造任何实验结果。

完成后，在终端输出：

* 使用的数据路径；
* 选择的作者数量；
* 成功和失败调用数量；
* 六种条件的 mean B³ F1；
* 最优条件；
* 报告路径；
* 图表目录；
* 结果目录。
