请在当前项目中完成一个“字段组合有效性解释”的 pilot 实验分析。目标是基于此前已经完成的、包含 3 个作者 block 的不同字段组合 LLM clustering 实验结果，系统分析并解释为什么 `title + coauthor + organization`（下文简称 `T+O+C`）表现最好。

请先仔细检查项目目录结构、已有的实验脚本、结果文件格式、ground truth 格式、论文原始 metadata 格式，以及此前字段组合实验的保存位置。不要假设文件名或 JSON schema；优先复用现有代码、已有结果和已有评估逻辑。

---

# 总目标

完成以下三个实验，并生成所有中间结果、图表和一份中文 Markdown 实验报告。

实验对象仅限此前已有字段组合实验结果的 3 个作者 block。不要擅自扩展到完整 80 个作者数据集。

如果某些数据缺失、字段为空、预测结果不完整，必须：

1. 在日志和报告中明确说明；
2. 尽量采用稳健的降级处理；
3. 不要伪造结果；
4. 不要在证据不足时声称 `T+O+C` 一定最优。

所有新生成的 JSON 实验数据必须保存到：

```text
artifacts/results/field_selection/analysis/
```

报告可以保存到：

```text
artifacts/results/field_selection/analysis/
```

图表可以保存到：

```text
artifacts/figures/field_selection/analysis/
```

建议目录结构：

```text
artifacts/results/field_selection/analysis/
├── field_combination_summary.json
├── pairwise_similarity_records.json
├── field_contribution_title.json
├── field_contribution_coauthor.json
├── field_contribution_organization.json
└── pilot_experiment_report_zh.md

artifacts/figures/field_selection/analysis/
├── combination_metrics.png
├── similarity_distribution_title.png
├── similarity_distribution_abstract.png
├── similarity_distribution_organization.png
├── coauthor_overlap_distribution.png
├── field_error_repair_summary.png
└── ...
```

文件名可以略作调整，但必须保持语义清晰，并全部位于 `artifacts/results/field_selection/analysis/` 下。

---

# 0. 先做数据与代码审计

请先执行以下工作：

1. 找到此前小范围 3 个作者的字段组合实验结果；
2. 确认每个实验结果对应的字段组合，例如：

   * `T`
   * `C`
   * `O`
   * `T+C`
   * `T+O`
   * `C+O`
   * `T+O+C`
   * 如已有，也包含 `A`、`T+A+O`、`T+A+C`、`T+O+C+A` 等；
3. 找到每个作者 block 的 ground truth clustering；
4. 找到对应论文的原始 metadata，至少尽量获得：

   * paper id
   * title
   * abstract
   * coauthor
   * organization / affiliation
5. 理解现有 prediction clustering 的输出格式；
6. 复用现有评估代码，确认 Pairwise Precision / Recall / F1 与 B³ Precision / Recall / F1 的计算逻辑。

请在脚本开头或 README 注释中写清楚你找到的路径、使用的文件和关键 schema。

如果项目中已有相似的评测或绘图脚本，优先扩展而不是重新实现一套互不兼容的逻辑。

---

# 实验 1：字段组合效果验证

目标：基于已有实验结果，验证在这 3 个作者 block 上，`T+O+C` 是否在主要指标上表现最佳或最稳定。

## 1.1 汇总指标

针对每个作者 block、每种字段组合，整理至少以下指标：

```text
author_block
field_combination
pairwise_precision
pairwise_recall
pairwise_f1
b3_precision
b3_recall
b3_f1
predicted_cluster_count
ground_truth_cluster_count
```

如果已有其他指标，例如 exact cluster match、ARI、NMI，也可以一并保存，但不要取代 Pairwise F1 和 B³ F1。

将结构化结果保存为：

```text
artifacts/results/field_selection/analysis/field_combination_summary.json
```

同时建议生成 CSV，便于人工查看。

## 1.2 聚合方式

请同时计算：

1. 每个作者 block 的单独结果；
2. 3 个 block 的 macro-average；
3. 每种字段组合的 median；
4. 如有必要，给出 min / max，避免平均值掩盖极端失败案例。

注意：只有 3 个 block，报告中不要进行夸张的统计显著性宣称。应表述为 pilot observations / pilot findings。

## 1.3 图表

请生成至少两张图：

1. 字段组合 × Pairwise F1 的 grouped bar chart；
2. 字段组合 × B³ F1 的 grouped bar chart。

要求：

* 横轴：字段组合；
* 图中清晰区分 3 个作者 block；
* 额外显示 macro-average；
* `T+O+C` 用视觉方式突出，但不要通过夸张颜色误导；
* 图片保存到 `artifacts/results/field_selection/analysis/plots/`。

如果已有 abstract 相关组合，也请额外生成一张只比较下列组合的图：

```text
T+O+C
T+A+O
T+A+C
T+O+C+A
A
```

---

# 实验 2：字段相似度与区分能力分析

目标：从论文 pair 的角度，解释不同字段对于区分“同一真实作者”和“不同真实作者”的能力。

核心原则：

* 只使用 ground truth 来标注 pair 是否同人；
* 该实验不是 LLM clustering 结果；
* 该实验分析字段本身提供的可分性；
* 所有 pair 应在每个 author block 内生成，不跨 block 配对。

## 2.1 Pair 构造

对于每个作者 block 中的 (n) 篇论文，构造全部无序论文对：

```text
(paper_i, paper_j), where i < j
```

根据 ground truth cluster，生成：

```text
gt_same = true / false
```

## 2.2 相似度特征

请为每个 pair 计算以下字段级特征。缺失字段必须明确标记，而不是当作正常的零相似度混在一起。

### Title

使用本地 embedding 模型编码 title，并计算 cosine similarity。

优先选择项目环境中可直接使用、下载成本合理的 sentence-transformers 模型。若没有现成模型，可优先尝试一个轻量、通用、适合英语学术文本的本地模型，例如：

```text
sentence-transformers/all-MiniLM-L6-v2
```

但必须把实际使用的模型名称、版本、加载方式、是否联网下载写入报告。

字段：

```text
title_available
title_cosine_similarity
```

### Abstract

如果 abstract 可用，也做相同分析：

```text
abstract_available
abstract_cosine_similarity
```

abstract 可能很长。请采用一致、可复现的截断策略，例如保留前 N 个字符或前 N 个 tokens，并在报告中说明。

### Organization / Affiliation

请尽可能进行轻量标准化，例如：

* 转小写；
* 去除首尾空格；
* 统一连续空白；
* 处理简单缩写；
* 保留原始文本和标准化文本；
* 不要构造复杂且不可解释的人工映射规则。

至少计算：

```text
organization_available
organization_exact_match
organization_text_similarity
```

其中 `organization_text_similarity` 可以使用 embedding cosine similarity，或使用可解释的字符串相似度；如用 embedding，请与 title/abstract 模型保持一致或在报告中说明差异。

### Coauthor

将每篇论文的 coauthor 表示为作者名集合。请尽量：

* 排除目标歧义姓名本人；
* 统一大小写、空格和常见标点；
* 保留清洗前后数量；
* 避免把空集合误判为完全匹配。

至少计算：

```text
coauthor_available_i
coauthor_available_j
shared_coauthor_count
coauthor_jaccard_similarity
coauthor_overlap_indicator
```

其中：

```text
coauthor_jaccard_similarity =
|C_i ∩ C_j| / |C_i ∪ C_j|
```

只有在两个集合均非空时才计算正常 Jaccard；否则记录缺失或不可用状态。

## 2.3 输出数据

将所有 pair 的字段特征保存为：

```text
artifacts/results/field_selection/analysis/pairwise_similarity_records.json
```

每条记录建议至少包含：

```json
{
  "author_block": "...",
  "paper_i": "...",
  "paper_j": "...",
  "gt_same": true,
  "title_available": true,
  "title_cosine_similarity": 0.0,
  "abstract_available": true,
  "abstract_cosine_similarity": 0.0,
  "organization_available": true,
  "organization_exact_match": false,
  "organization_text_similarity": 0.0,
  "coauthor_available_i": true,
  "coauthor_available_j": true,
  "shared_coauthor_count": 0,
  "coauthor_jaccard_similarity": 0.0,
  "coauthor_overlap_indicator": false
}
```

实际 schema 可以补充字段，但请保持清晰稳定。

## 2.4 分析与图表

对每种字段，分别比较：

```text
same-author pairs
different-author pairs
```

至少生成：

1. title cosine similarity 的分布图；
2. abstract cosine similarity 的分布图（如果 abstract 覆盖率足够）；
3. organization text similarity 的分布图；
4. coauthor Jaccard similarity 或 shared coauthor count 的分布图。

推荐使用：

* violin plot + strip/jitter；
* 或 histogram / KDE；
* 但需要让 same-author 和 different-author 两组可直接比较。

请同时计算描述性统计：

```text
field
available_pair_count
same_pair_count
different_pair_count
same_mean
different_mean
same_median
different_median
separation_gap = same_mean - different_mean
```

如样本规模允许，也可以计算每个单字段特征用于预测 `gt_same` 的 ROC-AUC；但只有在正负样本数量足够、实现可靠时才做。若不适合，不要硬做。

重要要求：

* 不要把字段相似度高直接等同于“该字段一定导致 LLM 判断正确”；
* 只应说该字段具有更强或更弱的 pair-level discriminative signal；
* 对 coauthor 和 organization，应同时报告 coverage / missingness，因为高区分性但低覆盖率也是重要结论。

---

# 实验 3：T+O+C 中每个字段修复了哪些错误

目标：以完整组合 `T+O+C` 为中心，分别分析 title、coauthor、organization 的边际贡献，以及各自修复或引入的 pairwise 错误。

不要把这个实验理解为让 LLM 逐 pair 分类。你需要比较不同整体 clustering 结果导出的 pairwise relation。

## 3.1 三组 leave-one-field-out 对照

请分别比较：

```text
Full: T+O+C
Without title: O+C
Without coauthor: T+O
Without organization: T+C
```

对应含义：

```text
O+C -> T+O+C：title 的边际贡献
T+O -> T+O+C：coauthor 的边际贡献
T+C -> T+O+C：organization 的边际贡献
```

若某个必要实验组合在已有结果中不存在：

1. 先检查是否可以复用已有 pipeline 重新运行；
2. 若可以低成本运行，则补跑该组合；
3. 若不能可靠补跑，则在报告中明确标记该字段贡献分析不可完成；
4. 不要用不兼容或不同配置的结果硬比较。

重新运行时必须保持与原实验一致的：

* LLM model；
* prompt template；
* temperature；
* max tokens；
* 输出格式；
* post-processing；
* 论文排序策略；
* 其他实验配置。

## 3.2 Clustering 转换为 pairwise labels

对于每个 author block、每一组 full/ablated 结果，将 clustering 转成每个论文 pair 的：

```text
gt_same
pred_without_field_same
pred_full_same
```

对于每一个字段，找出 prediction 发生变化的 pair：

```text
pred_without_field_same != pred_full_same
```

然后把所有 pair 分类为以下四类。

### Fixed false split

```text
gt_same = true
pred_without_field_same = false
pred_full_same = true
```

解释：加入该字段后，原来被错误拆开的同一作者论文被正确合并。

### Fixed false merge

```text
gt_same = false
pred_without_field_same = true
pred_full_same = false
```

解释：加入该字段后，原来被错误合并的不同作者论文被正确拆开。

### Introduced false split

```text
gt_same = true
pred_without_field_same = true
pred_full_same = false
```

解释：加入该字段后，原本正确的同作者关系被错误拆开。

### Introduced false merge

```text
gt_same = false
pred_without_field_same = false
pred_full_same = true
```

解释：加入该字段后，原本正确的不同作者关系被错误合并。

另外统计 unchanged pairs。

注意：由于 clustering 是整体输出，pairwise relation 的改变可能来自 cluster 结构的间接变化。报告中请使用严谨表述，例如：

```text
“After adding coauthor information, the resulting clustering corrected the pairwise relation...”
```

不要写成：

```text
“The model directly classified this pair correctly because of coauthor.”
```

## 3.3 每个字段的输出 JSON

分别生成：

```text
artifacts/results/field_selection/analysis/field_contribution_title.json
artifacts/results/field_selection/analysis/field_contribution_coauthor.json
artifacts/results/field_selection/analysis/field_contribution_organization.json
```

每条记录建议包含：

```json
{
  "author_block": "...",
  "field_added": "coauthor",
  "paper_i": "...",
  "paper_j": "...",
  "gt_same": true,
  "pred_without_field_same": false,
  "pred_full_same": true,
  "change_type": "fixed_false_split",
  "title_cosine_similarity": 0.0,
  "abstract_cosine_similarity": 0.0,
  "organization_match": true,
  "organization_similarity": 0.0,
  "shared_coauthor_count": 2,
  "coauthor_jaccard_similarity": 0.4
}
```

如果字段不存在，则使用明确的 `null` 或 `available=false`，不要伪造数值。

## 3.4 汇总表

生成一个字段错误修复汇总表，至少包括：

```text
field_added
fixed_false_splits
fixed_false_merges
introduced_false_splits
introduced_false_merges
unchanged_pairs
net_pairwise_gain
```

其中：

```text
net_pairwise_gain =
fixed_false_splits
+ fixed_false_merges
- introduced_false_splits
- introduced_false_merges
```

请同时按：

1. 全部 3 个 block 汇总；
2. 每个作者 block 单独统计。

生成一张图，例如 grouped / stacked bar chart，展示三个字段分别修复和引入了多少错误：

```text
artifacts/results/field_selection/analysis/plots/field_error_repair_summary.png
```

## 3.5 代表性案例分析

对于每个字段，自动筛选并人工可读地输出若干最具代表性的变化 pair：

* 最多 3 个 fixed false split；
* 最多 3 个 fixed false merge；
* 如存在，也最多 2 个 introduced errors。

每个案例应提供：

```text
author block
paper ids
ground truth relation
without-field prediction
full prediction
两个论文的 title
coauthor overlap 信息
organization 信息
必要时的 abstract 截断摘要
```

请保存一个可读 Markdown 或 JSON 文件，例如：

```text
artifacts/results/field_selection/analysis/field_contribution_cases_zh.md
```

不要在报告中直接堆大量原始长 abstract；仅保留必要摘要或截断文本。

---

# 报告要求

完成全部实验后，生成：

```text
artifacts/results/field_selection/analysis/pilot_experiment_report_zh.md
```

报告必须是中文、Markdown 格式、可直接阅读，不要只是日志汇总。

建议使用以下结构：

```markdown
# Pilot 实验报告：字段组合有效性与互补性分析

## 1. 实验目标

## 2. 数据与实验设置
- 三个作者 block 的基本信息
- 使用的字段组合
- 复用的历史实验结果
- LLM 配置
- embedding 模型与版本
- 缺失数据与限制

## 3. 实验一：字段组合效果比较
- 表格
- 图表
- 对 Pairwise F1 和 B³ F1 的分析
- 是否支持 T+O+C 在这三个 block 上最优或最稳定
- 不允许超出三作者 pilot 的结论范围

## 4. 实验二：字段的 pair-level 区分能力
- title、abstract、coauthor、organization 的覆盖率
- same-author / different-author 相似度分布
- separation gap 等描述性统计
- 解释三类字段分别提供的语义、关系、机构证据
- 明确说明 abstract 是否信息丰富但身份特异性较弱

## 5. 实验三：字段错误修复机制
- leave-one-field-out 设置
- title、coauthor、organization 分别修复的 false split / false merge
- 引入错误的情况及其原因
- 代表性案例表
- 说明 clustering 的 pairwise 改变可能具有间接性

## 6. 综合结论
- 严格基于结果总结 T+O+C 是否最好
- 若 T+O+C 并非所有 block 都第一，应诚实说明“总体最好 / 最稳定 / 在若干 block 最好”等
- 解释其优势是否来自字段互补，而不是单纯字段数量更多
- 提出后续扩展到 80 个作者 block 的实验建议

## 7. 局限性
- 样本仅有 3 个作者 block
- LLM clustering 可能受 prompt / 顺序 / 输出不稳定影响
- metadata 缺失和组织名称不规范
- embedding 相似度只反映字段可分性，不是 LLM 因果解释
```

报告中的结论必须符合真实结果。

特别要求：

* 不要在结果不足时写“证明”；
* 应使用“在该 3-author pilot 中支持”“观察到”“初步表明”等严谨措辞；
* 如果 `T+O+C` 不是最佳组合，明确报告真实结果，并分析可能原因；
* 如果指标、数据或字段覆盖率不足，请明确指出；
* 每张图、每个表都要在报告中解释，不要只插入文件链接；
* 报告中给出所有生成文件的相对路径。

---

# 工程要求

1. 尽量新增独立、可复现的 Python 脚本，例如：

```text
scripts/run_pilot_field_analysis.py
scripts/plot_pilot_results.py
```

也可以根据现有项目结构调整。

2. 所有随机性必须固定 seed。

3. 代码中加入必要的日志、异常处理和 schema 检查。

4. 不要覆盖或破坏已有实验结果。

5. 对缺失字段、空字符串、无 coauthor、无 affiliation 的论文做显式处理。

6. 若需要安装依赖，请优先使用已有环境；若新增依赖，最小化新增内容，并把依赖写入 requirements 或报告。

7. 最后在终端输出一段简洁摘要，包含：

   * 找到的输入数据路径；
   * 成功完成的实验；
   * 未完成部分及原因；
   * 最终报告路径；
   * `T+O+C` 在三作者 pilot 中的主要结果。

请直接开始检查仓库并执行，不要只给出计划或伪代码。
