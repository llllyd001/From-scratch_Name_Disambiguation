# 任务：构建 T+O+C 的 LLM 高可靠作者筛选器

## 目标

当前固定：

```text
字段组合：title + organization + coauthor（T+O+C）
LLM、prompt、已有 clustering 结果：保持不变
```

研究问题：

> 能否在运行 T+O+C LLM clustering 之前，仅根据一个作者名 block 的原始论文数据，筛选出预计能够被高质量消歧的作者？

定义：

$
\text{Reliable} = \mathbb{1}(B^3F1 \ge 0.85)
$

最终需要得到一个筛选规则：

```text
原始 author-name block
        ↓
预测 P(B³ F1 ≥ 0.85)
        ↓
概率高于阈值：接受，运行 T+O+C LLM clustering
概率低于阈值：拒绝，标记为 high-risk，留给后续补救策略
```

本任务只研究“如何筛选 T+O+C 预计能成功的作者”，不比较字段组合，不重新运行 LLM，不实现后续补救方法。

---

## 已有结果：必须复用

项目已经有约 80 个作者的：

* T+O+C LLM clustering 输出；
* ground truth；
* Pairwise 与 B³ 指标；
* 部分分析 CSV、Markdown 报告与 Plotly HTML 图。

请先检查并复用已有文件、已有指标计算代码和已有 HTML 图。
**不要重新调用 LLM，不要重复生成已经存在的同类图表。**

---

## 数据划分

固定划分：

```text
训练集：64 个作者
测试集：16 个作者
random_state = 42
```

要求：

* 必须按 `name` 划分；
* 训练/测试作者名单保存到：

```text
artifacts/results/router/train_names.json
artifacts/results/router/test_names.json
```

* 训练集可以使用真实 F1、ground truth、预测 cluster 等所有后验信息做分析和监督学习；
* 测试集只能在模型、特征、阈值全部冻结后评估一次；
* 不得用测试集修改模型、特征或 threshold。

---

# 实验 1：训练集后验分析与特征表构建

## 目标

先用训练集 64 个作者的已有 T+O+C 结果回答：

> 高 F1 作者和低 F1 作者有什么差异？哪些原始数据特征值得用于提前筛选？

## A. 训练集后验分析

对每位训练作者汇总：

```text
name
paper_count

Pairwise precision / recall / F1
B³ precision / recall / F1

true_cluster_count
predicted_cluster_count
predicted_to_true_cluster_ratio

title_coverage
organization_coverage
coauthor_coverage
```

定义：

```python
reliable_085 = int(b3_f1 >= 0.85)
hard_080 = int(b3_f1 < 0.80)
catastrophic_050 = int(b3_f1 < 0.50)
```

重点分析：

1. B³ F1 的均值、中位数、分布；
2. reliable / hard / catastrophic 的数量；
3. 低分案例主要是：

   * Pairwise Precision 低：可能 over-merging；
   * Pairwise Recall 低：可能 over-splitting；
   * 两者都低：整体困难；
4. paper count、coverage、cluster ratio、标题符号密度等是否与 F1 有明显关系。

已有 HTML 图可以直接在报告中引用；只有缺少的必要图才新增。

## B. 构建可部署特征表

构建：

```text
artifacts/results/router/pre_llm_features_toc.csv
```

每行一个作者。所有特征必须在 LLM clustering 前，从原始 title、organization、coauthor 数据中计算。

只提取以下特征，保持简单。

### 1. 基础特征

```text
paper_count
title_coverage
organization_coverage
coauthor_coverage
overall_coverage
estimated_prompt_tokens
tokens_per_paper
```

### 2. 单字段相似度结构

```text
title: TF-IDF cosine similarity
organization: normalized token Jaccard similarity
coauthor: Jaccard similarity
```

对每个字段统计：

```text
pairwise_similarity_mean
pairwise_similarity_std
nearest_neighbor_similarity_mean
nearest_neighbor_margin_mean
```

其中：

$\text{nearest-neighbor margin} =  \text{最高相似度} - \text{第二高相似度}$


### 3. 三字段邻居一致性

对每篇论文，从 title、organization、coauthor 各取 Top-3 相似邻居，计算邻居集合之间的 Jaccard overlap：

```text
title_organization_neighbor_agreement
title_coauthor_neighbor_agreement
organization_coauthor_neighbor_agreement
mean_cross_view_agreement
```

### 4. 标题符号特征

```text
title_digit_ratio
title_non_alphabetic_ratio
title_uppercase_token_ratio
title_formula_like_token_ratio
```

这部分仅用于探索“符号密集标题是否与低 F1 或 oversplitting 有关联”，不得将具体领域硬编码为规则。

---

# 实验 2：训练集监督学习筛选器

## 目标

使用训练集的真实 B³ F1 作为监督标签，训练一个模型预测：

$
P(B^3F1 \ge 0.85)
$

即：预测一个作者是否值得直接交给 T+O+C LLM。

## 方法

在训练集 64 位作者上进行 5-fold cross-validation。

比较四种筛选机制：

| 模型                       | 输入特征                        | 用途         |
| ------------------------ | --------------------------- | ---------- |
| Random ranking           | 随机分数                        | 下界         |
| Paper-count-only         | paper_count，可选 token 数      | 判断规模是否足够   |
| Coverage-only            | title/org/coauthor coverage | 判断数据缺失是否足够 |
| Full Logistic Regression | 所有 pre-LLM 特征               | 主模型        |

Logistic Regression 必须包含：

```text
缺失值 imputation
特征标准化
class_weight = balanced
固定 random_state
```

不要进行复杂超参数搜索。Random Forest 可以作为可选对照，但不是必须。

保存所有模型的训练集 out-of-fold 预测：

```text
artifacts/results/router/cv_predictions_toc.csv
```

至少包含：

```text
name
actual_b3_f1
reliable_085
predicted_probability
model_name
```

## 重点评估

不要主要看普通 accuracy。重点看：

$
\text{Accepted Precision} = 
P(B^3F1 \ge 0.85 \mid \text{accepted})
$

$
\text{Coverage} = \frac{\text{accepted authors}}{\text{all authors}}$

$
\text{Hard-block Recall} = P(\text{rejected} \mid B^3F1 < 0.80)
$

并按预测概率从高到低，在 coverage 为：

```text
20%, 40%, 60%, 80%, 100%
```

时比较各模型的：

```text
accepted mean B³ F1
accepted precision
hard-block recall
```

核心要回答：

> Full router 是否比随机、只看 paper count、只看 coverage 更能挑出真实高质量的 T+O+C 作者？

生成必要的新图：

```text
artifacts/results/router/interactive/risk_coverage_curve.html
artifacts/results/router/interactive/cv_probability_vs_actual_b3_f1.html
artifacts/results/router/interactive/feature_importance.html
```

---

# 实验 3：确定筛选规则并冻结测试集验证

## A. 在训练集选择 threshold

只使用训练集 out-of-fold 预测，扫描：

```text
0.50, 0.55, ..., 0.95
```

每个 threshold 计算：

```text
accepted_precision
coverage
hard_block_recall
accepted_mean_b3_f1
```

选择原则：

```text
优先选择 accepted_precision >= 0.90；
若多个阈值满足，选择 coverage 最大者。
```

若无法达到 0.90，诚实报告最佳 precision–coverage trade-off。

保存：

```text
artifacts/results/router/router_config.json
```

内容：

```text
final_model
feature_columns
threshold
training_oof_metrics
random_seed
```

生成：

```text
artifacts/results/router/interactive/threshold_tradeoff.html
```

## B. 冻结后测试集验证

完成模型和阈值选择后：

1. 使用全部 64 个训练作者拟合最终模型；
2. 对 16 个测试作者，仅使用原始 pre-LLM 特征；
3. 输出：

```text
name
predicted_reliability_probability
accepted_or_rejected
```

4. 用已有测试集真实 B³ F1 做一次最终评估：

```text
test accepted_precision
test coverage
test hard_block_recall
test accepted_mean_b3_f1
```

保存：

```text
artifacts/results/router/test_predictions_toc.csv
artifacts/results/router/router_model_toc.joblib
```

---

# 输出文件：保持最少

只生成：

```text
artifacts/results/router/
├── train_names.json
├── test_names.json
├── pre_llm_features_toc.csv
├── cv_predictions_toc.csv
├── test_predictions_toc.csv
├── router_model_toc.joblib
├── router_config.json
├── report_zh.md
└── interactive/
    ├── index.html
    ├── risk_coverage_curve.html
    ├── cv_probability_vs_actual_b3_f1.html
    ├── threshold_tradeoff.html
    ├── feature_importance.html
    └── [仅新增缺少的必要图]
```

新增代码尽量只放在：

```text
src/router/toc_reliability_router.py
```

如确实需要，可额外加入：

```text
src/router/utils.py
```

---

# 中文报告

生成：

```text
artifacts/results/router/report_zh.md
```

报告只需包含以下六部分：

1. **研究问题**
   固定 T+O+C，提前筛选预计 B³ F1 ≥ 0.85 的作者。

2. **数据与已有实验结果**
   说明复用了哪些已有结果，训练/测试划分，以及训练集 F1 分布。

3. **训练集后验分析**
   说明高分/低分作者的差异、主要失败模式，以及哪些观察只是相关趋势。

4. **筛选器设计**
   说明训练时使用真实 F1 作为标签；部署时只使用原始数据特征。

5. **筛选效果与最终规则**
   对比四个模型，解释 risk–coverage 曲线，给出最终 threshold 和接受规则。

6. **冻结测试集结果与局限**
   清楚区分训练和测试；说明样本规模小、当前仅验证 T+O+C，下一阶段再研究被拒绝作者如何补救。

完成后，请简要列出：

* 复用了哪些已有 T+O+C 结果；
* 实际训练/测试作者数；
* 最终 threshold；
* 报告路径；
* HTML 首页路径；
* 是否完成冻结测试集评估；
* 因数据缺失而未完成的项目。
