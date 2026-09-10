# T+O+C LLM 高可靠作者筛选器

## 1. 研究问题

在之前的实验中，已知LLM对于同名作者论文消歧的能力非常强大，本实验聚焦于在某一个已经证明LLM消歧表现较好但不完全适用所有作者block的字段组合下，是否能够通过某种提前的筛选机制，在LLM真实进行消歧前LLM预测出消歧结果较好的block和消歧效果不太好的block。

本实验固定使用 `title + organization + coauthor`（T+O+C）已有的 LLM聚类结果，不重新调用 LLM。目标是在聚类前，仅利用一个姓名 block 的原始论文数据，
预测

$$
P(Y=1\mid x),\qquad Y=\mathbb{1}(B^3F1\ge 0.85).
$$

概率达到冻结阈值的姓名进入 T+O+C 聚类；其余姓名标记为 high-risk。本实验不处理被
拒绝姓名，也不证明某个特征与聚类效果存在因果关系。

当前使用的小范围数据集：whoiswho基于AMiner手工标注的v3数据集中的valid集合（总共80位作者）。理想情况为使用train数据集训练，valid数据集验证，但当前需先在小数据集上完成作者block特征数目选择，具体训练流程。

## 2. 数据与已有实验结果

实验复用了 `artifacts/results/baseline_clusters/` 中 80 位作者的 T+O+C 聚类结果、NA_Demo
ground truth，以及 `artifacts/figures/field_selection/overview/analysis_summary.csv` 中已计算
的 Pairwise 与 B³ 指标。按排序后的作者名使用 `random_state=42` 随机打乱，
得到 64 位训练作者和 16 位测试作者。划分过程不使用 F1 标签。

训练集 B³ F1 均值为 **0.851**，中位数为
**0.908**；Reliable 作者 45/64，
Hard 作者 17/64，Catastrophic 作者
3/64。测试集在模型、特征和阈值冻结前没有参与
模型选择。

## 3. 训练集后验分析

Hard 案例按 Pairwise precision/recall 是否低于 0.80 作描述性划分：

- 类 over-merging（precision 低、recall 不低）：6；
- 类 over-splitting（recall 低、precision 不低）：10；
- 两者都低：1；
- 两者均不低但 B³ F1 仍低于 0.80：0。

这里的“类 over-merging/over-splitting”是指标现象，不是对 LLM 内部原因的断言。
训练集上与 B³ F1 绝对 Spearman 相关性最大的 pre-LLM 特征为：

- `coauthor_pairwise_similarity_mean`：Spearman $\rho=0.351$
- `paper_count`：Spearman $\rho=-0.294$
- `coauthor_pairwise_similarity_std`：Spearman $\rho=0.267$
- `estimated_prompt_tokens`：Spearman $\rho=-0.225$
- `title_nearest_neighbor_margin_mean`：Spearman $\rho=0.200$
- `title_pairwise_similarity_std`：Spearman $\rho=0.169$
- `coauthor_nearest_neighbor_margin_mean`：Spearman $\rho=-0.168$
- `tokens_per_paper`：Spearman $\rho=0.166$

Spearman 相关系数只比较单调排序关系：

$$
\rho_s=\operatorname{corr}(\operatorname{rank}(X),
\operatorname{rank}(B^3F1)).
$$

64 个训练样本较少，相关性可能受极端作者影响，因此只用于解释，不据此手写领域规则。

## 4. 筛选器设计

### 同名作者block特征提取

部署特征全部由原始 T+O+C 数据计算。Coverage 是字段非空论文比例；
`overall_coverage` 是三个 coverage 的均值。Title 使用 block 内 TF-IDF：

$$
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)
\left[\log\frac{1+N}{1+df(t)}+1\right],
$$

- `t`：某个 token，例如 "graph"；
- `d`：一篇标题；
- `N`：这个 block 内标题总数；
- `tf(t,d)`：词 t 在标题 d 中出现次数；
- `df(t)`：包含这个词的标题数量。

论文标题相似度为 L2 归一化向量的余弦：

$$
s_{title}(i,j)=\frac{v_i^\top v_j}{\lVert v_i\rVert_2\lVert v_j\rVert_2}.
$$

Organization 和 coauthor 被规范化为 token/name 集合，使用 Jaccard：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

每个字段统计全部有效论文对的均值和标准差，并对每篇论文计算最高邻居相似度及
margin：

$$
m_i=s_{i,(1)}-s_{i,(2)}.
$$

三视图一致性比较各字段 Top-3 邻居集合的 Jaccard，再在论文和视图对上取平均。
标题符号特征按非空字符或 token 计数；formula-like token 定义为同时含字母和数字，
或同时含字母数字与 `= + * / ^` 运算符的 token。

最终，一个作者名 block 就从“很多篇文本记录”变成一行数值表。

例如：

|特征|值|
|---|---|
|paper_count|4|
|title_coverage|1.00|
|org_coverage|1.00|
|coauthor_coverage|1.00|
|title_pairwise_similarity_mean|0.238|
|title_nearest_neighbor_margin_mean|0.61|
|org_pairwise_similarity_mean|0.34|
|coauthor_pairwise_similarity_mean|0.17|
|mean_cross_view_agreement|0.67|
|title_digit_ratio|0.04|

这整行，就是 Logistic Regression 的输入 x 或标准化后的 z。

### 模型训练

主模型为 Logistic Regression：

$$
P(Y=1\mid x)=\sigma(\beta_0+\beta^\top z),\qquad
\sigma(a)=\frac{1}{1+e^{-a}},
$$

其中 $z$ 是仅在训练 fold 上完成中位数填补和标准化后的特征。模型最小化带 L2
正则和类别平衡权重的负对数似然：

$$
\mathcal L=-\sum_i w_{y_i}[y_i\log p_i+(1-y_i)\log(1-p_i)]
+\lambda\lVert\beta\rVert_2^2.
$$

5-fold out-of-fold（OOF）保证每位训练作者的概率由未见过该作者的 fold 模型生成，
比训练集拟合概率更接近部署误差。

### 具体过程解释

#### 1. 目标

我们想预测一个作者 block 在 T+O+C 下是否能被 LLM 高质量消歧：

$
Y=\mathbb{1}(B^3F1\ge 0.85)
$

也就是：

$
Y=\begin{cases}1, & B^3F1\ge0.85 \quad \text{可靠}\\0, & B^3F1<0.85 \quad \text{不可靠}\end{cases}
$

模型目标：

$
P(Y=1\mid x)
$

即：给定这个作者 block 的原始数据特征 \(x\)，预测它可靠的概率。

---

#### 2. 原始数据如何变成特征

##### 2.1 title 相似度

每篇论文标题先变成 TF-IDF 向量 $v_i$。

两篇论文标题相似度：

$
s_{title}(i,j)=\frac{v_i^\top v_j}{\|v_i\|_2\|v_j\|_2}
$

含义：两个标题越相似，值越接近 1。

---

##### 2.2 organization / coauthor 相似度

把 organization 或 coauthor 变成集合 \(A,B\)。

Jaccard 相似度：

$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$

含义：两个集合重叠越多，值越接近 1。

---

##### 2.3 最近邻 margin

对论文 $i$，找它最相似和第二相似的论文：

$
m_i=s_{i,(1)}-s_{i,(2)}
$

含义：

- margin 大：最像的对象很明确，容易判断；
- margin 小：多个候选都差不多，容易混淆。

---

##### 2.4 汇总成 block-level 特征

前面的公式都是论文 pair 层面的。

对一个作者 block，把所有 pair 的相似度汇总成特征，例如：

$
\text{title\_sim\_mean}=\frac{1}{\binom{n}{2}}\sum_{i<j}s_{title}(i,j)
$

$
\text{title\_margin\_mean}=\frac{1}{n}\sum_i m_i
$

organization、coauthor 也同理。

最终每个作者 block 变成一个特征向量：

$
x=(x_1,x_2,\ldots,x_k)
$

例如：

$
x=(\text{paper\_count},\text{title\_coverage},\text{coauthor\_sim\_mean},\text{title\_margin\_mean},\ldots)
$

---

##### 2.5 真实标签

现在每个作者 block 会被压成一行数值特征表，文件在：

`artifacts/results/router/pre_llm_features_toc.csv`

目前是 80 行 x 28 列：其中 name 是作者 block 名，其余 27 列是特征。

- 基础规模与字段完整度

  - `paper_count`
  - `title_coverage`
  - `organization_coverage`
  - `coauthor_coverage`
  - `overall_coverage`
  - `estimated_prompt_tokens`
  - `tokens_per_paper`

  含义大致是：这个 block 有多少论文、title/org/coauthor 三个字段各自有多少比例非空、整体字段完整度、估算 prompt token 数、平均每篇论文 token 数。

- title / organization / coauthor 各自的相似度统计

  每个字段都有 4 个统计量：

  - `title_pairwise_similarity_mean`
  - `title_pairwise_similarity_std`
  - `title_nearest_neighbor_similarity_mean`
  - `title_nearest_neighbor_margin_mean`

  - `organization_pairwise_similarity_mean`
  - `organization_pairwise_similarity_std`
  - `organization_nearest_neighbor_similarity_mean`
  - `organization_nearest_neighbor_margin_mean`

  - `coauthor_pairwise_similarity_mean`
  - `coauthor_pairwise_similarity_std`
  - `coauthor_nearest_neighbor_similarity_mean`
  - `coauthor_nearest_neighbor_margin_mean`

  其中 title 用 TF-IDF cosine similarity；organization 和 coauthor 用集合Jaccard similarity。

  `pairwise_similarity_mean/std` 看整个 block 内论文两两之间平均有多像、差异有多大。

  `nearest_neighbor_similarity_mean` 看每篇论文最像的邻居平均有多像。

  `nearest_neighbor_margin_mean` 看“最像的论文”和“第二像的论文”差距大不大。margin大通常表示局部判断更清楚，margin 小表示容易混淆。

- 三个视角之间的一致性

  - `title_organization_neighbor_agreement`
  - `title_coauthor_neighbor_agreement`
  - `organization_coauthor_neighbor_agreement`
  - `mean_cross_view_agreement`

  这个是在比较：如果 title 认为 A 论文最接近 B/C/D，coauthor 或 organization 是否也认为它们接近。也就是不同字段给出的“近邻关系”是否一致。

- 标题符号/复杂度特征

  - `title_digit_ratio`
  - `title_non_alphabetic_ratio`
  - `title_uppercase_token_ratio`
  - `title_formula_like_token_ratio`

  这些主要是为了捕捉 title 是否含大量数字、符号、大写缩写、化学式/公式式 token。它们可能和 LLM 对某些学科标题的理解难度有关，比如化学、材料、生物
  化学里常见的复杂缩写和符号。

  目前这个表没有放入任何 ground truth 或 F1 指标，所以它本身是 pre-LLM / pre-evaluation 的特征表；b3_f1、pairwise_f1 只是在后续训练 router 时作为标签合并进去，不属于这个原始 block 数值表。

---

#### 3. 标准化

在每个训练 fold 内，对特征做标准化：

$
z_j=\frac{x_j-\mu_j}{\sigma_j}
$

其中：

- $\mu_j$：训练 fold 中第 $j$ 个特征的均值；
- $\sigma_j$：训练 fold 中第 $j$ 个特征的标准差；
- $z_j$：标准化后的特征。

所以模型真正使用的是：

$
z=(z_1,z_2,\ldots,z_k)
$

---

#### 4. Logistic Regression 预测公式

模型先做线性加权：

$
a=\beta_0+\beta^\top z
$

展开就是：

$
a=\beta_0+\beta_1z_1+\beta_2z_2+\cdots+\beta_kz_k
$

然后用 sigmoid 把线性分数变成概率：

$
p=P(Y=1\mid x)=\sigma(a)
$

$
\sigma(a)=\frac{1}{1+e^{-a}}
$

所以完整预测公式是：

$
P(Y=1\mid x)=\frac{1}{1+e^{-(\beta_0+\beta^\top z)}}
$

---

#### 5. 损失函数：如何学习 $\beta$

对于训练作者 $i$：

- 真实标签：$y_i\in\{0,1\}$
- 模型预测：$p_i=P(Y_i=1\mid x_i)$

单个样本的交叉熵损失：

$
\ell_i=-\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$

含义：

- 若 $y_i=1$，希望 $p_i$ 越接近 1 越好；
- 若 $y_i=0$，希望 $p_i$ 越接近 0 越好。

加入类别权重：

$
\ell_i=-w_{y_i}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$

加入 L2 正则：

$
\mathcal L=-\sum_i w_{y_i}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]+\lambda\|\beta\|_2^2
$

其中：

$
\|\beta\|_2^2=\beta_1^2+\beta_2^2+\cdots+\beta_k^2
$

最终训练就是求：

$
\beta^*=\arg\min_{\beta}\mathcal L(\beta)
$

含义：

> 找到一组 $\beta$，让训练作者的预测尽量接近真实标签，同时不要让权重过大。

---

#### 6. 5-fold OOF 怎么做

训练集有 64 个作者，分成 5 份：

$
D=D_1\cup D_2\cup D_3\cup D_4\cup D_5
$

第 $r$ 轮：

$
\text{Train}^{(r)}=D\setminus D_r
$

$
\text{Valid}^{(r)}=D_r
$

在 $\text{Train}^{(r)}$ 上训练：

$
\beta^{(r)}=\arg\min_\beta \mathcal L_{\text{Train}^{(r)}}(\beta)
$

然后对验证 fold 中每个作者 $i\in D_r$ 预测：

$
p_i^{OOF}=\sigma\left(\beta_0^{(r)}+{\beta^{(r)}}^\top z_i^{(r)}\right)
$

注意：

- 每个 fold 都有一组自己的 $\beta^{(r)}$；
- 每个作者的 OOF 概率都来自“没见过它”的模型；
- OOF 用来评估模型和选择 threshold。

因此，每个训练作者都会得到一个 OOF 概率，并且这个概率来自一个“训练时没有见过该作者”的模型。

OOF 的作用是：

> 在训练集内部尽量模拟测试集情形，用来评估筛选器和选择 threshold。

它不是最终部署模型；最终部署模型会在 threshold 确定后，用全部 64 个训练作者重新训练。

---

#### 7. 选择 threshold

**Threshold的选择标准：对于训练集合（64个作者），每一轮OOF再分成80%的训练集和20%的测试集。在80%的训练集上训练出所有特征权重$\beta_i$，然后在剩下的20%的测试集上循环测试出表现最好的threshold具体数值**

得到所有训练作者的 OOF 概率后，扫描阈值 $t$：

$
\hat{Y}_i(t)=\mathbb{1}(p_i^{OOF}\ge t)
$

也就是说：

$
p_i^{OOF} \ge t\Rightarrow\text{accept}
$

$
p_i^{OOF} < t\Rightarrow\text{reject / high-risk}
$

则接受该作者；否则拒绝。

这里的 threshold $t$ 不是人工拍脑袋决定的，而是在训练集 OOF 预测上扫描得到的。本实验扫描：

$
t \in \{0.50, 0.55, 0.60, \ldots, 0.95\}
$

对于每一个候选 threshold，计算以下三个核心指标。

第一，Accepted Precision：

$
\text{Accepted Precision}=\frac{\#\{p_i^{OOF}\ge t \land y_i=1\}}{\#\{p_i^{OOF}\ge t\}}
$

它表示：

> 被筛选器接受的作者中，有多少比例真实达到了 \(B^3F1 \ge 0.85\)。

第二，Coverage：

$
\text{Coverage}=\frac{\#\{p_i^{OOF}\ge t\}}{\#\{i\}}
$

它表示：

> 全部作者中，有多少比例被筛选器接受。

第三，Hard-block Recall：

$
\text{Hard Recall}=\frac{\#\{p_i^{OOF}< t \land B^3F1_i<0.80\}}{\#\{B^3F1_i<0.80\}}
$

它表示：

> 所有真实 hard 作者中，有多少被筛选器成功拒绝。

这三个指标反映的是不同目标：

| 指标 | 含义 | 希望 |
|---|---|---|
| Accepted Precision | 接受作者的质量纯度 | 越高越好 |
| Coverage | 接受作者的比例 | 不能太低 |
| Hard-block Recall | 困难作者的拦截能力 | 越高越好 |

threshold 越高，模型越保守：

- Accepted Precision 通常会上升；
- Coverage 通常会下降；
- Hard-block Recall 通常会上升。

threshold 越低，模型越宽松：

- Coverage 通常会上升；
- 但可能接受更多低质量作者；
- Accepted Precision 可能下降；
- Hard-block Recall 可能下降。

本实验的 threshold 选择规则是：

1. 优先寻找 Accepted Precision \(\ge 0.90\) 的 threshold；
2. 如果存在多个满足条件的 threshold，则选择 Coverage 最大的那个；
3. 如果没有任何 threshold 达到 0.90，则选择训练 OOF 上 Accepted Precision 最高的 threshold；若并列，再选择 Coverage 最大者。

因此，threshold 的选择完全基于训练集 OOF 概率和训练集真实 F1，不使用测试集。

---

#### 8. 最终测试集预测

**这里用之前已经计算好的threshold $t$，结合用所有64个训练作者计算出来的总权重$\beta_i$，在测试集上进行测试**

OOF 结束后，用全部 64 个训练作者重新训练最终模型：

$
\beta^{final}=\arg\min_\beta \mathcal L_{\text{all train}}(\beta)
$

对测试作者 $q$：

$
p_q=\sigma\left(\beta_0^{final}+{\beta^{final}}^\top z_q\right)
$

最终规则：

$
p_q\ge t\Rightarrow\text{accept}
$

$
p_q<t\Rightarrow\text{reject / high-risk}
$

完整流程是：

```text
训练集 64 作者
    ↓
5-fold OOF 生成每个训练作者的未见过预测概率
    ↓
用 OOF 概率和真实 F1 选择 threshold
    ↓
threshold 冻结
    ↓
用全部 64 个训练作者重新训练 final model
    ↓
对 16 个测试作者预测概率
    ↓
使用冻结 threshold 判断 accept / reject
```

## 5. 筛选效果与最终规则

本节评估的问题不是普通分类 accuracy，而是：

> 如果筛选器只接受它认为最可靠的一部分作者，这些被接受作者的真实 T+O+C 消歧质量是否更高？

因此，本实验主要观察三个指标。

第一，Accepted Precision：

$
\text{Accepted Precision}=\frac{\#\{\text{accepted} \land B^3F1 \ge 0.85\}}{\#\{\text{accepted}\}}
$

它表示：

> 被筛选器放行的作者中，有多少真实达到了可靠标准。

第二，Coverage：

$
\text{Coverage}=\frac{\#\{\text{accepted}\}}{\#\{\text{all authors}\}}
$

它表示：

> 筛选器接受了多少比例的作者。

第三，Hard-block Recall：

$
\text{Hard-block Recall}=\frac{\#\{\text{rejected} \land B^3F1 < 0.80\}}{\#\{B^3F1 < 0.80\}}
$

它表示：

> 真实困难作者中，有多少被筛选器成功拦截。

---

### 5.1 60% coverage 下的排序能力比较

首先，为了比较不同模型的排序能力，本实验固定 coverage = 60%。

这一步的含义是：

> 不管模型自己的概率阈值是多少，都让每个模型接受它认为最可靠的前 60% 作者，然后比较这些 accepted 作者的真实质量。

训练集共有 64 位作者，因此 60% coverage 大约表示接受 38 位作者。

结果如下：

| 模型 | Accepted precision | Accepted mean B³ F1 | Hard-block recall |
|---|---:|---:|---:|
| `random_ranking` | 0.667 | 0.835 | 0.353 |
| `paper_count_only` | 0.821 | 0.881 | 0.647 |
| `coverage_only` | 0.744 | 0.864 | 0.529 |
| `full_logistic_regression` | 0.718 | 0.862 | 0.471 |

从表中可以看出：

1. `random_ranking` 是随机排序基线。它在 60% coverage 下的 Accepted precision 为 0.667，表示随机接受的作者中约 66.7% 真实可靠。

2. `paper_count_only` 表现最好。它的 Accepted precision 为 0.821，Accepted mean B³ F1 为 0.881，Hard-block recall 为 0.647。说明在当前训练集 OOF 中，论文数量本身就是一个较强的风险信号：论文数量较大的 block 往往更容易低 F1。

3. `coverage_only` 比随机更好，但不如 `paper_count_only`。这说明字段完整度有一定解释力，但在本次实验中不如论文数量稳定。

4. `full_logistic_regression` 使用了全部 pre-LLM 特征，但在 60% coverage 下只达到 0.718 Accepted precision，低于 `paper_count_only` 的 0.821，也只略高于随机基线的 0.667。

因此，在当前训练集上，复杂 pre-LLM 特征并没有带来比简单 `paper_count` 更稳定的筛选能力。

---

### 5.2 训练 OOF 上的 threshold 选择结果

随后，本实验只在 Full Logistic Regression 的训练 OOF 概率上选择最终部署 threshold。

扫描范围为：

$
t \in \{0.50, 0.55, \ldots, 0.95\}
$

原始目标是找到一个 threshold，使得：

$
\text{Accepted Precision} \ge 0.90
$

也就是说，希望被筛选器接受的作者中，至少 90% 真实满足：

$
B^3F1 \ge 0.85
$

但是，在训练 OOF 结果中，没有任何候选 threshold 达到 0.90 Accepted precision。

因此，按照预设规则，最终选择训练 OOF 上 Accepted precision 最高的 threshold；若出现并列，则选择 Coverage 最大者。最终得到：

$
t = 0.90
$

在该 threshold 下，训练 OOF 结果为：

| 指标 | 数值 |
|---|---:|
| Accepted precision | 0.727 |
| Coverage | 0.172 |
| Hard-block recall | 0.882 |
| Accepted mean B³ F1 | 0.901 |

这意味着：

- 筛选器只接受约 17.2% 的训练作者；
- 被接受作者中，约 72.7% 真实达到 \(B^3F1 \ge 0.85\)；
- 真实 hard 作者中，约 88.2% 被筛选器拒绝；
- 被接受作者的平均 B³ F1 为 0.901。

这个结果说明该 threshold 非常保守：它拒绝了大部分作者，因此能拦截较多 hard block；但它并没有达到原本希望的 0.90 Accepted precision。

因此，threshold = 0.90 不是一个成功达到目标的阈值，而是在当前候选范围和 OOF 结果下得到的最佳折中点。

---

### 5.3 最终接受规则

根据训练 OOF 阶段选择的 threshold，最终筛选规则为：

```text
使用冻结的 pre-LLM 特征管线计算概率
P(B³ F1 >= 0.85) >= 0.90 -> accepted
否则 -> rejected / high-risk
```

Risk-coverage 曲线不是普通 accuracy：它展示逐步增加接受作者时，质量纯度与可覆盖
工作量之间的交换。相关交互图位于 `artifacts/figures/router/index.html`。

## 6. 冻结测试集结果与局限

阈值冻结后，使用全部 64 位训练作者重新拟合一次 Full Logistic Regression，并只用
16 位测试作者的 pre-LLM 特征预测。测试集接受 0/16 位作者：

- Test accepted precision：**N/A**；
- Test coverage：**0.000**；
- Test hard-block recall：**1.000**；
- Test accepted mean B³ F1：**N/A**。

测试概率最高值仍低于冻结阈值，因此模型拒绝全部测试作者。这里的 hard-block recall
虽然是 1.000，却来自 coverage=0 的退化策略，不能视为成功；Accepted precision 和
Accepted mean B³ F1 也因没有 accepted 样本而不可定义。按照冻结测试原则，本实验
不再根据这 16 位作者降低阈值或修改特征。

这只是一次小样本冻结测试。它仅验证当前 80 个作者、当前 T+O+C prompt/model 的
可筛选性；不同模型、prompt、数据域或可靠阈值需要重新校准。特征系数是条件相关，
不能解释为因果贡献。测试集仅有 16 人，因此一个作者就会明显改变 precision；
后续应扩大独立作者样本，并单独研究 rejected 作者的补救策略。

