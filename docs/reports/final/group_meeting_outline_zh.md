# 组会汇报纲要：LLM 姓名消歧实验总结

## 1. 本次汇报目标

本次组会主要汇报我负责的 LLM 姓名消歧部分，目标包括三点：

1. 说明我负责的研究问题是什么
2. 总结前期文献调研中学习到的传统姓名消歧方法
3. 汇报围绕 LLM 调用方式做过的实验、当前结论和后续可以继续推进的方向

可以先用一句话概括：

> 我的工作主要是在 from-scratch author name disambiguation 任务中，探索如何调用 LLM 才能更准确、更稳定地根据论文信息完成作者聚类。

## 2. 研究问题

本项目研究的是 from-scratch author name disambiguation：给定同名或同拼音作者的一组论文，要求根据论文元信息判断哪些论文属于同一位真实作者。

最开始的问题是：

> LLM 能不能直接做姓名消歧？

经过前期实验后，问题进一步拆成三个更具体的问题：

1. 应该给 LLM 哪些论文字段？
2. 应该如何组织这些论文信息？
3. prompt 应该如何写，才能让 LLM 更稳定地输出高质量聚类结果？

当前实验主要围绕 `title`、`abstract`、`keywords`、`coauthors`、`organization`、`venue` 等字段展开，并使用 Pairwise F1 与 B³ F1 评估聚类质量。

## 3. 文献调研：传统姓名消歧方法

在尝试直接调用 LLM 之前，我先调研了几种更传统、更数学化的姓名消歧方法，主要包括 WhoIsWho、BOND 和 MCCG。调研这些方法的目的不是完全复现它们，而是理解已有方法如何利用 `title`、`coauthors`、`organization`、`venue` 等信息，以及它们的局限在哪里。

这三类方法的共同特点是：先把论文元信息转化为显式特征、相似度矩阵或图结构，再使用聚类算法或图神经网络得到作者簇。

### 3.1 WhoIsWho：规则特征 + 距离矩阵 + DBSCAN

WhoIsWho 是比较典型的传统姓名消歧工具包。它的核心思路是：

1. 对同名作者 block 内的论文提取 semantic feature 和 relational feature
2. 对每两篇论文计算距离或相似度
3. 构造论文之间的距离矩阵
4. 使用 DBSCAN 进行聚类
5. 对 DBSCAN 产生的 outlier 再做规则后处理

可以这样介绍：

> WhoIsWho 不是直接理解论文内容，而是先把论文之间的关系转成显式距离。例如，两篇论文是否有共同合作者、机构是否相似、venue 是否相似、标题词是否重叠。然后把这些距离组合成矩阵，再交给 DBSCAN 聚类。

WhoIsWho 中需要区分两种任务：

- SND：from-scratch name disambiguation，没有已有作者 profile，只在同名 block 内做论文聚类
- RND：real-time name disambiguation，把新论文匹配到已有作者 profile

本项目更关注 SND，因为我们的目标是从一批同名论文中直接判断真实作者簇。

WhoIsWho 的优点：

- 规则清楚
- 可解释
- 可复现
- DBSCAN 不需要预先指定真实作者数量

WhoIsWho 的局限：

- 依赖人工特征和权重
- 字符串规则难以理解深层语义
- 字段缺失或写法不一致时效果会下降
- 跨领域时可能需要重新调参

一句话总结：

> WhoIsWho 告诉我们，coauthor、organization、venue、title 都是有用身份线索，但它主要把这些线索处理成字符串相似度和距离矩阵，语义理解能力有限。

### 3.2 BOND：规则构图 + GAT + 聚类自训练

BOND 是比 WhoIsWho 更进一步的图学习方法。它的核心思想是：

1. 每篇论文是图中的一个节点
2. 如果两篇论文共享合作者、机构或 venue，就建立边
3. 边属性表示关系强弱
4. 使用 GAT 学习论文 embedding
5. 在 embedding 空间中使用 DBSCAN 聚类
6. 训练过程中反复用 DBSCAN 结果生成 pseudo label，继续优化模型

可以这样介绍：

> BOND 和 WhoIsWho 的区别在于，它不是只在人工距离矩阵上聚类一次，而是先用规则构造论文图，再用图注意力网络学习每篇论文的表示。DBSCAN 不只是最后一步，它还会在训练过程中不断产生伪标签，帮助模型继续优化 embedding。

讲 BOND 时可以重点解释两个问题。

第一，GAT 在做什么？

- 初始规则边决定哪些论文之间可以传递信息
- GAT 根据邻居信息更新论文节点 embedding
- 注意力机制会学习哪些邻居更重要
- 它不是简单更新边权，而是在学习更适合聚类的论文表示

第二，二元交叉熵如何优化 embedding？

- DBSCAN 当前结果会被转成论文对标签
- 如果两篇论文在同一伪簇中，标签为 1
- 如果两篇论文不在同一伪簇中，标签为 0
- 模型预测两篇论文是否同簇
- 预测错误会通过二元交叉熵反向传播到节点 embedding
- 每篇论文 embedding 会受到它和其他论文 pairwise 关系的共同约束

BOND 的优点：

- 比纯规则距离矩阵更灵活
- 能利用图结构
- 能通过训练学习更好的论文表示
- 能结合规则图、embedding 和聚类

BOND 的局限：

- 需要构图规则
- 需要训练模型
- 依赖初始 paper embedding 和 DBSCAN pseudo label
- 如果初始图噪声大，伪标签可能会放大错误
- 工程成本比直接调用 LLM 高

一句话总结：

> BOND 的价值在于把规则关系、图神经网络和聚类自训练结合起来，但它仍然依赖人工构图、embedding 质量和训练过程。

### 3.3 MCCG：多视图对比学习 + 聚类引导

MCCG 是更复杂的 representation learning 方法。它的核心思想是：

1. 把同名作者 block 构造成论文图
2. 每篇论文是节点
3. coauthor、organization、venue 等关系形成边
4. 从同一个原始图生成两个增强视图
5. 用多视图对比学习让同一论文在不同扰动下表示保持一致
6. 再用 cluster-guided learning 让表示更适合聚类

可以这样介绍：

> MCCG 关注的问题是，启发式构图本身可能有噪声。如果只依赖一张固定图，模型可能过度相信某些错误边。因此它对同一个图做两次随机增强，得到两个视图，然后让模型学习在不同视图下仍然稳定的表示。

两个增强视图可以这样解释：

- 原始图由论文节点特征矩阵和邻接矩阵组成
- 两个视图保留相同论文节点
- 但会随机删除部分边
- 也会随机遮蔽部分节点特征
- 这样模型不会过度依赖某一条边或某一个特征维度

projector 可以这样解释：

- GNN 先得到节点 embedding
- projector 是一个小的神经网络
- 它把 embedding 映射到对比学习空间或聚类空间
- multi-view representation 用于比较两个增强视图中同一节点是否一致
- cluster representation 用于让节点表示更接近聚类目标

MCCG 的优点：

- 考虑图噪声
- 通过多视图增强提升鲁棒性
- 把 representation learning 和 clustering 联系起来
- 比单纯规则或单视图图学习更复杂、更稳健

MCCG 的局限：

- 方法复杂
- 训练成本高
- 需要 embedding、图增强、projector、contrastive loss、cluster loss 等多个模块
- 对实现和参数比较敏感
- 对当前想快速探索 LLM 如何利用字段的目标来说，成本较重

一句话总结：

> MCCG 的重点是通过多视图对比学习获得更鲁棒的论文表示，但它也进一步增加了训练和工程复杂度。

### 3.4 三类方法横向比较

| 方法 | 核心机制 | 优点 | 局限 |
|:---:|:---:|:---:|:---:|
| WhoIsWho | 手工特征 + 距离矩阵 + DBSCAN | 清楚、可解释、容易理解 | 依赖规则和权重，语义能力有限 |
| BOND | 规则图 + GAT + DBSCAN 自训练 | 能学习论文 embedding，利用图结构 | 依赖构图、训练和伪标签质量 |
| MCCG | 多视图图增强 + 对比学习 + 聚类引导 | 更重视鲁棒表示和图噪声 | 训练复杂，工程成本高 |

可以这样串起来：

> 这三类方法都在回答同一个问题：如何把论文中的 title、coauthor、organization、venue 等信息转成可以聚类的表示。区别在于，WhoIsWho 主要靠手工规则，BOND 加入图神经网络学习，MCCG 进一步加入多视图对比学习。

### 3.5 为什么调研后转向 LLM

传统方法虽然数学结构清晰，但都需要把论文信息先压缩成规则、距离或 embedding。这样做的问题是，很多语义判断仍然很难通过简单规则表达。例如：

- 同一作者可能跨机构
- 同一作者可能跨主题
- 不同作者可能在同一机构、同一领域工作
- 合作者网络可能变化或缺失
- 机构和姓名字符串可能存在多种写法

转向 LLM 的原因是：

- LLM 可以直接读取结构化论文信息
- LLM 能综合 `title`、`coauthors`、`organization` 等多种证据
- 不需要重新训练图模型
- 更适合快速探索不同字段、context 和 prompt 设计
- 但需要解决输出稳定性、token 成本和本地校验问题

可以用一句话结束文献调研部分：

> 调研传统方法后，我的理解是：传统方法已经证明了 coauthor、organization、title 等字段是有效身份线索；我的工作是尝试把这些线索直接交给 LLM，让 LLM 做综合判断，并重点研究如何设计字段、context 和 prompt 才能让 LLM 更稳定地完成姓名消歧。

## 4. LLM 实验设计

LLM 实验主要分成三个方向。

### 4.1 字段组合实验

测试不同字段单独或组合输入 LLM 时的聚类效果。

涉及字段包括：

- `title`
- `abstract`
- `keywords`
- `coauthors`
- `organization`
- `venue`

核心问题：

> 哪些字段组合最适合让 LLM 做姓名消歧？

### 4.2 信息组织实验

固定字段为 `title + coauthors + organization`，只改变论文在 prompt 中的组织方式。

测试过的组织方式包括：

- `original_flat`
- `random_order`
- `chronological_order`
- `organization_grouped`
- `coauthor_network`
- `multifield_similarity`

核心问题：

> 在字段完全相同的情况下，context organization 是否会影响 LLM 聚类结果？

### 4.3 Prompt 实验

固定字段和论文顺序，只改变 prompt 说明。

测试过的 prompt 条件包括：

- `minimal`
- `task_definition`
- `evidence_guidance`
- `error_aware`
- `analyze_then_cluster`

核心问题：

> 什么样的 prompt 能让 LLM 明确理解这是姓名消歧，而不是普通主题聚类？

## 5. 主要实验结果

### 5.1 字段组合：T+O+C 是当前最合适的主组合

最重要结论：

> `title + coauthors + organization` 是当前最合适的默认字段组合。

原因：

- `title` 提供研究主题信号
- `coauthors` 提供合作关系信号
- `organization` 提供机构身份信号
- 三者互补，比单字段稳定
- 不默认加入 `abstract`，因为 `abstract` token 成本高，而且可能引入更强主题噪声

关键数字：

- NA_Demo 三作者 pilot 中，T+O+C 的 macro B³ F1 达到 0.9814
- v3 validation 80 作者中，T+O+C 的平均 B³ F1 为 0.8292，中位数为 0.9076
- 80 个作者中，B³ F1 不低于 0.85 的有 55 个

### 5.2 信息组织：复杂组织可能提高局部分数，但稳定性风险更高

信息组织实验说明：

- `organization_grouped` 的成功样本 mean B³ F1 最高，为 0.9020
- `original_flat` 的 mean B³ F1 为 0.8871
- 但 `organization_grouped` 出现 parse failure 和论文缺失问题
- `random_order` 平均效果下降，说明 LLM 对输入顺序敏感

因此结论是：

- 默认使用稳定 flat context
- 不随机打乱
- 复杂分组只能作为异常 block 的补充策略
- 所有输出必须做 full-coverage 校验

### 5.3 Prompt：task definition 稳定优于 minimal

Prompt 实验说明：

- `minimal` mean B³ F1 为 0.7150
- `task_definition` mean B³ F1 为 0.7816
- `error_aware` 成功样本最高，但失败调用更多

因此更稳的结论是：

- prompt 必须明确这是 author name disambiguation
- 要提醒 LLM 不要把任务理解成普通主题聚类
- 可以加入简短错误提醒
- 但不要要求长篇分析，否则容易消耗输出预算或导致 JSON 不完整

## 6. 当前推荐的 LLM 调用方式

当前最推荐的默认调用方式是：

```text
字段：title + coauthors + organization
context：stable flat context，每篇论文一条 record
prompt：明确姓名消歧任务 + 简洁证据解释 + 错误提醒
输出：compact JSON assignment
参数：temperature=0
校验：每篇论文必须恰好出现一次
```

可以压缩成一句话：

> T+O+C + flat context + task-aware / concise error-aware prompt + compact JSON output，是当前最稳妥、性价比最高的 LLM 调用方案。

## 7. 结果验证实验

为了验证推荐流程不是只来自总体均值，我挑选了一些原始结果没有达到 0.95 的作者进行比较。

其中成功改善明显的案例包括：

| 作者 | 原始 B³ F1 | 验证 B³ F1 | 原始 Pairwise F1 | 验证 Pairwise F1 |
|:---:|:---:|:---:|:---:|:---:|
| `zheng_hu` | 0.9299 | 0.9913 | 0.8663 | 0.9874 |
| `yi_qian` | 0.0567 | 0.9389 | 0.0000 | 0.9647 |

这说明：

- 一部分看似失败的 LLM 聚类结果可以通过更稳定的调用方式修正
- 但不是所有失败作者都能靠重跑解决
- 后续需要研究如何判断 LLM 输出是否可信，以及什么时候触发修正

## 8. 主要失败模式

目前观察到的主要失败模式有三类。

### 8.1 Over-splitting

同一真实作者被拆成大量 singleton，导致 Pairwise Recall 接近 0。

典型案例：

- `bo_zou`
- `yi_qian`
- `jun_wang`

可能原因：

- block 太大
- 主题跨度大
- title 中缩写和专业实体太多
- LLM 没有充分利用 coauthor 和 organization 连续性

### 8.2 Over-merging

不同真实作者被合并进同一个大簇。

这会损害 precision，也会增加后续人工检查或二次修正的压力。

### 8.3 输出完整性失败

包括：

- 缺失论文
- 重复分配
- 跨 cluster 重复
- JSON 不完整
- API incomplete read

这类失败和聚类质量不同，必须单独统计。

## 9. 后续研究方向

当前更有研究价值的方向不是继续无条件增加字段，而是在 LLM 调用前后加入判断和修正机制。

### 9.1 调用前判断

研究哪些作者 block 适合直接交给 LLM。

可以考虑的信号：

- paper count
- coauthor 连通性
- organization 覆盖率
- title 主题分散度
- 候选块规模

### 9.2 调用前预处理

对复杂 block 进行确定性预处理，例如：

- 按共同合作者局部分组
- 按 organization 排序
- 按年份排序
- 用 title / coauthor / organization 相似度做局部排序

目标是降低 LLM 在长上下文中自行发现结构的难度。

### 9.3 调用后质量判断与修正

LLM 输出后，不仅要检查 JSON 格式，还要判断聚类结果是否可信。

可以检查：

- 预测簇数是否异常大
- singleton 比例是否异常高
- 最大簇是否接近 catch-all
- 同一簇内 coauthor / organization 是否完全断裂
- 不同簇之间是否存在大量共同合作者或相似机构

如果结果不可信，可以触发：

- singleton 合并检查
- 大簇内部精分
- 缺失论文重跑
- 复杂 block 分块处理

## 10. 组会上可以重点讨论的问题

最后可以把讨论引向下面几个问题：

1. 最终汇报中，文献调研部分需要讲多详细？
2. LLM 实验部分是突出字段组合，还是突出“如何稳定调用 LLM”？
3. 失败案例是否要作为重点展示？
4. 后续方向要不要继续推进 router / 预处理 / 调用后修正？
5. 最后汇报 slides 中，我负责的部分应该占多大篇幅？

## 11. 最短口头总结

如果时间很紧，可以这样总结：

> 我负责的部分主要研究如何调用 LLM 做 from-scratch 姓名消歧。在文献调研中，我学习了 WhoIsWho、BOND 和 MCCG，它们分别代表规则距离、图神经网络和多视图对比学习方法。传统方法证明了 coauthor、organization、title 等字段是有效身份线索，但也依赖人工规则、embedding 训练和图构造。基于这一点，我尝试把这些身份线索直接交给 LLM，并围绕字段组合、context organization 和 prompt 设计做实验。当前结论是：title + coauthors + organization 是最合适的默认字段组合；context 应保持稳定 flat；prompt 需要明确这是姓名消歧而不是主题聚类；输出必须使用 compact JSON 并做 full-coverage 校验。后续更有价值的方向是在 LLM 调用前后加入可靠性判断、预处理和错误修正机制。
