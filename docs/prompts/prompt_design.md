请在当前姓名消歧项目中完成第二个小规模实验，研究：

> 在输入作者、论文顺序和 title + organization + coauthor（T+O+C）字段完全相同的情况下，不同的 prompt 和 context instruction 是否会影响 LLM 的作者姓名消歧效果。

本实验只改变提供给 LLM 的任务说明和判断指导，不改变论文数据本身的顺序、字段和内容。

---

## 1. 目录要求

代码放在：

```text
src/prompt_context/
```

实验结果放在：

```text
artifacts/results/prompt_design/
```

简洁中文实验报告放在：

```text
docs/reports/prompt_design/
```

交互图表放在：

```text
artifacts/figures/prompt_design/
```

不要修改或覆盖已有 T+O+C 实验和 `context_orgnization` 实验的结果。

---

## 2. 优先复用现有代码

开始前检查并复用项目已有的：

* validation 数据和 ground truth；
* 原始 T+O+C prompt；
* LLM 调用代码；
* JSON 输出解析代码；
* Pairwise 和 B³ 评测代码；
* 第一个 `context_orgnization` 实验中的断点续跑、结果保存和绘图逻辑。

不要无必要地重新实现已有功能。

所有条件固定：

* 相同模型；
* 相同 temperature；
* 相同 token 设置；
* 相同作者；
* 相同论文顺序；
* 相同 T+O+C 字段；
* 相同字段格式；
* 相同输出 JSON schema；
* 相同重试和解析机制。

不同条件只能改变任务 prompt 中的说明和判断指导。

---

## 3. 作者选择

为了缩短运行时间，只选择 3 个作者：

* 1 个 easy：原 T+O+C 的 B³ F1 ≥ 0.90；
* 1 个 medium：0.80 ≤ B³ F1 < 0.90；
* 1 个 hard：B³ F1 < 0.80 (但不要太难，控制F1在0.6以上)。

优先选择论文数量不太大的作者，以控制 token 和运行时间。

建议优先选择：

* `paper_count` 不超过整个 validation 集中位数；
* 同时具备 title、organization 和 coauthor 信息；
* 原 T+O+C 结果能够正常解析；
* easy、medium、hard 三组都有代表性。

使用固定随机种子完成选择，并保存：

```text
artifacts/results/prompt_design/selected_authors.json
```

文件中记录：

* author_id；
* difficulty_group；
* baseline B³ F1；
* paper_count；
* organization coverage；
* coauthor coverage

---

## 4. 固定论文 Context

所有 prompt 条件必须使用完全相同的论文 context。

直接采用原 T+O+C 实验使用的原始平铺形式，并保持相同论文顺序，例如：

```text
Paper ID: P1
Title: ...
Organization: ...
Coauthors: ...

Paper ID: P2
Title: ...
Organization: ...
Coauthors: ...
```

禁止：

* 随机排序；
* 按时间排序；
* 按机构或合作者分组；
* 增加相似度信息；
* 增加字段统计摘要；
* 增加外部数据。

本实验需要与第一个信息组织实验严格区分。

---

## 5. 五种 Prompt 条件

> 在输入作者、论文顺序和 title + organization + coauthor（T+O+C）字段完全相同的情况下，不同的 prompt 和 context instruction 是否会影响 LLM 的作者姓名消歧效果。

使用以下稳定 condition ID：

```text
minimal
task_definition
evidence_guidance
error_aware
analyze_then_cluster
```

所有条件使用相同的最终输出要求，例如：

```json
{
  "clusters": [
    ["P1", "P3"],
    ["P2", "P4"]
  ]
}
```

不得要求模型在最终输出中加入长篇解释。即使某些条件要求先分析，最终也只保存统一的结构化聚类结果，或者将简短分析与 clusters 分字段保存。

---

### 5.1 minimal

作用：最简单的 baseline prompt。

Prompt 只需要说明：

* 这些论文具有相同或相似的作者姓名；
* 请判断它们分别属于哪些真实作者；
* 按要求输出聚类 JSON。

不要解释字段意义，也不要提醒常见错误。

示意：

```text
The following publications share the same or a similar author name.
Group the publications by their actual authors using the provided information.
Return only the required JSON clustering result.
```

---

### 5.2 task_definition

作用：测试更明确的任务定义是否有帮助。

在 `minimal` 基础上增加：

* 这是 author name disambiguation 任务；
* 同一个姓名可能对应多个不同的人；
* 同一个人也可能出现研究主题、机构或合作者变化；
* 需要综合判断，而不是根据单一字段直接分类。

核心说明：

```text
This is an author name disambiguation task. Publications with the same displayed
name may belong to different people. At the same time, one person may change
organization, research topic, or collaborators over time. Use all available
evidence jointly and do not make decisions based on only one field.
```

除此之外不要增加具体字段优先级。

---

### 5.3 evidence_guidance

作用：测试显式解释 T+O+C 三类证据的使用方式是否有帮助。

在 `task_definition` 基础上增加简洁的字段指导：

* title：反映研究主题，但同一作者的研究方向可能变化，不同作者也可能研究相似主题；
* organization：通常是重要证据，但作者可能转机构，机构字符串也可能缺失或存在不同写法；
* coauthor：重复合作者是重要关联证据，但合作者网络也可能随时间变化；
* 不应由任意单一字段直接决定聚类。

Prompt 中不要写死具体权重，不要声明 coauthor 一定比 organization 更重要。

核心说明：

```text
Use the fields as follows:

- Titles provide topic evidence, but one author may change topics and different
  authors may work on similar topics.
- Organizations provide affiliation evidence, but authors may move and the same
  organization may appear in different forms.
- Coauthors provide relationship evidence, but collaboration networks may also
  change.

Combine these signals instead of treating any single field as decisive.
```

---

### 5.4 error_aware

作用：测试明确提醒常见错误是否能够改变模型的 overmerge / oversplit 倾向。

在 `task_definition` 基础上加入以下提醒：

* 不要仅因机构不同就拆分；
* 不要仅因机构相同就合并；
* 不要仅因标题主题不同就拆分；
* 不要仅因标题主题相似就合并；
* 共享合作者是支持证据，但不是绝对证明；
* 既要避免把一个人拆成过多 cluster，也要避免把不同人合并。

保持中性，不要只强调 precision 或只强调 recall。

核心说明：

```text
Avoid common mistakes:

- Do not split publications only because organizations or topics differ.
- Do not merge publications only because organizations or topics are similar.
- Shared coauthors are useful evidence, but are not absolute proof.
- Avoid both excessive splitting of one person and excessive merging of
  different people.
```

---

### 5.5 analyze_then_cluster

作用：测试先进行内部证据分析，再形成全局聚类是否有帮助。

使用与 `evidence_guidance` 相同的字段解释，并增加一个简短的分阶段要求：

1. 先识别论文之间支持属于同一作者的证据；
2. 再识别明显冲突；
3. 检查聚类的全局一致性；
4. 最后输出聚类。

不要要求模型列举所有论文对，也不要生成非常长的 chain-of-thought。

可以要求简短的结构化 evidence summary，例如：

```json
{
  "brief_evidence": [
    "P1 and P3 share organization and coauthors.",
    "P2 has conflicting organization and research context."
  ],
  "clusters": [
    ["P1", "P3"],
    ["P2"]
  ]
}
```

如果现有解析流程只支持 clusters，则允许要求模型在内部完成分析，但最终只输出 clusters：

```text
First examine supporting and conflicting evidence and check global consistency.
Then return only the final JSON clustering result.
```

优先采用后者，以减少输出 token。

---

## 6. Prompt 构造要求

实现统一接口，例如：

```python
build_prompt(
    papers,
    condition,
    base_output_schema,
) -> str
```

建议将 prompt 拆成：

1. 固定 system prompt；
2. 根据 condition 变化的 task instruction；
3. 完全固定的论文 context；
4. 完全固定的 output requirement。

对同一作者，五个条件必须通过自动检查确认：

* paper IDs 完全相同；
* paper 顺序完全相同；
* title 完全相同；
* organization 完全相同；
* coauthor 完全相同；
* 输出 JSON schema 完全相同；
* 只有 instruction 部分不同。

把每个完整 prompt 保存到：

```text
artifacts/results/prompt_design/prompts/
```

---

## 7. 实验执行

建议提供主命令：

```bash
python -m src.prompt_context.run_experiment
```

流程：

1. 检查并复用现有数据和 LLM 配置；
2. 选择并保存 9 个作者；
3. 为每个作者生成 5 种 prompt；
4. 自动检查输入数据一致性；
5. 保存完整 prompts；
6. 调用 LLM；
7. 保存原始响应；
8. 解析聚类；
9. 计算 Pairwise 和 B³ 指标；
10. 生成汇总表和交互图；
11. 生成简洁中文报告。

总调用量预计为：

```text
9 authors × 5 conditions = 45 calls
```

暂时不要求每个 prompt 重复多次，以减少运行时间。

必须支持断点续跑：

* 已成功得到结果的 author-condition 不重复调用；
* 保存失败信息；
* 再次运行时只补跑缺失结果。

---

## 8. 保存结果

至少生成：

```text
artifacts/results/prompt_design/author_condition_metrics.csv
artifacts/results/prompt_design/condition_summary.csv
artifacts/results/prompt_design/author_deltas.csv
```

### author_condition_metrics.csv

每行对应一个：

```text
author × prompt condition
```

至少包含：

* author_id；
* difficulty_group；
* condition；
* paper_count；
* Pairwise Precision；
* Pairwise Recall；
* Pairwise F1；
* B³ Precision；
* B³ Recall；
* B³ F1；
* predicted cluster count；
* true cluster count；
* cluster count bias；
* input tokens；
* output tokens；
* total tokens；
* latency；
* parse success；
* retry count。

### condition_summary.csv

每个 condition 汇总：

* mean B³ Precision；
* mean B³ Recall；
* mean B³ F1；
* median B³ F1；
* mean Pairwise F1；
* mean predicted cluster count；
* mean cluster count bias；
* mean input tokens；
* mean output tokens；
* parse failure count。

### author_deltas.csv

以 `minimal` 为 baseline，计算：

```text
delta_b3_f1 = 当前条件 B³ F1 - minimal B³ F1
delta_pairwise_f1 = 当前条件 Pairwise F1 - minimal Pairwise F1
```

并保存每个作者、每个非 baseline 条件的变化。

---

## 9. 结果分析

不需要复杂统计建模。

重点比较以下问题：

### 9.1 明确任务定义是否有帮助

比较：

```text
task_definition vs minimal
```

判断只增加任务背景是否能改善结果。

### 9.2 字段使用指导是否有帮助

比较：

```text
evidence_guidance vs task_definition
```

判断模型是否需要显式了解 title、organization 和 coauthor 的局限。

### 9.3 错误提醒是否改变 precision–recall 平衡

比较：

```text
error_aware vs task_definition
```

重点观察：

* Pairwise Precision；
* Pairwise Recall；
* predicted cluster count；
* cluster count bias。

判断 error-aware prompt 是否减少：

* oversplitting；
* overmerging。

### 9.4 分阶段分析是否有效

比较：

```text
analyze_then_cluster vs evidence_guidance
```

同时考虑：

* B³ F1 提升；
* output token 增加；
* latency 增加。

判断额外推理过程是否值得。

### 9.5 不同难度作者是否表现不同

分别汇总 easy、medium、hard 三组的 mean B³ F1 和 mean delta。

重点观察：

* 简单作者是否几乎不受 prompt 影响；
* 困难作者是否从更详细 prompt 中获益；
* 详细 prompt 是否会错误干预原本已经正确的 easy 作者。

---

## 10. 交互图表

使用 Plotly 生成以下 3 张 HTML 图，放在：

```text
artifacts/figures/prompt_design/
```

### 10.1 condition_performance.html

展示五种 prompt 的 B³ F1 分布：

* box plot；
* 叠加作者点；
* hover 显示 author_id、difficulty、B³ F1、Pairwise F1。

### 10.2 paired_prompt_comparison.html

比较各 prompt 与 `minimal`：

* x 轴：minimal B³ F1；
* y 轴：当前 prompt B³ F1；
* 添加 y=x 参考线；
* 可切换 condition；
* hover 显示 author_id 和 delta。

### 10.3 author_prompt_heatmap.html

* 行：author；
* 列：五种 prompt；
* 值：B³ F1 或相对 minimal 的 delta；
* hover 显示各项指标；
* 按 easy、medium、hard 排序。

不需要生成更多图表。

---

## 11. 简洁实验报告

生成：

```text
docs/reports/prompt_design/prompt_context_experiment_report_zh.md
```

报告尽量简洁，只包含以下内容。

### 1. 实验目的

说明：

* 固定作者、论文顺序和 T+O+C 数据；
* 只改变 prompt 和 context instruction；
* 研究任务说明、证据指导、错误提醒和分阶段分析的影响。

### 2. 实验设置

简要说明：

* 作者数量；
* easy / medium / hard 分布；
* 模型和 temperature；
* 五种 prompt；
* 主要评测指标。

### 3. Prompt 条件

每种 prompt 用 1–2 句话介绍，不要复制完整 prompt。

### 4. 主要结果

提供一张表格，包含：

* condition；
* mean B³ Precision；
* mean B³ Recall；
* mean B³ F1；
* median B³ F1；
* mean Pairwise F1；
* mean cluster count bias；
* mean input/output tokens。

### 5. 主要发现

根据真实结果回答：

* prompt 内容是否明显影响消歧；
* 哪个 prompt 平均最好；
* 详细 prompt 是否主要帮助 hard 作者；
* error-aware prompt 是否改变 oversplitting / overmerging；
* analyze-then-cluster 的收益是否值得额外成本；
* 是否存在某些作者在不同 prompt 下结果差异很大。

### 6. 典型案例

选择 2–3 个作者进行简洁分析：

* 更详细 prompt 明显改善；
* 更详细 prompt 反而恶化；
* 不同 prompt 改变 cluster 数量或 precision–recall 平衡。

### 7. 结论

如实总结，不要预设详细 prompt 一定更好。

如果差异很小，可以得出：

> 在当前小规模样本、固定 T+O+C 字段和模型下，增加任务说明并未稳定改善结果，模型性能更可能受证据质量限制。

如果不同 prompt 对不同作者效果不一致，可以指出：

> Prompt 设计没有统一最优方案，其效果与作者 block 难度和字段证据特征有关。

---

## 12. 代码与实验要求

* 不硬编码 API key；
* 不覆盖已有实验；
* 使用固定随机种子；
* 保存完整 prompt 和原始响应；
* 支持断点续跑；
* 不使用 ground truth 构造 prompt；
* ground truth 只能用于作者分层和最终评测；
* 不手工修改 LLM 输出；
* 不伪造任何结果；
* 分析和绘图必须可以仅使用已保存结果重新运行。

完成后在终端输出：

* 实际使用的数据路径；
* 选取的作者名单；
* 成功和失败调用数量；
* 五种 prompt 的 mean B³ F1；
* 最优平均 prompt；
* 报告路径；
* 图表目录；
* 结果目录。
