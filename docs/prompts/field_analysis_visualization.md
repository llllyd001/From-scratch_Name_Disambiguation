请为这个姓名消歧项目实现一个“交互式实验结果分析”工具，优先复用仓库中已有的 LLM clustering 输出（目前在artifacts/results/baseline_clusters/*）、ground truth、report 逻辑和字段定义，不要重写已有评估算法。

目标：针对约 80 个 author-name block，生成 Plotly 交互式 HTML 图表。每个点代表一个作者 block；鼠标悬停时必须显示作者名，以及该 block 的完整关键指标。

实现要求：

1. 使用 Python、pandas、Plotly；生成独立可打开的 `.html` 文件，不依赖 notebook 或服务器。
2. 新建合适的脚本，例如：
   `src/stage1_field_combinations/analyze_interactive.py`
3. 支持命令行参数：
   * `--data-dir`
   * `--result-dir`
   * `--ground-truth`
   * `--output-dir`
   * 可选的 `--combination`
   * 可选的 `--f1-metric`，默认 `b3_f1`
4. 自动扫描每个作者的预测结果和 ground truth，为每个 block 汇总一行数据，并写出：
   `analysis_summary.csv`
5. 不要假设 JSON 字段结构；先检查现有 `llm_cluster.py`、`report.py` 和结果文件格式，再复用或抽取已有的评估逻辑。
6. 缺失字段、无效预测、无法计算指标的作者不能让脚本崩溃；应在 CSV 中记录状态和原因，并跳过不适用的图点。

每个 block 尽可能计算这些列：

* `name`
* `status`
* `paper_count`
* `prompt_tokens`；若已有日志中没有，允许为空
* `tokens_per_paper`
* `pairwise_precision`
* `pairwise_recall`
* `pairwise_f1`
* `b3_precision`
* `b3_recall`
* `b3_f1`
* `true_cluster_count`
* `predicted_cluster_count`
* `cluster_count_ratio = predicted / true`
* `log2_cluster_count_ratio`
* `true_singleton_ratio`
* `predicted_singleton_ratio`
* `true_largest_cluster_ratio`
* `predicted_largest_cluster_ratio`
* `true_cluster_size_entropy`
* 各字段 coverage：至少包括 `title_coverage`、`organization_coverage`、`coauthor_coverage`，以及可选 `overall_coverage`

生成以下交互式图表；每张图单独输出一个 HTML：

1. `f1_ranked_distribution.html`

   * 按 B³ F1 从低到高排序的 scatter/lollipop 图。
   * y 轴为 B³ F1，x 轴为排序序号。
   * hover 显示作者名、Pairwise F1、B³ F1、论文数、真实/预测 cluster 数、coverage。
   * 可加默认 F1=0.90 的水平参考线。

2. `pairwise_precision_recall.html`

   * x=Pairwise Precision，y=Pairwise Recall。
   * 点大小可表示 paper_count，颜色表示 B³ F1。
   * 加 x=0.9、y=0.9 参考线。
   * hover 显示作者名、所有 F1、paper_count、coverage、cluster count ratio。
   * 图标题说明右上角是容易 block，左上偏 over-merging，右下偏 over-splitting。

3. `coverage_vs_f1.html`

   * 至少生成 overall/title/organization/coauthor coverage 与 B³ F1 的散点图。
   * hover 显示作者名、paper_count、cluster 数、Pairwise P/R/F1。
   * 若某 coverage 列不存在或全为空，跳过对应图。

4. `paper_count_vs_f1.html`

   * x=paper_count，y=B³ F1。
   * 点颜色表示 true_cluster_count，点大小表示 overall_coverage。
   * hover 显示作者名和完整指标。

5. `true_cluster_count_vs_f1.html`

   * x=true_cluster_count，y=B³ F1。
   * 对 x 轴较大的情况使用合理刻度；必要时允许 log scale。
   * hover 显示作者名、paper_count、coverage、cluster-size entropy。

6. `cluster_ratio_vs_f1.html`

   * x=log2(predicted_cluster_count / true_cluster_count)，y=B³ F1。
   * 加 x=0 的竖线，标注 “predicted = true”。
   * 左侧说明倾向 over-merging，右侧说明倾向 over-splitting。
   * hover 显示作者名、真实/预测 cluster 数、singleton ratio、largest-cluster ratio、Pairwise P/R。

7. `token_count_vs_f1.html`

   * 若 token 信息存在，生成：

     * prompt_tokens vs B³ F1
     * tokens_per_paper vs B³ F1
   * hover 显示作者名、paper_count、coverage、true_cluster_count。
   * 若 token 信息不存在，输出清晰提示，不要报错。

8. `singleton_ratio_comparison.html`

   * x=true_singleton_ratio，y=predicted_singleton_ratio。
   * 加 y=x 对角线。
   * hover 显示作者名、B³ F1、cluster_count_ratio。
   * 用于诊断 over-splitting 或 singleton 被错误合并。

9. `largest_cluster_ratio_comparison.html`

   * x=true_largest_cluster_ratio，y=predicted_largest_cluster_ratio。
   * 加 y=x 对角线。
   * hover 显示作者名、B³ F1、Pairwise P/R。
   * 用于诊断是否产生异常大的预测 cluster。

10. `true_entropy_vs_f1.html`

    * x=true_cluster_size_entropy，y=B³ F1。
    * hover 显示作者名、true_cluster_count、paper_count、largest cluster ratio。
    * 若 entropy 不适合现有 ground truth 结构，跳过并记录原因。

11. `feature_heatmap.html`

    * 行是作者 block，按 B³ F1 从低到高排序。
    * 列至少包括：B³ F1、Pairwise F1、paper_count、true/predicted cluster count、cluster_count_ratio、各 coverage、singleton ratio、largest cluster ratio、entropy。
    * hover 显示作者名和具体数值。
    * 数值做适当标准化，但 hover 必须保留原始值。

12. `index.html`

    * 创建一个简单 HTML 首页，链接到所有生成的图。
    * 顶部说明：每个点代表一个 author-name block；hover 可查看作者名和详细指标。
    * 列出低于 F1 阈值的 hardest blocks，并链接或注明其名称。

额外要求：

* 所有图都要有清楚的英文标题、坐标轴标签、hover template。
* 作者名必须在 hover 中显著显示。
* 图表应支持缩放、框选、保存图片等 Plotly 默认交互。
* 不要把真实标签相关特征误写成可部署的输入特征；在 README 或脚本注释中明确区分：

  * 仅供事后分析：true cluster count、true entropy、F1 等；
  * 可用于未来 hard-block routing：paper count、coverage、token count、预测 cluster 结构等。
* 增加一个简短 README 使用示例。
* 先实现正确、稳健的数据汇总，再实现图表；完成后运行一次基本自检，确认输出目录中至少有 CSV、index.html 和可打开的图表 HTML。
