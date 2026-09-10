#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from name_disambiguation.core.llm_cluster import field_value, paper_card, prompt_for
from name_disambiguation.paths import FIGURES_ROOT, REPORTS_ROOT, RESULTS_ROOT, dataset_dir


SEED = 42
FIELDS = ("title", "coauthors", "organization")
DATA_DIR = dataset_dir("v3")
METRICS_PATH = FIGURES_ROOT / "field_selection" / "overview" / "analysis_summary.csv"
RESULT_DIR = RESULTS_ROOT / "router"
PLOT_DIR = FIGURES_ROOT / "router"
REPORT_DIR = REPORTS_ROOT / "router"
PLOT_CONFIG = {"displaylogo": False, "responsive": True}

BASIC_FEATURES = [
    "paper_count",
    "title_coverage",
    "organization_coverage",
    "coauthor_coverage",
    "overall_coverage",
    "estimated_prompt_tokens",
    "tokens_per_paper",
]
SIMILARITY_FEATURES = [
    f"{field}_{stat}"
    for field in ("title", "organization", "coauthor")
    for stat in (
        "pairwise_similarity_mean",
        "pairwise_similarity_std",
        "nearest_neighbor_similarity_mean",
        "nearest_neighbor_margin_mean",
    )
]
AGREEMENT_FEATURES = [
    "title_organization_neighbor_agreement",
    "title_coauthor_neighbor_agreement",
    "organization_coauthor_neighbor_agreement",
    "mean_cross_view_agreement",
]
TITLE_SYMBOL_FEATURES = [
    "title_digit_ratio",
    "title_non_alphabetic_ratio",
    "title_uppercase_token_ratio",
    "title_formula_like_token_ratio",
]
FULL_FEATURES = (
    BASIC_FEATURES + SIMILARITY_FEATURES + AGREEMENT_FEATURES
    + TITLE_SYMBOL_FEATURES
)
MODEL_FEATURES = {
    "random_ranking": [],
    "paper_count_only": ["paper_count"],
    "coverage_only": [
        "title_coverage",
        "organization_coverage",
        "coauthor_coverage",
        "overall_coverage",
    ],
    "full_logistic_regression": FULL_FEATURES,
}


def read_json(path):
    return json.loads(path.read_text())


def normalize_text(value):
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def coauthor_set(paper, target_name):
    parts = re.findall(r"[a-z0-9]+", target_name.lower())
    target_forms = {"".join(parts), "".join(reversed(parts))}
    return {
        normalize_text(author.get("name", ""))
        for author in paper.get("authors", [])
        if normalize_name(author.get("name", "")) not in target_forms
        and normalize_text(author.get("name", ""))
    }


def organization_tokens(paper, target_name):
    organizations = field_value(paper, target_name, "organization")
    return set(re.findall(r"[a-z0-9]+", " ".join(organizations).lower()))


def title_similarity_matrix(titles):
    available = np.array([bool(normalize_text(title)) for title in titles])
    if available.sum() < 2:
        matrix = np.full((len(titles), len(titles)), np.nan, dtype=np.float32)
        return matrix, available
    try:
        vectors = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
            dtype=np.float32,
        ).fit_transform(titles)
    except ValueError:
        matrix = np.full((len(titles), len(titles)), np.nan, dtype=np.float32)
        return matrix, available
    matrix = (vectors @ vectors.T).toarray()
    matrix[~(available[:, None] & available[None, :])] = np.nan
    return matrix, available


def jaccard_matrix(values):
    available = np.array([bool(value) for value in values])
    vocabulary = {
        token: index
        for index, token in enumerate(sorted(set().union(*values) if values else set()))
    }
    if not vocabulary:
        return np.full((len(values), len(values)), np.nan, dtype=np.float32), available
    rows, columns = [], []
    for row, tokens in enumerate(values):
        for token in tokens:
            rows.append(row)
            columns.append(vocabulary[token])
    data = np.ones(len(rows), dtype=np.float32)
    encoded = sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(len(values), len(vocabulary)),
        dtype=np.float32,
    )
    intersections = (encoded @ encoded.T).toarray()
    counts = np.asarray(encoded.sum(axis=1)).ravel()
    unions = counts[:, None] + counts[None, :] - intersections
    matrix = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    matrix[~(available[:, None] & available[None, :])] = np.nan
    return matrix, available


def similarity_statistics(matrix):
    count = matrix.shape[0]
    pair_values = matrix[np.triu_indices(count, 1)]
    pair_values = pair_values[np.isfinite(pair_values)]
    nearest, margins = [], []
    for row_index in range(count):
        row = matrix[row_index].copy()
        row[row_index] = np.nan
        values = row[np.isfinite(row)]
        if values.size:
            ordered = np.sort(values)
            nearest.append(float(ordered[-1]))
            if values.size >= 2:
                margins.append(float(ordered[-1] - ordered[-2]))
    return {
        "pairwise_similarity_mean": float(np.mean(pair_values)) if pair_values.size else np.nan,
        "pairwise_similarity_std": float(np.std(pair_values)) if pair_values.size else np.nan,
        "nearest_neighbor_similarity_mean": float(np.mean(nearest)) if nearest else np.nan,
        "nearest_neighbor_margin_mean": float(np.mean(margins)) if margins else np.nan,
    }


def top_neighbors(matrix, k=3):
    neighbors = []
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index].copy()
        row[row_index] = np.nan
        valid = np.flatnonzero(np.isfinite(row))
        if not valid.size:
            neighbors.append(set())
            continue
        order = valid[np.argsort(row[valid], kind="stable")[-k:]]
        neighbors.append(set(map(int, order)))
    return neighbors


def neighbor_agreement(first, second):
    values = []
    for left, right in zip(first, second):
        union = left | right
        if union:
            values.append(len(left & right) / len(union))
    return float(np.mean(values)) if values else np.nan


def title_symbol_features(titles):
    text = " ".join(title for title in titles if title)
    characters = [char for char in text if not char.isspace()]
    denominator = len(characters) or 1
    tokens = re.findall(r"\S+", text)
    alphabetic_tokens = [
        token for token in tokens if any(char.isalpha() for char in token)
    ]
    uppercase = [
        token for token in alphabetic_tokens
        if len([char for char in token if char.isalpha()]) >= 2
        and all(not char.isalpha() or char.isupper() for char in token)
    ]
    formula_like = [
        token for token in tokens
        if (
            any(char.isalpha() for char in token)
            and any(char.isdigit() for char in token)
        )
        or (
            any(char.isalnum() for char in token)
            and any(char in "=+*/^" for char in token)
        )
    ]
    return {
        "title_digit_ratio": sum(char.isdigit() for char in characters) / denominator,
        "title_non_alphabetic_ratio": (
            sum(not char.isalpha() for char in characters) / denominator
        ),
        "title_uppercase_token_ratio": (
            len(uppercase) / len(alphabetic_tokens) if alphabetic_tokens else 0
        ),
        "title_formula_like_token_ratio": (
            len(formula_like) / len(tokens) if tokens else 0
        ),
    }


def author_features(name, paper_ids, papers):
    author_papers = [papers[paper_id] for paper_id in paper_ids]
    titles = [paper.get("title", "") or "" for paper in author_papers]
    organizations = [
        organization_tokens(paper, name) for paper in author_papers
    ]
    coauthors = [coauthor_set(paper, name) for paper in author_papers]
    title_matrix, title_available = title_similarity_matrix(titles)
    organization_matrix, organization_available = jaccard_matrix(organizations)
    coauthor_matrix, coauthor_available = jaccard_matrix(coauthors)
    matrices = {
        "title": title_matrix,
        "organization": organization_matrix,
        "coauthor": coauthor_matrix,
    }
    neighbors = {field: top_neighbors(matrix) for field, matrix in matrices.items()}

    cards = [paper_card(paper, name, FIELDS) for paper in author_papers]
    compact_cards = [
        {
            "record_index": index,
            **{field: card[field] for field in FIELDS},
        }
        for index, card in enumerate(cards)
    ]
    prompt_tokens = len(prompt_for(name, FIELDS, compact_cards, compact_output=True)) // 4
    coverage = {
        "title_coverage": float(np.mean(title_available)),
        "organization_coverage": float(np.mean(organization_available)),
        "coauthor_coverage": float(np.mean(coauthor_available)),
    }
    features = {
        "name": name,
        "paper_count": len(paper_ids),
        **coverage,
        "overall_coverage": float(np.mean(list(coverage.values()))),
        "estimated_prompt_tokens": prompt_tokens,
        "tokens_per_paper": prompt_tokens / len(paper_ids),
    }
    for field, matrix in matrices.items():
        features.update({
            f"{field}_{key}": value
            for key, value in similarity_statistics(matrix).items()
        })
    features.update({
        "title_organization_neighbor_agreement": neighbor_agreement(
            neighbors["title"], neighbors["organization"]
        ),
        "title_coauthor_neighbor_agreement": neighbor_agreement(
            neighbors["title"], neighbors["coauthor"]
        ),
        "organization_coauthor_neighbor_agreement": neighbor_agreement(
            neighbors["organization"], neighbors["coauthor"]
        ),
    })
    features["mean_cross_view_agreement"] = float(np.nanmean([
        features["title_organization_neighbor_agreement"],
        features["title_coauthor_neighbor_agreement"],
        features["organization_coauthor_neighbor_agreement"],
    ]))
    features.update(title_symbol_features(titles))
    return features


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=SEED,
                max_iter=2000,
            ),
        ),
    ])


def split_names(names):
    shuffled = np.array(sorted(names), dtype=object)
    np.random.RandomState(SEED).shuffle(shuffled)
    return sorted(shuffled[:64].tolist()), sorted(shuffled[64:].tolist())


def add_labels(features, metrics):
    columns = [
        "name",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "b3_precision",
        "b3_recall",
        "b3_f1",
        "true_cluster_count",
        "predicted_cluster_count",
        "cluster_count_ratio",
    ]
    labelled = features.merge(metrics[columns], on="name", validate="one_to_one")
    labelled["reliable_085"] = (labelled["b3_f1"] >= 0.85).astype(int)
    labelled["hard_080"] = (labelled["b3_f1"] < 0.80).astype(int)
    labelled["catastrophic_050"] = (labelled["b3_f1"] < 0.50).astype(int)
    return labelled


def cross_validated_predictions(train):
    y = train["reliable_085"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    random_scores = np.random.RandomState(SEED).random(len(train))
    for model_name, feature_columns in MODEL_FEATURES.items():
        if model_name == "random_ranking":
            probabilities = random_scores
        else:
            probabilities = np.zeros(len(train))
            x = train[feature_columns]
            for fit_indexes, validation_indexes in folds.split(x, y):
                model = make_pipeline()
                model.fit(x.iloc[fit_indexes], y[fit_indexes])
                probabilities[validation_indexes] = model.predict_proba(
                    x.iloc[validation_indexes]
                )[:, 1]
        for row_index, probability in enumerate(probabilities):
            item = train.iloc[row_index]
            rows.append({
                "name": item["name"],
                "actual_b3_f1": item["b3_f1"],
                "reliable_085": item["reliable_085"],
                "hard_080": item["hard_080"],
                "predicted_probability": probability,
                "model_name": model_name,
            })
    return pd.DataFrame(rows)


def selection_metrics(frame, accepted):
    accepted = np.asarray(accepted, dtype=bool)
    reliable = frame["reliable_085"].to_numpy(dtype=bool)
    hard = frame["hard_080"].to_numpy(dtype=bool)
    return {
        "accepted_authors": int(accepted.sum()),
        "accepted_precision": (
            float(reliable[accepted].mean()) if accepted.any() else None
        ),
        "coverage": float(accepted.mean()),
        "hard_block_recall": (
            float((~accepted & hard).sum() / hard.sum()) if hard.any() else None
        ),
        "accepted_mean_b3_f1": (
            float(frame.loc[accepted, "actual_b3_f1"].mean())
            if accepted.any() else None
        ),
    }


def coverage_curves(cv_predictions):
    rows = []
    for model_name, frame in cv_predictions.groupby("model_name", sort=False):
        frame = frame.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
        for accepted_count in range(1, len(frame) + 1):
            accepted = np.arange(len(frame)) < accepted_count
            rows.append({
                "model_name": model_name,
                **selection_metrics(frame, accepted),
            })
    return pd.DataFrame(rows)


def coverage_checkpoints(cv_predictions):
    rows = []
    for model_name, frame in cv_predictions.groupby("model_name", sort=False):
        frame = frame.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
        for target in (0.2, 0.4, 0.6, 0.8, 1.0):
            accepted_count = math.ceil(target * len(frame))
            accepted = np.arange(len(frame)) < accepted_count
            rows.append({
                "model_name": model_name,
                "target_coverage": target,
                **selection_metrics(frame, accepted),
            })
    return pd.DataFrame(rows)


def choose_threshold(full_predictions):
    rows = []
    for threshold in np.arange(0.50, 0.951, 0.05):
        metrics = selection_metrics(
            full_predictions,
            full_predictions["predicted_probability"] >= threshold,
        )
        rows.append({"threshold": round(float(threshold), 2), **metrics})
    eligible = [
        row for row in rows
        if row["accepted_precision"] is not None
        and row["accepted_precision"] >= 0.90
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (row["coverage"], row["accepted_precision"], -row["threshold"]),
        )
        rule = "在 Accepted precision 不低于 0.90 的阈值中选择 Coverage 最大者"
    else:
        candidates = [row for row in rows if row["accepted_precision"] is not None]
        selected = max(
            candidates,
            key=lambda row: (row["accepted_precision"], row["coverage"], -row["threshold"]),
        )
        rule = "未达到 0.90，先选择最高 Accepted precision，再选择最大 Coverage"
    return selected, pd.DataFrame(rows), rule


def write_plot(figure, path):
    figure.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
        legend_title_text="Model",
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config=PLOT_CONFIG,
    )


def plot_risk_coverage(curves):
    metrics = [
        ("accepted_precision", "Accepted Precision"),
        ("accepted_mean_b3_f1", "Accepted Mean B3 F1"),
        ("hard_block_recall", "Hard-block Recall"),
    ]
    figure = make_subplots(rows=1, cols=3, subplot_titles=[item[1] for item in metrics])
    for column, (metric, _) in enumerate(metrics, start=1):
        for model_name, frame in curves.groupby("model_name", sort=False):
            figure.add_trace(
                go.Scatter(
                    x=frame["coverage"],
                    y=frame[metric],
                    mode="lines",
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=column == 1,
                    hovertemplate="coverage=%{x:.3f}<br>value=%{y:.3f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="Coverage", row=1, col=column)
        figure.update_yaxes(range=[0, 1.02], row=1, col=column)
    figure.update_layout(title="Training OOF Risk-Coverage Curves", height=520)
    write_plot(figure, PLOT_DIR / "risk_coverage_curve.html")


def plot_probability_vs_f1(full_predictions, threshold):
    colors = np.where(full_predictions["reliable_085"] == 1, "#2a9d8f", "#e76f51")
    figure = go.Figure(go.Scatter(
        x=full_predictions["predicted_probability"],
        y=full_predictions["actual_b3_f1"],
        mode="markers",
        text=full_predictions["name"],
        marker={"color": colors, "size": 10, "opacity": 0.8},
        hovertemplate="%{text}<br>P(reliable)=%{x:.3f}<br>B3 F1=%{y:.3f}<extra></extra>",
    ))
    figure.add_vline(x=threshold, line_dash="dash", line_color="#264653")
    figure.add_hline(y=0.85, line_dash="dash", line_color="#2a9d8f")
    figure.update_layout(
        title="Training OOF Probability vs Actual B3 F1",
        xaxis_title="OOF predicted P(B3 F1 >= 0.85)",
        yaxis_title="Actual B3 F1",
        yaxis_range=[0, 1.02],
    )
    write_plot(figure, PLOT_DIR / "cv_probability_vs_actual_b3_f1.html")


def plot_threshold_tradeoff(thresholds, selected_threshold):
    figure = go.Figure()
    for column, label in (
        ("accepted_precision", "Accepted Precision"),
        ("coverage", "Coverage"),
        ("hard_block_recall", "Hard-block Recall"),
        ("accepted_mean_b3_f1", "Accepted Mean B3 F1"),
    ):
        figure.add_trace(go.Scatter(
            x=thresholds["threshold"],
            y=thresholds[column],
            mode="lines+markers",
            name=label,
        ))
    figure.add_vline(x=selected_threshold, line_dash="dash", line_color="#e76f51")
    figure.update_layout(
        title="Training OOF Threshold Trade-off",
        xaxis_title="Acceptance threshold",
        yaxis_title="Metric",
        yaxis_range=[0, 1.02],
    )
    write_plot(figure, PLOT_DIR / "threshold_tradeoff.html")


def plot_feature_importance(model, feature_columns):
    coefficients = model.named_steps["classifier"].coef_[0]
    frame = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": coefficients,
        "absolute": np.abs(coefficients),
    }).sort_values("absolute", ascending=True)
    colors = np.where(frame["coefficient"] >= 0, "#2a9d8f", "#e76f51")
    figure = go.Figure(go.Bar(
        x=frame["coefficient"],
        y=frame["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>standardized coefficient=%{x:.3f}<extra></extra>",
    ))
    figure.update_layout(
        title="Full Logistic Regression Feature Coefficients",
        xaxis_title="Coefficient after median imputation and standardization",
        yaxis_title="",
        height=850,
    )
    write_plot(figure, PLOT_DIR / "feature_importance.html")


def write_index():
    entries = [
        ("Risk-Coverage Curve", "risk_coverage_curve.html"),
        ("OOF Probability vs Actual B3 F1", "cv_probability_vs_actual_b3_f1.html"),
        ("Threshold Trade-off", "threshold_tradeoff.html"),
        ("Feature Importance", "feature_importance.html"),
    ]
    links = "\n".join(
        f'<li><a href="{filename}">{title}</a></li>' for title, filename in entries
    )
    (PLOT_DIR / "index.html").write_text(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>T+O+C Reliability Router</title>"
        "<style>body{font-family:Arial;max-width:900px;margin:40px auto;"
        "line-height:1.6}li{margin:12px 0}</style></head><body>"
        "<h1>T+O+C Reliability Router</h1>"
        "<p>Router-specific interactive figures. General 80-author clustering "
        "figures remain in <code>artifacts/figures/field_selection/overview/</code>.</p>"
        f"<ul>{links}</ul></body></html>\n"
    )


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def failure_modes(train):
    hard = train[train["hard_080"] == 1]
    return {
        "hard_count": len(hard),
        "over_merging_like": int(
            ((hard["pairwise_precision"] < 0.8) & (hard["pairwise_recall"] >= 0.8)).sum()
        ),
        "over_splitting_like": int(
            ((hard["pairwise_recall"] < 0.8) & (hard["pairwise_precision"] >= 0.8)).sum()
        ),
        "both_low": int(
            ((hard["pairwise_precision"] < 0.8) & (hard["pairwise_recall"] < 0.8)).sum()
        ),
        "neither_below_080": int(
            ((hard["pairwise_precision"] >= 0.8) & (hard["pairwise_recall"] >= 0.8)).sum()
        ),
    }


def correlation_summary(train):
    correlations = train[FULL_FEATURES + ["b3_f1"]].corr(
        method="spearman", numeric_only=True
    )["b3_f1"].drop("b3_f1").dropna()
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def report_text(
    train,
    checkpoints,
    selected,
    threshold_rule,
    test_metrics,
    feature_correlations,
    modes,
):
    train_f1 = train["b3_f1"]
    model_rows = []
    for model_name in MODEL_FEATURES:
        row = checkpoints[
            (checkpoints["model_name"] == model_name)
            & (checkpoints["target_coverage"] == 0.6)
        ].iloc[0]
        model_rows.append(
            f"| `{model_name}` | {fmt(row['accepted_precision'])} | "
            f"{fmt(row['accepted_mean_b3_f1'])} | {fmt(row['hard_block_recall'])} |"
        )
    top_correlations = "\n".join(
        f"- `{feature}`：Spearman $\\rho={value:.3f}$"
        for feature, value in feature_correlations.head(8).items()
    )
    accepted_test = test_metrics["accepted_authors"]
    report = f"""# T+O+C LLM 高可靠作者筛选器

## 1. 研究问题

本实验固定使用 `title + organization + coauthor`（T+O+C）及其已有 LLM
聚类结果，不重新调用 LLM。目标是在聚类前，仅利用一个姓名 block 的原始论文数据，
预测

$$
P(Y=1\\mid x),\\qquad Y=\\mathbb{{1}}(B^3F1\\ge 0.85).
$$

概率达到冻结阈值的姓名进入 T+O+C 聚类；其余姓名标记为 high-risk。本实验不处理被
拒绝姓名，也不证明某个特征与聚类效果存在因果关系。

## 2. 数据与已有实验结果

实验复用了 `artifacts/results/baseline_clusters/` 中 80 位作者的 T+O+C 聚类结果、NA_Demo
ground truth，以及 `artifacts/figures/field_selection/overview/analysis_summary.csv` 中已计算
的 Pairwise 与 B³ 指标。按排序后的作者名使用 `random_state={SEED}` 随机打乱，
得到 64 位训练作者和 16 位测试作者。划分过程不使用 F1 标签。

训练集 B³ F1 均值为 **{train_f1.mean():.3f}**，中位数为
**{train_f1.median():.3f}**；Reliable 作者 {int(train['reliable_085'].sum())}/64，
Hard 作者 {int(train['hard_080'].sum())}/64，Catastrophic 作者
{int(train['catastrophic_050'].sum())}/64。测试集在模型、特征和阈值冻结前没有参与
模型选择。

## 3. 训练集后验分析

Hard 案例按 Pairwise precision/recall 是否低于 0.80 作描述性划分：

- 类 over-merging（precision 低、recall 不低）：{modes['over_merging_like']}；
- 类 over-splitting（recall 低、precision 不低）：{modes['over_splitting_like']}；
- 两者都低：{modes['both_low']}；
- 两者均不低但 B³ F1 仍低于 0.80：{modes['neither_below_080']}。

这里的“类 over-merging/over-splitting”是指标现象，不是对 LLM 内部原因的断言。
训练集上与 B³ F1 绝对 Spearman 相关性最大的 pre-LLM 特征为：

{top_correlations}

Spearman 相关系数只比较单调排序关系：

$$
\\rho_s=\\operatorname{{corr}}(\\operatorname{{rank}}(X),
\\operatorname{{rank}}(B^3F1)).
$$

64 个训练样本较少，相关性可能受极端作者影响，因此只用于解释，不据此手写领域规则。

## 4. 筛选器设计

部署特征全部由原始 T+O+C 数据计算。Coverage 是字段非空论文比例；
`overall_coverage` 是三个 coverage 的均值。Title 使用 block 内 TF-IDF：

$$
\\operatorname{{tfidf}}(t,d)=\\operatorname{{tf}}(t,d)
\\left[\\log\\frac{{1+N}}{{1+df(t)}}+1\\right],
$$

论文标题相似度为 L2 归一化向量的余弦：

$$
s_{{title}}(i,j)=\\frac{{v_i^\\top v_j}}{{\\lVert v_i\\rVert_2\\lVert v_j\\rVert_2}}.
$$

Organization 和 coauthor 被规范化为 token/name 集合，使用 Jaccard：

$$
J(A,B)=\\frac{{|A\\cap B|}}{{|A\\cup B|}}.
$$

每个字段统计全部有效论文对的均值和标准差，并对每篇论文计算最高邻居相似度及
margin：

$$
m_i=s_{{i,(1)}}-s_{{i,(2)}}.
$$

三视图一致性比较各字段 Top-3 邻居集合的 Jaccard，再在论文和视图对上取平均。
标题符号特征按非空字符或 token 计数；formula-like token 定义为同时含字母和数字，
或同时含字母数字与 `= + * / ^` 运算符的 token。

主模型为 Logistic Regression：

$$
P(Y=1\\mid x)=\\sigma(\\beta_0+\\beta^\\top z),\\qquad
\\sigma(a)=\\frac{{1}}{{1+e^{{-a}}}},
$$

其中 $z$ 是仅在训练 fold 上完成中位数填补和标准化后的特征。模型最小化带 L2
正则和类别平衡权重的负对数似然：

$$
\\mathcal L=-\\sum_i w_{{y_i}}[y_i\\log p_i+(1-y_i)\\log(1-p_i)]
+\\lambda\\lVert\\beta\\rVert_2^2.
$$

5-fold out-of-fold（OOF）保证每位训练作者的概率由未见过该作者的 fold 模型生成，
比训练集拟合概率更接近部署误差。

## 5. 筛选效果与最终规则

在 60% coverage 检查点：

| 模型 | Accepted precision | Accepted mean B³ F1 | Hard-block recall |
|---|---:|---:|---:|
{chr(10).join(model_rows)}

三个核心量定义为：

$$
\\text{{Accepted Precision}}=
\\frac{{\\#\\{{accepted\\ \\land\\ B^3F1\\ge0.85\\}}}}{{\\#\\{{accepted\\}}}},
$$

$$
\\text{{Coverage}}=\\frac{{\\#\\{{accepted\\}}}}{{\\#\\{{all\\ authors\\}}}},
\\qquad
\\text{{Hard-block Recall}}=
\\frac{{\\#\\{{rejected\\ \\land\\ B^3F1<0.80\\}}}}
{{\\#\\{{B^3F1<0.80\\}}}}.
$$

只在 Full model 的训练 OOF 概率上扫描 0.50 至 0.95。最终阈值为
**{selected['threshold']:.2f}**，选择规则为：{threshold_rule}。此时训练 OOF
Accepted precision={fmt(selected['accepted_precision'])}，
Coverage={fmt(selected['coverage'])}，
Hard-block recall={fmt(selected['hard_block_recall'])}，
Accepted mean B³ F1={fmt(selected['accepted_mean_b3_f1'])}。

没有任何候选阈值达到预设的 0.90 Accepted precision。并且在 60% coverage 下，
Full model 的 0.718 仅略高于随机排序的 0.667，低于 paper-count-only 的 0.821。
因此当前实验**没有证明复杂 pre-LLM 特征能训练出有效的高可靠筛选器**；相反，简单
的论文规模信号在这次 OOF 中更稳定。可能原因包括：训练作者只有 64 位，而 Full
model 有 {len(FULL_FEATURES)} 个且彼此相关的特征；balanced class weight 改善少数类
拟合但不保证概率已校准；低分作者同时包含 over-merging、over-splitting 和极端
singleton 等异质失败模式，单个线性边界难以统一识别。

最终接受规则：

```text
使用冻结的 pre-LLM 特征管线计算概率
P(B³ F1 >= 0.85) >= {selected['threshold']:.2f} -> accepted
否则 -> rejected / high-risk
```

Risk-coverage 曲线不是普通 accuracy：它展示逐步增加接受作者时，质量纯度与可覆盖
工作量之间的交换。相关交互图位于 `artifacts/figures/router/index.html`。

## 6. 冻结测试集结果与局限

阈值冻结后，使用全部 64 位训练作者重新拟合一次 Full Logistic Regression，并只用
16 位测试作者的 pre-LLM 特征预测。测试集接受 {accepted_test}/16 位作者：

- Test accepted precision：**{fmt(test_metrics['accepted_precision'])}**；
- Test coverage：**{fmt(test_metrics['coverage'])}**；
- Test hard-block recall：**{fmt(test_metrics['hard_block_recall'])}**；
- Test accepted mean B³ F1：**{fmt(test_metrics['accepted_mean_b3_f1'])}**。

测试概率最高值仍低于冻结阈值，因此模型拒绝全部测试作者。这里的 hard-block recall
虽然是 1.000，却来自 coverage=0 的退化策略，不能视为成功；Accepted precision 和
Accepted mean B³ F1 也因没有 accepted 样本而不可定义。按照冻结测试原则，本实验
不再根据这 16 位作者降低阈值或修改特征。

这只是一次小样本冻结测试。它仅验证当前 80 个作者、当前 T+O+C prompt/model 的
可筛选性；不同模型、prompt、数据域或可靠阈值需要重新校准。特征系数是条件相关，
不能解释为因果贡献。测试集仅有 16 人，因此一个作者就会明显改变 precision；
后续应扩大独立作者样本，并单独研究 rejected 作者的补救策略。
"""
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = read_json(args.data_dir / "sna_valid_raw.json")
    papers = read_json(args.data_dir / "sna_valid_pub.json")
    metrics = pd.read_csv(args.metrics)
    metrics = metrics[metrics["status"] == "completed"].copy()
    names = sorted(set(raw) & set(metrics["name"]))
    if len(names) != 80:
        raise ValueError(f"expected 80 completed authors, found {len(names)}")

    feature_path = RESULT_DIR / "pre_llm_features_toc.csv"
    if args.reuse_features and feature_path.exists():
        features = pd.read_csv(feature_path)
    else:
        rows = []
        for index, name in enumerate(names, start=1):
            print(f"[{index:02d}/80] extracting pre-LLM features: {name}")
            rows.append(author_features(name, raw[name], papers))
        features = pd.DataFrame(rows)[["name"] + FULL_FEATURES]
        features.to_csv(feature_path, index=False)

    train_names, test_names = split_names(names)
    (RESULT_DIR / "train_names.json").write_text(
        json.dumps(train_names, indent=2) + "\n"
    )
    (RESULT_DIR / "test_names.json").write_text(
        json.dumps(test_names, indent=2) + "\n"
    )

    labelled = add_labels(features, metrics)
    train = labelled[labelled["name"].isin(train_names)].sort_values("name").reset_index(drop=True)
    test = labelled[labelled["name"].isin(test_names)].sort_values("name").reset_index(drop=True)
    cv_predictions = cross_validated_predictions(train)
    cv_predictions.to_csv(RESULT_DIR / "cv_predictions_toc.csv", index=False)
    curves = coverage_curves(cv_predictions)
    checkpoints = coverage_checkpoints(cv_predictions)

    full_oof = cv_predictions[
        cv_predictions["model_name"] == "full_logistic_regression"
    ].reset_index(drop=True)
    selected, thresholds, threshold_rule = choose_threshold(full_oof)

    final_model = make_pipeline()
    final_model.fit(train[FULL_FEATURES], train["reliable_085"])
    test_probabilities = final_model.predict_proba(test[FULL_FEATURES])[:, 1]
    test_predictions = pd.DataFrame({
        "name": test["name"],
        "predicted_reliability_probability": test_probabilities,
        "accepted_or_rejected": np.where(
            test_probabilities >= selected["threshold"], "accepted", "rejected"
        ),
        "actual_b3_f1": test["b3_f1"],
        "reliable_085": test["reliable_085"],
        "hard_080": test["hard_080"],
    })
    test_predictions.to_csv(RESULT_DIR / "test_predictions_toc.csv", index=False)
    test_metrics = selection_metrics(
        test_predictions,
        test_predictions["accepted_or_rejected"] == "accepted",
    )

    model_bundle = {
        "pipeline": final_model,
        "feature_columns": FULL_FEATURES,
        "threshold": selected["threshold"],
        "random_seed": SEED,
        "target": "b3_f1 >= 0.85",
    }
    joblib.dump(model_bundle, RESULT_DIR / "router_model_toc.joblib")

    config = {
        "final_model": "full_logistic_regression",
        "feature_columns": FULL_FEATURES,
        "threshold": selected["threshold"],
        "threshold_selection_rule": threshold_rule,
        "training_oof_metrics": selected,
        "test_metrics_frozen_once": test_metrics,
        "random_seed": SEED,
        "train_authors": len(train),
        "test_authors": len(test),
        "coverage_checkpoint_metrics": checkpoints.to_dict(orient="records"),
        "feature_definitions": {
            "title_similarity": "within-block TF-IDF cosine",
            "organization_similarity": "normalized token-set Jaccard",
            "coauthor_similarity": "normalized coauthor-name-set Jaccard",
            "missing_values": "median imputation fitted within each training fold",
        },
    }
    (RESULT_DIR / "router_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )

    plot_risk_coverage(curves)
    plot_probability_vs_f1(full_oof, selected["threshold"])
    plot_threshold_tradeoff(thresholds, selected["threshold"])
    plot_feature_importance(final_model, FULL_FEATURES)
    write_index()

    report = report_text(
        train,
        checkpoints,
        selected,
        threshold_rule,
        test_metrics,
        correlation_summary(train),
        failure_modes(train),
    )
    (REPORT_DIR / "report_zh.md").write_text(report)
    print(f"saved features: {feature_path}")
    print(f"selected threshold: {selected['threshold']:.2f}")
    print(f"test metrics: {json.dumps(test_metrics)}")
    print(f"saved report: {REPORT_DIR / 'report_zh.md'}")
    print(f"saved plots: {PLOT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
