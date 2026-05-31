from pathlib import Path

from cluster import cluster_one_target
from data_io import dump_json, ground_truth_index, load_json, raw_candidate_sets
from LLM import LocalLLMJudge


def run_snd_clustering(
    pub_path,
    raw_path,
    truth_path,
    out_dir,
    threshold,
    use_llm=False,
    llm_model=None,
    llm_max_new_tokens=8,
    max_papers_per_name=None,
):
    papers = load_json(pub_path)
    raw = load_json(raw_path)
    truth = ground_truth_index(load_json(truth_path))
    llm_judge = LocalLLMJudge(
        model_name=llm_model,
        max_new_tokens=llm_max_new_tokens,
        enabled=use_llm,
    )

    result = {}
    details = {}

    for target_key, paper_ids in raw_candidate_sets(raw):
        paper_ids = [paper_id for paper_id in paper_ids if paper_id in papers]
        if max_papers_per_name is not None:
            paper_ids = paper_ids[:max_papers_per_name]

        groups = cluster_one_target(
            papers=papers,
            paper_ids=paper_ids,
            target_key=target_key,
            threshold=threshold,
            llm_judge=llm_judge,
        )
        result[target_key] = groups

        for group_index, group in enumerate(groups):
            cluster_key = f"{target_key}_{group_index:04d}"
            for paper_id in group:
                item = dict(papers[paper_id])
                item["predicted_cluster_key"] = cluster_key
                item["target_name"] = target_key
                item.update(truth.get(paper_id, {
                    "true_author_index": None,
                    "true_author_key": None,
                }))
                details[paper_id] = item

    out_dir = Path(out_dir)
    result_path = out_dir / "sna_valid_result.json"
    detail_path = out_dir / "sna_valid_pub_with_author.json"

    dump_json(result, result_path)
    dump_json(details, detail_path)

    return {
        "clusters": sum(len(groups) for groups in result.values()),
        "papers": len(details),
        "result_path": str(result_path),
        "detail_path": str(detail_path),
    }
