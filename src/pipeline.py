from pathlib import Path

from data_io import dump_json, ground_truth_index, load_json, raw_candidate_sets
from LLM import OneShotLLMClusterer


def run_snd_clustering(
    pub_path,
    raw_path,
    truth_path,
    out_dir,
    llm_model=None,
    llm_max_new_tokens=512,
    max_papers_per_name=None,
    author_number=None,
    max_retries=5,
    retry_delay=30.0,
):
    papers = load_json(pub_path)
    raw = load_json(raw_path)
    truth = ground_truth_index(load_json(truth_path))
    clusterer = OneShotLLMClusterer(
        model_name=llm_model,
        max_new_tokens=llm_max_new_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    result = {}
    cluster_analysis = {}

    for author_index, (target_key, paper_ids) in enumerate(raw_candidate_sets(raw)):
        if author_number is not None and author_index >= author_number:
            break

        paper_ids = [paper_id for paper_id in paper_ids if paper_id in papers]
        if max_papers_per_name is not None:
            paper_ids = paper_ids[:max_papers_per_name]

        print(f"Clustering {target_key}: {len(paper_ids)} papers", flush=True)
        groups = clusterer.cluster_target(papers, paper_ids, target_key)
        print(f"Finished {target_key}: {len(groups)} clusters", flush=True)
        result[target_key] = groups
        cluster_analysis[target_key] = {}

        for group_index, group in enumerate(groups):
            cluster_key = f"{target_key}_{group_index:04d}"
            cluster_analysis[target_key][cluster_key] = {}
            for paper_id in group:
                item = dict(papers[paper_id])
                item["predicted_cluster_key"] = cluster_key
                item["target_name"] = target_key
                item.update(truth.get(paper_id, {
                    "true_author_index": None,
                    "true_author_key": None,
                }))

                true_key = item["true_author_key"] or "unknown_true_author"
                if true_key not in cluster_analysis[target_key][cluster_key]:
                    cluster_analysis[target_key][cluster_key][true_key] = {
                        "true_author_index": item["true_author_index"],
                        "papers": [],
                    }
                cluster_analysis[target_key][cluster_key][true_key]["papers"].append(item)

    out_dir = Path(out_dir)
    result_path = out_dir / "sna_valid_result.json"
    analysis_path = out_dir / "sna_valid_cluster_analysis.json"

    dump_json(result, result_path)
    dump_json(cluster_analysis, analysis_path)

    return {
        "clusters": sum(len(groups) for groups in result.values()),
        "papers": sum(len(group) for groups in result.values() for group in groups),
        "result_path": str(result_path),
        "analysis_path": str(analysis_path),
    }
