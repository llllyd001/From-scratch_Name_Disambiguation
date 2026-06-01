from pathlib import Path

from data_io import dump_json, load_json, raw_candidate_sets
from LLM import CandidateLLM
from .analysis import build_cluster_analysis
from .candidates import collect_candidates
from .clustering import apply_cluster_merges, initial_profile_clusters, summarize_cluster
from .profiles import build_paper_profile, fill_missing_broad_topics


def run_pipeline(
    pub_path,
    raw_path,
    out_dir,
    llm_model=None,
    llm_max_tokens=2048,
    author_number=None,
    max_papers_per_name=None,
    merge_threshold=0.75,
):
    papers = load_json(pub_path)
    raw = load_json(raw_path)
    ground_truth_path = Path(raw_path).with_name("sna_valid_ground_truth.json")
    ground_truth = load_json(ground_truth_path) if ground_truth_path.exists() else None
    llm = CandidateLLM(model_name=llm_model, max_tokens=llm_max_tokens)
    profile_output = {}
    result = {}
    cluster_analysis = {}

    for author_index, (target_key, paper_ids) in enumerate(raw_candidate_sets(raw)):
        if author_number is not None and author_index >= author_number:
            break
        paper_ids = [pid for pid in paper_ids if pid in papers]
        if max_papers_per_name is not None:
            original_count = len(paper_ids)
            paper_ids = paper_ids[:max_papers_per_name]
            if len(paper_ids) < original_count:
                print(
                    f"Limited {target_key}: {len(paper_ids)}/{original_count} papers. "
                    "This output is for smoke tests, not full-ground-truth F1.",
                    flush=True,
                )

        print(f"Building scholar profiles for {target_key}: {len(paper_ids)} papers", flush=True)
        candidates = collect_candidates(llm, papers, paper_ids, target_key)
        profiles = {
            pid: build_paper_profile(pid, papers[pid], target_key, candidates)
            for pid in paper_ids
        }
        fill_missing_broad_topics(llm, profiles, candidates)
        initial_clusters = initial_profile_clusters(profiles)
        summaries = [
            summarize_cluster(f"c{index}", cluster, profiles, candidates)
            for index, cluster in enumerate(initial_clusters, start=1)
        ]
        merges = llm.suggest_cluster_merges(target_key, summaries, merge_threshold)
        groups = apply_cluster_merges(summaries, merges)

        profile_output[target_key] = {
            "candidates": candidates,
            "papers": profiles,
            "initial_clusters": summaries,
            "llm_merges": merges,
        }
        result[target_key] = groups
        cluster_analysis[target_key] = build_cluster_analysis(
            target_key,
            groups,
            papers,
            profiles,
            ground_truth,
        )
        print(f"Finished {target_key}: {len(initial_clusters)} -> {len(groups)} clusters", flush=True)

    out_dir = Path(out_dir)
    profile_path = out_dir / "sna_valid_scholar_profiles.json"
    result_path = out_dir / "sna_valid_result.json"
    analysis_path = out_dir / "sna_valid_cluster_analysis.json"
    dump_json(profile_output, profile_path)
    dump_json(result, result_path)
    dump_json(cluster_analysis, analysis_path)
    return {
        "targets": len(profile_output),
        "profile_path": str(profile_path),
        "result_path": str(result_path),
        "analysis_path": str(analysis_path),
    }
