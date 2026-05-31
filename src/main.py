import argparse

from pipeline import run_snd_clustering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub", default="dataset/data/NA_Demo/SND/valid/sna_valid_pub.json")
    parser.add_argument("--raw", default="dataset/data/NA_Demo/SND/valid/sna_valid_raw.json")
    parser.add_argument("--truth", default="dataset/data/NA_Demo/SND/valid/sna_valid_ground_truth.json")
    parser.add_argument("--out-dir", default="result")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--llm-max-new-tokens", type=int, default=8)
    parser.add_argument("--max-papers-per-name", type=int, default=None)
    args = parser.parse_args()

    summary = run_snd_clustering(
        pub_path=args.pub,
        raw_path=args.raw,
        truth_path=args.truth,
        out_dir=args.out_dir,
        threshold=args.threshold,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_max_new_tokens=args.llm_max_new_tokens,
        max_papers_per_name=args.max_papers_per_name,
    )

    print(f"clusters: {summary['clusters']}")
    print(f"papers: {summary['papers']}")
    print(f"wrote: {summary['result_path']}")
    print(f"wrote: {summary['detail_path']}")


if __name__ == "__main__":
    main()
