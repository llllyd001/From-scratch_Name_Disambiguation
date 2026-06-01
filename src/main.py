import argparse

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pub", default="dataset/data/NA_Demo/SND/valid/sna_valid_pub.json")
    parser.add_argument("--raw", default="dataset/data/NA_Demo/SND/valid/sna_valid_raw.json")
    parser.add_argument("--out-dir", default="result")
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--author-number", type=int, default=None)
    parser.add_argument("--max-papers-per-name", type=int, default=None)
    parser.add_argument("--merge-threshold", type=float, default=0.75)
    args = parser.parse_args()

    summary = run_pipeline(
        pub_path=args.pub,
        raw_path=args.raw,
        out_dir=args.out_dir,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
        author_number=args.author_number,
        max_papers_per_name=args.max_papers_per_name,
        merge_threshold=args.merge_threshold,
    )

    print(f"targets: {summary['targets']}")
    print(f"profiles: {summary['profile_path']}")
    print(f"result: {summary['result_path']}")
    print(f"analysis: {summary['analysis_path']}")


if __name__ == "__main__":
    main()
