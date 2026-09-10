from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "whoiswho" / "data"
RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "results"
FIGURES_ROOT = PROJECT_ROOT / "artifacts" / "figures"
REPORTS_ROOT = PROJECT_ROOT / "docs" / "reports"


def dataset_dir(dataset="NA_Demo", split="valid"):
    return DATA_ROOT / dataset / "SND" / split
