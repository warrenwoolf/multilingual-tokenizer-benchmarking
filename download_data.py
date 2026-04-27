"""Stream FineWeb / FineWeb 2 corpora for each language.

Edit the constants below to change the run. Outputs:
    data/{lang}/train.txt
    data/{lang}/eval.txt
"""

from pathlib import Path

from src.tools.download_data import download_all_languages

# --- config ----------------------------------------------------------------
LANGUAGES = ["en", "zh", "tr"]
TRAIN_BUDGET_MB = 500.0
EVAL_BUDGET_MB = 25.0
DATA_DIR = Path("data")
SKIP_VERIFY = False  # set True to skip the fineweb-2 config name check

if __name__ == "__main__":
    download_all_languages(
        languages=LANGUAGES,
        data_dir=DATA_DIR,
        train_budget_mb=TRAIN_BUDGET_MB,
        eval_budget_mb=EVAL_BUDGET_MB,
        skip_verify=SKIP_VERIFY,
    )
