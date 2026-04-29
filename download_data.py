"""Stream FineWeb / FineWeb 2 corpora for each language.

Edit the constants below to change the run. Outputs:
    data/{lang}/train.txt
    data/{lang}/eval.txt
"""

from pathlib import Path

from src.tools.download_data import download_all_languages

# --- config ----------------------------------------------------------------
LANGUAGES = ["en", "zh", "tr"]
MAX_TRAIN_ROWS = 500_000
MAX_EVAL_ROWS = 25_000
DATA_DIR = Path("data")
SKIP_VERIFY = False  # set True to skip the fineweb-2 config name check

if __name__ == "__main__":
    download_all_languages(
        languages=LANGUAGES,
        data_dir=DATA_DIR,
        max_train_rows=MAX_TRAIN_ROWS,
        max_eval_rows=MAX_EVAL_ROWS,
        skip_verify=SKIP_VERIFY,
    )
