"""Evaluate every trained tokenizer artifact and write results.csv.

Run after ``generate_tokenizers.py``. Edit the constants below to change
the run.
"""

from pathlib import Path

from src.tools.evaluate_tokenizer import evaluate_all_tokenizers

# --- config ----------------------------------------------------------------
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
RESULTS_PATH = Path("results.csv")
CONTINUE_ON_ERROR = False

if __name__ == "__main__":
    evaluate_all_tokenizers(
        data_dir=DATA_DIR,
        artifact_dir=ARTIFACT_DIR,
        results_path=RESULTS_PATH,
        continue_on_error=CONTINUE_ON_ERROR,
    )
