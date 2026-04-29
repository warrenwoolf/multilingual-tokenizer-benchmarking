"""Train tokenizer artifacts for every (language, algorithm, vocab_size) triple.

Run after ``download_data.py``. Edit the constants below to change the run.
Outputs ``artifacts/{lang}_{algorithm}_v{vocab_size}/tokenizer.json``.
"""

from pathlib import Path

from src.tools.create_tokenizer import generate_all_tokenizers

# --- config ----------------------------------------------------------------
# v1 paper focus: BPE, SuperBPE, tiktoken-style, and (probably) MorphBPE.
# WordPiece / Unigram / ByT5 are kept available as fallback / baselines.
LANGUAGES = ["en", "zh", "tr"]
ALGORITHMS = ["bpe", "superbpe", "tiktoken", "morphbpe"]
# MorphBPE only runs for languages with a configured morpheme segmenter
# (currently {"en", "tr"}). Mandarin is super-analytic and out of scope —
# (zh, morphbpe) is filtered automatically by ``iter_jobs``.
VOCAB_SIZES = [8_000, 16_000, 32_000, 64_000]
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
CONTINUE_ON_ERROR = False  # True = log + continue when one combo fails

if __name__ == "__main__":
    generate_all_tokenizers(
        languages=LANGUAGES,
        algorithms=ALGORITHMS,
        vocab_sizes=VOCAB_SIZES,
        data_dir=DATA_DIR,
        artifact_dir=ARTIFACT_DIR,
        continue_on_error=CONTINUE_ON_ERROR,
    )
