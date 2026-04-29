"""Train one small LM per tokenizer artifact and write llm_results.csv.

This is the *downstream* tokenizer evaluation: train a fixed-architecture
~50M-param GPT with each tokenizer (held-fixed token budget per the spec),
then measure held-out perplexity + bits-per-byte. BPB is the
cross-tokenizer-comparable metric — raw perplexity isn't.

Run after ``generate_tokenizers.py``. Requires the ``llm`` extras:

    pip install -e ".[llm]"

Edit the constants below to change the run.
"""

from pathlib import Path

from src.tools.train_llm import train_all_llms
from src.utils.llm_training import LLMConfig

# --- config ----------------------------------------------------------------
DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
RESULTS_PATH = Path("llm_results.csv")
CONTINUE_ON_ERROR = True

# Which artifacts to include. None = all that exist under ARTIFACT_DIR.
ONLY_LANGUAGES: list[str] | None = None      # e.g. ["en", "tr"]
ONLY_ALGORITHMS: list[str] | None = None     # e.g. ["bpe", "tiktoken"]
ONLY_VOCAB_SIZES: list[int] | None = None    # e.g. [32_000]

# LM training/architecture config. Defaults: ~50M params at 32k vocab,
# 50M training tokens. Tune down for smoke tests, up for the real sweep.
LLM_CONFIG = LLMConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    ctx_len=512,
    train_tokens=50_000_000,
    batch_size=32,
    learning_rate=3e-4,
)

if __name__ == "__main__":
    train_all_llms(
        data_dir=DATA_DIR,
        artifact_dir=ARTIFACT_DIR,
        results_path=RESULTS_PATH,
        config=LLM_CONFIG,
        continue_on_error=CONTINUE_ON_ERROR,
        only_languages=ONLY_LANGUAGES,
        only_algorithms=ONLY_ALGORITHMS,
        only_vocab_sizes=ONLY_VOCAB_SIZES,
    )
