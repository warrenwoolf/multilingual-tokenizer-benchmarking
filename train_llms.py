"""Train one small LM per tokenizer artifact and write llm_results.csv.

This is the *downstream* tokenizer evaluation: train a fixed-architecture
~50M-param GPT with each tokenizer (training-token budget held fixed per spec),
then score held-out PPL + bits-per-byte on two eval sets:
  - the in-domain FineWeb test split, and
  - FLORES-200 devtest (out-of-distribution).

Each LM is monolingual.

Run after ``generate_tokenizers.py``. Requires the ``llm`` extras:

    pip install -e ".[llm]"

Optional Weights & Biases logging: set ``WANDB_PROJECT`` below (or drop a key
into ``tokens/wandb.token`` / set ``WANDB_API_KEY``). One W&B run per
(language, algorithm, vocab_size) combo. By default each run also uploads
the tokenizer dir and the trained model state_dict as W&B Artifacts — turn
``wandb_log_*_artifact`` off in ``LLM_CONFIG`` for large sweeps if storage
is a concern (each model is ~200 MB at 50M params).

The output CSV includes per-run row counts and a ``train_bytes_per_row``
column — sanity-check it against your expected per-language byte size to
make sure the data slice is what you think it is.

Edit the constants below to change the run.
"""

import os
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

# Eval FLORES-200 in addition to the test set. Needs network on first run
# (HF dataset is cached afterwards).
EVAL_FLORES = True

# W&B: None disables logging entirely. Otherwise the named project is created
# in your default W&B account; entity overrides the account if needed.
WANDB_PROJECT: str | None = os.environ.get("WANDB_PROJECT")  # e.g. "tokenizer-bench"
WANDB_ENTITY: str | None = os.environ.get("WANDB_ENTITY")

# LM training/architecture config. Defaults: ~50M params at 32k vocab, 1B
# training tokens (~Chinchilla-optimal for this size; ~65 min on A100 40GB).
# Tune `train_tokens` down for smoke tests / a sweep that won't ruin the cluster.
LLM_CONFIG = LLMConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    ctx_len=512,
    train_tokens=1_000_000_000,
    batch_size=32,
    learning_rate=3e-4,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
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
        eval_flores=EVAL_FLORES,
    )
