#@title Multilingual Tokenizer Benchmarking — Colab quickstart
# ---------------------------------------------------------------------------
# Paste this entire file into a Google Colab cell and run it. Uses Colab's
# IPython-style ``!`` for shell commands, so it's not valid as a plain
# ``python colab.py`` invocation — it's a Colab cell, not a script.
#
# Steps performed:
#   1. Clone the repo (skipped if already present)
#   2. Install dependencies in editable mode
#   3. Stream FineWeb / FineWeb 2 corpora to data/
#   4. Train tokenizer artifacts to artifacts/
#   5. Evaluate and write results.csv
#   6. Display results as a pandas DataFrame
#
# Tweak via the SETTINGS block. Defaults are conservative for free-tier Colab
# (small budget so a full run finishes in roughly 10 minutes).
# ---------------------------------------------------------------------------

# === SETTINGS ==============================================================
REPO_URL = "https://github.com/warrenwoolf/multilingual-tokenizer-benchmarking.git"
BRANCH = "main" # Change this if using an unmerged branch
REPO_DIR = "multilingual-tokenizer-benchmarking"

LANGUAGES = "en,zh,tr"
ALGORITHMS = "bpe,superbpe,tiktoken,morphbpe,wordpiece,unigram,byt5"
VOCAB_SIZES = "8000,16000,32000"                    # 64000 added if budget permits
MAX_TRAIN_ROWS = 100_000                            # rows per language for a quick run
MAX_EVAL_ROWS = 5_000

# Downstream LLM evaluation (heavy; needs a GPU runtime + extra deps).
# Flip ON only on a GPU runtime; defaults are tuned for a smoke run.
RUN_LLM_EVAL = False
LLM_TRAIN_TOKENS = 50_000_000      # 1B for the real Chinchilla-optimal sweep
WANDB_PROJECT = "tokenizer-bench"  # set to None to disable W&B logging
# ===========================================================================

import os
import sys

# 1. Clone the repo
if not os.path.isdir(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# 2. Persist HuggingFace token from Colab Secrets into tokens/hf_token
try:
    from google.colab import userdata as _userdata
    _hf_token = _userdata.get('HF_TOKEN')
    if _hf_token:
        with open('tokens/hf_token', 'w') as _fh:
            _fh.write(_hf_token)
        os.environ['HF_TOKEN'] = _hf_token
except Exception:
    pass  # not running in Colab or secret not set

# 3. Install dependencies (editable so any tweaks take effect immediately)
!pip install -q -e .
if RUN_LLM_EVAL:
    !pip install -q -e ".[llm]" wandb

# 2b. Load the W&B API key from Colab Secrets if available, into tokens/wandb.token.
#     Add a secret named WANDB_API_KEY in the Colab "key" sidebar before running.
if RUN_LLM_EVAL and WANDB_PROJECT:
    try:
        from google.colab import userdata  # type: ignore
        wandb_key = userdata.get("WANDB_API_KEY")
    except Exception:
        wandb_key = None
    if wandb_key:
        os.makedirs("tokens", exist_ok=True)
        with open("tokens/wandb.token", "w") as fh:
            fh.write(wandb_key.strip())
        os.environ["WANDB_API_KEY"] = wandb_key.strip()
        print("Loaded WANDB_API_KEY from Colab secrets.")
    else:
        print(
            "WANDB_API_KEY not set in Colab secrets — W&B logging will be skipped. "
            "Add it via the 'key' icon in the Colab sidebar to enable."
        )
        WANDB_PROJECT = None

# 4. Override the per-script config via env-var-friendly Python overrides.
#    Each script is config-only at the top, so a tiny shim keeps the run
#    parameterized without editing the file in the repo.
shim = f"""
import sys
sys.path.insert(0, '.')

# --- download ---
from src.tools.download_data import download_all_languages
download_all_languages(
    languages={LANGUAGES.split(',')!r},
    data_dir='data',
    max_train_rows={MAX_TRAIN_ROWS},
    max_eval_rows={MAX_EVAL_ROWS},
)

# --- train ---
from src.tools.create_tokenizer import generate_all_tokenizers
generate_all_tokenizers(
    languages={LANGUAGES.split(',')!r},
    algorithms={ALGORITHMS.split(',')!r},
    vocab_sizes=[int(v) for v in {VOCAB_SIZES.split(',')!r}],
    data_dir='data',
    artifact_dir='artifacts',
    continue_on_error=True,
)

# --- evaluate (intrinsic metrics) ---
from src.tools.evaluate_tokenizer import evaluate_all_tokenizers
evaluate_all_tokenizers(
    data_dir='data',
    artifact_dir='artifacts',
    results_path='results.csv',
    continue_on_error=True,
)

# --- evaluate (downstream LLM PPL/BPB) ---
if {RUN_LLM_EVAL}:
    from src.tools.train_llm import train_all_llms
    from src.utils.llm_training import LLMConfig
    cfg = LLMConfig(
        train_tokens={LLM_TRAIN_TOKENS},
        wandb_project={WANDB_PROJECT!r},
    )
    train_all_llms(
        data_dir='data',
        artifact_dir='artifacts',
        results_path='llm_results.csv',
        config=cfg,
        continue_on_error=True,
        eval_flores=True,
    )
"""
with open("/tmp/run_pipeline.py", "w") as fh:
    fh.write(shim)
!python /tmp/run_pipeline.py

# 5. Display results
import pandas as pd
df = pd.read_csv("results.csv")
print(df.to_string(index=False))

# Also display LLM results if the downstream eval ran. Look at
# `train_bytes_per_row` to sanity-check the per-language data slice.
if RUN_LLM_EVAL and os.path.exists("llm_results.csv"):
    print("\n=== LLM results (PPL + BPB) ===")
    llm_df = pd.read_csv("llm_results.csv")
    print(llm_df.to_string(index=False))

df  # last expression renders as a Colab table
