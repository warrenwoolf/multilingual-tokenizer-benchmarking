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
BRANCH = "claude/setup-paper-codebase-aKac5"  # change to "main" after merge
REPO_DIR = "multilingual-tokenizer-benchmarking"

LANGUAGES = "en,zh,tr"
ALGORITHMS = "bpe,tiktoken,morphbpe,wordpiece,unigram,byt5"  # SuperBPE needs extra setup
VOCAB_SIZES = "8000,16000,32000"                    # 64000 added if budget permits
TRAIN_BUDGET_MB = 100                               # ~100 MB per language for a quick run
EVAL_BUDGET_MB = 5
# ===========================================================================

import os
import sys

# 1. Clone the repo
if not os.path.isdir(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# 2. Install dependencies (editable so any tweaks take effect immediately).
#    morfessor is pulled in transitively via pyproject.toml, but listing it
#    explicitly here makes the MorphBPE dependency obvious.
!pip install -q -e . morfessor

# 3. Override the per-script config via env-var-friendly Python overrides.
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
    train_budget_mb={TRAIN_BUDGET_MB},
    eval_budget_mb={EVAL_BUDGET_MB},
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

# --- evaluate ---
from src.tools.evaluate_tokenizer import evaluate_all_tokenizers
evaluate_all_tokenizers(
    data_dir='data',
    artifact_dir='artifacts',
    results_path='results.csv',
    continue_on_error=True,
)
"""
with open("/tmp/run_pipeline.py", "w") as fh:
    fh.write(shim)
!python /tmp/run_pipeline.py

# 4. Display results
import pandas as pd
df = pd.read_csv("results.csv")
print(df.to_string(index=False))
df  # last expression renders as a Colab table
