#@title MorphBPE English 32k — case-sensitivity fix validation run
# ---------------------------------------------------------------------------
# Paste this entire file into a Google Colab cell and run it. Requires a GPU
# runtime for the LLM training step.
# Trains ONLY MorphBPE on English with a 32k vocab using the
# fix/morphbpe-case-sensitivity branch, runs intrinsic evaluation, then
# trains the downstream LLM and reports BPB.
# ---------------------------------------------------------------------------

# === SETTINGS ==============================================================
REPO_URL = "https://github.com/warrenwoolf/multilingual-tokenizer-benchmarking.git"
BRANCH = "fix/morphbpe-case-sensitivity"
REPO_DIR = "multilingual-tokenizer-benchmarking"

LANGUAGES = "en"
ALGORITHMS = "morphbpe"
VOCAB_SIZES = "32000"
MAX_TRAIN_ROWS = 100_000
MAX_EVAL_ROWS = 5_000

LLM_TRAIN_TOKENS = 50_000_000      # 1B for a full Chinchilla-optimal run
WANDB_PROJECT = "tokenizer-bench"  # set to None to disable W&B logging
# ===========================================================================

import os
import sys

# 1. Clone the repo
if not os.path.isdir(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}

# 2. Persist HuggingFace token from Colab Secrets
try:
    from google.colab import userdata as _userdata
    _hf_token = _userdata.get('HF_TOKEN')
    if _hf_token:
        with open('tokens/hf_token', 'w') as _fh:
            _fh.write(_hf_token)
        os.environ['HF_TOKEN'] = _hf_token
except Exception:
    pass

# 3. Load W&B key from Colab Secrets
_wandb_project = WANDB_PROJECT
if _wandb_project:
    try:
        from google.colab import userdata as _userdata
        _wandb_key = _userdata.get('WANDB_API_KEY')
    except Exception:
        _wandb_key = None
    if _wandb_key:
        os.makedirs("tokens", exist_ok=True)
        with open("tokens/wandb.token", "w") as fh:
            fh.write(_wandb_key.strip())
        os.environ["WANDB_API_KEY"] = _wandb_key.strip()
        print("Loaded WANDB_API_KEY from Colab secrets.")
    else:
        print("WANDB_API_KEY not set — W&B logging disabled.")
        _wandb_project = None

# 4. Install dependencies
print("[colab] Installing dependencies...", flush=True)
!pip install -q -e ".[llm]" wandb

# 5. Run pipeline
shim = f"""
import sys
sys.path.insert(0, '.')

from src.tools.download_data import download_all_languages
download_all_languages(
    languages={LANGUAGES.split(',')!r},
    data_dir='data',
    max_train_rows={MAX_TRAIN_ROWS},
    max_eval_rows={MAX_EVAL_ROWS},
)

from src.tools.create_tokenizer import generate_all_tokenizers
generate_all_tokenizers(
    languages={LANGUAGES.split(',')!r},
    algorithms={ALGORITHMS.split(',')!r},
    vocab_sizes=[int(v) for v in {VOCAB_SIZES.split(',')!r}],
    data_dir='data',
    artifact_dir='artifacts',
    continue_on_error=False,
    evaluate_each=True,
)

from src.tools.evaluate_tokenizer import evaluate_all_tokenizers
evaluate_all_tokenizers(
    data_dir='data',
    artifact_dir='artifacts',
    results_path='results.csv',
    continue_on_error=False,
)

from src.tools.train_llm import train_all_llms
from src.utils.llm_training import LLMConfig
cfg = LLMConfig(
    train_tokens={LLM_TRAIN_TOKENS},
    wandb_project={_wandb_project!r},
)
train_all_llms(
    data_dir='data',
    artifact_dir='artifacts',
    results_path='llm_results.csv',
    config=cfg,
    continue_on_error=False,
    eval_flores=True,
)
"""
with open("/tmp/run_pipeline.py", "w") as fh:
    fh.write(shim)
!python /tmp/run_pipeline.py

# 6. Display results
import pandas as pd
if os.path.exists("results.csv"):
    print("=== Intrinsic metrics ===")
    df = pd.read_csv("results.csv")
    print(df.to_string(index=False))
else:
    df = None
    print("[colab] results.csv not found.")

if os.path.exists("llm_results.csv"):
    print("\n=== LLM results (BPB) ===")
    llm_df = pd.read_csv("llm_results.csv")
    print(llm_df.to_string(index=False))

df
