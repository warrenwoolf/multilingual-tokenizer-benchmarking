#@title MorphBPE English 32k — case-sensitivity fix validation run
# ---------------------------------------------------------------------------
# Paste this entire file into a Google Colab cell and run it.
# Trains ONLY MorphBPE on English with a 32k vocab using the
# fix/morphbpe-case-sensitivity branch, then runs intrinsic evaluation.
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

RUN_LLM_EVAL = False
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

# 3. Install dependencies
print("[colab] Installing dependencies...", flush=True)
!pip install -q -e .

# 4. Run pipeline
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
"""
with open("/tmp/run_pipeline.py", "w") as fh:
    fh.write(shim)
!python /tmp/run_pipeline.py

# 5. Display results
import pandas as pd
if os.path.exists("results.csv"):
    df = pd.read_csv("results.csv")
    print(df.to_string(index=False))
else:
    df = None
    print("[colab] results.csv not found.")

df
