"""Train MAGNET models and evaluate bits-per-byte.

MAGNET (Ahia et al. 2024) trains tokenisation end-to-end with the LM:
a boundary predictor learns where to split byte sequences into segments,
and the resulting variable-length segments are processed by a compressed
transformer stack.  The boundary predictor is trained via a Binomial prior
that controls the target compression rate per language/script.

BPB is the only metric reported — it is the cross-tokeniser-comparable
figure for byte-level models (BPB = mean_CE_nats / ln(2)).

Usage
-----
    python train_magnet.py

Prerequisites:
    pip install torch transformers datasets   # or: make install-llm
    python download_data.py                   # data/{lang}/train.txt + eval.txt

Outputs:
    magnet_results.csv — one row per language with test_bits_per_byte
                         (and flores_bits_per_byte if FLORES is reachable).
"""

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these to change the sweep
# ---------------------------------------------------------------------------

LANGUAGES = ["en", "zh", "tr"]

DATA_DIR = Path("data")
RESULTS_PATH = Path("magnet_results.csv")

# Set to True to also evaluate on FLORES-200 devtest (requires network).
EVAL_FLORES = False

# MagnetConfig overrides — set to None to use defaults from MagnetConfig.
MODEL_OVERRIDES: dict = {
    # Architecture
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 1024,
    "pre_layers": 4,
    "shortened_layers": 4,
    "post_layers": 0,
    "ctx_len": 512,
    # Boundary predictor — prior controls compression (0.25 → ~4× compression)
    "boundary_prior": 0.25,
    "boundary_temp": 1.0,
    "boundary_lambda": 1.0,
    # Training
    "train_tokens": 100_000_000,
    "batch_size": 16,
    "learning_rate": 3e-4,
}

CONTINUE_ON_ERROR = True

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.magnet_training import MagnetConfig, train_and_evaluate_magnet

    results: list[dict] = []
    first_write = True

    for lang in LANGUAGES:
        train_path = DATA_DIR / lang / "train.txt"
        eval_path = DATA_DIR / lang / "eval.txt"

        if not train_path.exists():
            print(
                f"[{lang}] SKIP — {train_path} not found. "
                "Run download_data.py first.",
                file=sys.stderr,
            )
            continue

        if not eval_path.exists():
            print(
                f"[{lang}] SKIP — {eval_path} not found. "
                "Run download_data.py first.",
                file=sys.stderr,
            )
            continue

        cfg = MagnetConfig(**MODEL_OVERRIDES)

        print(f"\n{'='*60}")
        print(f"MAGNET  language={lang}")
        print(f"{'='*60}")

        try:
            metrics = train_and_evaluate_magnet(
                train_corpus_path=train_path,
                eval_corpus_path=eval_path,
                cfg=cfg,
                language=lang,
                eval_flores=EVAL_FLORES,
            )
        except Exception as exc:
            msg = f"[{lang}] FAILED: {exc}"
            print(msg, file=sys.stderr)
            if not CONTINUE_ON_ERROR:
                raise
            continue

        row = {"language": lang, **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}}
        results.append(row)

        # Incremental CSV write so partial runs are recoverable.
        all_keys = list(row.keys())
        write_header = first_write or not RESULTS_PATH.exists()
        with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        first_write = False

        bpb = metrics.get("test_bits_per_byte", "N/A")
        print(f"[{lang}] test BPB = {bpb}")

    print(f"\nResults written to {RESULTS_PATH}")
