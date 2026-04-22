"""Download raw FineWeb(-2) corpora and prepare train/eval splits.

Run after `make install`. All configuration is at the top of this file —
edit here to change the run.

Outputs:
    data/raw/{lang}.raw.txt        — one document per line, byte-budgeted
    data/prepared/{lang}/train.txt — shuffled, deduped, 95% of usable lines
    data/prepared/{lang}/eval.txt  — remaining 5%
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.prepare_data.download_datasets import (
    LANGUAGE_CONFIGS,
    download_language,
)
from src.prepare_data.prepare_datasets import prepare_corpus

# --- config ---------------------------------------------------------------
LANGUAGES = ["en", "ru", "hi", "tr"]
BYTE_BUDGET_MB = 500
TRAIN_FRACTION = 0.95
MAX_SENTENCES: int | None = None
DATA_DIR = Path("data")
RAW_SUBDIR = "raw"
PREPARED_SUBDIR = "prepared"


def _verify_fineweb2_configs() -> None:
    """Sanity-check that FineWeb 2 exposes the FLORES configs we expect.

    Caches as a no-op if the dataset is temporarily unreachable — we don't
    want to fail the whole pipeline on a transient network hiccup if the
    configs are correct.
    """
    try:
        from datasets import get_dataset_config_names
    except ImportError:
        return
    try:
        configs = set(get_dataset_config_names("HuggingFaceFW/fineweb-2"))
    except Exception as exc:
        print(f"[warn] could not verify fineweb-2 configs: {exc}")
        return
    missing = []
    for lang, cfg in LANGUAGE_CONFIGS.items():
        if cfg["repo"] == "HuggingFaceFW/fineweb-2" and cfg["config"] not in configs:
            missing.append((lang, cfg["config"]))
    if missing:
        raise RuntimeError(
            f"Expected FineWeb 2 configs not found: {missing}. "
            f"Available samples: {sorted(configs)[:10]}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", type=str, default=",".join(LANGUAGES),
                        help="Comma-separated language codes")
    parser.add_argument("--byte-budget-mb", type=int, default=BYTE_BUDGET_MB)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--max-sentences", type=int, default=MAX_SENTENCES)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip FineWeb 2 config verification")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    raw_dir = args.data_dir / RAW_SUBDIR
    prepared_dir = args.data_dir / PREPARED_SUBDIR

    if not args.skip_verify:
        _verify_fineweb2_configs()

    for lang in languages:
        print(f"[{lang}] downloading up to {args.byte_budget_mb} MB ...")
        raw_path = download_language(lang, raw_dir, byte_budget_mb=args.byte_budget_mb)
        print(f"[{lang}] raw: {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")

        print(f"[{lang}] preparing train/eval splits ...")
        splits = prepare_corpus(
            raw_path=raw_path,
            output_dir=prepared_dir / lang,
            train_fraction=args.train_fraction,
            max_sentences=args.max_sentences,
        )
        train_n = sum(1 for _ in splits["train"].open(encoding="utf-8"))
        eval_n = sum(1 for _ in splits["eval"].open(encoding="utf-8"))
        print(f"[{lang}] train={train_n} lines, eval={eval_n} lines")


if __name__ == "__main__":
    main()
