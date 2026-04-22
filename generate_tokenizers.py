"""Train tokenizer artifacts for every (language, algorithm, vocab_size) triple.

Run after `download_and_prepare.py`. All configuration is at the top of this
file — edit here to change the run.

Outputs:
    artifacts/{lang}_{algorithm}_v{vocab_size}/tokenizer.json
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src.tools.create_tokenizer import create_tokenizer_artifact
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS

# --- config ---------------------------------------------------------------
LANGUAGES = ["en", "ru", "hi", "tr"]
ALGORITHMS = list(SUPPORTED_ALGORITHMS)  # bpe, superbpe, wordpiece, unigram, byt5
VOCAB_SIZES = [8_000, 16_000, 32_000, 64_000]
DATA_DIR = Path("data") / "prepared"
ARTIFACT_DIR = Path("artifacts")

# ByT5 has a fixed ~259-id vocab; training is a no-op, so only one artifact per
# language instead of the full vocab sweep.
SINGLE_SIZE_ALGORITHMS = {"byt5"}


def _iter_jobs(
    languages: list[str], algorithms: list[str], vocab_sizes: list[int]
):
    for lang in languages:
        for algo in algorithms:
            sizes = [0] if algo in SINGLE_SIZE_ALGORITHMS else vocab_sizes
            for vs in sizes:
                yield lang, algo, vs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", type=str, default=",".join(LANGUAGES))
    parser.add_argument("--algorithms", type=str, default=",".join(ALGORITHMS))
    parser.add_argument("--vocab-sizes", type=str,
                        default=",".join(str(v) for v in VOCAB_SIZES))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Don't abort if one combination fails")
    args = parser.parse_args()

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    algorithms = [s.strip() for s in args.algorithms.split(",") if s.strip()]
    vocab_sizes = [int(s.strip()) for s in args.vocab_sizes.split(",") if s.strip()]

    unknown = set(algorithms) - set(SUPPORTED_ALGORITHMS)
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Supported: {SUPPORTED_ALGORITHMS}")

    failures: list[tuple[str, str, int, str]] = []
    for lang, algo, vs in _iter_jobs(languages, algorithms, vocab_sizes):
        corpus = args.data_dir / lang / "train.txt"
        if not corpus.exists():
            print(f"[{lang}/{algo}/v{vs}] SKIP: {corpus} does not exist "
                  "(run download_and_prepare.py first)")
            continue
        label = f"{lang}/{algo}/v{vs if vs else 'fixed'}"
        print(f"[{label}] training ...")
        try:
            artifact = create_tokenizer_artifact(
                corpus_path=corpus,
                algorithm=algo,
                vocab_size=vs,
                output_dir=args.artifact_dir,
                language=lang,
            )
            print(f"[{label}] wrote {artifact}")
        except Exception as exc:
            failures.append((lang, algo, vs, repr(exc)))
            print(f"[{label}] FAILED: {exc}")
            if not args.continue_on_error:
                traceback.print_exc()
                sys.exit(1)

    if failures:
        print(f"\n{len(failures)} failures:")
        for lang, algo, vs, msg in failures:
            print(f"  {lang}/{algo}/v{vs}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
