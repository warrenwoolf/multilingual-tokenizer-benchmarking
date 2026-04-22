# Multilingual Tokenizer Benchmarking

Compare tokenizer algorithms across languages with different morphological
typology — measuring fertility, coverage, and split rate on FineWeb / FineWeb 2.

## Hypothesis

**SuperBPE wins on analytic languages (English) and only ties BPE on synthetic
languages (Russian, Hindi, Turkish).**

SuperBPE's key innovation is learning merges that cross whitespace (multi-word
"superword" tokens). This pays off in analytic languages where whitespace
reliably separates semantic units, but should not help (and may hurt) in
agglutinative or fusional languages where the interesting morphological
structure lives *inside* words rather than across whitespace.

## Algorithms (v1)

| | Algorithm | Source |
|---|---|---|
| 1 | BPE | `tokenizers.models.BPE` |
| 2 | SuperBPE | [PythonNut/superbpe](https://github.com/PythonNut/superbpe) (shell-out) |
| 3 | WordPiece | `tokenizers.models.WordPiece` |
| 4 | Unigram | `tokenizers.models.Unigram` |
| 5 | ByT5 (byte-level baseline) | `transformers.ByT5Tokenizer` |

Scoped out of v1 with reasons documented in [REFERENCES.md](REFERENCES.md):
MAGNET (no public code; architecture-level), MorphBPE (thin official repo, time),
byte-level BPE / tiktoken (can add later).

## Languages & Data

| Code | Language  | Typology             | Source |
|------|-----------|----------------------|--------|
| en   | English   | Analytic             | `HuggingFaceFW/fineweb` (sample-10BT) |
| ru   | Russian   | Fusional / synthetic | `HuggingFaceFW/fineweb-2` (`rus_Cyrl`) |
| hi   | Hindi     | Fusional / synthetic | `HuggingFaceFW/fineweb-2` (`hin_Deva`) |
| tr   | Turkish   | Agglutinative        | `HuggingFaceFW/fineweb-2` (`tur_Latn`) |

FineWeb 2 deliberately excludes English (it was built from the non-English
residual of the original FineWeb), so EN is drawn from the original FineWeb.

## Metrics

Implemented in `src/utils/evaluation_metrics.py`:

- **Fertility** — average tokens per whitespace-delimited word. Lower = more efficient.
- **Vocabulary coverage** — fraction of eval-set word types that appear as a single token in the vocab.
- **% continued words** — fraction of eval-set words that are split into more than one token.

## Quick start

```bash
# 1. Install deps (Python 3.11+; SuperBPE requires 3.12 and Rust — optional)
make install

# 2. Run the full test suite (offline; ~1.5s)
make test

# 3. Stream FineWeb data and prepare train/eval splits (writes to data/)
python download_and_prepare.py --languages en,ru,hi,tr --byte-budget-mb 500

# 4. Train every tokenizer (writes to artifacts/)
python generate_tokenizers.py

# 5. Evaluate (writes results.csv)
python evaluate_tokenizers.py
```

To enable SuperBPE:

```bash
# requires Python 3.12 and a Rust toolchain
make install-superbpe
export SUPERBPE_REPO=third_party/superbpe
python generate_tokenizers.py --algorithms superbpe
```

## Configuration

All runtime config lives at the top of the three top-level scripts
(`download_and_prepare.py`, `generate_tokenizers.py`, `evaluate_tokenizers.py`)
as module-level Python — edit in place. Same values are overridable via CLI
flags; see `--help` on each script.

## Repository layout

```
.
├── download_and_prepare.py      top-level: download + split
├── generate_tokenizers.py       top-level: train tokenizers
├── evaluate_tokenizers.py       top-level: compute metrics → results.csv
├── pyproject.toml
├── Makefile
├── REFERENCES.md                paper + implementation citations
├── src/
│   ├── prepare_data/
│   │   ├── download_datasets.py   FineWeb streaming + byte budgeting
│   │   └── prepare_datasets.py    clean / dedupe / shuffle / split
│   ├── tools/
│   │   ├── create_tokenizer.py    high-level: train one artifact
│   │   └── evaluate_tokenizer.py  high-level: evaluate one artifact
│   └── utils/
│       ├── tokenizer_algorithms.py 5 adapters (BPE, SuperBPE, WordPiece, Unigram, ByT5)
│       └── evaluation_metrics.py   fertility, coverage, split rate
├── tests/
│   ├── conftest.py                 tiny multilingual corpus fixture
│   ├── test_tokenizer_contract.py  contract tests run against every adapter
│   ├── test_data_pipeline.py       offline tests for download + prepare
│   └── test_evaluation_metrics.py  unit tests for the 3 metrics
├── data/                           downloaded + prepared corpora (gitignored)
└── artifacts/                      trained tokenizers (gitignored)
```

## Compute budget

4 languages × 4 BPE-family algorithms × 4 vocab sizes × ~500 MB per language
≈ 10–40 CPU-hours total. ByT5 is free. SuperBPE's stage-2 step is the slowest.
Fits on a single 8-core VM overnight.
