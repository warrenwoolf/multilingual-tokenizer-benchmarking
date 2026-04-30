# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install        # pip install -e ".[dev]"  (core + pytest)
make install-llm    # pip install -e ".[llm]"  (adds torch, wandb, numpy)
make test           # pytest tests/ -v
make download       # python download_data.py
make train          # python generate_tokenizers.py
make evaluate       # python evaluate_tokenizers.py
make train-llms     # python train_llms.py
make clean          # remove data/, artifacts/, results CSVs
```

Run a single test file: `pytest tests/test_evaluation_metrics.py -v`
Run a specific test: `pytest tests/test_tokenizer_contract.py::test_vocab_size -v`
Parallel tests: `pytest tests/ -n auto` (pytest-xdist is installed)

Python ≥ 3.11 required.

## Pipeline Overview

The project benchmarks 7 tokenizer algorithms across 3 languages that represent different morphological typologies (English/analytic, Mandarin/super-analytic, Turkish/agglutinative). The full pipeline is:

```
download_data.py → data/{lang}/{train,eval}.txt
generate_tokenizers.py → artifacts/{lang}_{algo}_v{vocab}/tokenizer.json
evaluate_tokenizers.py → results.csv  (intrinsic metrics)
train_llms.py → llm_results.csv      (downstream BPB, ~40 A100-hours)
```

Configuration for each stage lives as module-level constants at the top of the corresponding top-level script — not in a config file.

## Architecture

### Tokenizer layer (`src/utils/tokenizer_algorithms.py`)

All algorithms expose a common interface via `HFAdapter` (or `ByT5Adapter` for the byte-level baseline):

- `encode(text)`, `decode(ids)`, `encode_batch(texts)`
- `get_vocab()`, `vocab_size`, `token_to_id()`, `save()`
- Boolean flags: `is_byte_level`, `special_tokens`

`train_tokenizer(algo, lang, corpus_path, vocab_size)` dispatches to per-algorithm `_train_*()` helpers. `load_tokenizer(artifact_dir)` reverses it. Artifacts are named `{lang}_{algo}_v{vocab_size}`.

The 7 algorithms: **BPE**, **SuperBPE** (two-stage: word-boundary BPE → cross-word extension), **tiktoken-style** (byte-level BPE), **MorphBPE** (Morfessor pre-segmentation → BPE), **WordPiece**, **Unigram** (SentencePiece), **ByT5** (fixed 256-byte vocab, no training).

Unsupported combinations are skipped silently (e.g., `zh + morphbpe` — Mandarin is excluded from Morfessor segmentation).

### Evaluation layer (`src/utils/evaluation_metrics.py`)

Three intrinsic metrics computed over a sentence corpus:
- `fertility` — avg tokens per whitespace-delimited word (lower = better)
- `vocabulary_coverage` — fraction of word types representable without UNK (byte-level tokenizers always return 1.0)
- `pct_continued_words` — fraction of words split into >1 token

All three via `compute_all_metrics(tokenizer, sentences) → dict`.

### MorphBPE segmentation (`src/utils/morpheme_segmentation.py`)

Wraps Morfessor 2.0 with a per-word-type cache (~200k entries). Morfessor is trained on the corpus before BPE runs. Mandarin is excluded. Edge cases (empty morphemes, pure punctuation) are handled explicitly.

### LLM layer (`src/utils/llm_training.py`)

Pre-LN GPT decoder (~50M params): `d_model=512, n_layers=8, n_heads=8, ctx_len=512`. Training: 1B tokens, AdamW, cosine LR + warmup, batch size 512. Evaluation reports **bits-per-byte (BPB)** — tokenizer-invariant by normalizing cross-entropy by raw UTF-8 byte count. W&B integration is optional (token in `tokens/wandb.token`).

FLORES-200 OOD eval uses ISO 639-3 + script codes: `eng_Latn`, `zho_Hans`, `tur_Latn`.

### Tools layer (`src/tools/`)

Thin orchestration scripts that iterate job combinations and call the utils layer:
- `create_tokenizer.py` — `iter_jobs()` yields `(lang, algo, vocab_size)` triples; `generate_all_tokenizers()` runs all, collects errors
- `evaluate_tokenizer.py` — `parse_artifact_name()` regex-parses artifact directory names; writes CSV
- `download_data.py` — verifies FineWeb-2 config availability before streaming

### Tests (`tests/`)

173 contract/unit tests, ~10s total. `conftest.py` provides a multilingual corpus fixture (English, Russian, Hindi, Turkish samples). `test_tokenizer_contract.py` is parametrized over all trainable algorithms and verifies encode/decode round-trip, vocab size, pickling, and batch consistency. SuperBPE and MorphBPE are skipped in contract tests.

## Data & Artifacts

`data/` and `artifacts/` are gitignored and generated locally. `tokens/` holds secrets (`wandb.token`, `huggingface.token`) which are also gitignored — only the README and `.gitkeep` are tracked.
