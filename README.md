# Multilingual Tokenizer Benchmarking

Compare tokenizer algorithms across languages with different morphological
typology — measuring fertility, coverage, and split rate on FineWeb / FineWeb 2.

## Hypothesis

**SuperBPE wins on analytic languages and only ties BPE on synthetic
languages.**

SuperBPE's key innovation is learning merges that cross whitespace (multi-word
"superword" tokens). This pays off in analytic languages where whitespace
reliably separates semantic units, but should not help (and may hurt) in
agglutinative or fusional languages where the interesting morphological
structure lives *inside* words rather than across whitespace.

## Algorithms

v1 paper focus is on **BPE, SuperBPE, tiktoken-style byte-level BPE**, and
**MorphBPE**. WordPiece, Unigram, and ByT5 are kept available as fallback /
baselines.

| | Algorithm | Source |
|---|---|---|
| 1 | BPE | `tokenizers.models.BPE` |
| 2 | SuperBPE | [PythonNut/superbpe](https://github.com/PythonNut/superbpe) (shell-out, opt-in) |
| 3 | tiktoken-style BPE | byte-level BPE via `tokenizers` (GPT-2 / GPT-4 style) |
| 4 | MorphBPE | from-scratch implementation of [Asgari et al. 2025](https://arxiv.org/abs/2502.00894), Algorithm 1 — see notes |
| 5 | WordPiece | `tokenizers.models.WordPiece` |
| 6 | Unigram | `tokenizers.models.Unigram` |
| 7 | ByT5 (byte-level baseline) | `transformers.ByT5Tokenizer` |

Out of v1 with reasons documented in [REFERENCES.md](REFERENCES.md): MAGNET
(architecture-level; no public code).

**MorphBPE notes.** Algorithm 1 says "merge the most frequent byte pair
without crossing morpheme boundaries". We implement that by morpheme-
segmenting the corpus first and joining morphemes with single spaces, then
training standard HF BPE — the Whitespace pre-tokenizer guarantees no merge
ever observes a cross-morpheme pair. The official `llm-lab-org/MorphBPE`
repo was an empty placeholder at the time of writing, so this is a from-
scratch implementation rather than a wrapper. Per-language segmenters
(`src/utils/morpheme_segmentation.py`):

- **English & Turkish:** unsupervised Morfessor 2.0 trained on the corpus.
- **Mandarin:** not supported — Mandarin is super-analytic and lacks the
  inflectional morphology MorphBPE was designed to exploit. The (zh,
  morphbpe) sweep cell is auto-skipped by ``generate_tokenizers``.

## Languages & Data

| Code | Language | Typology         | Source |
|------|----------|------------------|--------|
| en   | English  | Analytic         | `HuggingFaceFW/fineweb` (sample-10BT) |
| zh   | Mandarin | Super-analytic   | `HuggingFaceFW/fineweb-2` (`cmn_Hani`) |
| tr   | Turkish  | Agglutinative    | `HuggingFaceFW/fineweb-2` (`tur_Latn`) |

Russian (`ru` / `rus_Cyrl`) and Hindi (`hi` / `hin_Deva`) are kept configured
in `LANGUAGE_CONFIGS` for ad-hoc use; they're just not in the default
`LANGUAGES` list.

FineWeb 2 deliberately excludes English (it was built from the non-English
residual of the original FineWeb), so EN is drawn from the original FineWeb.
FineWeb / FineWeb 2 are already deduplicated, so `download_language` writes
``train.txt`` and ``eval.txt`` straight from the stream — no separate
preparation pass.

## Metrics

### Intrinsic (cheap; no model training)

Implemented in `src/utils/evaluation_metrics.py`:

- **Fertility** — average tokens per whitespace-delimited word. Lower = more efficient.
- **Vocabulary coverage** — fraction of eval-set word types that appear as a single token in the vocab.
- **% continued words** — fraction of eval-set words that are split into more than one token.

### Downstream LLM (expensive; predictive)

Implemented in `src/utils/llm_training.py`. Each tokenizer trains a fresh
**monolingual** ~50M-param GPT on its language's corpus, then we score two
held-out eval sets:

- **Test set** (in-domain) — `data/{lang}/eval.txt` from FineWeb / FineWeb 2.
- **FLORES-200 devtest** (out-of-distribution) — `facebook/flores`,
  professionally translated, the standard cross-lingual generalisation probe.

For each, we report:

- **Bits-per-byte (BPB)** — held-out cross-entropy normalised by raw UTF-8
  bytes of the eval text. **This is the cross-tokenizer-comparable metric.**
  Lower = the tokenizer enables a better LM at fixed compute.
- **Perplexity** — kept for diagnostics; *not* comparable across tokenizers
  because PPL depends on vocab/segmentation.

Architecture: pre-LN GPT decoder, d_model=512, 8 layers, 8 heads, d_ff=2048,
ctx=512, weight-tied head. Param count varies with vocab (the embedding
table dominates):

| vocab | params |
|------:|-------:|
|  8 k  | ~30 M  |
| 32 k  | ~42 M  |
| 64 k  | ~58 M  |

**Wall-clock** (1B-token Chinchilla-optimal run, A100 40GB at ~250K tok/s
bf16): ~65 min per LM. Full default sweep of 3 langs × 3 algos × 4 vocab
sizes ≈ 36 LMs ≈ ~40 A100-hours. Shrink the vocab sweep before kicking off.

**Compute fairness caveat.** We fix the *training-token budget* across all
tokenizers (default 1B tokens, ~Chinchilla-optimal for 50M params). This
is intentionally imperfect: a high-fertility tokenizer sees less *content*
for the same token count, and a larger-vocab tokenizer has a bigger
embedding so more FLOPs/token. Fixing wall-clock or training-FLOPs would
be more rigorous; for v1 we trade rigor for reproducibility and document
the caveat. See `llm_training.py` docstring.

**Logging.** Every (lang, algo, vocab) run optionally streams loss + LR +
final eval metrics to Weights & Biases. Set `WANDB_PROJECT` (env var or in
`train_llms.py`); the API key is loaded from `WANDB_API_KEY` env var or
`tokens/wandb.token` (gitignored). In Colab, `colab.py` reads the
`WANDB_API_KEY` Colab Secret and writes it to the token slot for you.

## Quick start

### Local

```bash
make install            # python deps
make test               # 173 contract tests, ~10s, no network
python download_data.py # streams FineWeb(-2) for the configured languages
python generate_tokenizers.py
python evaluate_tokenizers.py
cat results.csv

# Optional: downstream perplexity / BPB by training a small LM per tokenizer.
# Heavy (needs torch + a GPU for the default 50M-token budget); opt-in.
make install-llm
python train_llms.py
cat llm_results.csv
```

### Colab

Paste [`colab.py`](colab.py) into a Google Colab cell and run.

### Optional: enable SuperBPE

Requires Python 3.12 and a Rust toolchain.

```bash
make install-superbpe
export SUPERBPE_REPO=third_party/superbpe
# then add "superbpe" to ALGORITHMS in generate_tokenizers.py
```

## Configuration

All runtime config lives at the top of the three top-level scripts
(`download_data.py`, `generate_tokenizers.py`, `evaluate_tokenizers.py`) as
plain Python module-level constants. The actual orchestration logic
(iteration, error handling, path management) lives in `src/tools/` so the
top-level scripts stay short and readable — edit them in place to change a
run.

## Repository layout

```
.
├── colab.py                       paste-into-Colab one-cell quickstart
├── download_data.py               top-level: config for streaming FineWeb(-2)
├── generate_tokenizers.py         top-level: config for training tokenizers
├── evaluate_tokenizers.py         top-level: intrinsic metrics + CSV
├── train_llms.py                  top-level: downstream LLM PPL/BPB + CSV
├── pyproject.toml
├── Makefile
├── REFERENCES.md                  paper + implementation citations
├── src/
│   ├── prepare_data/
│   │   └── download_datasets.py     FineWeb streaming + train/eval split
│   ├── tools/
│   │   ├── download_data.py         iteration over LANGUAGES
│   │   ├── create_tokenizer.py      single-shot + full sweep training
│   │   ├── evaluate_tokenizer.py    single-shot + full sweep + CSV
│   │   └── train_llm.py             sweep: train small LM per artifact + CSV
│   └── utils/
│       ├── tokenizer_algorithms.py  7 adapters
│       ├── evaluation_metrics.py    fertility, coverage, split rate
│       └── llm_training.py          ~50M GPT, train loop, perplexity + BPB
├── tests/
│   ├── conftest.py                  tiny multilingual corpus fixture
│   ├── test_tokenizer_contract.py   contract tests against every adapter
│   ├── test_data_pipeline.py        offline tests for download
│   └── test_evaluation_metrics.py   unit tests for the 3 metrics
├── data/                            downloaded corpora (gitignored)
└── artifacts/                       trained tokenizers (gitignored)
```

## Compute budget

3 languages × 3 BPE-family algorithms × 4 vocab sizes × ~500 MB per language
≈ 5–20 CPU-hours. ByT5 is free. SuperBPE's stage-2 step is the slowest. Fits
on a single 8-core VM in a few hours.
