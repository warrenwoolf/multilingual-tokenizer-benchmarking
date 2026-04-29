# References

Papers, datasets, and reference implementations used in this benchmark.

## Algorithms

### BPE (Byte-Pair Encoding)
- **Paper:** Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- **Implementation:** [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — `tokenizers.models.BPE` + `BpeTrainer`.

### SuperBPE
- **Paper:** Nut, Liu, Zettlemoyer & Smith (2025). *SuperBPE: Space Travel for Language Models*. [arXiv:2503.13423](https://arxiv.org/abs/2503.13423)
- **Official code:** [PythonNut/superbpe](https://github.com/PythonNut/superbpe)
- **Patched `tokenizers` fork (train-time dep):** [alisawuffles/tokenizers-superbpe](https://github.com/alisawuffles/tokenizers-superbpe)
- **Released model:** [allenai/superbpe-experimental_v0.1.0](https://huggingface.co/allenai/superbpe-experimental_v0.1.0)

### WordPiece
- **Paper:** Devlin, Chang, Lee & Toutanova (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Implementation:** [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — `tokenizers.models.WordPiece` + `WordPieceTrainer`.

### Unigram LM / SentencePiece
- **Papers:**
  - Kudo (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)
  - Kudo & Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- **Implementations:**
  - [google/sentencepiece](https://github.com/google/sentencepiece) — canonical C++ + Python bindings.
  - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — pure-Rust `Unigram` model + `UnigramTrainer` (what we use here).

### ByT5 (pure byte-level baseline)
- **Paper:** Xue, Barua, Constant, Al-Rfou, Narang, Kale, Roberts & Raffel (2022). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*. [arXiv:2105.13626](https://arxiv.org/abs/2105.13626)
- **Implementation:** [huggingface/transformers](https://github.com/huggingface/transformers) — `transformers.ByT5Tokenizer`. No training needed; each UTF-8 byte is a token ID.

### MorphBPE
- **Paper:** Asgari, El Kheir & Sadraei Javaheri (2025). *MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies*. [arXiv:2502.00894](https://arxiv.org/abs/2502.00894)
- **Official code:** [llm-lab-org/MorphBPE](https://github.com/llm-lab-org/MorphBPE) — empty placeholder at the time of writing, so we re-implement Algorithm 1 from scratch in `src/utils/morpheme_segmentation.py` + `src/utils/tokenizer_algorithms.py:_train_morphbpe`.
- **Morpheme segmenter:** [Morfessor 2.0](https://github.com/aalto-speech/morfessor) (Virpioja et al. 2013, [paper](https://aaltodoc.aalto.fi/server/api/core/bitstreams/619206b3-7eef-4940-a2c9-89bf9e85c8b6/content)) — unsupervised, MDL-based, designed for agglutinative languages. Trained per-language on the same corpus used for the BPE step.
- **Not to be confused with** [h9-tec/MorphBPE](https://github.com/h9-tec/MorphBPE), which is a different paper (Fanar, [arXiv:2501.13944](https://arxiv.org/abs/2501.13944)).

## Deferred to future work

The following algorithms were scoped out of v1. Keeping citations here for the paper's Related Work section.

### MAGNET
- **Paper:** Ahia, Kumar, Gonen, Hoffmann, Limisiewicz, Tsvetkov & Smith (2024). *MAGNET: Improving the Multilingual Fairness of Language Models with Adaptive Gradient-Based Tokenization*. [arXiv:2407.08818](https://arxiv.org/abs/2407.08818) — NeurIPS 2024.
- **Status:** No public code release. MAGNET is architecture-level (gradient-based boundary predictors integrated with the LM), not a standalone vocab+merges tokenizer — it cannot be compared on fertility/coverage metrics without training a full LM.

### Byte-level BPE (GPT-2 style)
- **Paper:** Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* ([PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
- **Implementation:** `tokenizers.ByteLevelBPETokenizer` — easy to add in v2.

### tiktoken-style BPE
- **Implementation:** [openai/tiktoken](https://github.com/openai/tiktoken)
- **Status:** `tiktoken` is encoding-only with OpenAI's preset vocabs (cl100k_base etc.); no production training path.

## Data

### FineWeb & FineWeb 2
- **Paper:** Penedo, Kydlíček, Lozhkov, Mitchell, Raffel, Von Werra, Wolf et al. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- **English:** [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — fastText-filtered at English score ≥ 0.65.
- **Non-English (RU/HI/TR):** [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) — multilingual successor, built from the non-English residual of FineWeb. Uses FLORES `{iso639-3}_{script}` config names.
- **FineWeb 2 processing pipeline:** [huggingface/fineweb-2](https://github.com/huggingface/fineweb-2)
