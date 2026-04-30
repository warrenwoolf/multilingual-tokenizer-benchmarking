"""Morpheme segmentation for MorphBPE.

MorphBPE (Asgari et al. 2025, arXiv:2502.00894) constrains BPE so that no
merge crosses a morpheme boundary. We implement that constraint at the
corpus-preprocessing layer: each whitespace-delimited word is replaced by
its morpheme decomposition with morphemes joined by single spaces, then
standard BPE is trained on the rewritten corpus. Because the Whitespace
pre-tokenizer splits on those new spaces, BPE never observes a symbol pair
that crosses a morpheme boundary — so it cannot count one and cannot merge
one. This is equivalent to Algorithm 1 in the paper without needing a
custom trainer.

Per-language morpheme segmenters
--------------------------------
- en (English):  Morfessor 2.0 trained unsupervisedly on the corpus.
- tr (Turkish):  Morfessor 2.0 trained unsupervisedly on the corpus.
                 (Morfessor was originally designed for Finnish, another
                 agglutinative language; it works well for Turkish too.)
- zh (Mandarin): not supported. Mandarin is super-analytic — its words
                 don't decompose into the inflectional/derivational
                 morphemes MorphBPE was designed to exploit — so we raise
                 a clear error rather than silently degrading to BPE.

Adding a language amounts to adding it to ``SUPPORTED_LANGUAGES`` and
verifying Morfessor produces sensible segmentations on a sample.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "tr")

# Cap how many word types we feed to Morfessor. Morfessor's MDL objective
# scales with vocabulary, and on a 500MB corpus the type count is ~1-3M;
# 200k covers >99% of token mass for both English and Turkish.
DEFAULT_MAX_TYPES = 200_000

# Cap on lines read for word-frequency collection. None = read whole file.
DEFAULT_MAX_LINES: int | None = None


def is_supported(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES


def _collect_word_counts(
    corpus_path: Path, max_lines: int | None = DEFAULT_MAX_LINES
) -> Counter[str]:
    counts: Counter[str] = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            for w in line.split():
                counts[w] += 1
    return counts


def _train_morfessor(word_counts: Counter[str], max_types: int):
    import morfessor

    model = morfessor.BaselineModel()
    data = [(c, w) for w, c in word_counts.most_common(max_types)]
    model.load_data(data)
    model.train_batch()
    return model


# A morpheme is a non-empty string. We rejoin morphemes with a single ASCII
# space so the downstream Whitespace pre-tokenizer treats each as its own
# unit. Empty morphemes (rare; Morfessor occasionally returns them on edge
# cases) are dropped to avoid emitting double spaces.
_PUNCT_RE = re.compile(r"^\W+$", flags=re.UNICODE)


def _segment_word(model, word: str, cache: dict[str, list[str]]) -> list[str]:
    cached = cache.get(word)
    if cached is not None:
        return cached
    # Pure-punctuation tokens are not morphemes; leave them intact so BPE
    # can still learn punctuation merges within them.
    if _PUNCT_RE.match(word):
        cache[word] = [word]
        return cache[word]
    morphemes, _ = model.viterbi_segment(word)
    morphemes = [m for m in morphemes if m]
    if not morphemes:
        morphemes = [word]
    cache[word] = morphemes
    return morphemes


def segment_corpus(
    corpus_path: Path,
    output_path: Path,
    language: str,
    max_types: int = DEFAULT_MAX_TYPES,
    max_lines: int | None = DEFAULT_MAX_LINES,
) -> None:
    """Write a morpheme-segmented copy of corpus_path to output_path.

    Each whitespace-delimited word is replaced by its morphemes joined
    by single spaces. Original line structure is preserved.
    """
    if not is_supported(language):
        raise NotImplementedError(
            f"MorphBPE is not configured for language {language!r}. "
            f"Supported: {SUPPORTED_LANGUAGES}. Mandarin (zh) is excluded "
            "because it lacks the inflectional morphology MorphBPE relies "
            "on. To add another language, register a Morfessor model (or "
            "another segmenter) in src/utils/morpheme_segmentation.py."
        )

    counts = _collect_word_counts(corpus_path, max_lines=max_lines)
    if not counts:
        raise ValueError(f"Corpus at {corpus_path} contains no whitespace-delimited tokens")

    model = _train_morfessor(counts, max_types=max_types)

    # Cache one segmentation per type. For a typical corpus this is
    # ~200k entries — small compared to the corpus itself.
    cache: dict[str, list[str]] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_split = 0  # word tokens that were split into 2+ morphemes
    n_total = 0
    with (
        open(corpus_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            words = line.split()
            if not words:
                fout.write("\n")
                continue
            pieces: list[str] = []
            for w in words:
                morphemes = _segment_word(model, w, cache)
                pieces.extend(morphemes)
                n_total += 1
                if len(morphemes) > 1:
                    n_split += 1
            fout.write(" ".join(pieces))
            fout.write("\n")

    if n_total > 0 and n_split == 0:
        warnings.warn(
            f"Morfessor produced no morpheme splits for language {language!r} "
            f"({n_total} word tokens checked). MorphBPE will be equivalent to "
            f"plain BPE for this corpus. Morfessor works best on agglutinative "
            f"languages (Turkish, Finnish) with large, morphologically diverse "
            f"corpora. For English the paper (Asgari et al. 2025) uses gold "
            f"SIGMORPHON segmentations rather than Morfessor.",
            stacklevel=2,
        )
