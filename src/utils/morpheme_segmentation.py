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
- en (English):  MorphyNet (Batsuren et al. 2021) inflectional segmentations
                 — a ~650k-entry gold lookup built from Wiktionary. Words not
                 in the table are kept unsplit. The file is downloaded once to
                 ``~/.cache/morphynet/`` (or a caller-supplied directory) and
                 reused on subsequent runs. This matches the paper's use of
                 gold SIGMORPHON/Wiktionary segmentations for English rather
                 than unsupervised Morfessor, which rarely splits English words.
- hu (Hungarian):Morfessor 2.0 trained unsupervisedly on the corpus.
                 (Morfessor was originally designed for Finnish, another
                 Uralic agglutinative language; it generalises well to
                 Hungarian given its similarly rich suffixal morphology.)
- zh (Mandarin): not supported. Mandarin is super-analytic — its words
                 don't decompose into the inflectional/derivational
                 morphemes MorphBPE was designed to exploit — so we raise
                 a clear error rather than silently degrading to BPE.

Adding a language amounts to adding it to ``SUPPORTED_LANGUAGES`` and
wiring up a segmenter in ``segment_corpus``.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "hu")

# MorphyNet English inflectional data (Batsuren et al. 2021, CC-BY-SA 3.0).
MORPHYNET_URL = (
    "https://raw.githubusercontent.com/kbatsuren/MorphyNet/master/"
    "eng/eng.inflectional.v1.tsv"
)
MORPHYNET_DEFAULT_CACHE: Path = Path.home() / ".cache" / "morphynet"

# Cap how many word types we feed to Morfessor. Morfessor's MDL objective
# scales with vocabulary, and on a 500MB corpus the type count is ~1-3M;
# 200k covers >99% of token mass for Hungarian.
DEFAULT_MAX_TYPES = 200_000

# Cap on lines read for word-frequency collection. None = read whole file.
DEFAULT_MAX_LINES: int | None = None


def is_supported(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# English: MorphyNet lookup
# ---------------------------------------------------------------------------


def _load_morphynet_english(cache_dir: Path) -> dict[str, list[str]]:
    """Return word → [morpheme, ...] from MorphyNet English inflectional TSV.

    Downloads the ~20 MB file once to cache_dir and reuses it on subsequent
    calls. Words with irregular segmentations (column 4 == "-") are skipped;
    callers fall back to [word] for those.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = cache_dir / "eng.inflectional.v1.tsv"
    if not tsv_path.exists():
        import urllib.request

        urllib.request.urlretrieve(MORPHYNET_URL, tsv_path)

    lookup: dict[str, list[str]] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[3] == "-":
                continue
            morphemes = [m for m in parts[3].split("|") if m]
            word = parts[1]
            if morphemes and word not in lookup:
                lookup[word] = morphemes
    return lookup


# ---------------------------------------------------------------------------
# Hungarian: Morfessor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared word-segmentation helper
# ---------------------------------------------------------------------------

# A morpheme is a non-empty string. We rejoin morphemes with a single ASCII
# space so the downstream Whitespace pre-tokenizer treats each as its own
# unit. Empty morphemes (rare; Morfessor occasionally returns them on edge
# cases) are dropped to avoid emitting double spaces.
_PUNCT_RE = re.compile(r"^\W+$", flags=re.UNICODE)


def _segment_word(
    segmenter_fn: Callable[[str], list[str]],
    word: str,
    cache: dict[str, list[str]],
) -> list[str]:
    cached = cache.get(word)
    if cached is not None:
        return cached
    # Pure-punctuation tokens are not morphemes; leave them intact so BPE
    # can still learn punctuation merges within them.
    if _PUNCT_RE.match(word):
        cache[word] = [word]
        return cache[word]
    morphemes = [m for m in segmenter_fn(word) if m]
    if not morphemes:
        morphemes = [word]
    cache[word] = morphemes
    return morphemes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_corpus(
    corpus_path: Path,
    output_path: Path,
    language: str,
    max_types: int = DEFAULT_MAX_TYPES,
    max_lines: int | None = DEFAULT_MAX_LINES,
    morphynet_cache_dir: Path | None = None,
) -> None:
    """Write a morpheme-segmented copy of corpus_path to output_path.

    Each whitespace-delimited word is replaced by its morphemes joined
    by single spaces. Original line structure is preserved.

    Args:
        morphynet_cache_dir: Directory to cache the MorphyNet TSV (English
            only). Defaults to ``~/.cache/morphynet``. Pass a tmp directory
            in tests to avoid network access.
    """
    if not is_supported(language):
        raise NotImplementedError(
            f"MorphBPE is not configured for language {language!r}. "
            f"Supported: {SUPPORTED_LANGUAGES}. Mandarin (zh) is excluded "
            "because it lacks the inflectional morphology MorphBPE relies "
            "on. To add another language, register a segmenter in "
            "src/utils/morpheme_segmentation.py."
        )

    if language == "en":
        cache_dir = morphynet_cache_dir if morphynet_cache_dir is not None else MORPHYNET_DEFAULT_CACHE
        lookup = _load_morphynet_english(cache_dir)
        segmenter_fn: Callable[[str], list[str]] = lambda w: lookup.get(w, [w])
    else:
        counts = _collect_word_counts(corpus_path, max_lines=max_lines)
        if not counts:
            raise ValueError(
                f"Corpus at {corpus_path} contains no whitespace-delimited tokens"
            )
        model = _train_morfessor(counts, max_types=max_types)
        segmenter_fn = lambda w: list(model.viterbi_segment(w)[0])

    cache: dict[str, list[str]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_split = 0
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
                morphemes = _segment_word(segmenter_fn, w, cache)
                pieces.extend(morphemes)
                n_total += 1
                if len(morphemes) > 1:
                    n_split += 1
            fout.write(" ".join(pieces))
            fout.write("\n")

    if n_total > 0 and n_split == 0 and language != "en":
        warnings.warn(
            f"Morfessor produced no morpheme splits for language {language!r} "
            f"({n_total} word tokens checked). MorphBPE will be equivalent to "
            f"plain BPE for this corpus. Morfessor works best on agglutinative "
            f"languages (Hungarian, Finnish) with large, morphologically diverse "
            f"corpora.",
            stacklevel=2,
        )
