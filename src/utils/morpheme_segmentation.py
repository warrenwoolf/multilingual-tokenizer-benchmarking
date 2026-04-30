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
Both supported languages use MorphyNet (Batsuren et al. 2021) gold
inflectional segmentations derived from Wiktionary (CC-BY-SA 3.0). This
matches the paper (Asgari et al. 2025), which sources segmentations for
English, Hungarian, and Russian from the SIGMORPHON 2022 Shared Task on
Morpheme Segmentation (Batsuren et al. 2022) — which is itself built on
MorphyNet.

- en (English):  eng/eng.inflectional.v1.tsv          (~650k entries)
- hu (Hungarian):hun/hu.inflectional.segmentation.v1.tsv (~1M entries)

Words not in the lookup are kept unsplit (OOV fallback).

- zh (Mandarin): not supported. Mandarin is super-analytic — its words
                 don't decompose into the inflectional/derivational
                 morphemes MorphBPE was designed to exploit — so we raise
                 a clear error rather than silently degrading to BPE.

Adding a language amounts to adding it to ``SUPPORTED_LANGUAGES`` and
``_MORPHYNET_FILES``, then verifying that the MorphyNet TSV exists at the
expected path in the repository.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "hu")

# MorphyNet file registry: our 2-letter code → (subdir, filename) within the repo.
# File format (TSV, 4 columns): lemma TAB inflected TAB features TAB morphemes
# where morphemes are pipe-separated and "-" means irregular (no segmentation).
_MORPHYNET_FILES: dict[str, tuple[str, str]] = {
    "en": ("eng", "eng.inflectional.v1.tsv"),
    "hu": ("hun", "hu.inflectional.segmentation.v1.tsv"),
}
MORPHYNET_BASE_URL = (
    "https://raw.githubusercontent.com/kbatsuren/MorphyNet/master/"
)
MORPHYNET_DEFAULT_CACHE: Path = Path.home() / ".cache" / "morphynet"

# Kept for optional use and tests that exercise Morfessor internals directly.
DEFAULT_MAX_TYPES = 200_000
DEFAULT_MAX_LINES: int | None = None


def is_supported(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# MorphyNet lookup (production path for all supported languages)
# ---------------------------------------------------------------------------


def _load_morphynet(language: str, cache_dir: Path) -> dict[str, list[str]]:
    """Return word → [morpheme, ...] from the MorphyNet inflectional TSV.

    Downloads the file once to cache_dir and reuses it on subsequent calls.
    Words with irregular segmentations (column 4 == "-") are skipped; callers
    fall back to [word] for those.
    """
    subdir, filename = _MORPHYNET_FILES[language]
    url = f"{MORPHYNET_BASE_URL}{subdir}/{filename}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = cache_dir / filename
    if not tsv_path.exists():
        import urllib.request

        urllib.request.urlretrieve(url, tsv_path)
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
# Morfessor utilities — kept for tests that exercise internals directly
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
# unit. Empty morphemes are dropped to avoid emitting double spaces.
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
    morphynet_cache_dir: Path | None = None,
    # kept for API compatibility; unused now that MorphyNet replaces Morfessor
    max_types: int = DEFAULT_MAX_TYPES,
    max_lines: int | None = DEFAULT_MAX_LINES,
) -> None:
    """Write a morpheme-segmented copy of corpus_path to output_path.

    Each whitespace-delimited word is replaced by its morphemes joined
    by single spaces. Original line structure is preserved.

    Args:
        morphynet_cache_dir: Directory to cache downloaded MorphyNet TSV
            files. Defaults to ``~/.cache/morphynet``. Pass a tmp directory
            in tests to avoid network access.
    """
    if not is_supported(language):
        raise NotImplementedError(
            f"MorphBPE is not configured for language {language!r}. "
            f"Supported: {SUPPORTED_LANGUAGES}. Mandarin (zh) is excluded "
            "because it lacks the inflectional morphology MorphBPE relies "
            "on. To add another language, add it to SUPPORTED_LANGUAGES and "
            "_MORPHYNET_FILES in src/utils/morpheme_segmentation.py."
        )

    cache_dir = morphynet_cache_dir if morphynet_cache_dir is not None else MORPHYNET_DEFAULT_CACHE
    lookup = _load_morphynet(language, cache_dir)
    segmenter_fn: Callable[[str], list[str]] = lambda w: lookup.get(w, [w])

    cache: dict[str, list[str]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
                pieces.extend(_segment_word(segmenter_fn, w, cache))
            fout.write(" ".join(pieces))
            fout.write("\n")
