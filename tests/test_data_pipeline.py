"""Tests for data downloading and preparation (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prepare_data.download_datasets import (
    LANGUAGE_CONFIGS,
    download_language,
)
from src.prepare_data.prepare_datasets import prepare_corpus


# ---------- download_datasets ----------------------------------------------


def test_language_configs_covers_target_languages():
    assert set(LANGUAGE_CONFIGS.keys()) == {"en", "ru", "hi", "tr"}


def test_language_configs_fineweb_split():
    # English comes from the original FineWeb; others from FineWeb 2.
    assert LANGUAGE_CONFIGS["en"]["repo"] == "HuggingFaceFW/fineweb"
    for lang in ("ru", "hi", "tr"):
        assert LANGUAGE_CONFIGS[lang]["repo"] == "HuggingFaceFW/fineweb-2"


def test_language_configs_flores_codes():
    # FineWeb 2 uses FLORES {iso639-3}_{script} codes.
    assert LANGUAGE_CONFIGS["ru"]["config"] == "rus_Cyrl"
    assert LANGUAGE_CONFIGS["hi"]["config"] == "hin_Deva"
    assert LANGUAGE_CONFIGS["tr"]["config"] == "tur_Latn"


def test_download_language_rejects_unknown_language(tmp_path):
    with pytest.raises(ValueError, match="not configured"):
        download_language("xx", tmp_path)


def test_download_language_streams_and_respects_budget(tmp_path, monkeypatch):
    """Mock datasets.load_dataset to verify streaming + byte budget logic."""
    mock_docs = [
        {"text": "a" * 100_000},  # ~100 KB
        {"text": "b" * 100_000},
        {"text": "c" * 100_000},
        {"text": "d" * 100_000},
        {"text": "e" * 100_000},  # would overshoot
    ]

    def fake_load_dataset(repo, name, split, streaming):
        assert streaming is True
        return iter(mock_docs)

    import src.prepare_data.download_datasets as mod

    # `datasets.load_dataset` is imported lazily inside the function, so patch
    # the datasets module directly.
    import sys
    import types

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    out = download_language("en", tmp_path, byte_budget_mb=0.3)  # ~300 KB cap
    assert out.exists()
    written = out.read_bytes()
    # Budget is 300 KB; we should fit 2 or 3 docs of 100 KB (plus newlines).
    assert len(written) <= 0.3 * 1024 * 1024 + 1000  # small overshoot tolerance
    assert written.count(b"\n") <= 4
    assert written.startswith(b"a" * 100)


# ---------- prepare_datasets -----------------------------------------------


@pytest.fixture
def raw_corpus(tmp_path) -> Path:
    """Write a small raw corpus with duplicates and short lines."""
    lines = [
        "This is a long enough sentence for inclusion in training.",
        "This is a long enough sentence for inclusion in training.",  # dup
        "Another unique line that is definitely sufficient in length.",
        "short",  # dropped: < 16 chars
        "",  # dropped: empty
        "Yet another long enough sentence for corpus preparation tests.",
        "One more sentence, distinct from the others in this corpus.",
        "Fifth long unique sentence, intended for the eval split probably.",
    ]
    path = tmp_path / "raw.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_prepare_corpus_drops_short_and_empty_lines(raw_corpus, tmp_path):
    out_dir = tmp_path / "prepared"
    paths = prepare_corpus(raw_corpus, out_dir, train_fraction=0.8)
    total = paths["train"].read_text().splitlines() + paths["eval"].read_text().splitlines()
    # 5 unique usable lines expected.
    assert len(total) == 5
    assert "short" not in total
    assert "" not in total


def test_prepare_corpus_deduplicates(raw_corpus, tmp_path):
    out_dir = tmp_path / "prepared"
    paths = prepare_corpus(raw_corpus, out_dir, train_fraction=0.8)
    total = paths["train"].read_text().splitlines() + paths["eval"].read_text().splitlines()
    assert len(set(total)) == len(total)


def test_prepare_corpus_split_ratio(raw_corpus, tmp_path):
    out_dir = tmp_path / "prepared"
    paths = prepare_corpus(raw_corpus, out_dir, train_fraction=0.8)
    train_n = len(paths["train"].read_text().splitlines())
    eval_n = len(paths["eval"].read_text().splitlines())
    total = train_n + eval_n
    # 5 lines × 0.8 = 4 train, 1 eval
    assert train_n == 4 and eval_n == 1
    assert total == 5


def test_prepare_corpus_max_sentences_caps_total(raw_corpus, tmp_path):
    out_dir = tmp_path / "prepared"
    paths = prepare_corpus(raw_corpus, out_dir, train_fraction=0.8, max_sentences=3)
    total = len(paths["train"].read_text().splitlines()) + len(paths["eval"].read_text().splitlines())
    assert total == 3


def test_prepare_corpus_is_deterministic(raw_corpus, tmp_path):
    """Same seed → same shuffle."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    a = prepare_corpus(raw_corpus, out_a, train_fraction=0.8)
    b = prepare_corpus(raw_corpus, out_b, train_fraction=0.8)
    assert a["train"].read_text() == b["train"].read_text()
    assert a["eval"].read_text() == b["eval"].read_text()


def test_prepare_corpus_rejects_invalid_fraction(raw_corpus, tmp_path):
    with pytest.raises(ValueError, match="train_fraction"):
        prepare_corpus(raw_corpus, tmp_path / "x", train_fraction=1.0)
    with pytest.raises(ValueError, match="train_fraction"):
        prepare_corpus(raw_corpus, tmp_path / "y", train_fraction=0.0)
