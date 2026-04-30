"""Tests for the data download path (offline; HF mocked)."""

from __future__ import annotations

import pytest

from src.prepare_data.download_datasets import (
    LANGUAGE_CONFIGS,
    download_language,
)


# ---------- LANGUAGE_CONFIGS contract --------------------------------------


def test_language_configs_includes_paper_languages():
    # English + super-analytic Mandarin + agglutinative Turkish are the
    # v1 focus. ru/hi remain available as fallbacks.
    for lang in ("en", "zh", "tr"):
        assert lang in LANGUAGE_CONFIGS


def test_language_configs_fineweb_split():
    # English comes from the original FineWeb (FineWeb 2 excludes English).
    assert LANGUAGE_CONFIGS["en"]["repo"] == "HuggingFaceFW/fineweb"
    for lang in ("zh", "tr", "ru", "hi"):
        assert LANGUAGE_CONFIGS[lang]["repo"] == "HuggingFaceFW/fineweb-2"


def test_language_configs_flores_codes():
    # FineWeb 2 uses FLORES {iso639-3}_{script} codes.
    assert LANGUAGE_CONFIGS["zh"]["config"] == "cmn_Hani"
    assert LANGUAGE_CONFIGS["tr"]["config"] == "tur_Latn"
    assert LANGUAGE_CONFIGS["ru"]["config"] == "rus_Cyrl"
    assert LANGUAGE_CONFIGS["hi"]["config"] == "hin_Deva"


# ---------- download_language behaviour ------------------------------------


def test_download_language_rejects_unknown_language(tmp_path):
    with pytest.raises(ValueError, match="not configured"):
        download_language("xx", tmp_path)


def test_download_language_writes_train_and_eval(tmp_path, monkeypatch):
    """Verify streaming + row-count split between train.txt and eval.txt."""
    docs = [{"text": chr(ord("a") + i) * 100_000} for i in range(8)]

    def fake_load_dataset(repo, name=None, split=None, streaming=None, token=None):
        assert streaming is True
        return iter(docs)

    import src.prepare_data.download_datasets as dl_mod
    monkeypatch.setattr(dl_mod, "load_dataset", fake_load_dataset)

    paths = download_language("en", tmp_path, max_train_rows=3, max_eval_rows=2)
    assert paths["train"].exists() and paths["eval"].exists()

    train_text = paths["train"].read_text(encoding="utf-8")
    eval_text = paths["eval"].read_text(encoding="utf-8")

    # Exactly 3 rows in train, 2 in eval.
    train_lines = [l for l in train_text.splitlines() if l]
    eval_lines = [l for l in eval_text.splitlines() if l]
    assert len(train_lines) == 3
    assert len(eval_lines) == 2

    # Train starts with doc 0 (aaa...), eval starts with doc 3 (ddd...).
    assert train_text.startswith("a" * 100)
    assert eval_text.startswith("d" * 100)
