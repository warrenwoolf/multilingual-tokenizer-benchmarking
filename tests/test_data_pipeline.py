"""Tests for the data download path (offline; HF mocked)."""

from __future__ import annotations

import sys
import types

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
    """Verify streaming + budget split between train.txt and eval.txt."""
    docs = [{"text": chr(ord("a") + i) * 100_000} for i in range(8)]  # ~800 KB

    def fake_load_dataset(repo, name, split, streaming):
        assert streaming is True
        return iter(docs)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    paths = download_language(
        "en", tmp_path, train_budget_mb=0.3, eval_budget_mb=0.15
    )
    assert paths["train"].exists() and paths["eval"].exists()

    train_bytes = paths["train"].read_bytes()
    eval_bytes = paths["eval"].read_bytes()
    # train ~ 300 KB cap, eval ~ 150 KB cap; each doc is 100 KB.
    assert len(train_bytes) <= 0.3 * 1024 * 1024 + 1000
    assert len(eval_bytes) <= 0.15 * 1024 * 1024 + 1000
    # Train is filled first, so eval starts at a later doc.
    assert train_bytes.startswith(b"a" * 100)
    assert eval_bytes[:1] != b"a"
