"""Download raw corpora for each target language from FineWeb / FineWeb 2.

Target languages
-----------------
en  – English    (analytic)       -- HuggingFaceFW/fineweb (sample-10BT)
ru  – Russian    (synthetic)      -- HuggingFaceFW/fineweb-2 (rus_Cyrl)
hi  – Hindi      (synthetic)      -- HuggingFaceFW/fineweb-2 (hin_Deva)
tr  – Turkish    (agglutinative)  -- HuggingFaceFW/fineweb-2 (tur_Latn)

FineWeb 2 deliberately excludes English (it was built from the non-English
residual of the original FineWeb), so English is drawn from the original
FineWeb dataset.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE_CONFIGS: dict[str, dict] = {
    "en": {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT"},
    "ru": {"repo": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl"},
    "hi": {"repo": "HuggingFaceFW/fineweb-2", "config": "hin_Deva"},
    "tr": {"repo": "HuggingFaceFW/fineweb-2", "config": "tur_Latn"},
}

DEFAULT_BYTE_BUDGET_MB = 500


def download_language(
    language: str,
    output_dir: str | Path,
    byte_budget_mb: int = DEFAULT_BYTE_BUDGET_MB,
) -> Path:
    """Stream one language from HF and write text to a raw file.

    Reads from the HuggingFace Hub in streaming mode and writes one document
    per line to ``{output_dir}/{language}.raw.txt`` until the byte budget is
    exhausted.

    Args:
        language: ISO 639-1 code (must be in LANGUAGE_CONFIGS).
        output_dir: Directory where the raw file will be written.
        byte_budget_mb: Approximate size cap for the output file in megabytes.

    Returns:
        Path to the written raw text file.
    """
    if language not in LANGUAGE_CONFIGS:
        raise ValueError(
            f"Language {language!r} not configured. Keys: {list(LANGUAGE_CONFIGS)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{language}.raw.txt"

    cfg = LANGUAGE_CONFIGS[language]
    from datasets import load_dataset

    stream = load_dataset(
        cfg["repo"],
        name=cfg["config"],
        split="train",
        streaming=True,
    )

    byte_budget = byte_budget_mb * 1024 * 1024
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for example in stream:
            text = example.get("text", "").strip()
            if not text:
                continue
            # One doc per line; replace internal newlines so the line-based
            # downstream splitter stays simple.
            line = text.replace("\n", " ") + "\n"
            b = line.encode("utf-8")
            if written + len(b) > byte_budget:
                break
            fh.write(line)
            written += len(b)

    return out_path


def download_all(
    output_dir: str | Path,
    byte_budget_mb: int = DEFAULT_BYTE_BUDGET_MB,
) -> dict[str, Path]:
    """Download every configured language; return a map of language → path."""
    return {
        lang: download_language(lang, output_dir, byte_budget_mb=byte_budget_mb)
        for lang in LANGUAGE_CONFIGS
    }
