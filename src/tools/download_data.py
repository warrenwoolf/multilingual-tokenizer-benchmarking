"""Iteration entry point for downloading every configured language."""

from __future__ import annotations

from pathlib import Path

from src.prepare_data.download_datasets import (
    LANGUAGE_CONFIGS,
    download_language,
    verify_fineweb2_configs,
)


def download_all_languages(
    languages: list[str],
    data_dir: Path,
    max_train_rows: int = 500_000,
    max_eval_rows: int = 25_000,
    skip_verify: bool = False,
) -> dict[str, dict[str, Path]]:
    """Download each language; return a map of ``lang -> {train, eval}`` paths.

    Verifies FineWeb 2 config availability up front (one HF call) unless
    ``skip_verify`` is set; this catches typos in LANGUAGE_CONFIGS before
    spending bandwidth on streaming the wrong shard.
    """
    unknown = set(languages) - set(LANGUAGE_CONFIGS)
    if unknown:
        raise ValueError(
            f"Unknown languages: {unknown}. "
            f"Known: {list(LANGUAGE_CONFIGS)}"
        )

    if not skip_verify:
        verify_fineweb2_configs(languages)

    data_dir = Path(data_dir)
    out: dict[str, dict[str, Path]] = {}
    for lang in languages:
        print(
            f"[{lang}] streaming up to {max_train_rows:,} train rows + "
            f"{max_eval_rows:,} eval rows ..."
        )
        paths = download_language(
            language=lang,
            output_dir=data_dir,
            max_train_rows=max_train_rows,
            max_eval_rows=max_eval_rows,
        )
        train_size = paths["train"].stat().st_size / 1e6
        eval_size = paths["eval"].stat().st_size / 1e6
        print(
            f"[{lang}] train={paths['train']} ({train_size:.1f} MB), "
            f"eval={paths['eval']} ({eval_size:.1f} MB)"
        )
        out[lang] = paths
    return out
