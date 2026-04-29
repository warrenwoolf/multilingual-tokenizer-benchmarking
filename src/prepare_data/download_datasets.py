"""Stream raw FineWeb / FineWeb 2 corpora into train + eval text files.

Target languages
-----------------
en  – English   (analytic)        -- HuggingFaceFW/fineweb (sample-10BT)
zh  – Mandarin  (super-analytic)  -- HuggingFaceFW/fineweb-2 (cmn_Hani)
tr  – Turkish   (agglutinative)   -- HuggingFaceFW/fineweb-2 (tur_Latn)

Russian and Hindi remain configured for ad-hoc use:
ru  – Russian   (fusional)        -- HuggingFaceFW/fineweb-2 (rus_Cyrl)
hi  – Hindi     (fusional)        -- HuggingFaceFW/fineweb-2 (hin_Deva)
"""

from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

LANGUAGE_CONFIGS: dict[str, dict] = {
    "en": {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT"},
    "zh": {"repo": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani"},
    "tr": {"repo": "HuggingFaceFW/fineweb-2", "config": "tur_Latn"},
    "ru": {"repo": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl"},
    "hi": {"repo": "HuggingFaceFW/fineweb-2", "config": "hin_Deva"},
}

DEFAULT_MAX_TRAIN_ROWS = 500_000
DEFAULT_MAX_EVAL_ROWS = 25_000
BATCH_SIZE = 1_000_000


def _load_token() -> str | None:
    """Return HF token from HF_TOKEN env var or tokens/hf_token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path("tokens/hf_token")
    if token_path.is_file():
        token = token_path.read_text().strip()
        return token or None
    return None


def download_language(
    language: str,
    output_dir: str | Path,
    train_budget_mb: float | None = None,
    eval_budget_mb: float | None = None,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
) -> dict[str, Path]:
    """Stream one language from HF and write train.txt + eval.txt.

    Args:
        language: Code in LANGUAGE_CONFIGS (en/zh/tr/ru/hi).
        output_dir: Directory; we create ``{output_dir}/{language}/``.
        train_budget_mb: Ignored (kept for backwards compatibility).
        eval_budget_mb: Ignored (kept for backwards compatibility).
        max_train_rows: Max number of documents written to train.txt.
        max_eval_rows: Max number of documents written to eval.txt.

    Returns:
        ``{"train": Path, "eval": Path}``.
    """
    if language not in LANGUAGE_CONFIGS:
        raise ValueError(
            f"Language {language!r} not configured. Keys: {list(LANGUAGE_CONFIGS)}"
        )

    out_dir = Path(output_dir) / language
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    eval_path = out_dir / "eval.txt"

    cfg = LANGUAGE_CONFIGS[language]
    token = _load_token()

    print(f"Downloading '{cfg['config']}' from '{cfg['repo']}'...")

    if cfg["config"] is None:
        ds = load_dataset(cfg["repo"], split="train", streaming=True, token=token)
    else:
        ds = load_dataset(
            cfg["repo"],
            name=cfg["config"],
            split="train",
            streaming=True,
            token=token,
        )

    train_buffer: list[str] = []
    eval_buffer: list[str] = []
    train_count = 0
    eval_count = 0

    with (
        train_path.open("w", encoding="utf-8") as train_fh,
        eval_path.open("w", encoding="utf-8") as eval_fh,
    ):
        for sample in tqdm(ds, desc=f"  {language}", mininterval=5):
            text = sample.get("text", "").strip()
            if not text:
                continue

            line = text.replace("\n", " ")

            if train_count < max_train_rows:
                train_buffer.append(line)
                train_count += 1
                if len(train_buffer) % BATCH_SIZE == 0:
                    train_fh.write("\n".join(train_buffer) + "\n")
                    print(f"  [{language}] train: {train_count} rows so far...")
                    train_buffer.clear()
            elif eval_count < max_eval_rows:
                eval_buffer.append(line)
                eval_count += 1
                if len(eval_buffer) % BATCH_SIZE == 0:
                    eval_fh.write("\n".join(eval_buffer) + "\n")
                    print(f"  [{language}] eval: {eval_count} rows so far...")
                    eval_buffer.clear()
            else:
                break

        if train_buffer:
            train_fh.write("\n".join(train_buffer) + "\n")
        if eval_buffer:
            eval_fh.write("\n".join(eval_buffer) + "\n")

    print(f"  [{language}] done — train: {train_count} rows, eval: {eval_count} rows")
    print(f"  train -> {train_path}")
    print(f"  eval  -> {eval_path}")

    return {"train": train_path, "eval": eval_path}


def download_all(
    output_dir: str | Path,
    languages: list[str] | None = None,
    train_budget_mb: float | None = None,
    eval_budget_mb: float | None = None,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    max_eval_rows: int = DEFAULT_MAX_EVAL_ROWS,
) -> dict[str, dict[str, Path]]:
    """Download every language in ``languages`` (default: all configured)."""
    if languages is None:
        languages = list(LANGUAGE_CONFIGS)
    return {
        lang: download_language(
            lang,
            output_dir,
            max_train_rows=max_train_rows,
            max_eval_rows=max_eval_rows,
        )
        for lang in languages
    }


def verify_fineweb2_configs(languages: list[str]) -> None:
    """Sanity-check that FineWeb 2 exposes the FLORES configs we expect."""
    needed = [
        cfg["config"]
        for lang, cfg in LANGUAGE_CONFIGS.items()
        if lang in languages and cfg["repo"] == "HuggingFaceFW/fineweb-2"
    ]
    if not needed:
        return
    try:
        from datasets import get_dataset_config_names

        configs = set(
            get_dataset_config_names("HuggingFaceFW/fineweb-2", token=_load_token())
        )
    except Exception as exc:
        print(f"[warn] could not verify fineweb-2 configs: {exc}")
        return
    missing = [c for c in needed if c not in configs]
    if missing:
        raise RuntimeError(
            f"Expected FineWeb 2 configs not found: {missing}. "
            f"Sample of available: {sorted(configs)[:10]}..."
        )
