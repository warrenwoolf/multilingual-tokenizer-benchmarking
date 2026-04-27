"""Stream raw FineWeb / FineWeb 2 corpora into train + eval text files.

Target languages
-----------------
en  – English   (analytic)        -- HuggingFaceFW/fineweb (sample-10BT)
zh  – Mandarin  (super-analytic)  -- HuggingFaceFW/fineweb-2 (cmn_Hani)
tr  – Turkish   (agglutinative)   -- HuggingFaceFW/fineweb-2 (tur_Latn)

Russian and Hindi remain configured for ad-hoc use:
ru  – Russian   (fusional)        -- HuggingFaceFW/fineweb-2 (rus_Cyrl)
hi  – Hindi     (fusional)        -- HuggingFaceFW/fineweb-2 (hin_Deva)

FineWeb 2 deliberately excludes English (built from the non-English residual
of the original FineWeb), so EN is drawn from the original FineWeb dataset.

FineWeb is already deduplicated and shuffled, so we don't reshuffle or hash —
we simply stream a train byte-budget into train.txt and a smaller eval
byte-budget into eval.txt, sequentially.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE_CONFIGS: dict[str, dict] = {
    "en": {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT"},
    "zh": {"repo": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani"},
    "tr": {"repo": "HuggingFaceFW/fineweb-2", "config": "tur_Latn"},
    "ru": {"repo": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl"},
    "hi": {"repo": "HuggingFaceFW/fineweb-2", "config": "hin_Deva"},
}

DEFAULT_TRAIN_BUDGET_MB = 500
DEFAULT_EVAL_BUDGET_MB = 25


def download_language(
    language: str,
    output_dir: str | Path,
    train_budget_mb: float = DEFAULT_TRAIN_BUDGET_MB,
    eval_budget_mb: float = DEFAULT_EVAL_BUDGET_MB,
) -> dict[str, Path]:
    """Stream one language from HF and write train.txt + eval.txt.

    Train docs come first, then eval docs from later in the same stream.
    Both files contain one document per line (internal newlines stripped).

    Args:
        language: Code in LANGUAGE_CONFIGS (en/zh/tr/ru/hi).
        output_dir: Directory; we create ``{output_dir}/{language}/``.
        train_budget_mb: Approximate cap for train.txt in MB.
        eval_budget_mb: Approximate cap for eval.txt in MB.

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
    from datasets import load_dataset

    stream = load_dataset(
        cfg["repo"],
        name=cfg["config"],
        split="train",
        streaming=True,
    )

    train_budget = int(train_budget_mb * 1024 * 1024)
    eval_budget = int(eval_budget_mb * 1024 * 1024)

    train_written = 0
    eval_written = 0
    train_full = False

    with train_path.open("w", encoding="utf-8") as train_fh, \
         eval_path.open("w", encoding="utf-8") as eval_fh:
        for example in stream:
            text = example.get("text", "").strip()
            if not text:
                continue
            line = text.replace("\n", " ") + "\n"
            n = len(line.encode("utf-8"))

            if not train_full:
                if train_written + n > train_budget:
                    train_full = True
                else:
                    train_fh.write(line)
                    train_written += n
                    continue

            # Train budget hit; fill eval.
            if eval_written + n > eval_budget:
                # Stream doc sizes vary, so a single oversized doc shouldn't
                # terminate the loop — skip it and let smaller later docs fit.
                if eval_written >= eval_budget:
                    break
                continue
            eval_fh.write(line)
            eval_written += n

    return {"train": train_path, "eval": eval_path}


def download_all(
    output_dir: str | Path,
    languages: list[str] | None = None,
    train_budget_mb: float = DEFAULT_TRAIN_BUDGET_MB,
    eval_budget_mb: float = DEFAULT_EVAL_BUDGET_MB,
) -> dict[str, dict[str, Path]]:
    """Download every language in ``languages`` (default: all configured)."""
    if languages is None:
        languages = list(LANGUAGE_CONFIGS)
    return {
        lang: download_language(
            lang, output_dir,
            train_budget_mb=train_budget_mb,
            eval_budget_mb=eval_budget_mb,
        )
        for lang in languages
    }


def verify_fineweb2_configs(languages: list[str]) -> None:
    """Sanity-check that FineWeb 2 exposes the FLORES configs we expect.

    No-op for languages that don't use fineweb-2.
    """
    needed = [
        cfg["config"]
        for lang, cfg in LANGUAGE_CONFIGS.items()
        if lang in languages and cfg["repo"] == "HuggingFaceFW/fineweb-2"
    ]
    if not needed:
        return
    try:
        from datasets import get_dataset_config_names
        configs = set(get_dataset_config_names("HuggingFaceFW/fineweb-2"))
    except Exception as exc:
        print(f"[warn] could not verify fineweb-2 configs: {exc}")
        return
    missing = [c for c in needed if c not in configs]
    if missing:
        raise RuntimeError(
            f"Expected FineWeb 2 configs not found: {missing}. "
            f"Sample of available: {sorted(configs)[:10]}..."
        )
