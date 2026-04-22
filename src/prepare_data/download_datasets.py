"""Download raw corpora for each target language.

Target languages (initial set)
-------------------------------
en  – English    (analytic)
ru  – Russian    (synthetic / fusional)
hi  – Hindi      (synthetic / fusional)
+ others TBD

Primary sources: Wikipedia dumps, CC-100, OSCAR, or similar open corpora.
"""

from pathlib import Path

# Map ISO 639-1 code -> dataset identifiers (to be filled in per source)
LANGUAGE_CONFIGS: dict[str, dict] = {
    "en": {"source": "cc100", "subset": "en"},
    "ru": {"source": "cc100", "subset": "ru"},
    "hi": {"source": "cc100", "subset": "hi"},
}


def download_language(language: str, output_dir: str | Path) -> Path:
    """Download the raw corpus for one language and save it to output_dir.

    Args:
        language: ISO 639-1 code (must be a key in LANGUAGE_CONFIGS).
        output_dir: Directory where the raw corpus file will be written.

    Returns:
        Path to the downloaded file.
    """
    if language not in LANGUAGE_CONFIGS:
        raise ValueError(
            f"Language '{language}' not configured. Add it to LANGUAGE_CONFIGS."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"Downloading {language} not yet implemented")


def download_all(output_dir: str | Path) -> dict[str, Path]:
    """Download all configured languages and return a map of language -> path."""
    return {lang: download_language(lang, output_dir) for lang in LANGUAGE_CONFIGS}
