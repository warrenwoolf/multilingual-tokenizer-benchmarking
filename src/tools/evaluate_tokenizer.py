"""High-level operation: evaluate one tokenizer artifact against a corpus."""

from pathlib import Path

from src.utils.evaluation_metrics import compute_all_metrics
from src.utils.tokenizer_algorithms import load_tokenizer


def evaluate_tokenizer_artifact(
    artifact_dir: str | Path,
    eval_corpus_path: str | Path,
    algorithm: str,
) -> dict:
    """Load a tokenizer artifact and compute evaluation metrics on a corpus.

    Args:
        artifact_dir: Directory produced by create_tokenizer_artifact.
        eval_corpus_path: Plain-text corpus to evaluate on.
        algorithm: Algorithm string needed to dispatch the correct loader.

    Returns:
        Dict mapping metric name to value, e.g. {"fertility": 1.42, ...}.
    """
    tokenizer = load_tokenizer(Path(artifact_dir), algorithm=algorithm)
    corpus = Path(eval_corpus_path).read_text(encoding="utf-8")
    sentences = [s for s in corpus.splitlines() if s.strip()]
    return compute_all_metrics(tokenizer=tokenizer, sentences=sentences)
