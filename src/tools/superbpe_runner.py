"""Small wrapper around the official SuperBPE repository.

The official code lives in https://github.com/PythonNut/superbpe and
depends on the patched Rust-backed tokenizers fork under the hood. This
module keeps that dependency isolated in its own checkout and virtualenv,
so the main project never needs to embed the old manual implementation.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


SUPERBPE_STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SUPERBPE_STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# Stage-2 byte budget default (50 MB). Stage-2 pretokens are paragraph-sized
# because the regex doesn't split on letters, so cost is O(bytes × stage1_merges).
# 200 MB was the previous default but OOMs Colab (~12 GB RAM) at vocab ≥ 8 k.
# Override via SUPERBPE_STAGE2_BYTES env var.
_DEFAULT_STAGE2_BYTES = 5 * 10**7


class SuperBPESetupError(RuntimeError):
    """Raised when the external SuperBPE environment is missing or incomplete."""


def _repo_path() -> Path:
    return Path(os.environ.get("SUPERBPE_REPO", "third_party/superbpe")).expanduser().resolve()


def _venv_bin_dir(repo: Path) -> Path:
    return repo / ".venv" / "bin"


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def _repo_python(repo: Path) -> Path:
    python = _venv_python(repo)
    if python.exists():
        return python
    raise SuperBPESetupError(
        f"SuperBPE virtualenv not found at {python}. Run scripts/install_superbpe.sh first."
    )


def _run_repo_python(repo: Path, args: Sequence[str]) -> None:
    env = os.environ.copy()
    venv_bin = _venv_bin_dir(repo)
    if venv_bin.exists():
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
    cmd = [str(_repo_python(repo)), *map(str, args)]
    print("Running SuperBPE command:", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo), env=env, check=True)


def run_script(script: str, args: Sequence[str] | None = None) -> int:
    """Run a shell script inside the SuperBPE repo.

    The script is executed with `bash` and PATH is adjusted so that the repo
    venv's `bin` directory is first on PATH (so the repo's venv python is used).
    """
    repo = _repo_path()
    if not repo.exists():
        raise FileNotFoundError(f"SUPERBPE repo not found at {repo}; run scripts/install_superbpe.sh")

    script_path = repo / script
    if not script_path.exists():
        raise FileNotFoundError(f"Script {script} not found in {repo}")

    env = os.environ.copy()
    venv_bin = _venv_bin_dir(repo)
    if venv_bin.exists():
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")

    cmd = ["bash", str(script_path)]
    if args:
        cmd += list(args)

    print("Running SuperBPE script:", shlex.join(cmd))
    return subprocess.call(cmd, env=env)


def _prepare_superbpe_corpus(src: Path, dst: Path) -> None:
    """Copy corpus to dst, truncating lines above the p99 length threshold.

    Very long documents become giant single pretokens in stage 2 (the regex
    doesn't split on letters), disproportionately slowing merge computation.
    We replicate the original paper's top-1% truncation by computing the 99th
    percentile of line lengths and capping any line that exceeds it.

    Uses two streaming passes to avoid holding the entire corpus in RAM.
    """
    # Pass 1: collect lengths only (ints, not strings)
    lengths = []
    with src.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            lengths.append(len(line))

    if not lengths:
        dst.write_text("", encoding="utf-8")
        return

    lengths_sorted = sorted(lengths)
    p99_idx = max(0, int(len(lengths_sorted) * 0.99) - 1)
    threshold = lengths_sorted[p99_idx]
    del lengths_sorted

    # Pass 2: stream-copy, truncating long lines in place
    truncated = 0
    total = len(lengths)
    total_chars = sum(lengths)
    del lengths

    with src.open(encoding="utf-8", errors="replace") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if len(line) > threshold:
                line = line[:threshold].rstrip() + "\n"
                truncated += 1
            fout.write(line)

    mean_len = total_chars / total if total else 0
    print(
        f"[superbpe corpus] {total} documents — "
        f"p99 threshold {threshold:,} chars, "
        f"mean {mean_len:.0f} chars, "
        f"truncated {truncated} ({truncated / total:.1%})"
    )


def train_superbpe(
    corpus_path: str | Path,
    vocab_size: int,
    output_dir: str | Path,
    stage2_bytes: int | None = None,
) -> None:
    """Train SuperBPE using the official PythonNut/superbpe repo.

    The official trainer performs stage 1 and stage 2 through its own
    patched tokenizers dependency. We run the repo's `train_tokenizer`
    module twice: once with whitespace-aware pretokenization and once with
    whitespace disabled, reusing the same output directory so stage 2
    resumes from the stage-1 merges.

    stage2_bytes caps how much corpus stage 2 sees (default 200 MB via
    SUPERBPE_STAGE2_BYTES env var). Stage 2 cost is O(bytes × stage1_merges)
    because the regex doesn't split on letters, so unbounded corpus OOMs.
    Stage 1 always sees the full corpus.
    """
    repo = _repo_path()
    if not repo.exists():
        raise SuperBPESetupError(
            f"SuperBPE repo not found at {repo}. Run scripts/install_superbpe.sh or make install-superbpe first."
        )

    if stage2_bytes is None:
        stage2_bytes = int(os.environ.get("SUPERBPE_STAGE2_BYTES", _DEFAULT_STAGE2_BYTES))

    corpus_path = Path(corpus_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep a stable corpus directory under the artifact so retries/debugging
    # can rerun the exact same command without depending on a transient /tmp path.
    corpus_dir = output_dir / ".superbpe_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    copied_corpus = corpus_dir / "train.txt"
    _prepare_superbpe_corpus(corpus_path, copied_corpus)
    full_bytes = copied_corpus.stat().st_size
    stage2_bytes = min(stage2_bytes, full_bytes)
    stage1_vocab = max(1, int(vocab_size * 0.9))

    print(
        f"[superbpe] stage 1 — num_bytes={full_bytes:,} vocab={stage1_vocab}"
    )
    _run_repo_python(
        repo,
        [
            "-m",
            "train_tokenizer",
            "--output_dir",
            str(output_dir),
            "--corpus_dir",
            str(corpus_dir),
            "--num_bytes",
            str(full_bytes),
            "--vocab_size",
            str(stage1_vocab),
            "--regex_string",
            SUPERBPE_STAGE1_REGEX,
        ],
    )

    # Stage 2 must re-select corpus files at the smaller byte budget.
    # The upstream trainer reuses meta.json from stage 1 when it exists,
    # which would ignore --num_bytes and use the full stage-1 selection.
    # Delete it so stage 2 triggers a fresh, smaller selection.
    meta_json = output_dir / "meta.json"
    if meta_json.exists():
        meta_json.unlink()

    print(
        f"[superbpe] stage 2 — num_bytes={stage2_bytes:,} vocab={vocab_size}"
    )
    _run_repo_python(
        repo,
        [
            "-m",
            "train_tokenizer",
            "--output_dir",
            str(output_dir),
            "--corpus_dir",
            str(corpus_dir),
            "--num_bytes",
            str(stage2_bytes),
            "--vocab_size",
            str(vocab_size),
            "--regex_string",
            SUPERBPE_STAGE2_REGEX,
        ],
    )


def train_stage1(repo: Path | None = None) -> int:
    return run_script("scripts/train_tokenizer.sh")


def extend_stage2(repo: Path | None = None) -> int:
    return run_script("scripts/extend_tokenizer.sh")
