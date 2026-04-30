"""Small wrapper around the official SuperBPE repository.

The official code lives in https://github.com/PythonNut/superbpe and
depends on the patched Rust-backed tokenizers fork under the hood. This
module keeps that dependency isolated in its own checkout and virtualenv,
so the main project never needs to embed the old manual implementation.
"""
from __future__ import annotations

import os
import shutil
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


SUPERBPE_STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]"
    r"+?[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SUPERBPE_STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def _repo_path() -> Path:
    return Path(os.environ.get("SUPERBPE_REPO", "third_party/superbpe"))


def _venv_bin_dir(repo: Path) -> Path:
    return repo / ".venv" / "bin"


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def _repo_python(repo: Path) -> Path:
    python = _venv_python(repo)
    if python.exists():
        return python
    raise FileNotFoundError(
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


def train_superbpe(corpus_path: str | Path, vocab_size: int, output_dir: str | Path) -> None:
    """Train SuperBPE using the official PythonNut/superbpe repo.

    The official trainer performs stage 1 and stage 2 through its own
    patched tokenizers dependency. We run the repo's `train_tokenizer`
    module twice: once with whitespace-aware pretokenization and once with
    whitespace disabled, reusing the same output directory so stage 2
    resumes from the stage-1 merges.
    """
    repo = _repo_path()
    if not repo.exists():
        raise RuntimeError(
            f"SuperBPE repo not found at {repo}. Run scripts/install_superbpe.sh or make install-superbpe first."
        )

    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="superbpe_corpus_") as tmpdir:
        corpus_dir = Path(tmpdir)
        copied_corpus = corpus_dir / corpus_path.name
        shutil.copy2(corpus_path, copied_corpus)
        num_bytes = corpus_path.stat().st_size
        stage1_vocab = max(1, int(vocab_size * 0.9))

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
                str(num_bytes),
                "--vocab_size",
                str(stage1_vocab),
                "--regex_string",
                SUPERBPE_STAGE1_REGEX,
            ],
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
                str(num_bytes),
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
