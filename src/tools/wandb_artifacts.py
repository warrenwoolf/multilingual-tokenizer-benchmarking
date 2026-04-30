"""Push and pull tokenizer artifacts to/from Weights & Biases.

Intended for a split CPU/GPU workflow: train tokenizers on a CPU node,
push artifacts to W&B, then pull them on a GPU node before LLM evaluation.

Usage::

    from src.tools.wandb_artifacts import push_tokenizer_artifacts, pull_tokenizer_artifacts

    # CPU node: after generate_all_tokenizers()
    push_tokenizer_artifacts("artifacts", project="tokenizer-bench")

    # GPU node: before train_all_llms()
    pull_tokenizer_artifacts("artifacts", project="tokenizer-bench")
"""
from __future__ import annotations

from pathlib import Path

from src.utils.llm_training import _ensure_wandb_login

ARTIFACT_TYPE = "tokenizer"


def push_tokenizer_artifacts(
    artifact_dir: str | Path,
    project: str,
    entity: str | None = None,
    run_name: str = "upload-tokenizers",
) -> None:
    """Upload every tokenizer artifact directory under artifact_dir to W&B.

    Each subdirectory that contains a tokenizer.json is uploaded as a
    separate W&B Artifact named after the directory (e.g. en_bpe_v8000).
    """
    import wandb

    _ensure_wandb_login()
    artifact_dir = Path(artifact_dir)
    candidates = sorted(
        d for d in artifact_dir.iterdir()
        if d.is_dir() and (d / "tokenizer.json").exists()
    )
    if not candidates:
        print(f"[wandb] No tokenizer artifacts found under {artifact_dir}")
        return

    print(f"[wandb] Uploading {len(candidates)} tokenizer artifact(s) to {project}...")
    with wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type="upload-tokenizers",
    ) as run:
        for artifact_path in candidates:
            art = wandb.Artifact(name=artifact_path.name, type=ARTIFACT_TYPE)
            # Exclude the hidden corpus copy — it can be hundreds of MB and
            # is reproducible from the source data.
            for item in artifact_path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    art.add_dir(str(item), name=item.name)
                else:
                    art.add_file(str(item))
            run.log_artifact(art)
            print(f"  uploaded: {artifact_path.name}")

    print("[wandb] Upload complete.")


def pull_tokenizer_artifacts(
    artifact_dir: str | Path,
    project: str,
    entity: str | None = None,
) -> None:
    """Download all tokenizer artifacts from W&B into artifact_dir.

    Each artifact is downloaded into artifact_dir/{artifact_name}/.
    Existing directories are skipped (idempotent).
    """
    import wandb

    _ensure_wandb_login()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    try:
        artifacts = api.artifacts(type_name=ARTIFACT_TYPE, project=project_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not list artifacts in W&B project '{project_path}'. "
            f"Check that the project exists and WANDB_API_KEY is set. Original error: {exc}"
        ) from exc

    pulled = 0
    skipped = 0
    for art in artifacts:
        dest = artifact_dir / art.name
        if dest.exists() and (dest / "tokenizer.json").exists():
            print(f"  skip (already present): {art.name}")
            skipped += 1
            continue
        dest.mkdir(parents=True, exist_ok=True)
        art.download(root=str(dest))
        print(f"  pulled: {art.name}")
        pulled += 1

    print(f"[wandb] Pull complete — {pulled} downloaded, {skipped} already present.")
