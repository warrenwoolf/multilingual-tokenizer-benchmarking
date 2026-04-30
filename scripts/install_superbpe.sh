#!/usr/bin/env bash
set -euo pipefail

# Installs the official SuperBPE repo and its patched tokenizers fork into
# the workspace. Defaults to cloning into third_party/superbpe. This creates
# a local virtualenv at $SUPERBPE_REPO/.venv and installs the project there.

SUPERBPE_REPO=${SUPERBPE_REPO:-third_party/superbpe}
PYTHON=${PYTHON:-python3}

echo "Installing SuperBPE into: $SUPERBPE_REPO"

mkdir -p "$(dirname "$SUPERBPE_REPO")"
if [ ! -d "$SUPERBPE_REPO" ]; then
  git clone --recurse-submodules https://github.com/PythonNut/superbpe.git "$SUPERBPE_REPO"
else
  echo "Directory exists, updating..."
  git -C "$SUPERBPE_REPO" pull --rebase || true
  git -C "$SUPERBPE_REPO" submodule update --init --recursive
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Warning: cargo (Rust) not found. Install Rust toolchain (rustup) before building native extensions."
fi

pushd "$SUPERBPE_REPO" >/dev/null

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv at $SUPERBPE_REPO/.venv"
  $PYTHON -m venv .venv
fi

VENV_PIP="$(pwd)/.venv/bin/pip"
VENV_PY="$(pwd)/.venv/bin/python"

echo "Upgrading pip in venv"
"$VENV_PIP" install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  echo "Installing Python requirements into venv"
  "$VENV_PIP" install -r requirements.txt
fi

# Build & install the patched tokenizers crate if present
if [ -d tokenizers_superbpe ] || [ -d tokenizers ]; then
  # common layout: tokenizers_superbpe/bindings/python or tokenizers/bindings/python
  if [ -d tokenizers_superbpe/bindings/python ]; then
    BINDINGS_DIR=tokenizers_superbpe/bindings/python
  else
    BINDINGS_DIR=tokenizers/bindings/python
  fi
  if [ -d "$BINDINGS_DIR" ]; then
    echo "Building Python wheel for patched tokenizers from $BINDINGS_DIR"
    mkdir -p wheels
    "$VENV_PIP" wheel "$BINDINGS_DIR" -w wheels
    echo "Installing built wheel into venv"
    "$VENV_PIP" install wheels/*.whl || true
  else
    echo "No Python bindings directory found for patched tokenizers. Skipping wheel build."
  fi
else
  echo "No tokenizers fork found as a subdirectory. The upstream repository uses a submodule; ensure submodules were initialized."
fi

echo "SuperBPE installed. To use it, activate: source $SUPERBPE_REPO/.venv/bin/activate"
echo "Set SUPERBPE_REPO=$SUPERBPE_REPO to point scripts to the installation location."

popd >/dev/null
