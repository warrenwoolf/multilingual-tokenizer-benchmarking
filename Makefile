.PHONY: install install-superbpe test download train evaluate all clean

PY ?= python3
SUPERBPE_REPO ?= third_party/superbpe

install:
	$(PY) -m pip install -e ".[dev]"

# SuperBPE needs Python 3.12, a Rust toolchain, and the patched tokenizers fork.
# This target clones the official repo under third_party/ and builds its deps.
install-superbpe:
	@command -v cargo >/dev/null 2>&1 || { echo "ERROR: Rust toolchain (cargo) not found. Install from https://rustup.rs"; exit 1; }
	@$(PY) -c "import sys; assert sys.version_info[:2] >= (3, 12), 'SuperBPE requires Python 3.12+'" \
		|| { echo "ERROR: SuperBPE requires Python 3.12+. Current: $$($(PY) --version)"; exit 1; }
	mkdir -p third_party
	@if [ ! -d "$(SUPERBPE_REPO)" ]; then \
		git clone --recurse-submodules https://github.com/PythonNut/superbpe.git $(SUPERBPE_REPO); \
	fi
	cd $(SUPERBPE_REPO) && $(PY) -m pip install -r requirements.txt
	$(PY) -m pip install "git+https://github.com/alisawuffles/tokenizers-superbpe.git@main#subdirectory=bindings/python"
	@echo "SuperBPE installed at $(SUPERBPE_REPO). Set SUPERBPE_REPO env var if you move it."

test:
	$(PY) -m pytest tests/ -v

download:
	$(PY) download_data.py

train:
	$(PY) generate_tokenizers.py

evaluate:
	$(PY) evaluate_tokenizers.py

all: download train evaluate

clean:
	rm -rf data/ artifacts/ results.csv .pytest_cache/ **/__pycache__
