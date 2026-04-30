.PHONY: install install-llm test download train evaluate train-llms all clean

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev]"

# Pulls torch + numpy for the downstream LLM perplexity evaluation.
install-llm:
	$(PY) -m pip install -e ".[llm]"

install-superbpe:
	@echo "Running SuperBPE installer script (may require git, cargo, rust, python3)"
	set -e; \
	bash ./scripts/install_superbpe.sh

test:
	$(PY) -m pytest tests/ -v

download:
	$(PY) download_data.py

train:
	$(PY) generate_tokenizers.py

evaluate:
	$(PY) evaluate_tokenizers.py

train-llms:
	$(PY) train_llms.py

all: download train evaluate

clean:
	rm -rf data/ artifacts/ results.csv llm_results.csv .pytest_cache/ **/__pycache__
