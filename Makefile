SHELL := /bin/bash
PYTHON := uv run python
COMPETITION := misconception-detection-in-k-12-student-math-workings

.PHONY: help setup data check report clean all

help:
	@echo "================================================================================"
	@echo " K-12 Math Misconceptions Champion Pipeline"
	@echo "================================================================================"
	@echo "  make setup     - Create virtual environment and sync dependencies via uv"
	@echo "  make data      - Ingest competition dataset via Kaggle CLI"
	@echo "  make check     - Run pre-flight integrity, secrets, and champion verification"
	@echo "  make report    - Verify standalone interactive HTML solution preview"
	@echo "  make clean     - Remove temporary caches and pycache"
	@echo "  make all       - Run end-to-end setup and verification suite"
	@echo "================================================================================"

setup:
	@echo "==> Setting up environment with uv..."
	uv sync

data:
	@echo "==> Ingesting dataset for $(COMPETITION)..."
	mkdir -p data/competition
	kaggle competitions download -c $(COMPETITION) -p data/
	unzip -o -q data/$(COMPETITION).zip -d data/competition/
	@echo "==> Extraction complete."

check:
	@echo "==> Running champion configuration & secrets check..."
	$(PYTHON) tools/checks/check_champion.py
	$(PYTHON) tools/checks/check_secrets.py

report:
	@echo "==> Verifying interactive HTML solution showcase..."
	@test -f reports/solution_preview.html && echo "Preview verified: reports/solution_preview.html"

clean:
	@echo "==> Cleaning transient caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .pytest_cache

all: setup check report
	@echo "==> End-to-end pipeline verification complete!"
