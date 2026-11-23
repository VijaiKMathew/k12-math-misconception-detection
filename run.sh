#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo " K-12 Math Misconceptions Champion Solution Pipeline"
echo "================================================================================"

echo "==> Step 1: Syncing virtual environment via uv..."
uv sync

echo "==> Step 2: Running champion metadata and secrets verification..."
uv run python tools/checks/check_champion.py
uv run python tools/checks/check_secrets.py

echo "==> Step 3: Checking interactive report and walkthrough notebook..."
test -f reports/solution_preview.html && echo "Verified: reports/solution_preview.html"
test -f notebooks/misconception_detection_champion_masterclass.ipynb && echo "Verified: notebooks/misconception_detection_champion_masterclass.ipynb"

echo "================================================================================"
echo " SUCCESS: Solution verified and ready for deployment!"
echo " Writeup: WRITEUP.md"
echo " Interactive HTML: reports/solution_preview.html"
echo " Masterclass Notebook: notebooks/misconception_detection_champion_masterclass.ipynb"
echo "================================================================================"
