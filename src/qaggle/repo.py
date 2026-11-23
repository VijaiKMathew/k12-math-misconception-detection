"""Repository paths and configuration loader."""

from __future__ import annotations

import json
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("pyproject.toml not found above qaggle.repo")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def competition() -> dict:
    return read_json(repo_root() / "configs" / "competition.json")


def data_root() -> Path:
    return repo_root() / "data" / "competition"


def train_path() -> Path:
    return data_root() / "train.csv"


def test_path() -> Path:
    return data_root() / "test.csv"


def taxonomy_path() -> Path:
    return data_root() / "taxonomy_v1.csv"


def sample_submission_path() -> Path:
    return data_root() / "sample_submission.csv"


def processed_root() -> Path:
    return repo_root() / "data" / "processed"


def train_parquet_path() -> Path:
    return processed_root() / "train.parquet"


def test_parquet_path() -> Path:
    return processed_root() / "test.parquet"
