"""Competition asset inventory helpers for student and question images."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

STUDENT_IMAGE_SUBDIRS = (
    "competition_data/images/train",
    "competition_data/images/test",
)
QUESTION_IMAGE_SUBDIRS = ("competition_data/images/questions",)
DEFAULT_IMAGE_SUBDIRS = STUDENT_IMAGE_SUBDIRS + QUESTION_IMAGE_SUBDIRS
IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png"})


def _column_values(df, column: str) -> set[str]:
    if column not in df.columns:
        return set()
    values = df[column].dropna()
    return {str(value) for value in values if str(value) != ""}


def expected_asset_files(train_df, test_df) -> dict[str, set[str]]:
    """Return the unique student and question image filenames referenced by the CSVs."""
    student_images: set[str] = set()
    question_images: set[str] = set()
    for df in (train_df, test_df):
        student_images |= _column_values(df, "image_file")
        question_images |= _column_values(df, "question_image_file")
    return {
        "student_images": student_images,
        "question_images": question_images,
    }


def _is_image(path: Path) -> bool:
    return not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS


def _index_files(root: Path, image_subdirs: Iterable[str]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for subdir in image_subdirs:
        directory = root / subdir
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and _is_image(path):
                index.setdefault(path.name, []).append(path)
    return index


def _partition_subdirs(
    image_subdirs: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    student: list[str] = []
    questions: list[str] = []
    for subdir in image_subdirs:
        if "questions" in Path(subdir).parts:
            questions.append(subdir)
        else:
            student.append(subdir)
    return tuple(student), tuple(questions)


def verify_asset_inventory(
    root: Path,
    train_df,
    test_df,
    image_subdirs: Iterable[str] = DEFAULT_IMAGE_SUBDIRS,
) -> dict:
    """Resolve expected filenames under their category-specific dirs without raising."""
    root = Path(root)
    image_subdirs = tuple(image_subdirs)
    student_subdirs, question_subdirs = _partition_subdirs(image_subdirs)
    expected = expected_asset_files(train_df, test_df)
    expected_names = expected["student_images"] | expected["question_images"]

    student_index = _index_files(root, student_subdirs)
    question_index = _index_files(root, question_subdirs)
    all_index = {**student_index, **question_index}

    missing_names = (expected["student_images"] - set(student_index)) | (
        expected["question_images"] - set(question_index)
    )
    missing = sorted(missing_names)
    found = len(expected_names - missing_names)
    extra_count = sum(1 for name in all_index if name not in expected_names)

    return {
        "expected": len(expected_names),
        "found": found,
        "missing": missing,
        "extra_count": extra_count,
        "ok": not missing,
    }


def summarize_assets(report: dict) -> str:
    status = "OK" if report.get("ok") else "MISSING"
    return (
        f"{status}: {report.get('found', 0)}/{report.get('expected', 0)} expected assets found"
        f" ({len(report.get('missing', []))} missing, {report.get('extra_count', 0)} extra)"
    )
