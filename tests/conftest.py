"""Shared test fixtures for talkengine tests."""

import shutil
from pathlib import Path
from typing import Union


TMP_PATH_EXAMPLES: str = "./tests/tmp"


def _copy_directory(
    src_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    exclude: list[str] | None = None,
) -> None:
    """Copy a directory to a destination.

    Args:
        src_dir: Source directory path
        dest_dir: Destination directory path
        exclude: list of file/directory names to exclude from copying
    """
    exclude = exclude or []
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if not dest_path.exists():
        dest_path.mkdir(parents=True)

    for item in src_path.iterdir():
        if item.name in exclude:
            continue

        if item.is_dir():
            _copy_directory(item, dest_path / item.name, exclude)
        else:
            shutil.copy2(item, dest_path / item.name)
