#!/usr/bin/env python3
"""create ID/OOD folders for GL-MCM from downloaded image sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Source:
    split: str
    name: str
    path: Path
    limit: int | None


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def stable_name(path: Path, source_name: str) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{source_name}_{path.stem}_{digest}{path.suffix.lower()}"


def copy_source(source: Source, dataset_root: Path, rows: list[dict[str, str]]) -> int:
    if source.split not in {"id", "ood"}:
        raise ValueError(f"Invalid split for {source.name}: {source.split}")
    if not source.path.exists():
        raise FileNotFoundError(f"Missing source path: {source.path}")

    target_dir = dataset_root / source.split / source.name
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in iter_images(source.path):
        if source.limit is not None and copied >= source.limit:
            break
        target_path = target_dir / stable_name(image_path, source.name)
        shutil.copy2(image_path, target_path)
        rows.append(
            {
                "split": source.split,
                "source": source.name,
                "original_path": str(image_path),
                "dataset_path": str(target_path),
            }
        )
        copied += 1
    return copied


def parse_source(value: str) -> Source:
    parts = value.split(":", 3)
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "Source must be split:name:path or split:name:path:limit"
        )
    split, name, raw_path = parts[:3]
    limit = int(parts[3]) if len(parts) == 4 else None
    return Source(split=split, name=name, path=Path(raw_path).expanduser(), limit=limit)


def write_manifest(dataset_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = dataset_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source", "original_path", "dataset_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create ID/OOD folders for GL-MCM from downloaded image sources."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Dataset root containing id/, ood/, raw/. Default: parent folder.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        help="Source as split:name:path[:limit]. Example: id:ppe:raw/ppe_dataset:300",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    rows: list[dict[str, str]] = []
    totals: dict[str, int] = {}

    for source in args.source:
        copied = copy_source(source, dataset_root, rows)
        totals[f"{source.split}/{source.name}"] = copied

    write_manifest(dataset_root, rows)

    print("Dataset built:")
    for name, copied in totals.items():
        print(f"  {name}: {copied} images")
    print(f"Manifest: {dataset_root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
