# Step nuovo, tra check_test_leakage.py e merge_kfold_manifest.py.
#
# Legge il leakage_report.csv (prodotto da check_test_leakage.py, match esatti
# dataset_clean vs dataset_test_isaac) e toglie quelle immagini dal manifest
# unito, senza toccare merged_manifest.csv originale: scrive una copia filtrata.
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_leaked_stems(leakage_report: Path) -> set[str]:
    stems: set[str] = set()
    with leakage_report.open(newline="") as handle:
        for row in csv.DictReader(handle):
            train_file = row["train_file"]
            stem = train_file.rsplit(".", 1)[0] if "." in train_file else train_file
            stems.add(stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toglie dal manifest le immagini segnalate come leakage col test set."
    )
    parser.add_argument(
        "--leakage-report",
        default="research_phases/06_kfold_pulito/manifest/leakage_report.csv",
    )
    parser.add_argument(
        "--manifest",
        default="research_phases/06_kfold_pulito/kfold/merged_manifest.csv",
    )
    parser.add_argument(
        "--out",
        default="research_phases/06_kfold_pulito/kfold/merged_manifest_no_leak.csv",
    )
    args = parser.parse_args()

    leaked_stems = load_leaked_stems(Path(args.leakage_report))
    print(f"Stem in leakage_report: {len(leaked_stems)}")

    manifest_path = Path(args.manifest)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [row for row in rows if row["stem"] not in leaked_stems]
    removed = [row for row in rows if row["stem"] in leaked_stems]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Manifest originale : {len(rows)} immagini")
    print(f"Rimosse (leakage)  : {len(removed)}")
    print(f"Rimaste            : {len(kept)}")
    print(f"Output: {out_path}")

    not_found = leaked_stems - {row["stem"] for row in removed}
    if not_found:
        print(
            f"\nNota: {len(not_found)} stem del leakage report non erano nel manifest "
            "(gia' rimossi altrove, o naming diverso)."
        )


if __name__ == "__main__":
    main()
