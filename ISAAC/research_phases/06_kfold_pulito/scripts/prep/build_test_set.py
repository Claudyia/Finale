# Step G (preparazione) - test set.
#
# Il test set e il dataset construction_ppe tenuto INTERAMENTE fuori dal training.
# Qui si prende la sua versione gia convertita (external_staging/construction_ppe),
# si verifica che non ci sia overlap di hash col training e si scrive dataset_test_v2/.
#
# Non addestra niente.
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


def load_hashes(csv_path: Path) -> set[str]:
    with csv_path.open(newline="") as handle:
        return {row["sha1"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step G: crea dataset_test_v2 dal dataset held-out.")
    parser.add_argument("--staging", default="research_phases/06_kfold_pulito/external_staging/construction_ppe")
    parser.add_argument("--manifest", default="research_phases/06_kfold_pulito/kfold/merged_manifest.csv")
    parser.add_argument("--out-dir", default="research_phases/06_kfold_pulito/dataset_test_v2")
    args = parser.parse_args()

    staging = Path(args.staging)
    src_csv = staging / "sources.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Manca {src_csv} (esegui prima convert_external_ppe.py).")

    train_hashes = load_hashes(Path(args.manifest)) if Path(args.manifest).exists() else set()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    rows_out: list[dict] = []
    image_by_stem = {p.stem: p for p in (staging / "images").iterdir() if p.is_file()}

    with src_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stats["seen"] += 1
            if row["sha1"] in train_hashes:
                stats["skip_in_train"] += 1
                continue
            image_path = image_by_stem.get(row["stem"])
            label_path = staging / "labels" / f"{row['stem']}.txt"
            if image_path is None or not label_path.exists():
                stats["skip_missing"] += 1
                continue
            shutil.copy2(image_path, out_dir / "images" / image_path.name)
            shutil.copy2(label_path, out_dir / "labels" / f"{row['stem']}.txt")
            stats["kept"] += 1
            stats["box_helmet"] += int(row["n_helmet"])
            stats["box_vest"] += int(row["n_vest"])
            if int(row["n_helmet"]) and int(row["n_vest"]):
                stats["img_mixed"] += 1
            rows_out.append(row)

    with (out_dir / "sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images\nval: images\ntest: images\n\n"
        "names:\n  0: helmet\n  1: vest\n"
    )

    print("--- TEST SET (construction_ppe, held-out) ---")
    for key in ("seen", "skip_in_train", "skip_missing", "kept"):
        print(f"  {key:16s} {stats[key]}")
    print(f"  immagini helmet+vest  {stats['img_mixed']}")
    print(f"  box helmet / vest     {stats['box_helmet']} / {stats['box_vest']}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
