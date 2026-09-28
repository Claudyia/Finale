# Step A (materializzazione).
#
# Copia le immagini uniche elencate in manifest_clean.csv in una cartella
# piatta (niente split train/val: lo split lo fa il k-fold).
# Copia anche le label; per le immagini negative scrive una label vuota.
#
# Non tocca il dataset sorgente.
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step A: crea dataset_clean/ dalle immagini uniche del manifest."
    )
    parser.add_argument(
        "--manifest",
        default="research_phases/06_kfold_pulito/manifest/manifest_clean.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="research_phases/06_kfold_pulito/dataset_clean",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Svuota la cartella di output se gia esiste.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest non trovato: {manifest_path}. Esegui prima build_clean_manifest.py"
        )

    out_dir = Path(args.out_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} non e vuota. Usa --overwrite per rifarla.")
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest_path.open(newline="")))
    sources_rows = []
    by_source: Counter[str] = Counter()
    missing_images = 0

    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.exists():
            missing_images += 1
            continue
        stem = row["stem"]
        dest_image = images_out / image_path.name
        shutil.copy2(image_path, dest_image)

        dest_label = labels_out / f"{stem}.txt"
        label_path = Path(row["label_path"]) if row["label_path"] else None
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_label)
        else:
            dest_label.write_text("")  # immagine negativa / hard negative

        by_source[row["source_effective"]] += 1
        sources_rows.append(
            {
                "image": dest_image.name,
                "stem": stem,
                "source": row["source_effective"],
                "group_id": row["group_id"],
                "n_helmet": row["n_helmet"],
                "n_vest": row["n_vest"],
                "is_empty_label": row["is_empty_label"],
                "sha1": row["sha1"],
            }
        )

    with (out_dir / "sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image", "stem", "source", "group_id",
                "n_helmet", "n_vest", "is_empty_label", "sha1",
            ],
        )
        writer.writeheader()
        writer.writerows(sources_rows)

    (out_dir / "data.yaml").write_text(
        "# Dataset base pulito. Lo split train/val lo generano i fold.\n"
        f"path: {out_dir.resolve()}\n"
        "train: images\n"
        "val: images\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: vest\n"
    )

    print(f"Immagini copiate : {len(sources_rows)}")
    if missing_images:
        print(f"Immagini mancanti (saltate): {missing_images}")
    for source, count in by_source.most_common():
        print(f"  {source:14s} {count}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
