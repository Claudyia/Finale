# Step C (parte 1).
#
# Unisce dataset_clean + tutte le cartelle external_staging/* in un unico manifest
# per il k-fold, con dedup GLOBALE per hash (una immagine identica presente in piu
# sorgenti viene tenuta una volta sola, secondo l'ordine di priorita).
#
# Non copia immagini: scrive solo un CSV con i percorsi. I fold poi leggono da qui.
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# Ordine di priorita: chi viene prima "vince" quando due sorgenti hanno la stessa
# immagine. dataset_clean per primo (curato), poi le sorgenti con scene miste.
# NB: construction_ppe NON e qui: e tenuto interamente fuori come test set
# (vedi dataset_test_v2 / build_test_set.py).
DEFAULT_SOURCES = [
    "research_phases/06_kfold_pulito/dataset_clean",
    "research_phases/06_kfold_pulito/external_staging/site_safety",
    "research_phases/06_kfold_pulito/external_staging/ppe_combined",
    "research_phases/06_kfold_pulito/external_staging/hardhat_kaggle",
    "research_phases/06_kfold_pulito/external_staging/hardneg_backpack1",
    "research_phases/06_kfold_pulito/external_staging/hardneg_backpack2",
    "research_phases/06_kfold_pulito/external_staging/hardneg_baseball",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step C.1: manifest unico per il k-fold, dedup globale.")
    parser.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES,
                        help="Cartelle con images/, labels/ e sources.csv, in ordine di priorita.")
    parser.add_argument("--out", default="research_phases/06_kfold_pulito/kfold/merged_manifest.csv")
    parser.add_argument("--exclude-sources", nargs="*", default=[],
                        help="Nomi di sorgente (colonna source) da NON includere.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}      # sha1 -> stem tenuto
    rows_out: list[dict] = []
    per_source_kept: Counter[str] = Counter()
    per_source_dropped: Counter[str] = Counter()
    cross_collisions: list[dict] = []

    for source_dir in args.sources:
        root = Path(source_dir)
        csv_path = root / "sources.csv"
        images_dir = root / "images"
        labels_dir = root / "labels"
        if not csv_path.exists():
            print(f"  ATTENZIONE: manca {csv_path}, salto")
            continue

        # Indicizza una volta sola i file immagine della cartella: stem -> path.
        image_by_stem: dict[str, Path] = {}
        if images_dir.is_dir():
            for path in images_dir.iterdir():
                if path.is_file():
                    image_by_stem.setdefault(path.stem, path)

        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                source = row["source"]
                if source in args.exclude_sources:
                    continue
                sha1 = row["sha1"]
                if sha1 in seen_hashes:
                    per_source_dropped[source] += 1
                    cross_collisions.append(
                        {"sha1": sha1, "kept": seen_hashes[sha1], "dropped": row["stem"], "dropped_source": source}
                    )
                    continue
                seen_hashes[sha1] = row["stem"]

                # Percorso immagine: il nome file puo avere estensione diversa dallo stem.
                image_path = image_by_stem.get(row["stem"], images_dir / row["image"])
                label_path = labels_dir / f"{row['stem']}.txt"

                rows_out.append(
                    {
                        "stem": row["stem"],
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "source": source,
                        "group_id": row["group_id"],
                        "n_helmet": int(row["n_helmet"]),
                        "n_vest": int(row["n_vest"]),
                        "is_empty_label": int(row["is_empty_label"]),
                        "sha1": sha1,
                    }
                )
                per_source_kept[source] += 1

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stem", "image_path", "label_path", "source", "group_id",
                        "n_helmet", "n_vest", "is_empty_label", "sha1"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    collisions_path = out_path.with_name("cross_source_collisions.csv")
    with collisions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha1", "kept", "dropped", "dropped_source"])
        writer.writeheader()
        writer.writerows(cross_collisions)

    # --- Riepilogo ---
    total = len(rows_out)
    helmet_boxes = sum(r["n_helmet"] for r in rows_out)
    vest_boxes = sum(r["n_vest"] for r in rows_out)
    imgs_helmet = sum(1 for r in rows_out if r["n_helmet"] > 0)
    imgs_vest = sum(1 for r in rows_out if r["n_vest"] > 0)
    imgs_mixed = sum(1 for r in rows_out if r["n_helmet"] > 0 and r["n_vest"] > 0)
    imgs_neg = sum(1 for r in rows_out if r["is_empty_label"])
    groups = len({r["group_id"] for r in rows_out})

    lines = []
    lines.append(f"Immagini uniche totali : {total}")
    lines.append("")
    lines.append("Per sorgente (tenute / scartate come duplicato cross-source):")
    for source in sorted(set(per_source_kept) | set(per_source_dropped)):
        lines.append(f"  {source:16s} {per_source_kept[source]:6d}  / {per_source_dropped[source]}")
    lines.append("")
    lines.append("Contenuto:")
    lines.append(f"  box helmet : {helmet_boxes}   (in {imgs_helmet} immagini)")
    lines.append(f"  box vest   : {vest_boxes}   (in {imgs_vest} immagini)")
    lines.append(f"  immagini helmet+vest : {imgs_mixed}")
    lines.append(f"  immagini negative    : {imgs_neg}")
    if vest_boxes:
        lines.append(f"  rapporto helmet/vest : {helmet_boxes / vest_boxes:.2f}")
    lines.append(f"  gruppi (group_id)    : {groups}")
    lines.append("")
    lines.append(f"Manifest    : {out_path}")
    lines.append(f"Collisioni  : {collisions_path} ({len(cross_collisions)} righe)")

    summary = "\n".join(lines)
    out_path.with_name("merge_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
