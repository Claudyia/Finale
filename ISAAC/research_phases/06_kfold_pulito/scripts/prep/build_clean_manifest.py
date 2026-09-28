# Step A del k-fold pulito.
#
# Legge un dataset YOLO gia assemblato, unisce train + val, rimuove i duplicati
# byte-identici e produce un manifest con sorgente e group_id per ogni immagine
# unica. Non copia e non modifica nessun file: scrive solo dei CSV di analisi.
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "val")

# Marcatori che Roboflow aggiunge in coda al nome della foto sorgente.
# Tutto quello che viene dopo identifica la singola variante di augmentation,
# non una foto diversa.
ROBOFLOW_MARKERS = ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", ".rf.")


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash SHA-1 del contenuto del file, letto a blocchi."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(dataset_root: Path) -> list[tuple[Path, str]]:
    """Elenca (percorso_immagine, split_originale) per train e val."""
    found: list[tuple[Path, str]] = []
    for split in SPLITS:
        split_dir = dataset_root / "images" / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                found.append((path, split))
    return found


def label_for(dataset_root: Path, image_path: Path, split: str) -> Path:
    """Percorso del file di label YOLO corrispondente all'immagine."""
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def count_classes(label_path: Path) -> tuple[int, int, bool, bool]:
    """Ritorna (n_helmet, n_vest, label_presente, label_vuota)."""
    if not label_path.exists():
        return 0, 0, False, True
    n_helmet = 0
    n_vest = 0
    has_any = False
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        has_any = True
        if class_id == 0:
            n_helmet += 1
        elif class_id == 1:
            n_vest += 1
    return n_helmet, n_vest, True, not has_any


def classify_source(stem: str) -> str:
    """Sorgente logica dell'immagine, dedotta dal prefisso del nome file."""
    if stem.startswith("oversample_vest"):
        return "vest_oversample"
    if stem.startswith("vestboost_"):
        return "vest_ext"
    if stem.startswith("ppe_"):
        return "ppe_orig"
    if stem.startswith("oodneg_") or stem.startswith("ood_negative_"):
        return "oodneg"
    return "unknown"


def strip_roboflow(stem: str) -> str:
    """Toglie il suffisso Roboflow: le varianti aug. condividono lo stesso prefisso."""
    for marker in ROBOFLOW_MARKERS:
        index = stem.find(marker)
        if index != -1:
            return stem[:index]
    return stem


def derive_group_id(stem: str, source: str) -> str:
    """
    Chiave che tiene insieme le immagini che vengono dalla stessa foto/clip:
    varianti di augmentation, crop multipli, frame dello stesso video.
    Le foto senza varianti finiscono ognuna in un gruppo da sola (corretto).
    """
    base = strip_roboflow(stem)

    if source == "oodneg":
        # ood_negative_backpack_person_001_1a2b3c...  -> ...person_001
        base = re.sub(r"_[0-9a-f]{6,}$", "", base)
        # oodneg_00007_2009_000059  -> 2009_000059  (via l'indice progressivo)
        base = re.sub(r"^oodneg_\d+_", "", base)
        # frame dello stesso video: autox3_mp4-187 / autox3_mp4-277 -> autox3_mp4
        base = re.sub(r"[-_]\d+$", "", base) if re.search(r"(mp4|video|youtube)", base) else base
    elif source == "vest_ext":
        base = re.sub(r"^vestboost_(train|val)_\d+_", "", base)
        base = re.sub(r"^jhboyo_(train|val)_\d+_", "", base)
    elif source == "vest_oversample":
        base = re.sub(r"^oversample_vest[_-]?\d*[_-]?", "", base)
    elif source == "ppe_orig":
        base = re.sub(r"^ppe_(train|val)_image[_-]?\d*[_-]?", "", base)

    base = base.strip("_-") or stem
    return f"{source}:{base}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step A: manifest pulito (dedup + sorgente + group_id) da un dataset YOLO."
    )
    parser.add_argument(
        "--dataset",
        default="scripts/finetunig/dataset/finetune_dataset_ood_hardneg_vestboost_helmetboost",
        help="Dataset YOLO con images/train, images/val, labels/train, labels/val.",
    )
    parser.add_argument(
        "--out-dir",
        default="research_phases/06_kfold_pulito/manifest",
    )
    parser.add_argument(
        "--keep-oversample",
        action="store_true",
        help="Non trattare le immagini oversample_vest come sorgente separata da scartare.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(dataset_root)
    if not images:
        raise FileNotFoundError(f"Nessuna immagine trovata in: {dataset_root}")

    # Prima passata: hash di tutte le immagini.
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for image_path, split in images:
        stem = image_path.stem
        source = classify_source(stem)
        label_path = label_for(dataset_root, image_path, split)
        n_helmet, n_vest, has_label, empty_label = count_classes(label_path)
        record = {
            "image_path": str(image_path),
            "label_path": str(label_path) if has_label else "",
            "stem": stem,
            "source": source,
            "group_id": derive_group_id(stem, source),
            "orig_split": split,
            "n_helmet": n_helmet,
            "n_vest": n_vest,
            "is_empty_label": int(empty_label),
            "has_label_file": int(has_label),
        }
        record["sha1"] = sha1_of_file(image_path)
        by_hash[record["sha1"]].append(record)

    # Seconda passata: tiene un record per hash, registra gli scarti.
    kept: list[dict] = []
    removed_rows: list[dict] = []
    for sha1, records in by_hash.items():
        # Ordina in modo stabile: prima le sorgenti "vere", poi l'oversample.
        records.sort(key=lambda r: (r["source"] == "vest_oversample", r["stem"]))
        keeper = records[0]
        shared_sources = sorted({r["source"] for r in records})
        keeper["dup_count"] = len(records)
        keeper["shared_sources"] = "|".join(shared_sources)
        kept.append(keeper)
        for dropped in records[1:]:
            removed_rows.append(
                {
                    "sha1": sha1,
                    "kept_file": keeper["stem"],
                    "kept_source": keeper["source"],
                    "removed_file": dropped["stem"],
                    "removed_source": dropped["source"],
                }
            )

    kept.sort(key=lambda r: r["stem"])

    # Se una foto vera e la sua copia oversample hanno hash diversi (re-encoding),
    # l'oversample resta come immagine a se. La marchiamo comunque come vest.
    for record in kept:
        if record["source"] == "vest_oversample" and not args.keep_oversample:
            record["source_effective"] = "vest_ext"
        else:
            record["source_effective"] = record["source"]

    # --- Scrittura CSV ---
    manifest_path = out_dir / "manifest_clean.csv"
    fieldnames = [
        "stem", "image_path", "label_path", "sha1", "source", "source_effective",
        "group_id", "orig_split", "n_helmet", "n_vest", "is_empty_label",
        "has_label_file", "dup_count", "shared_sources",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    dup_path = out_dir / "duplicates_removed.csv"
    with dup_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha1", "kept_file", "kept_source", "removed_file", "removed_source"],
        )
        writer.writeheader()
        writer.writerows(removed_rows)

    group_counter = Counter(r["group_id"] for r in kept)
    groups_path = out_dir / "group_sizes.csv"
    with groups_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "n_images"])
        for group_id, count in group_counter.most_common():
            if count > 1:
                writer.writerow([group_id, count])

    # --- Riepilogo ---
    total_files = len(images)
    unique_images = len(kept)
    by_source = Counter(r["source_effective"] for r in kept)
    helmet_instances = sum(r["n_helmet"] for r in kept)
    vest_instances = sum(r["n_vest"] for r in kept)
    imgs_with_helmet = sum(1 for r in kept if r["n_helmet"] > 0)
    imgs_with_vest = sum(1 for r in kept if r["n_vest"] > 0)
    empty_imgs = sum(1 for r in kept if r["is_empty_label"])
    multi_groups = sum(1 for c in group_counter.values() if c > 1)

    lines = []
    lines.append(f"Dataset sorgente : {dataset_root}")
    lines.append(f"File totali (train+val)      : {total_files}")
    lines.append(f"Immagini uniche (post dedup) : {unique_images}")
    lines.append(f"Duplicati rimossi            : {total_files - unique_images}")
    lines.append("")
    lines.append("Immagini uniche per sorgente:")
    for source, count in by_source.most_common():
        lines.append(f"  {source:16s} {count}")
    lines.append("")
    lines.append("Istanze (box) per classe, sulle immagini uniche:")
    lines.append(f"  helmet : {helmet_instances}  (in {imgs_with_helmet} immagini)")
    lines.append(f"  vest   : {vest_instances}  (in {imgs_with_vest} immagini)")
    lines.append(f"  negative (label vuota) : {empty_imgs} immagini")
    if vest_instances:
        lines.append(f"  rapporto helmet/vest : {helmet_instances / vest_instances:.2f}")
    lines.append("")
    lines.append(f"Gruppi totali        : {len(group_counter)}")
    lines.append(f"Gruppi con >1 immagine: {multi_groups}")
    lines.append("")
    lines.append("Output:")
    lines.append(f"  {manifest_path}")
    lines.append(f"  {dup_path}")
    lines.append(f"  {groups_path}")

    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
