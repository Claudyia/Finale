# Step A del k-fold pulito.
#
# Confronta le immagini del manifest di training con quelle di un set di test
# (default: research_phases/dataset_test_isaac) per trovare immagini che sono
# finite sia nel training sia nel test.
#
# Controllo esatto: hash SHA-1 del contenuto (sempre attivo).
# Controllo percettuale: average-hash 8x8, distanza di Hamming (richiede Pillow,
#   opzionale con --perceptual).
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]


def load_train_hashes(manifest_path: Path) -> dict[str, str]:
    """sha1 -> stem, dal manifest_clean.csv prodotto da build_clean_manifest.py."""
    mapping: dict[str, str] = {}
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["sha1"]] = row["stem"]
    return mapping


def average_hash(path: Path, size: int = 8) -> int | None:
    """Average hash percettuale: 64 bit. None se l'immagine non e leggibile."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("--perceptual richiede Pillow: pip install pillow")
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((size, size))
            pixels = list(img.getdata())
    except Exception:
        return None
    avg = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= avg:
            bits |= 1 << index
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step A: cerca immagini di test finite anche nel training."
    )
    parser.add_argument(
        "--train-manifest",
        default="research_phases/06_kfold_pulito/manifest/manifest_clean.csv",
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        default=None,
        help="Cartella di test da controllare (ripetibile). "
        "Default: research_phases/dataset_test_isaac/images",
    )
    parser.add_argument(
        "--out-dir",
        default="research_phases/06_kfold_pulito/manifest",
    )
    parser.add_argument(
        "--perceptual",
        action="store_true",
        help="Cerca anche i quasi-duplicati con average-hash (richiede Pillow).",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="Distanza di Hamming massima per considerare due immagini quasi uguali.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest non trovato: {manifest_path}. Esegui prima build_clean_manifest.py"
        )

    test_dirs = [Path(d) for d in (args.test_dir or ["research_phases/dataset_test_isaac/images"])]
    for test_dir in test_dirs:
        if not test_dir.exists():
            raise FileNotFoundError(f"Cartella di test non trovata: {test_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_hashes = load_train_hashes(manifest_path)
    train_sha_set = set(train_hashes)
    print(f"Immagini di training nel manifest: {len(train_sha_set)}")

    train_phash: dict[Path, int] = {}
    if args.perceptual:
        print("Calcolo average-hash delle immagini di training...")
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                ph = average_hash(Path(row["image_path"]))
                if ph is not None:
                    train_phash[Path(row["image_path"])] = ph

    exact_rows: list[dict] = []
    near_rows: list[dict] = []

    for test_dir in test_dirs:
        test_images = iter_images(test_dir)
        print(f"\n{test_dir}: {len(test_images)} immagini")
        for test_path in test_images:
            sha1 = sha1_of_file(test_path)
            if sha1 in train_sha_set:
                exact_rows.append(
                    {
                        "test_dir": str(test_dir),
                        "test_file": test_path.name,
                        "match_type": "exact",
                        "train_file": train_hashes[sha1],
                        "distance": 0,
                    }
                )
                continue
            if args.perceptual:
                ph = average_hash(test_path)
                if ph is None:
                    continue
                best_path = None
                best_distance = args.phash_threshold + 1
                for train_path, train_ph in train_phash.items():
                    distance = hamming(ph, train_ph)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = train_path
                if best_path is not None and best_distance <= args.phash_threshold:
                    near_rows.append(
                        {
                            "test_dir": str(test_dir),
                            "test_file": test_path.name,
                            "match_type": "near",
                            "train_file": best_path.name,
                            "distance": best_distance,
                        }
                    )

    report_path = out_dir / "leakage_report.csv"
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_dir", "test_file", "match_type", "train_file", "distance"],
        )
        writer.writeheader()
        writer.writerows(exact_rows + near_rows)

    print("\n--- RISULTATO ---")
    print(f"Match esatti (stessa immagine in train e test) : {len(exact_rows)}")
    if args.perceptual:
        print(f"Quasi-duplicati (Hamming <= {args.phash_threshold})      : {len(near_rows)}")
    print(f"Report: {report_path}")
    if exact_rows or near_rows:
        print("\nATTENZIONE: c'e leakage. Le immagini elencate vanno tolte dal training")
        print("(o il set di test va rifatto) prima di costruire i fold.")
    else:
        print("\nNessun leakage trovato con questi controlli.")


if __name__ == "__main__":
    main()
