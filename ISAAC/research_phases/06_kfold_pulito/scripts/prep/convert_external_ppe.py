# Step B.
#
# Converte un dataset PPE esterno (Roboflow YOLOv8 o Pascal VOC) alle 2 classi
# del progetto: 0 = helmet, 1 = vest.
#
# - rimappa i nomi classe verso helmet / vest tramite un dizionario di sinonimi
# - tiene solo quei box, riscrive le label in formato YOLO
# - salta le immagini gia presenti (per hash) nel dataset pulito / test set
# - dedup interno per hash
# - group_id derivato dal nome base Roboflow: le varianti di augmentation della
#   stessa foto restano nello stesso gruppo (serve al GroupKFold)
#
# Non addestra niente. Output in external_staging/<source-name>/.
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
ROBOFLOW_MARKERS = ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", ".rf.")

HELMET_SYNONYMS = {
    "helmet", "hardhat", "hard hat", "hard-hat", "hard_hat", "casco",
    "safety helmet", "safety_helmet", "head_helmet", "hard hat workers",
}
VEST_SYNONYMS = {
    "vest", "safety vest", "safety_vest", "safety-vest", "reflective vest",
    "hi-vis", "hi vis", "high-vis", "high visibility vest", "gilet",
    "reflective_vest",
}


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def map_class_name(name: str) -> int | None:
    key = normalize(name)
    if key in HELMET_SYNONYMS:
        return 0
    if key in VEST_SYNONYMS:
        return 1
    return None


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_known_hashes(*csv_paths: Path) -> set[str]:
    known: set[str] = set()
    for path in csv_paths:
        if path and path.exists():
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("sha1"):
                        known.add(row["sha1"])
    return known


def read_yaml_names(data_yaml: Path) -> list[str]:
    text = data_yaml.read_text()
    names: list[str] = []
    in_names = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            inline = stripped[len("names:"):].strip()
            if inline.startswith("["):
                return [n.strip().strip("'\"") for n in inline.strip("[]").split(",") if n.strip()]
            in_names = True
            continue
        if in_names:
            if stripped.startswith("- "):
                names.append(stripped[2:].strip().strip("'\""))
            elif re.match(r"^\d+\s*:", stripped):
                names.append(stripped.split(":", 1)[1].strip().strip("'\""))
            elif stripped and not line.startswith((" ", "\t", "-")):
                break
    return names


def group_id_from_name(stem: str) -> str:
    """Nome base della foto sorgente, prima del suffisso di augmentation Roboflow."""
    for marker in ROBOFLOW_MARKERS:
        index = stem.find(marker)
        if index != -1:
            return stem[:index]
    return stem


def yolo_lines_from_txt(label_path: Path, index_map: dict[int, int]) -> list[str]:
    out: list[str] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_class = int(float(parts[0]))
        except ValueError:
            continue
        if src_class not in index_map:
            continue
        out.append(" ".join([str(index_map[src_class])] + parts[1:5]))
    return out


def yolo_lines_from_voc(xml_path: Path) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    size = root.find("size")
    if size is None:
        return []
    width = float(size.findtext("width", "0") or 0)
    height = float(size.findtext("height", "0") or 0)
    if width <= 0 or height <= 0:
        return []
    out: list[str] = []
    for obj in root.findall("object"):
        target = map_class_name(obj.findtext("name", ""))
        if target is None:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        cx = (xmin + xmax) / 2 / width
        cy = (ymin + ymax) / 2 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        if bw <= 0 or bh <= 0:
            continue
        out.append(f"{target} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def collect_pairs(src: Path, splits: list[str]) -> list[tuple[Path, Path | None, str]]:
    """Ritorna (immagine, annotazione_o_None, formato) per gli split richiesti."""
    pairs: list[tuple[Path, Path | None, str]] = []

    split_dirs = [src / s for s in splits if (src / s).is_dir()]
    search_dirs = split_dirs or [src]

    for base in search_dirs:
        images_dirs = [p for p in base.rglob("images") if p.is_dir()] or [base]
        for images_dir in images_dirs:
            parent = images_dir.parent
            for image_path in sorted(images_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                txt = parent / "labels" / f"{image_path.stem}.txt"
                if not txt.exists():
                    txt = images_dir.parent / "labels" / f"{image_path.stem}.txt"
                if txt.exists():
                    pairs.append((image_path, txt, "yolo"))
                    continue
                xml = parent / "annotations" / f"{image_path.stem}.xml"
                if not xml.exists():
                    xml = images_dir.parent / "annotations" / f"{image_path.stem}.xml"
                if xml.exists():
                    pairs.append((image_path, xml, "voc"))
                    continue
                pairs.append((image_path, None, "none"))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B: converte un dataset PPE esterno alle classi helmet/vest."
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--source-name", required=True, help="Etichetta sorgente, es. construction_ppe.")
    parser.add_argument("--out-dir", default="research_phases/06_kfold_pulito/external_staging")
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--class-names", default=None, help="Nomi classe separati da virgola (formato YOLO).")
    parser.add_argument(
        "--splits",
        default="train,valid,train2017,val2017",
        help="Sottocartelle di split da usare (le altre, es. test, vengono ignorate). "
        "Se nessuna esiste, usa la radice.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Massimo immagini tenute (0 = nessun limite).")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Tieni come hard-negative le immagini senza box helmet/vest.",
    )
    parser.add_argument(
        "--known-hashes",
        nargs="*",
        default=[
            "research_phases/06_kfold_pulito/dataset_clean/sources.csv",
        ],
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Sorgente non trovata: {src}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pairs = collect_pairs(src, splits)
    if not pairs:
        raise SystemExit(f"Nessuna immagine trovata in {src}")

    has_yolo = any(fmt == "yolo" for _, _, fmt in pairs)
    index_map: dict[int, int] = {}
    if has_yolo:
        if args.class_names:
            class_names = [n.strip() for n in args.class_names.split(",")]
        else:
            data_yaml = Path(args.data_yaml) if args.data_yaml else next(src.rglob("data.yaml"), None)
            if not data_yaml or not data_yaml.exists():
                raise FileNotFoundError("Formato YOLO ma nessun data.yaml: passa --data-yaml o --class-names.")
            class_names = read_yaml_names(data_yaml)
        for idx, raw in enumerate(class_names):
            target = map_class_name(raw)
            if target is not None:
                index_map[idx] = target
        if not index_map:
            if not args.keep_empty:
                raise SystemExit(
                    f"Nessuna classe helmet/vest riconosciuta in: {class_names}\n"
                    "Aggiungi i sinonimi, oppure usa --keep-empty per prenderle come hard-negative."
                )
            print(f"Nessuna classe helmet/vest in {class_names}: sorgente usata come hard-negative.")
        else:
            print("Mappatura YOLO:", ", ".join(
                f"{class_names[k]}->{'helmet' if v == 0 else 'vest'}" for k, v in index_map.items()
            ))

    known_hashes = load_known_hashes(*(Path(p) for p in args.known_hashes))
    print(f"Immagini candidate: {len(pairs)} | hash gia noti da escludere: {len(known_hashes)}")

    out_dir = Path(args.out_dir) / args.source_name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    manifest_rows: list[dict] = []
    seen_hashes: set[str] = set()

    for image_path, annotation, fmt in pairs:
        stats["seen"] += 1
        if args.limit and stats["kept"] >= args.limit:
            stats["skip_limit"] += 1
            continue

        digest = sha1_of_file(image_path)
        if digest in known_hashes:
            stats["skip_known"] += 1
            continue
        if digest in seen_hashes:
            stats["skip_dup_internal"] += 1
            continue

        if fmt == "yolo" and annotation is not None:
            lines = yolo_lines_from_txt(annotation, index_map)
        elif fmt == "voc" and annotation is not None:
            lines = yolo_lines_from_voc(annotation)
        else:
            lines = []

        if not lines and not args.keep_empty:
            stats["skip_no_box"] += 1
            continue

        seen_hashes.add(digest)
        new_stem = f"{args.source_name}_{digest[:12]}"
        dest_image = out_dir / "images" / f"{new_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, dest_image)
        (out_dir / "labels" / f"{new_stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        n_helmet = sum(1 for line in lines if line.startswith("0 "))
        n_vest = sum(1 for line in lines if line.startswith("1 "))
        stats["kept"] += 1
        stats["box_helmet"] += n_helmet
        stats["box_vest"] += n_vest
        if n_helmet and n_vest:
            stats["img_mixed"] += 1

        manifest_rows.append(
            {
                "image": dest_image.name,
                "stem": new_stem,
                "source": args.source_name,
                "group_id": f"{args.source_name}:{group_id_from_name(image_path.stem)}",
                "n_helmet": n_helmet,
                "n_vest": n_vest,
                "is_empty_label": int(not lines),
                "sha1": digest,
            }
        )

    with (out_dir / "sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "stem", "source", "group_id",
                        "n_helmet", "n_vest", "is_empty_label", "sha1"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n--- RISULTATO ---")
    for key in ("seen", "skip_known", "skip_dup_internal", "skip_no_box", "skip_limit", "kept"):
        if stats[key]:
            print(f"  {key:20s} {stats[key]}")
    print(f"  immagini helmet+vest {stats['img_mixed']}")
    print(f"  box helmet aggiunti  {stats['box_helmet']}")
    print(f"  box vest aggiunti    {stats['box_vest']}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
