"""Carica i confronti gia' calcolati come run MLflow, incluse le immagini
(confusion matrix, curve P/R/F1/PR, batch di validazione).

Un experiment MLflow per modello candidato ("padre"), con dentro due run:
"base" (metriche standard su test set, da compare_models_local.py) e
"ood" (hallucination su immagini OOD, da evaluate_id_ood_comparison.py).

Uso (senza argomenti carica tutti e tre i padri; con argomenti solo quelli scelti,
per non duplicare run gia caricate):
    MLFLOW_TRACKING_URI=http://localhost:5001 python log_to_mlflow.py fase06
    MLFLOW_TRACKING_URI=http://localhost:5001 python log_to_mlflow.py fold2 fase06 terzo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

RESEARCH_PHASES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_PHASES.parent / "scripts"))
from mlflow_utils import init_mlflow  # noqa: E402


def log_images(image_dir: Path, artifact_path: str) -> None:
    """Carica tutte le png/jpg di una cartella come artifact (esclude predictions.json)."""
    if not image_dir.is_dir():
        return
    images = [
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    for img in images:
        mlflow.log_artifact(str(img), artifact_path=artifact_path)


def log_base_run(json_path: Path, baseline_img_dir: Path | None = None, candidate_img_dir: Path | None = None) -> None:
    """Run 'base': output di compare_models_local.py."""
    data = json.loads(json_path.read_text())
    baseline = data["baseline"]
    candidate = data["candidate"]

    with mlflow.start_run(run_name="base"):
        mlflow.set_tags(
            {
                "kind": "base",
                "dataset": data.get("dataset", ""),
                "split": data.get("split", ""),
                "baseline_model": baseline["model"],
                "candidate_model": candidate["model"],
            }
        )
        mlflow.log_params(data.get("configuration", {}))
        for name, value in baseline["metrics"].items():
            mlflow.log_metric(f"baseline_{name}", value)
        for name, value in candidate["metrics"].items():
            mlflow.log_metric(f"candidate_{name}", value)
        for name, comp in data.get("comparison", {}).items():
            mlflow.log_metric(f"delta_{name}", comp["delta"])
            mlflow.log_metric(f"delta_percent_{name}", comp["delta_percent"])
        mlflow.log_artifact(str(json_path))
        if baseline_img_dir:
            log_images(baseline_img_dir, "images/baseline")
        if candidate_img_dir:
            log_images(candidate_img_dir, "images/candidate")


def log_ood_run(snapshot_dir: Path) -> None:
    """Run 'ood': output di evaluate_id_ood_comparison.py, da una cartella snapshot."""
    json_path = snapshot_dir / "comparison_report.json"
    data = json.loads(json_path.read_text())
    baseline = data["baseline"]
    candidate = data["candidate"]

    with mlflow.start_run(run_name="ood"):
        mlflow.set_tags(
            {
                "kind": "ood",
                "baseline_model": baseline["report"]["model"],
                "candidate_model": candidate["report"]["model"],
            }
        )
        for name, value in baseline["metrics"].items():
            mlflow.log_metric(f"baseline_{name}", value)
        for name, value in candidate["metrics"].items():
            mlflow.log_metric(f"candidate_{name}", value)
        for name, value in data.get("comparison", {}).items():
            mlflow.log_metric(name, value)
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(snapshot_dir / "baseline_ood_hallucinations.csv"))
        mlflow.log_artifact(str(snapshot_dir / "candidate_ood_hallucinations.csv"))
        log_images(snapshot_dir / "id_validation_baseline", "images/baseline")
        log_images(snapshot_dir / "id_validation_candidate", "images/candidate")


def main() -> None:
    id_ood_dir = RESEARCH_PHASES / "07_testv2" / "id_ood_comparison_runs"
    only = set(sys.argv[1:])  # ponytail: vuoto = tutti

    if not only or "fold2" in only:
        # Padre 1: baseline vs fold2, sullo stesso test pulito (1619 img) degli altri padri.
        init_mlflow("baseline_vs_fold2")
        comparison_runs = RESEARCH_PHASES / "07_testv2" / "comparison_runs_fold2_pulito"
        log_base_run(
            next(comparison_runs.glob("comparison_*.json")),
            baseline_img_dir=comparison_runs / "baseline",
            candidate_img_dir=comparison_runs / "candidate",
        )
        log_ood_run(id_ood_dir / "_snapshot_fold2")
        print("loggato: baseline_vs_fold2 / base + ood (con immagini)")

    if not only or "fase06" in only:
        # Padre 2: baseline vs fase06. Base su dataset_test_v2 "pulito" (1619 img,
        # stesso sottoinsieme senza leakage usato per il padre 3, confronto equo).
        init_mlflow("baseline_vs_fase06")
        comparison_runs_fase06 = RESEARCH_PHASES / "07_testv2" / "comparison_runs_fase06_pulito"
        log_base_run(
            next(comparison_runs_fase06.glob("comparison_*.json")),
            baseline_img_dir=comparison_runs_fase06 / "baseline",
            candidate_img_dir=comparison_runs_fase06 / "candidate",
        )
        log_ood_run(id_ood_dir / "_snapshot_fase06")
        print("loggato: baseline_vs_fase06 / base + ood (con immagini)")

    if not only or "terzo" in only:
        # Padre 3: terzo modello (runs_vestplus/final_model).
        # Base su dataset_test_v2 "pulito", stesso test di fase06 (confronto equo).
        init_mlflow("baseline_vs_terzo_modello")
        comparison_runs_terzo = RESEARCH_PHASES / "07_testv2" / "comparison_runs_terzo_pulito"
        log_base_run(
            next(comparison_runs_terzo.glob("comparison_*.json")),
            baseline_img_dir=comparison_runs_terzo / "baseline",
            candidate_img_dir=comparison_runs_terzo / "candidate",
        )
        log_ood_run(id_ood_dir / "_snapshot_terzo")
        print("loggato: baseline_vs_terzo_modello / base + ood (con immagini)")



if __name__ == "__main__":
    main()
