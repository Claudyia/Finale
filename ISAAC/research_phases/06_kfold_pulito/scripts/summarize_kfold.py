# Step E - riassume i fold completati: media +/- deviazione standard.
#
# Funziona anche coi fold parziali: usa solo quelli con marker .done.
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ("precision", "recall", "map50", "map50_95")


def mean(values: list[float]) -> float:
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    values = [v for v in values if v == v]
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step E: media +/- std sui fold completati.")
    parser.add_argument("--runs-dir", default="research_phases/06_kfold_pulito/kfold/runs")
    parser.add_argument("--out-dir", default="research_phases/06_kfold_pulito/kfold")
    parser.add_argument("--mlflow-uri", default="file:research_phases/06_kfold_pulito/kfold/mlruns")
    parser.add_argument("--experiment", default="kfold_06_pulito")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for fold_dir in sorted(runs_dir.glob("fold_*")):
        if not (fold_dir / ".done").exists():
            print(f"  {fold_dir.name}: non completato, salto")
            continue
        metrics_file = fold_dir / "fold_metrics.json"
        if not metrics_file.exists():
            continue
        data = json.loads(metrics_file.read_text())
        data["fold"] = fold_dir.name
        rows.append(data)

    if not rows:
        raise SystemExit("Nessun fold completato trovato.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kfold_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "best_epoch", *METRICS])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in ["fold", "best_epoch", *METRICS]})

    lines = [f"Fold completati: {len(rows)} / 5", ""]
    agg = {}
    for metric in METRICS:
        values = [r.get(metric, float("nan")) for r in rows]
        m, s = mean(values), std(values)
        agg[metric] = (m, s)
        lines.append(f"  {metric:10s} {m:.4f} +/- {s:.4f}   " + " ".join(f"{v:.3f}" for v in values))
    summary = "\n".join(lines)
    (out_dir / "kfold_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"\n{csv_path}")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name="kfold_summary", tags={"pipeline": "summary"}):
                mlflow.log_param("folds_completati", len(rows))
                for metric, (m, s) in agg.items():
                    mlflow.log_metric(f"{metric}_mean", m)
                    mlflow.log_metric(f"{metric}_std", s)
                mlflow.log_artifact(str(csv_path))
            print("MLflow: summary registrato.")
        except Exception as exc:  # noqa: BLE001
            print(f"(MLflow ignorato: {exc})")


if __name__ == "__main__":
    main()
