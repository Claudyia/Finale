#codice per leggere i risultati dei k-fold di YOLO, 
# estrarre le metriche migliori per ogni fold, 
# e scrivere un file CSV di sintesi con i risultati di tutti i fold.
#poi calcola la media e la deviazione 
# standard delle metriche sui fold, 
# e stampa un riassunto dei risultati.
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def read_results(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key.strip(): float(value) for key, value in row.items() if value.strip()})
    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    """""Funzione che prende una lista di righe di risultati (dizionari con metriche) e 
    restituisce la riga con il miglior mAP50.
    perche la riga? Perché in YOLO i risultati sono registrati per ogni epoca, 
    quindi ogni riga rappresenta le metriche a una certa epoca."""
    return max(rows, key=lambda row: row.get(METRIC_COLUMNS["mAP50"], float("-inf")))


def mean(values: list[float]) -> float:
    """"usiamo una funzione per calcolare la media e non il semplice mean di statistics
    perché vogliamo ignorare i valori NaN, così facendo impostiamo mean a NaN 
    se la lista è vuota o se tutti i valori sono NaN.
    questo metodo lo hai copito dal progetto di primo anno"""
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    """_funzione per calcolare la deviazione standard_
    """
    if len(values) < 2:
        return 0.0 
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize YOLO k-fold results.")
    parser.add_argument("--runs-dir", default="finetune_runs/kfold")
    parser.add_argument("--name-prefix", default="ppe_ood_hardneg_vestboost_kfold")
    parser.add_argument("--output-csv", default="research_phases/04_kfold_validation/kfold_summary.csv")
    args = parser.parse_args()

    run_dirs = sorted(Path(args.runs_dir).glob(f"{args.name_prefix}_fold_*"))
    summaries = []
    for run_dir in run_dirs:
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            continue
        rows = read_results(results_path)
        if not rows:
            continue
        row = best_row(rows)
        summary = {
            "run": run_dir.name,
            "best_epoch": int(row.get("epoch", -1)),
        }
        for metric, column in METRIC_COLUMNS.items():
            summary[metric] = row.get(column, float("nan"))
        summaries.append(summary)

    if not summaries:
        raise FileNotFoundError(f"No k-fold results found in: {args.runs_dir}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "best_epoch", *METRIC_COLUMNS.keys()]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Runs: {len(summaries)}")
    for metric in METRIC_COLUMNS:
        values = [row[metric] for row in summaries]
        print(f"{metric}: mean={mean(values):.4f} std={std(values):.4f}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
