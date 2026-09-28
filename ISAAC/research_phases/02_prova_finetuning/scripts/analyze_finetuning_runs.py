#codice per analizzare i risultati dei run di fine-tuning YOLO, confrontando metriche chiave e fornendo diagnosi per miglioramenti o regressioni.

## Esecuzione da terminale:
"""cd /Users/claudia/Desktop/isaac_odd
/usr/local/bin/python3 scripts/analyze_finetuning_runs.py \
  --baseline ppe_e50_img640_b8_lr01_pat100_base_map055 \
  --candidate ppe_e80_img640_b8_lr001_pat20_oversamplevest_map047"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)", # mAP misura la media della precisione a diverse soglie di confidenza, con IoU >= 0.5
    "mAP50-95": "metrics/mAP50-95(B)", # mAP medio su soglie IoU da 0.50 a 0.95
    "train_box_loss": "train/box_loss", # perdita di localizzazione (bounding box) durante l'addestramento
    "val_box_loss": "val/box_loss",# perdita di localizzazione (bounding box) durante la validazione
    "val_cls_loss": "val/cls_loss", # perdita di classificazione durante la validazione
    "val_dfl_loss": "val/dfl_loss", # DFL (Distribution Focal Loss) è una perdita utilizzata in alcuni modelli di rilevamento per migliorare la precisione delle previsioni di bounding box, specialmente per oggetti piccoli o parzialmente visibili.
}


def clean_key(key: str) -> str: # rimuove spazi bianchi e caratteri di controllo dalle chiavi dei dizionari, assicurando che siano pulite e uniformi per l'accesso ai valori.
    return key.strip()


def read_results(path: Path) -> list[dict[str, float]]: # la funzione legge un file CSV contenente i risultati di addestramento e validazione, pulisce le chiavi e converte i valori in float, restituendo una lista di dizionari con metriche numeriche per ogni epoca.
    """se il file CSV contiene righe con chiavi o valori nulli, queste righe vengono ignorate. Inoltre, se un valore non può essere convertito in float, viene saltato senza interrompere l'intero processo di lettura.
    questo serve a garantire che solo le righe con dati validi vengano considerate nell'analisi, evitando errori dovuti a dati mancanti o malformati."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key is None or value is None:
                    continue
                try:
                    cleaned[clean_key(key)] = float(value)
                except ValueError:
                    pass
            if cleaned:
                rows.append(cleaned)
    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]: 
    """la funzione best_row prende una lista di dizionari, ciascuno rappresentante le metriche di un'epoca 
    di addestramento, e restituisce il dizionario che ha il valore più alto per la metrica mAP50.
    Se una riga non contiene la chiave mAP50, viene considerata con un valore predefinito di -1.0, garantendo che solo le
    righe con mAP50 valido vengano considerate per determinare la migliore epoca.
    map50 è una metrica chiave per valutare le prestazioni di un modello di rilevamento oggetti,
    misurando la media della precisione a diverse soglie di confidenza, con IoU >= 0.5. Restituire la riga con il valore più alto di mAP50 
    consente di identificare
    l'epoca in cui il modello ha raggiunto le migliori prestazioni complessive in termini di rilevamento degli oggetti."""    
    return max(rows, key=lambda row: row.get(METRICS["mAP50"], -1.0))


def final_row(rows: list[dict[str, float]]) -> dict[str, float]:
    """la funzione final_row restituisce l'ultima riga della lista, che rappresenta le metriche dell'ultima epoca di addestramento.
    Se la lista è vuota, viene restituito un dizionario vuoto. Questo permette
    di confrontare le prestazioni finali del modello con quelle dell'epoca migliore,
    fornendo una visione completa dell'andamento delle metriche durante l'intero processo di addestramento.
    """
    if not rows:
        return {}
    return rows[-1]


def get(row: dict[str, float], metric: str) -> float:
    """la funzione get accede a un valore specifico di una metrica da una riga di risultati, utilizzando 
    la mappatura definita in METRICS per tradurre il nome della metrica in chiave del dizionario.
    Se la chiave corrispondente alla metrica non è presente nella riga, viene restituito nan così da
    gestire in modo robusto i casi in cui alcune metriche potrebbero essere mancanti
    """
    return row.get(METRICS[metric], float("nan"))


def formatta(value: float) -> str:
    """la funzione formatta prende un valore numerico e lo formatta come stringa con tre cifre decimali.
    Questo è utile per presentare i risultati in modo leggibile, 
    evidenziando chiaramente quando una metrica non è stata calcolata o non è applicabile."""
    if value != value:
        return "n/a"
    return f"{value:.3f}"


def summarize_run(run_dir: Path) -> dict[str, object]:
    """la funzione summarize_run analizza una cartella di un run di addestramento YOLO, leggendo i risultati da un file CSV
    e restituendo un dizionario con informazioni chiave sul run, tra cui il numero di epoche completate, le metriche dell'epoca migliore e dell'ultima epoca, e la presenza dei file di pesi best.pt e last.pt.
    Se il file results.csv è mancante o non contiene righe valide,
    la funzione solleva un'eccezione, garantendo che solo i run con dati completi vengano anal"""
    
    
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    rows = read_results(results_path)
    if not rows:
        raise ValueError(f"No metric rows found in: {results_path}")

    best = best_row(rows)
    final = final_row(rows)
    weights_dir = run_dir / "weights"

    return {
        "name": run_dir.name,
        "path": run_dir,
        "epochs": len(rows),
        "best_epoch": int(best.get("epoch", -1)) if "epoch" in best else None,
        "final": final,
        "best": best,
        "has_best_pt": (weights_dir / "best.pt").exists(),
        "has_last_pt": (weights_dir / "last.pt").exists(),
        "confusion_matrix": (run_dir / "confusion_matrix.png").exists(),
    }


def print_run(summary: dict[str, object]) -> None:
    """"la funzione print_run prende un dizionario di riepilogo di un run di addestramento 
    e stampa in modo leggibile le informazioni chiave, tra cui il numero di epoche completate,
    le metriche dell'epoca migliore e dell'ultima epoca, e la presenza dei file di pesi best.pt e last.pt.
    Questo consente di avere una panoramica chiara e immediata delle prestazioni del modello durante 
    il processo di addestramento, facilitando il confronto tra diversi run"""
    final = summary["final"]
    best = summary["best"]
    assert isinstance(final, dict)
    assert isinstance(best, dict)

    print(f"\nRun: {summary['name']}")
    print(f"  path: {summary['path']}")
    print(f"  epochs completed: {summary['epochs']}")
    print(f"  best epoch by mAP50: {summary['best_epoch']}")
    print(f"  weights: best.pt={summary['has_best_pt']} last.pt={summary['has_last_pt']}")
    print(f"  confusion_matrix.png: {summary['confusion_matrix']}")
    print("  final metrics:")
    print(f"    precision: {formatta(get(final, 'precision'))}")
    print(f"    recall:    {formatta(get(final, 'recall'))}")
    print(f"    mAP50:     {formatta(get(final, 'mAP50'))}")
    print(f"    mAP50-95:  {formatta(get(final, 'mAP50-95'))}")
    print("  best metrics:")
    print(f"    precision: {formatta(get(best, 'precision'))}")
    print(f"    recall:    {formatta(get(best, 'recall'))}")
    print(f"    mAP50:     {formatta(get(best, 'mAP50'))}") 
    print(f"    mAP50-95:  {formatta(get(best, 'mAP50-95'))}")
    """differenzatra map50 e mAP50-95 è che
    mAP50 misura la media della precisione a una singola soglia di confidenza, 
    mentre mAP50-95 misura la media della precisione a più soglie di confidenza
    questo permette di valutare le prestazioni del modello in modo più completo,
    considerando sia le previsioni più sicure (soglie più alte) 
    che quelle meno sicure (soglie più basse), fornendo una visione più dettagliata delle capacità del 
    modello di rilevare oggetti in diverse condizioni."""
    
    


def compare(baseline: dict[str, object], candidate: dict[str, object]) -> None:
    """la funzione compare confronta le metriche chiave di due run di addestramento, 
    identificando miglioramenti o regressioni nelle prestazioni del modello.
    Confronta le metriche di precisione, recall, mAP50 e mAP50-95 tra il run di baseline e il run candidato,
    calcolando la differenza e indicando se c'è stato un miglioramento o un peggioramento.
   il baseline è il primo fine-tuning run che fatto ( finetuning_run 13-40) e il candidate è il nuovo run che vuoi confrontare con il baseline per vedere se ha migliorato le prestazioni del modello.
    """

    base_best = baseline["best"]
    cand_best = candidate["best"]
    assert isinstance(base_best, dict)
    assert isinstance(cand_best, dict)

    print("\nComparison against baseline best epoch")
    print(f"  baseline:  {baseline['name']}")
    print(f"  candidate: {candidate['name']}")

    worse = []
    better = []
    for metric in ("precision", "recall", "mAP50", "mAP50-95"):
        base_value = get(base_best, metric)
        cand_value = get(cand_best, metric)
        delta = cand_value - base_value
        direction = "improved" if delta > 0 else "worse"
        print(
            f"  {metric:9s}: baseline={formatta(base_value)} "
            f"candidate={formatta(cand_value)} delta={delta:+.3f} ({direction})"
        )
        if delta < -0.01:
            worse.append(metric)
        elif delta > 0.01:
            better.append(metric)

    print("\nDiagnosi")
    if "mAP50" in worse or "mAP50-95" in worse:
        print("  il fine-tuning non ha migliorato le prestazioni del detector complessivamente.")
    elif better:
        print("  il candidato ha migliorato almeno una metrica chiave. Controlla la matrice di confusione prima di sostituire il modello.")
    else:
        print("  il candidato è approssimativamente equivalente al baseline. Preferisci il modello più semplice o più stabile.")

    if "recall" in better and ("precision" in worse or "mAP50" in worse):
        print("  la recall è migliorata, ma precision/mAP è peggiorata. The model is finding more objects but making worse detections.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="confronto yolovfine-tuning runs e mostra i risultati."
    )
    parser.add_argument(
        "--runs-dir",
        default="/Users/claudia/Desktop/isaac_odd/finetune_runs",
        help="Directory containing YOLO run folders.",
    )
    parser.add_argument(
        "--baseline",
        default="ppe_finetune_50e-3",
        help="Baseline run folder name.",
    )
    parser.add_argument(
        "--candidate",
        default="ppe_finetune_vest_oversample",
        help="Candidate run folder name.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    baseline = summarize_run(runs_dir / args.baseline)
    candidate = summarize_run(runs_dir / args.candidate)

    print_run(baseline)
    print_run(candidate)
    compare(baseline, candidate)


if __name__ == "__main__":
    main()
