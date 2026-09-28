
from __future__ import annotations

import argparse

from pathlib import Path

from ultralytics import YOLO
import mlflow

# Importa alcune funzioni di supporto definite nel file mlflow_utils.py.
from scripts.mlflow_utils import (
    init_mlflow,
    safe_log_params,
    log_artifact_path,
    get_mlflow_env_info,
)


def iter_fold_data(folds_dir: Path) -> list[Path]:
    """
    Cerca tutti i file data.yaml presenti nelle cartelle dei fold.
"""


def main() -> None:

    # Crea il parser che gestisce gli argomenti passati da terminale.
    parser = argparse.ArgumentParser(
        description="Phase 2: fine-tune one YOLO model for each k-fold split."
    )

    # Cartella che contiene tutti i fold del dataset.
    #
    # Il valore indicato in default viene usato quando non viene passato
    # esplicitamente il parametro --folds-dir.
    parser.add_argument(
        "--model",
        default="yolov8n_run_1.2_classes_1_2.pt",
        help="Percorso del modello YOLO da usare come punto di partenza.",
    )

    # Cartella principale nella quale YOLO salverà i risultati
    # dei training relativi ai diversi fold.
    parser.add_argument(
        "--project",
        default="finetune_runs/phase_2_kfold",
        help="Cartella principale in cui salvare i risultati.",
    )

    # Prefisso usato per costruire il nome di ogni singola run.
    #
    # Al prefisso verrà aggiunto il nome del fold, ad esempio:
    # ppe_..._phase2_kfold_fold_0
    parser.add_argument(
        "--name-prefix",
        default="ppe_ood_hardneg_vestboost_helmetboost_phase2_kfold",
        help="Prefisso del nome assegnato alle run dei diversi fold.",
    )

    # Numero massimo di epoche di training.
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Numero massimo di epoche di addestramento.",
    )

    # Dimensione alla quale vengono ridimensionate le immagini.
    #
    # Con imgsz=640, le immagini vengono elaborate con dimensione
    # indicativa di 640 x 640 pixel.
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Dimensione delle immagini usate durante il training.",
    )

    # Numero di immagini elaborate insieme in ogni iterazione.
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Dimensione del batch.",
    )

    # Learning rate iniziale.
    #
    # Determina quanto vengono aggiornati i pesi del modello
    # durante le prime fasi del training.
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Learning rate iniziale.",
    )

    # Fattore finale del learning rate.
    #
    # Ultralytics usa questo valore per determinare il learning rate
    # finale rispetto a quello iniziale.
    #
    parser.add_argument(
        "--lrf",
        type=float,
        default=1.0,
        help="Fattore finale del learning rate.",
    )

    # Ottimizzatore usato per aggiornare i pesi della rete.
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        help="Ottimizzatore da usare durante il training.",
    )

    # Penalizzazione applicata ai pesi del modello.
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Valore di weight decay usato dall'ottimizzatore.",
    )

    # Percentuale di neuroni disattivati casualmente durante il training.
    #
    # Anche il dropout viene usato per ridurre il rischio di overfitting.
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Valore di dropout.",
    )
    #
    # Mosaic combina più immagini in una singola immagine di training.
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.5,
        help="Probabilità di applicare l'augmentazione Mosaic.",
    )

    # Intensità della trasformazione prospettica applicata alle immagini.
    parser.add_argument(
        "--perspective",
        type=float,
        default=0.01,
        help="Intensità della trasformazione prospettica.",
    )

    # Numero di epoche senza miglioramenti tollerate prima
    # di interrompere il training tramite early stopping.
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Numero di epoche senza miglioramento prima dell'early stopping.",
    )

    parser.add_argument(
        "--device",
        default="",
        help="Dispositivo da usare, ad esempio cpu, mps oppure 0.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Numero di worker usati dal DataLoader.",
    )


    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Permette di usare una cartella di output già esistente.",
    )
    parser.add_argument(
        "--only-fold",
        type=int,
        default=-1,
        help="Indice del solo fold da eseguire. -1 esegue tutti i fold.",
    )
    args = parser.parse_args()

    fold_data_files = iter_fold_data(Path(args.folds_dir))

    if not fold_data_files:
        raise FileNotFoundError(
            f"Nessun file data.yaml dei fold trovato in: {args.folds_dir}"
        )
    model_path = Path(args.model)

    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    # Configura MLflow.
    #
    # La funzione init_mlflow dovrebbe impostare il tracking URI
    # e selezionare o creare l'esperimento MLflow.
    #
    # Qui viene passato args.project come nome dell'esperimento.
    init_mlflow(args.project)

    # Scorre tutti i file data.yaml trovati.
    #
    # Ogni file corrisponde a uno specifico fold.
    for fold_data in fold_data_files:

        # Recupera il nome della cartella che contiene data.yaml.
 
        fold_name = fold_data.parent.name

        # Estrae il numero del fold dal suo nome.
     
        fold_index = int(fold_name.split("_")[-1])
        if args.only_fold >= 0 and fold_index != args.only_fold:
            continue
        run_name = f"{args.name_prefix}_{fold_name}"

        # Stampa alcune informazioni utili nel terminale.
        print(f"Training del fold: {fold_name}")
        print(f"Dataset: {fold_data}")
        print(f"Modello iniziale: {args.model}")
        print(f"Cartella della run: {Path(args.project) / run_name}")

        # Racchiude in un dizionario tutti i parametri
        # che verranno passati al metodo YOLO.train().
        train_kwargs = {
            # File data.yaml del fold corrente.
            "data": str(fold_data),

            # Numero massimo di epoche.
            "epochs": args.epochs,

            # Dimensione delle immagini.
            "imgsz": args.imgsz,

            # Dimensione del batch.
            "batch": args.batch,

            # Learning rate iniziale.
            "lr0": args.lr0,

            # Fattore finale del learning rate.
            "lrf": args.lrf,

            # Regolarizzazione weight decay.
            "weight_decay": args.weight_decay,

            # Valore di dropout.
            "dropout": args.dropout,

            # Probabilità di applicare Mosaic.
            "mosaic": args.mosaic,

            # Ottimizzatore selezionato.
            "optimizer": args.optimizer,

            # Intensità della trasformazione prospettica.
            "perspective": args.perspective,

            # Pazienza dell'early stopping.
            "patience": args.patience,

            # Cartella principale dei risultati.
            "project": args.project,

            # Nome della cartella specifica della run.
            "name": run_name,

            # Numero di processi usati per caricare i dati.
            "workers": args.workers,

            # Permette oppure impedisce l'uso di una cartella già esistente.
            "exist_ok": args.exist_ok,
        }

        if args.device:
            train_kwargs["device"] = args.device
        with mlflow.start_run(run_name=run_name):

            # Recupera alcune informazioni sull'ambiente di esecuzione.
            env_info = get_mlflow_env_info()

            # Registra le informazioni sull'ambiente come tag MLflow.
            mlflow.set_tags(
                {
                    key: value
                    for key, value in env_info.items()
                    if value is not None
                }
            )

            # Registra su MLflow i principali parametri del training.
            #
            # safe_log_params viene usata al posto di mlflow.log_params
            # probabilmente per gestire valori non validi, troppo lunghi
            # oppure già registrati.
            safe_log_params(
                {
                    # Nome del fold corrente.
                    "fold": fold_name,

                    # Numero massimo di epoche.
                    "epochs": args.epochs,

                    # Dimensione delle immagini.
                    "imgsz": args.imgsz,

                    # Dimensione del batch.
                    "batch": args.batch,

                    # Learning rate iniziale.
                    "lr0": args.lr0,

                    # Fattore finale del learning rate.
                    "lrf": args.lrf,

                    # Modello utilizzato come punto di partenza.
                    "model": args.model,

                    # File data.yaml usato per il fold corrente.
                    "data": str(fold_data),
                }
            )

            # Carica il modello YOLO indicato in args.model.
            YOLO(args.model).train(**train_kwargs)
            best_model_path = (
                Path(args.project)
                / run_name
                / "weights"
                / "best.pt"
            )

            # Registra best.pt come artifact della run MLflow.
            #
            # In questo modo il modello migliore può essere consultato
            # e recuperato direttamente dall'interfaccia di MLflow.
            log_artifact_path(best_model_path)

    main()
    
    
    
    
