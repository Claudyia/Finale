#codice per il fine-tuning del modello PPE su un dataset personalizzato.
from ultralytics import YOLO


#run= cd /Users/claudia/Desktop/isaac_odd && /Users/claudia/.pyenv/versions/3.14.4/bin/python scripts/finetunig/finetuning.py

from pathlib import Path
import shutil
""""Questo script esegue il fine-tuning del modello YOLOv8n per il rilevamento
di giubbotti di sicurezza (vest) utilizzando un dataset personalizzato. 
Il processo include l'identificazione dei file di etichetta che contengono la classe "vest"
,la duplicazione di questi file e delle relative immagini per bilanciare il dataset, 
e infine l'addestramento del modello con i nuovi dati. 
I risultati del fine-tuning vengono salvati in una directory specificata per ulteriori analisi e confronti con altri run di fine
-tuning.
parto da un modello pre-addestrato su PPE  e 
lo addestro per 80 epoche con una dimensione dell'immagine di 640, 
un batch size di 8, un learning rate iniziale di 0.001 e una pazienza di 20 epoche per
il monitoraggio della perdita di validazione.
"""

DATA = Path("scripts/finetunig/dataset/finetune_dataset") 
labels_dir = DATA / "labels" / "train"
images_dir = DATA / "images" / "train"

image_exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

vest_labels = []

for label_path in labels_dir.glob("*.txt"):
    """ Legge ogni file di etichetta e verifica se contiene la classe "vest" 
    (indicata da "1" come primo numero in una riga)."""
    lines = label_path.read_text().strip().splitlines()
    if any(line.split() and line.split()[0] == "1" for line in lines):
        vest_labels.append(label_path)

print(f"Found {len(vest_labels)} train label files containing vest")

created = 0

# Duplica 2 volte tutte le immagini che contengono la classe "vest"
for dup_idx in range(2):

    # Scorre tutti i file label selezionati
    for label_path in vest_labels:

        # Nome file senza estensione per costruire il nome dell'immagine corrispondente
        stem = label_path.stem

        # Variabile che conterrà il path dell'immagine trovata
        image_path = None

        # Cerca l'immagine corrispondente provando tutte le estensioni possibili (.jpg, .png, ecc.)
        for ext in image_exts:

            # Costruisce il path candidato
            candidate = images_dir / f"{stem}{ext}"

            # Se il file esiste, salva il path e interrompe il ciclo
            if candidate.exists():
                image_path = candidate
                break

        # Se non è stata trovata nessuna immagine associata
        if image_path is None:
            print(f"Missing image for {label_path.name}")
            continue

        # Crea un nuovo nome univoco per evitare sovrascritture 
        new_stem = f"oversample_vest{dup_idx + 1}_{stem}"

        # Path del nuovo file label
        new_label = labels_dir / f"{new_stem}.txt"

        # Path della nuova immagine mantenendo l'estensione originale
        new_image = images_dir / f"{new_stem}{image_path.suffix}"

        # Evita di sovrascrivere file già esistenti
        if new_label.exists() or new_image.exists():
            print(f"Skipping existing pair: {new_stem}")
            continue

        # Copia il file label
        shutil.copy2(label_path, new_label)

        # Copia l'immagine corrispondente
        shutil.copy2(image_path, new_image)

        # Incrementa il contatore delle coppie create
        created += 1

# Stampa il numero totale di nuove coppie create
print(f"Created {created} oversampled image/label pairs")

model = YOLO('/Users/claudia/Desktop/isaac_odd/models/yolov8n-ppe_run_1_classes_1_2.pt')

model.train(
    data='/Users/claudia/Desktop/isaac_odd/scripts/finetunig/dataset/finetune_dataset/data.yaml',
    epochs=80,
    imgsz=640,
    batch=8,
    lr0=0.001,
    patience=20,
    project='/Users/claudia/Desktop/isaac_odd/finetune_runs',
    name='ppe_finetune_vest_oversample',
)