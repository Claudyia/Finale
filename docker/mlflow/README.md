# MLflow tracking server (Docker)

Avvio: `docker compose up -d` (da questa cartella). UI su http://localhost:5000.

Stop: `docker compose down`.

Dati (SQLite + artifact) in `./data/`, non versionati in git.

Per puntarci uno script: `MLFLOW_TRACKING_URI=http://localhost:5000`.
