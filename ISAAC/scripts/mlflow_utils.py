from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlflow
from mlflow.entities import Run


def init_mlflow(
    experiment_name: str,
    artifact_location: str | None = None,
) -> str:
    """Configura MLflow e crea o seleziona un experiment.

    ``artifact_location`` viene usato soltanto quando l'experiment deve essere
    creato. Se non viene passato, viene letto da ``MLFLOW_ARTIFACT_ROOT``;
    in assenza di entrambi MLflow usa lo storage configurato dal tracking
    server.
    """

    if not experiment_name.strip():
        raise ValueError("experiment_name non può essere vuoto")

    # Non tocca il tracking URI qui: MLflow legge gia da solo MLFLOW_TRACKING_URI
    # se nessuno ha chiamato set_tracking_uri() prima. Rifarlo qui scavalcherebbe
    # silenziosamente un --tracking-uri esplicito passato dallo script chiamante.

    selected_artifact_location = artifact_location or os.environ.get(
        "MLFLOW_ARTIFACT_ROOT"
    )
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None and selected_artifact_location:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=selected_artifact_location,
        )
        experiment = mlflow.set_experiment(experiment_id=experiment_id)
    else:
        # Se non esiste e non è stato indicato uno storage personalizzato,
        # set_experiment lo crea usando lo storage predefinito del server.
        experiment = mlflow.set_experiment(experiment_name)

    return experiment.experiment_id


def get_mlflow_env_info() -> dict[str, str | None]:
    """Restituisce la configurazione MLflow corrente."""

    return {
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
        "MLFLOW_ARTIFACT_ROOT": os.environ.get("MLFLOW_ARTIFACT_ROOT"),
        "ACTIVE_TRACKING_URI": mlflow.get_tracking_uri(),
    }


def _require_active_run() -> None:
    """Evita la creazione implicita di run fuori dall'experiment previsto."""

    if mlflow.active_run() is None:
        raise RuntimeError(
            "Nessuna run MLflow attiva: usa 'with mlflow.start_run(...)' "
            "oppure 'with start_mlflow_run(...)'."
        )


def safe_log_params(params: Mapping[str, Any]) -> None:
    """Converte i parametri in valori accettati da MLflow."""

    _require_active_run()

    safe_params: dict[str, str | int | float | bool] = {}

    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)):
            safe_params[key] = value
        else:
            safe_params[key] = str(value)

    if safe_params:
        mlflow.log_params(safe_params)


def safe_log_metrics(metrics: Mapping[str, int | float | None]) -> None:
    """Registra metriche numeriche nella run attiva."""

    _require_active_run()

    safe_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }

    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def log_artifact_path(
    path: str | Path,
    artifact_path: str | None = None,
) -> None:
    """Registra un file o una directory come artifact."""

    _require_active_run()

    path = Path(path)

    if not path.exists():
        return

    if path.is_file():
        mlflow.log_artifact(
            str(path),
            artifact_path=artifact_path,
        )
    else:
        mlflow.log_artifacts(
            str(path),
            artifact_path=artifact_path or path.name,
        )


@contextmanager
def start_mlflow_run(
    experiment_name: str,
    run_name: str,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    artifact_location: str | None = None,
    nested: bool = False,
) -> Iterator[Run]:
    """Apre una run nell'experiment e la chiude automaticamente."""

    experiment_id = init_mlflow(experiment_name, artifact_location)

    safe_tags = (
        {key: str(value) for key, value in tags.items()}
        if tags
        else None
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=safe_tags,
        nested=nested,
    ) as run:
        if params:
            safe_log_params(params)

        yield run


@contextmanager
def optional_mlflow_run(
    enabled: bool,
    experiment_name: str,
    run_name: str,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    artifact_location: str | None = None,
    nested: bool = False,
) -> Iterator[Run | None]:
    """Come ``start_mlflow_run`` ma disattivabile.

    Con ``enabled=False`` non crea né experiment né run (utile per ``--no-log``):
    il blocco ``with`` viene comunque eseguito e ``log_metrics_if_active`` /
    ``log_artifact_if_active`` diventano no-op perché non c'è una run attiva.
    """

    if not enabled:
        yield None
        return

    with start_mlflow_run(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags=tags,
        artifact_location=artifact_location,
        nested=nested,
    ) as run:
        yield run


def log_metrics_if_active(metrics: Mapping[str, int | float | None]) -> None:
    """Registra le metriche solo se esiste una run MLflow attiva."""

    if mlflow.active_run() is None:
        return

    safe_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }
    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def log_artifact_if_active(
    path: str | Path,
    artifact_path: str | None = None,
) -> None:
    """Registra un file/directory come artifact solo se c'è una run attiva."""

    if mlflow.active_run() is None:
        return

    path = Path(path)
    if not path.exists():
        return

    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    else:
        mlflow.log_artifacts(
            str(path),
            artifact_path=artifact_path or path.name,
        )
