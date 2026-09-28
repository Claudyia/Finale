# Graph Report - ISAAC  (2026-09-28)

## Corpus Check
- Corpus is ~2,303 words - fits in a single context window. You may not need a graph.

## Summary
- 34 nodes · 43 edges · 6 communities (5 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.65)
- Token cost: 2,400 input · 900 output

## Community Hubs (Navigation)
- OOD/PPE Experiment Narrative
- MLflow Run Lifecycle
- MLflow Setup & Metrics
- MLflow Artifact Logging
- MLflow Metric Guardrails
- Project Package

## God Nodes (most connected - your core abstractions)
1. `start_mlflow_run()` - 7 edges
2. `_require_active_run()` - 5 edges
3. `safe_log_params()` - 5 edges
4. `optional_mlflow_run()` - 5 edges
5. `log_artifact_path()` - 4 edges
6. `ISaAC OOD/PPE Experiments` - 4 edges
7. `init_mlflow()` - 3 edges
8. `safe_log_metrics()` - 3 edges
9. `log_artifact_if_active()` - 3 edges
10. `Hard-negative OOD + vest boost` - 3 edges

## Surprising Connections (you probably didn't know these)
- `start_mlflow_run()` --calls--> `init_mlflow()`  [EXTRACTED]
  scripts/mlflow_utils.py → scripts/mlflow_utils.py  _Bridges community 2 → community 1_
- `log_artifact_path()` --calls--> `_require_active_run()`  [EXTRACTED]
  scripts/mlflow_utils.py → scripts/mlflow_utils.py  _Bridges community 4 → community 3_
- `safe_log_params()` --calls--> `_require_active_run()`  [EXTRACTED]
  scripts/mlflow_utils.py → scripts/mlflow_utils.py  _Bridges community 4 → community 1_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **OOD/PPE experiment phase pipeline** — readme_ood_problem, readme_hardneg_vestboost, readme_kfold_validation, readme_helmetboost_pipeline, readme_kfold_pulito, readme_phase07_comparison [EXTRACTED 0.90]

## Communities (6 total, 1 thin omitted)

### Community 0 - "OOD/PPE Experiment Narrative"
Cohesion: 0.22
Nodes (7): Hard-negative OOD + vest boost, Helmetboost training pipeline, Clean k-fold (GroupKFold, dedup, no leakage), K-fold validation, OOD false-detection problem (YOLO PPE), Baseline vs candidate comparison (phase 07), ISaAC OOD/PPE Experiments

### Community 1 - "MLflow Run Lifecycle"
Cohesion: 0.36
Nodes (8): Any, Run, optional_mlflow_run(), Apre una run nell'experiment e la chiude automaticamente., Come ``start_mlflow_run`` ma disattivabile. Con ``enabled=False`` non crea né…, Converte i parametri in valori accettati da MLflow., safe_log_params(), start_mlflow_run()

### Community 2 - "MLflow Setup & Metrics"
Cohesion: 0.29
Nodes (6): get_mlflow_env_info(), init_mlflow(), log_metrics_if_active(), Configura MLflow e crea o seleziona un experiment. ``artifact_location`` viene…, Registra le metriche solo se esiste una run MLflow attiva., Restituisce la configurazione MLflow corrente.

### Community 3 - "MLflow Artifact Logging"
Cohesion: 0.40
Nodes (5): Path, log_artifact_if_active(), log_artifact_path(), Registra un file o una directory come artifact., Registra un file/directory come artifact solo se c'è una run attiva.

### Community 4 - "MLflow Metric Guardrails"
Cohesion: 0.50
Nodes (4): Evita la creazione implicita di run fuori dall'experiment previsto., Registra metriche numeriche nella run attiva., _require_active_run(), safe_log_metrics()

## Knowledge Gaps
- **2 isolated node(s):** `isaac-cvmod`, `Baseline vs candidate comparison (phase 07)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hard-negative OOD + vest boost` connect `OOD/PPE Experiment Narrative` to `MLflow Setup & Metrics`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `ISaAC OOD/PPE Experiments` connect `OOD/PPE Experiment Narrative` to `MLflow Setup & Metrics`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **What connects `isaac-cvmod`, `Baseline vs candidate comparison (phase 07)` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._