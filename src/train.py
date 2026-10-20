"""
train.py — Stage 3: train a baseline + advanced model, track with MLflow,
register and promote the best. Imported by Model_Development_and_Tracking.ipynb.

Run:  python -m src.train
"""
import contextlib
import io
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import config as cfg

for _p in [".", "..", "src", "../src"]:
    if os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

try:
    import src.data_prep as dp
except ModuleNotFoundError:
    import data_prep as dp
try:
    from src.evaluate import compute_metrics, confusion
except ModuleNotFoundError:
    from evaluate import compute_metrics, confusion


REFERENCE_ROWS = 5000   # training rows saved as the drift-monitoring baseline


def _log_model_compat(sk_model, artifact_name="model", **kw):
    """mlflow>=3 uses name=, mlflow<3 uses artifact_path= — support both.
    serialization_format="cloudpickle" keeps mlflow off the skops serialiser, which
    refuses the numpy.dtype objects held inside a fitted ColumnTransformer."""
    import mlflow.sklearn
    attempts = [
        dict(name=artifact_name,          serialization_format="cloudpickle"),
        dict(artifact_path=artifact_name, serialization_format="cloudpickle"),
        dict(name=artifact_name,          skops_trusted_types=["numpy.dtype"]),
        dict(artifact_path=artifact_name),
    ]
    last = None
    for extra in attempts:
        try:
            return mlflow.sklearn.log_model(sk_model=sk_model, **extra, **kw)
        except TypeError as exc:
            last = exc
    raise last


def build_models(numeric, categorical, scale_pos_weight):
    """The two Stage 3 candidates, each as a FULL Pipeline (preprocess + classify).

    Imbalance is handled inside the estimator — class_weight='balanced' for the
    baseline, scale_pos_weight = neg/pos for XGBoost — so no rows are discarded.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    baseline = Pipeline([
        ("preprocess", dp.build_preprocessor(numeric, categorical)),
        ("classify", LogisticRegression(class_weight="balanced", max_iter=2000,
                                        solver="lbfgs", random_state=cfg.RANDOM_STATE)),
    ])
    advanced = Pipeline([
        ("preprocess", dp.build_preprocessor(numeric, categorical)),
        ("classify", XGBClassifier(scale_pos_weight=scale_pos_weight, **cfg.XGB_PARAMS)),
    ])
    return {"logistic_regression": baseline, "xgboost": advanced}


def train_and_log():
    """TODO (Stage 3):
    - Get splits + feature lists from src.data_prep; compute scale_pos_weight = neg / pos.
    - Build two FULL sklearn Pipelines (preprocessor + classifier):
        * LogisticRegression(class_weight='balanced')           (baseline)
        * XGBClassifier(scale_pos_weight=..., **cfg.XGB_PARAMS)  (advanced)
    - For each model: fit on train, evaluate on validation (src.evaluate.compute_metrics),
      and log params + metrics + the model to the MLflow experiment cfg.MLFLOW_EXPERIMENT.
    - Pick the best by validation ROC-AUC, evaluate on the test split, then register it
      (cfg.REGISTERED_MODEL) and promote to the 'production' alias.
    - Save artifacts: best_model.pkl, metrics.json, reference_sample.csv, input_columns.json.
    Return (best_model_name, test_metrics). Expected validation ROC-AUC ~0.65."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    # ---------------------------------------------- data: Stage 2 preparation, reused
    raw = dp.load_raw()
    clean_df = dp.clean(raw)
    fe = dp.engineer_features(clean_df)

    X_train, X_val, X_test, y_train, y_val, y_test = dp.get_splits(fe)
    numeric, categorical = dp.get_feature_lists(fe)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / pos
    print(f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}")
    print(f"scale_pos_weight = {neg:,}/{pos:,} = {scale_pos_weight:.4f}")

    # ---------------------------------------------------------------- MLflow setup
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    # the first touch of a new SQLite store prints the full alembic migration log to stderr
    with contextlib.redirect_stderr(io.StringIO()):
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)
        MlflowClient().search_registered_models()

    models = build_models(numeric, categorical, scale_pos_weight)
    input_example = X_train.head(5)
    results = {}

    # ------------------------------------- fit on train, evaluate on validation, log
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_prob)
        cm = confusion(y_val, val_prob)

        clf_params = model.named_steps["classify"].get_params()
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({str(k): str(v) for k, v in clf_params.items()})
            mlflow.log_params({
                "model_type": type(model.named_steps["classify"]).__name__,
                "random_state": cfg.RANDOM_STATE,
                "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                "n_numeric": len(numeric), "n_categorical": len(categorical),
                "train_pos_rate": round(float(y_train.mean()), 4),
                "imbalance_handling": clf_params.get("class_weight")
                or round(scale_pos_weight, 4),
            })
            mlflow.log_metrics({f"val_{k}": float(v) for k, v in metrics.items()})
            mlflow.log_metrics({f"val_{k}": float(v) for k, v in cm.items()})
            mlflow.set_tags({"stage": "3", "selection_metric": "val_roc_auc"})
            _log_model_compat(
                model, "model",
                signature=infer_signature(input_example,
                                          model.predict_proba(input_example)[:, 1]),
                input_example=input_example)
            results[name] = {"model": model, "metrics": metrics, "confusion": cm,
                             "run_id": run.info.run_id}
        print(f"logged {name:<20} val_roc_auc={metrics['roc_auc']:.4f} "
              f"val_pr_auc={metrics['pr_auc']:.4f} recall={metrics['recall']:.4f}")

    # --------------------------------------------- select on validation ROC-AUC
    best_model_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_model_name]
    print(f"\nbest by val ROC-AUC: {best_model_name} ({best['metrics']['roc_auc']:.4f})")

    # --------------------------------------------- register + promote to production
    mv = mlflow.register_model(model_uri=f"runs:/{best['run_id']}/model",
                               name=cfg.REGISTERED_MODEL)
    client = MlflowClient()
    for _ in range(30):
        if client.get_model_version(cfg.REGISTERED_MODEL, mv.version).status == "READY":
            break
        time.sleep(1)
    try:
        client.set_registered_model_alias(cfg.REGISTERED_MODEL, "production", mv.version)
    except AttributeError:
        client.transition_model_version_stage(cfg.REGISTERED_MODEL, mv.version,
                                              stage="Production",
                                              archive_existing_versions=True)
    client.set_model_version_tag(cfg.REGISTERED_MODEL, mv.version, "val_roc_auc",
                                 str(best["metrics"]["roc_auc"]))
    print(f"registered {cfg.REGISTERED_MODEL} v{mv.version} and promoted to 'production'")

    # ------------------------------- test split: touched exactly once, at the end
    test_prob = best["model"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    test_cm = confusion(y_test, test_prob)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_cm.items()})
        mlflow.set_tags({"registered_model": cfg.REGISTERED_MODEL,
                         "registered_version": str(mv.version), "alias": "production"})
    print("test metrics:", test_metrics)

    # --------------------------------------- artefacts for Stage 4 (serve + monitor)
    art = cfg.ARTIFACT_DIR
    joblib.dump(best["model"], art / "best_model.pkl")
    (art / "metrics.json").write_text(json.dumps({
        "best_model": best_model_name, "selection_metric": "val_roc_auc",
        "registered_model": cfg.REGISTERED_MODEL, "registered_version": int(mv.version),
        "run_id": best["run_id"],
        "validation": best["metrics"], "validation_confusion": best["confusion"],
        "test": test_metrics, "test_confusion": test_cm,
        "all_validation_metrics": {n: r["metrics"] for n, r in results.items()},
    }, indent=2))
    # 5,000 rows (the size of a monitoring batch): above 1,000 reference rows Evidently uses
    # Wasserstein / Jensen-Shannon distances, which do not explode on rare categories the
    # way chi-square does on a small reference.
    X_train.sample(min(REFERENCE_ROWS, len(X_train)), random_state=cfg.RANDOM_STATE) \
           .to_csv(art / "reference_sample.csv", index=False)
    # The serving contract is the ordered list of raw input columns (tests/test_app.py reads it
    # as a list); numeric/categorical roles live inside the fitted ColumnTransformer itself.
    (art / "input_columns.json").write_text(json.dumps(list(X_train.columns), indent=2))
    print("artefacts saved to", art)

    return best_model_name, test_metrics


if __name__ == "__main__":
    train_and_log()
