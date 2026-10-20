"""
app.py — Stage 4.1: FastAPI inference service.
Implement GET /health and POST /predict, then demonstrate it from the operations notebook.

Run:  uvicorn app:app --reload
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5

app = FastAPI(title="Hospital Readmission Predictor", version="1.0")


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw encounter features (column -> value)")


# TODO (Stage 4.1):
# - Lazily load artifacts/best_model.pkl and artifacts/input_columns.json.
# - GET  /health  -> {"status": ..., "model_loaded": bool}.
# - POST /predict -> align the incoming features to the training columns (missing -> NaN; the
#   pipeline imputes), return {"readmission_probability", "readmitted_30d", "threshold"}.
#   Validate the payload (reject an empty features dict with 422) and log every prediction to
#   artifacts/predictions.log (governance). Note: JSON has no NaN, so clients send null / omit fields.
# ===== SOLUTION START =====
# Lazily-loaded globals: the Stage 3 pipeline (preprocessor + classifier) and the
# training-time column contract. Loading on first request keeps container start-up fast
# and lets /health report honestly when the artefacts are missing.
_MODEL = None
_COLUMNS = None
_LOAD_ERROR = None

NAN = float("nan")
MODEL_PATH = ARTIFACTS / "best_model.pkl"
COLUMNS_PATH = ARTIFACTS / "input_columns.json"
PREDICTIONS_LOG = ARTIFACTS / "predictions.log"

# Counters that must never be negative in a real encounter -> reject with 422.
NON_NEGATIVE_FIELDS = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "service_utilization", "num_med_changes", "num_meds_prescribed", "age",
)


def _column_roles(model, all_cols):
    """Numeric / categorical column lists, read from the fitted ColumnTransformer itself."""
    roles = {"all": list(all_cols), "numeric": [], "categorical": []}
    try:
        for name, _, cols in model.named_steps["preprocess"].transformers_:
            if name == "num":
                roles["numeric"] = list(cols)
            elif name == "cat":
                roles["categorical"] = list(cols)
    except Exception:
        pass
    return roles


def _load():
    """Load artifacts/best_model.pkl + artifacts/input_columns.json once per process."""
    global _MODEL, _COLUMNS, _LOAD_ERROR
    if _MODEL is None:
        try:
            _MODEL = joblib.load(MODEL_PATH)
            _COLUMNS = _column_roles(_MODEL, json.loads(COLUMNS_PATH.read_text()))
            _LOAD_ERROR = None
        except Exception as exc:
            _MODEL, _COLUMNS = None, None
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    return _MODEL, _COLUMNS


def _align(features, columns):
    """Align a raw feature dict to the training columns.

    Any column the client omits (or sends as null) becomes NaN so that the imputers
    inside the fitted ColumnTransformer fill it exactly as they did during training.
    Extra keys the model never saw are ignored rather than silently changing the schema.
    """
    all_cols = columns["all"]
    row = {}
    for col in all_cols:
        value = features.get(col, NAN)
        row[col] = NAN if value is None else value

    X = pd.DataFrame([row], columns=all_cols)
    for col in columns["numeric"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in columns["categorical"]:
        X[col] = X[col].astype("object")
    return X


def _validate_payload(features):
    """Validate the request body on its own terms (no model needed) -> 422 on bad input."""
    if not isinstance(features, dict) or len(features) == 0:
        raise HTTPException(
            status_code=422,
            detail="'features' must be a non-empty object of column -> value pairs",
        )

    bad = []
    for key in NON_NEGATIVE_FIELDS:
        value = features.get(key)
        if value is None:
            continue
        try:
            if float(value) < 0:
                bad.append(f"{key}={value}")
        except (TypeError, ValueError):
            bad.append(f"{key}={value!r} (not numeric)")
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"invalid non-negative counter(s): {', '.join(bad)}",
        )


def _field_diff(features, columns):
    """Which training columns the client omitted, and which keys the model never saw."""
    all_cols = columns["all"]
    missing = [c for c in all_cols if c not in features or features.get(c) is None]
    unknown = [k for k in features if k not in all_cols]
    return missing, unknown


def _log_prediction(record):
    """Append one JSON line to artifacts/predictions.log (4.6.1 governance evidence).

    Logging is best-effort: an unwritable log must never take the service down.
    """
    try:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


@app.get("/health")
def health():
    """Liveness + readiness: reports whether the model artefact actually loaded."""
    model, columns = _load()
    payload = {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
        "n_expected_features": len(columns["all"]) if columns else 0,
    }
    if _LOAD_ERROR:
        payload["error"] = _LOAD_ERROR
    return payload


@app.post("/predict")
def predict(req: PredictRequest):
    """Score one encounter and return the probability, the label and the threshold used."""
    # Payload validation first: a malformed request is a client error (422) regardless
    # of whether the model happens to be loaded.
    features = req.features
    _validate_payload(features)

    model, columns = _load()
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded ({_LOAD_ERROR})")

    missing, unknown = _field_diff(features, columns)
    X = _align(features, columns)
    try:
        probability = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"inference failed: {type(exc).__name__}: {exc}",
        )

    response = {
        "readmission_probability": round(probability, 6),
        "readmitted_30d": int(probability >= THRESHOLD),
        "threshold": THRESHOLD,
        "n_missing_fields": len(missing),
        "n_unknown_fields": len(unknown),
    }

    _log_prediction({
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "endpoint": "/predict",
        "n_features_sent": len(features),
        "n_missing_fields": len(missing),
        "unknown_fields": unknown[:10],
        "readmission_probability": response["readmission_probability"],
        "readmitted_30d": response["readmitted_30d"],
        "threshold": THRESHOLD,
    })
    return response
# ===== SOLUTION END =====
