"""
monitoring.py — Stage 4.4: data + prediction drift monitoring.
Compares a production 'current' batch against the saved training reference.

Run:  python -m src.monitoring
"""
import config as cfg


def run_monitoring():
    """TODO (Stage 4.4):
    - Load artifacts/reference_sample.csv (training baseline) and artifacts/current_batch.csv.
    - Feature drift: run Evidently DataDriftPreset(reference vs current) and save drift_report.html;
      read number_of_drifted_columns / number_of_columns / dataset_drift.
    - Prediction drift: load artifacts/best_model.pkl and compute the PSI between the reference and
      current predicted probabilities.
    - Write artifacts/drift_summary.json and return the summary dict.
    Expected (reference run): ~6/39 features drifted, dataset_drift False, prediction PSI ~0.46."""
    # Feature drift runs Evidently's DataDriftPreset on explicitly typed columns and reads
    # DriftedColumnsCount + the per-column ValueDrift results; prediction drift is the PSI of
    # the production pipeline's scores. A SciPy KS/chi-square path is kept only as a labelled
    # fallback for environments where Evidently cannot be imported.
    # ===== SOLUTION START =====
    import json
    import joblib
    import numpy as np
    import pandas as pd

    art = cfg.ARTIFACT_DIR
    reference = pd.read_csv(art / "reference_sample.csv")
    current = pd.read_csv(art / "current_batch.csv")

    # Align the two frames to a common schema so drift is computed column-for-column.
    shared = [c for c in reference.columns if c in current.columns]
    reference, current = reference[shared].copy(), current[shared].copy()

    # Column roles come from the fitted pipeline, not from CSV dtype inference (the nominal
    # ID codes read back from CSV as integers but are categorical).
    model = joblib.load(art / "best_model.pkl")
    try:
        roles = {name: list(cols) for name, _, cols
                 in model.named_steps["preprocess"].transformers_}
        categorical = [c for c in roles.get("cat", []) if c in shared]
    except Exception:
        categorical = reference.select_dtypes(exclude=np.number).columns.tolist()
    numeric = [c for c in shared if c not in categorical]
    ref_typed, cur_typed = reference.copy(), current.copy()
    for col in categorical:
        ref_typed[col] = ref_typed[col].map(_as_label)
        cur_typed[col] = cur_typed[col].map(_as_label)

    feat = _feature_drift(ref_typed, cur_typed, art / "drift_report.html",
                          numeric=numeric, categorical=categorical)

    # ---- prediction drift: PSI between reference and current model scores ----
    ref_scores = model.predict_proba(reference)[:, 1]
    cur_scores = model.predict_proba(current)[:, 1]
    prediction_psi = psi(ref_scores, cur_scores)

    n_cols = int(feat["number_of_columns"])
    n_drifted = int(feat["number_of_drifted_columns"])
    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_reference_rows": int(len(reference)),
        "n_current_rows": int(len(current)),
        "number_of_columns": n_cols,
        "number_of_drifted_columns": n_drifted,
        "share_drifted": round(n_drifted / n_cols, 4) if n_cols else 0.0,
        "dataset_drift": bool(feat["dataset_drift"]),
        "drifted_columns": feat["drifted_columns"],
        "drift_method": feat["method"],
        "column_drift": feat.get("column_drift", []),
        "prediction_psi": round(float(prediction_psi), 4),
        "psi_interpretation": interpret_psi(prediction_psi),
        "reference_score_mean": round(float(np.mean(ref_scores)), 4),
        "current_score_mean": round(float(np.mean(cur_scores)), 4),
        "report_html": str((art / "drift_report.html").name),
    }

    (art / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"feature drift : {n_drifted}/{n_cols} columns "
        f"(share={summary['share_drifted']}, dataset_drift={summary['dataset_drift']})"
    )
    print(
        f"prediction PSI: {summary['prediction_psi']} "
        f"({summary['psi_interpretation']}) "
        f"mean {summary['reference_score_mean']} -> {summary['current_score_mean']}"
    )
    return summary
    # ===== SOLUTION END =====


# ===== SOLUTION START =====
# Helpers used by run_monitoring(). Kept at module level so the operations notebook can
# reuse psi() directly when it plots the score distributions (Stage 4.4.2).

PSI_BINS = 10


def psi(expected, actual, bins=PSI_BINS):
    """Population Stability Index between two score arrays.

    Quantile-bins the reference ('expected') scores, then compares the proportion of
    reference vs current observations falling in each bin:

        PSI = sum( (a_i - e_i) * ln(a_i / e_i) )

    A small epsilon floors empty bins so the log stays finite. Bands used here (aligned with
    the PSI > 0.2 retraining trigger): < 0.10 stable, 0.10-0.20 moderate, > 0.20 significant.
    """
    import numpy as np

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Quantile edges from the reference distribution; de-duplicate for tied scores.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    e_pct = np.histogram(expected, bins=edges)[0] / expected.size
    a_pct = np.histogram(actual, bins=edges)[0] / actual.size
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def interpret_psi(value):
    """Map a PSI value onto the conventional stability bands."""
    if value < 0.10:
        return "stable (<0.10)"
    if value <= 0.20:
        return "moderate shift (0.10-0.20)"
    return "significant shift (>0.20)"


def _as_label(v):
    """Categorical cell -> string label (1 and 1.0 both become '1'); NaN stays NaN."""
    import pandas as pd

    if pd.isna(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _statistical_drift(reference, current, alpha=0.05):
    """Fallback feature drift: KS test for numerics, chi-square for categoricals.

    Used only when Evidently is unavailable, so the stage still produces a defensible
    drift verdict instead of failing.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    drifted = []
    for col in reference.columns:
        ref_col, cur_col = reference[col].dropna(), current[col].dropna()
        if ref_col.empty or cur_col.empty:
            continue
        if pd.api.types.is_numeric_dtype(reference[col]):
            p = stats.ks_2samp(ref_col, cur_col).pvalue
        else:
            levels = sorted(set(ref_col.unique()) | set(cur_col.unique()))
            table = np.vstack([
                ref_col.value_counts().reindex(levels, fill_value=0).to_numpy(),
                cur_col.value_counts().reindex(levels, fill_value=0).to_numpy(),
            ])
            table = table[:, table.sum(axis=0) > 0]
            p = stats.chi2_contingency(table).pvalue if table.shape[1] > 1 else 1.0
        if p < alpha:
            drifted.append(col)
    return drifted


#: Evidently's DataDriftPreset default: the dataset is "drifted" when at least this share
#: of columns drift.
DATASET_DRIFT_SHARE = 0.5


def _feature_drift(reference, current, html_path, numeric=None, categorical=None):
    """Run Evidently's DataDriftPreset and normalise the result across API versions.

    Evidently renamed its API between 0.4 and 0.7, so each known shape is tried in turn
    and a pure-SciPy equivalent is the final fallback. Returns a dict with
    number_of_columns, number_of_drifted_columns, dataset_drift, drifted_columns,
    column_drift (per-column method/score/threshold) and method.
    """
    n_cols = int(reference.shape[1])

    # --- Evidently >= 0.7: Dataset/Report([...]) ---
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        Report = None
    if Report is not None:
        definition = DataDefinition(numerical_columns=list(numeric or []),
                                    categorical_columns=list(categorical or []))
        ref_ds = Dataset.from_pandas(reference, data_definition=definition)
        cur_ds = Dataset.from_pandas(current, data_definition=definition)
        result = Report([DataDriftPreset(drift_share=DATASET_DRIFT_SHARE)]).run(
            reference_data=ref_ds, current_data=cur_ds)
        result.save_html(str(html_path))
        parsed = _parse_evidently_v07(result.dict(), n_cols)
        parsed["method"] = "evidently 0.7 DataDriftPreset"
        return parsed

    # --- Evidently 0.4-0.6: Report(metrics=[...]) ---
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        try:
            report.save_html(str(html_path))
        except Exception:
            pass
        parsed = _parse_evidently(report.as_dict(), n_cols)
        parsed["method"] = "evidently 0.4-0.6 DataDriftPreset"
        return parsed
    except Exception:
        pass

    # --- fallback: KS / chi-square ---
    drifted = _statistical_drift(reference, current)
    share = len(drifted) / n_cols if n_cols else 0.0
    html_path.write_text(
        "<h2>Data drift (SciPy fallback)</h2>"
        f"<p>Evidently unavailable. KS (numeric) / chi-square (categorical), alpha=0.05.</p>"
        f"<p>{len(drifted)} of {n_cols} columns drifted (share={share:.3f}).</p>"
        f"<p>Drifted: {', '.join(drifted) if drifted else 'none'}</p>"
    )
    return {
        "number_of_columns": n_cols,
        "number_of_drifted_columns": len(drifted),
        "dataset_drift": bool(share > 0.5),
        "drifted_columns": drifted,
        "method": "scipy KS/chi-square fallback",
    }


def _parse_evidently_v07(payload, n_cols):
    """Read an Evidently 0.7 report: DriftedColumnsCount -> {count, share}; one ValueDrift
    metric per column whose value is a p-value (drift if < threshold) or a distance
    (drift if >= threshold), depending on the test Evidently picked for that column."""
    count, share, columns = None, None, []
    for metric in payload.get("metrics", []):
        conf = metric.get("config", {})
        kind = str(conf.get("type", ""))
        value = metric.get("value")
        if kind.endswith("DriftedColumnsCount"):
            count, share = int(value["count"]), float(value["share"])
        elif kind.endswith("ValueDrift"):
            method = str(conf.get("method", ""))
            threshold = float(conf.get("threshold"))
            score = float(value)
            is_p_value = "p_value" in method
            drifted = score < threshold if is_p_value else score >= threshold
            columns.append({"column": conf.get("column"), "method": method,
                            "score": round(score, 6), "threshold": threshold,
                            "drift_detected": bool(drifted)})
    if count is None:
        count = sum(c["drift_detected"] for c in columns)
        share = count / n_cols if n_cols else 0.0
    return {
        "number_of_columns": n_cols,
        "number_of_drifted_columns": count,
        "dataset_drift": bool(share >= DATASET_DRIFT_SHARE),
        "drifted_columns": [c["column"] for c in columns if c["drift_detected"]],
        "column_drift": columns,
    }


def _parse_evidently(payload, n_cols):
    """Pull the drift counts out of an Evidently 0.4-0.6 payload, whatever its nesting."""
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("number_of_drifted_columns", "number_of_columns",
                           "dataset_drift", "share_of_drifted_columns") \
                        and not isinstance(value, (dict, list)):
                    found.setdefault(key, value)
                if key in ("drifted_columns", "drift_by_columns") and isinstance(value, dict):
                    found.setdefault("drifted_columns", [
                        c for c, v in value.items()
                        if isinstance(v, dict) and v.get("drift_detected")
                    ])
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)

    total = int(found.get("number_of_columns") or n_cols)
    if "number_of_drifted_columns" in found:
        n_drifted = int(found["number_of_drifted_columns"])
    elif "share_of_drifted_columns" in found:
        n_drifted = int(round(float(found["share_of_drifted_columns"]) * total))
    else:
        n_drifted = len(found.get("drifted_columns", []))

    return {
        "number_of_columns": total,
        "number_of_drifted_columns": n_drifted,
        "dataset_drift": bool(found.get("dataset_drift", False)),
        "drifted_columns": found.get("drifted_columns", []),
    }
# ===== SOLUTION END =====



if __name__ == "__main__":
    run_monitoring()
