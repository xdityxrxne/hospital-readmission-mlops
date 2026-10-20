"""
retrain.py (STARTER) — Stage 4.5: multi-signal retraining trigger + workflow.

Run:  python -m src.retrain
"""
import config as cfg

PSI_THRESHOLD = 0.20            # representative example for instruction
DRIFT_SHARE_THRESHOLD = 0.30   # representative example for instruction


def decide(summary):
    """TODO: return a list of human-readable reasons if any trigger fires —
    prediction_psi > PSI_THRESHOLD, share_drifted > DRIFT_SHARE_THRESHOLD, or dataset_drift is True.
    An empty list means no retraining is needed."""
    # ===== SOLUTION START =====
    reasons = []

    psi = float(summary.get("prediction_psi") or 0.0)
    share = float(summary.get("share_drifted") or 0.0)
    dataset_drift = bool(summary.get("dataset_drift", False))

    if psi > PSI_THRESHOLD:
        reasons.append(
            f"prediction drift: PSI {psi:.4f} > {PSI_THRESHOLD} "
            f"({summary.get('psi_interpretation', 'score distribution moved')})"
        )
    if share > DRIFT_SHARE_THRESHOLD:
        reasons.append(
            f"feature drift: {summary.get('number_of_drifted_columns', '?')}/"
            f"{summary.get('number_of_columns', '?')} columns drifted "
            f"(share {share:.4f} > {DRIFT_SHARE_THRESHOLD})"
        )
    if dataset_drift:
        reasons.append("dataset drift: Evidently dataset_drift flag is True")

    return reasons
    # ===== SOLUTION END =====



def run_retraining_workflow():
    """TODO (Stage 4.5): read artifacts/drift_summary.json; if decide() returns reasons, retrain
    (call src.train.train_and_log, which registers + promotes a new version) and record
    artifacts/retraining_decision.json with the reasons + new metrics. Return the decision dict.
    In production the retrain would use freshly LABELLED data; here it re-runs the pipeline to
    demonstrate the automated workflow + versioning."""
    # ===== SOLUTION START =====
    import json
    import pandas as pd

    art = cfg.ARTIFACT_DIR
    summary_path = art / "drift_summary.json"

    # The trigger is driven by the monitoring output; generate it if it is not there yet.
    if not summary_path.exists():
        from src.monitoring import run_monitoring
        summary = run_monitoring()
    else:
        summary = json.loads(summary_path.read_text())

    reasons = decide(summary)
    decision = {
        "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "thresholds": {
            "psi_threshold": PSI_THRESHOLD,
            "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        },
        "signals": {
            "prediction_psi": summary.get("prediction_psi"),
            "share_drifted": summary.get("share_drifted"),
            "number_of_drifted_columns": summary.get("number_of_drifted_columns"),
            "number_of_columns": summary.get("number_of_columns"),
            "dataset_drift": summary.get("dataset_drift"),
        },
        "retrain_triggered": bool(reasons),
        "reasons": reasons,
    }

    if not reasons:
        decision["action"] = "no_retrain"
        decision["note"] = "no trigger fired; the production version is retained"
    else:
        print("retraining triggered:")
        for reason in reasons:
            print("  -", reason)
        try:
            # train_and_log() retrains, logs to MLflow, registers a new version and
            # promotes it to the 'production' alias. In production this would run on
            # freshly LABELLED encounters; here it re-runs the pipeline to demonstrate
            # the automated workflow and the resulting version bump.
            result = train_and_log()
            if isinstance(result, dict):
                best_name = result.get("best_model") or result.get("model")
                new_metrics = result.get("test") or result.get("metrics") or {}
                new_version = result.get("registered_version")
            else:
                best_name, new_metrics = result
                new_version = None

            if new_version is None:
                metrics_path = art / "metrics.json"
                if metrics_path.exists():
                    new_version = json.loads(metrics_path.read_text()).get(
                        "registered_version"
                    )

            decision["action"] = "retrained_and_registered"
            decision["new_model"] = best_name
            decision["new_metrics"] = new_metrics
            decision["new_registered_version"] = new_version
            decision["registered_model"] = cfg.REGISTERED_MODEL
        except Exception as exc:
            decision["action"] = "retrain_failed"
            decision["error"] = f"{type(exc).__name__}: {exc}"
            print("retraining failed:", decision["error"])

    (art / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
    print(f"\ndecision: {decision['action']} -> artifacts/retraining_decision.json")
    return decision
    # ===== SOLUTION END =====


# ===== SOLUTION START =====
def train_and_log():
    """Thin indirection onto src.train.train_and_log (imported lazily).

    Keeps this module importable even when MLflow / XGBoost are absent, so the
    trigger logic in decide() can be unit-tested on its own.
    """
    from src.train import train_and_log as _train_and_log
    return _train_and_log()
# ===== SOLUTION END =====



if __name__ == "__main__":
    run_retraining_workflow()
