"""
benchmark_models.py
====================
Comprehensive benchmark script that:
  1. Loads all 4 pre-trained models (Random Forest, XGBoost, LightGBM, Logistic Regression)
  2. Evaluates each model on the held-out test set with full metrics + CV
  3. Saves per-model results to  model_benchmarks/<ModelName>/
  4. Generates comparison charts (HTML) and a master summary CSV
  5. Identifies and prints the BEST model by ROC-AUC
  6. Writes best_model.txt so the Streamlit app knows which model to use

Run from the project root:
    python benchmark_models.py
"""

import os
import sys
import io
import json
import time
import pickle
import warnings
import numpy  as np
import pandas as pd
from pathlib import Path

# ─── sklearn / xgboost / lightgbm ────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


def _write_html(fig, path: Path):
    """Windows-safe plotly HTML writer.

    plotly's built-in write_html uses pathlib.write_text which fails on
    Windows when the path contains spaces (OSError: [Errno 22] Invalid
    argument).  We work around it by generating the HTML string and writing
    with a plain open() call instead.
    """
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(html_str)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
OUTPUT_DIR  = ROOT / "model_benchmarks"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model registry – name → filename stem
MODEL_REGISTRY = {
    "Random Forest":        "random_forest",
    "XGBoost":              "xgboost",
    "LightGBM":             "lightgbm",
    "Logistic Regression":  "logistic_regression",
}

# Primary ranking metric
RANK_METRIC = "roc_auc"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_data():
    print("📂  Loading processed test data …")
    X_train  = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test   = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train  = pd.read_csv(DATA_DIR / "y_train.csv").squeeze().astype(int)
    y_test   = pd.read_csv(DATA_DIR / "y_test.csv").squeeze().astype(int)
    print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"   Fraud rate – train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def _retrain_and_save(stem: str, X_train, y_train):
    """Train a fresh model from scratch and save it to models/<stem>.pkl."""
    print(f"   🔄  Retraining {stem} from scratch (stale pkl detected)…")
    if stem == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                       random_state=42, n_jobs=-1)
    elif stem == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=6,
                               learning_rate=0.1, random_state=42,
                               eval_metric="logloss", verbosity=0)
    elif stem == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, random_state=42,
                                verbose=-1)
    elif stem == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0, max_iter=1000,
                                    random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model stem: {stem}")

    model.fit(X_train, y_train)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / f"{stem}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"   ✅  {stem}.pkl saved.")
    return model


def load_model(stem: str, X_train=None, y_train=None):
    """Load model from disk, retraining from scratch if the pkl is stale/broken."""
    path = MODELS_DIR / f"{stem}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            # Quick sanity-check: can we call get_params()?
            model.get_params()
            # For forests, check the new sklearn 1.3+ attribute
            if hasattr(model, "estimators_"):
                _ = model.estimators_[0].__class__.__name__
                if not hasattr(model.estimators_[0], "monotonic_cst"):
                    raise AttributeError("stale sklearn pkl")
            return model
        except Exception as exc:
            print(f"   ⚠️  {stem}.pkl is stale ({exc}). Deleting and retraining…")
            path.unlink(missing_ok=True)

    # Either never existed or just deleted
    if X_train is not None and y_train is not None:
        return _retrain_and_save(stem, X_train, y_train)
    return None


def optimize_threshold(y_true, y_proba, metric="f1"):
    """Scan thresholds 0.10–0.90 and return the one that maximises *metric*."""
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_hat = (y_proba >= thr).astype(int)
        score = {
            "f1":        f1_score(y_true, y_hat, zero_division=0),
            "recall":    recall_score(y_true, y_hat, zero_division=0),
            "precision": precision_score(y_true, y_hat, zero_division=0),
            "accuracy":  accuracy_score(y_true, y_hat),
        }[metric]
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Full evaluation: threshold search, test metrics, 5-fold CV."""
    print(f"\n{'─'*55}")
    print(f"   Evaluating  →  {name}")
    print(f"{'─'*55}")

    # --- Raw probabilities ---
    t0 = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]
    inf_ms  = (time.time() - t0) * 1000

    # --- Optimal threshold (F1) ---
    best_thr = optimize_threshold(y_test, y_proba, metric="f1")
    y_pred   = (y_proba >= best_thr).astype(int)

    # --- PR-AUC ---
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec_arr, prec_arr)

    # --- Core metrics ---
    metrics = {
        "accuracy":       float(accuracy_score(y_test, y_pred)),
        "precision":      float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":             float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":        float(roc_auc_score(y_test, y_proba)),
        "pr_auc":         float(pr_auc),
        "best_threshold": float(best_thr),
        "inference_ms":   round(inf_ms, 2),
    }

    # --- 5-fold Stratified CV on *test* set (small, so feasible) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scoring = {
        "accuracy":  "accuracy",
        "precision": "precision",
        "recall":    "recall",
        "f1":        "f1",
        "roc_auc":   "roc_auc",
    }
    cv_res = cross_validate(model, X_test, y_test, cv=cv, scoring=cv_scoring, n_jobs=-1)
    for m in cv_scoring:
        metrics[f"cv_{m}_mean"] = float(cv_res[f"test_{m}"].mean())
        metrics[f"cv_{m}_std"]  = float(cv_res[f"test_{m}"].std())

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # --- Feature importance (if available) ---
    feature_names = list(X_test.columns)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = dict(sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        ))
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0])
        fi = dict(sorted(zip(feature_names, coef.tolist()), key=lambda x: x[1], reverse=True))
    else:
        fi = {}
    metrics["feature_importance"] = fi

    # --- Classification report ---
    cr = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
    metrics["classification_report"] = cr

    # --- Print summary ---
    print(f"   Accuracy    : {metrics['accuracy']:.4f}")
    print(f"   Precision   : {metrics['precision']:.4f}")
    print(f"   Recall      : {metrics['recall']:.4f}")
    print(f"   F1-Score    : {metrics['f1']:.4f}")
    print(f"   ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC      : {metrics['pr_auc']:.4f}")
    print(f"   Best Thresh : {metrics['best_threshold']:.2f}")
    print(f"   CV ROC-AUC  : {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"   TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"   Inference   : {inf_ms:.1f} ms (full test set)")

    return metrics, y_proba, y_pred, feature_names


def save_model_results(name, metrics, y_test, y_proba, y_pred, feature_names):
    """Save all per-model artifacts to  model_benchmarks/<ModelName>/"""
    safe_name = name.replace(" ", "_")
    model_dir = OUTPUT_DIR / safe_name
    model_dir.mkdir(exist_ok=True)

    # ── 1. metrics.json ───────────────────────────────────────────────────────
    # Remove nested objects for clean JSON
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("feature_importance", "classification_report")}
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(serialisable, f, indent=4)

    # ── 2. feature_importance.csv ─────────────────────────────────────────────
    fi = metrics.get("feature_importance", {})
    if fi:
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
        fi_df.to_csv(model_dir / "feature_importance.csv", index=False)

    # ── 3. classification_report.csv ─────────────────────────────────────────
    cr = metrics.get("classification_report", {})
    if cr:
        cr_df = pd.DataFrame(cr).T
        cr_df.to_csv(model_dir / "classification_report.csv")

    # ── 4. predictions.csv ───────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "y_true":  y_test.values,
        "y_proba": y_proba,
        "y_pred":  y_pred,
    })
    pred_df.to_csv(model_dir / "predictions.csv", index=False)

    # ── 5. ROC curve plot ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = metrics["roc_auc"]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"{name} (AUC={roc_auc_val:.4f})",
        line=dict(color="#6366f1", width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="gray", dash="dash")
    ))
    fig_roc.update_layout(
        title=f"ROC Curve — {name}",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        template="plotly_dark", height=500
    )
    _write_html(fig_roc, model_dir / "roc_curve.html")

    # ── 6. Precision-Recall curve plot ───────────────────────────────────────
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr, mode="lines",
        name=f"{name} (PR-AUC={metrics['pr_auc']:.4f})",
        line=dict(color="#22d3ee", width=3)
    ))
    fig_pr.update_layout(
        title=f"Precision-Recall Curve — {name}",
        xaxis_title="Recall", yaxis_title="Precision",
        template="plotly_dark", height=500
    )
    _write_html(fig_pr, model_dir / "precision_recall_curve.html")

    # ── 7. Confusion matrix plot ──────────────────────────────────────────────
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    cm = [[tn, fp], [fn, tp]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred: Legit", "Pred: Fraud"],
        y=["Actual: Legit", "Actual: Fraud"],
        text=[[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]],
        texttemplate="%{text}",
        colorscale="Blues", showscale=True
    ))
    fig_cm.update_layout(
        title=f"Confusion Matrix — {name}",
        template="plotly_dark", height=450
    )
    _write_html(fig_cm, model_dir / "confusion_matrix.html")

    # ── 8. Feature importance bar chart ──────────────────────────────────────
    if fi:
        top_fi = dict(list(fi.items())[:25])
        fig_fi = go.Figure(go.Bar(
            x=list(top_fi.values()),
            y=list(top_fi.keys()),
            orientation="h",
            marker=dict(
                color=list(top_fi.values()),
                colorscale="Viridis"
            )
        ))
        fig_fi.update_layout(
            title=f"Top-25 Feature Importances — {name}",
            xaxis_title="Importance", template="plotly_dark",
            height=600, yaxis=dict(autorange="reversed")
        )
        _write_html(fig_fi, model_dir / "feature_importance.html")

    print(f"   ✅  Results saved →  model_benchmarks/{safe_name}/")


def generate_comparison_report(all_metrics: dict, y_test, all_probas: dict):
    """Master comparison: summary CSV + HTML dashboard."""
    print(f"\n{'='*55}")
    print("   GENERATING MASTER COMPARISON REPORT")
    print(f"{'='*55}")

    # ── Summary table ─────────────────────────────────────────────────────────
    core_cols = ["accuracy", "precision", "recall", "f1", "roc_auc",
                 "pr_auc", "best_threshold", "cv_roc_auc_mean",
                 "cv_roc_auc_std", "inference_ms", "tp", "fp", "tn", "fn"]
    rows = []
    for name, m in all_metrics.items():
        row = {"Model": name}
        for c in core_cols:
            row[c] = m.get(c, np.nan)
        rows.append(row)
    summary_df = pd.DataFrame(rows).set_index("Model")
    summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv")
    print(f"   Saved: model_benchmarks/model_comparison_summary.csv")

    # ── Bar chart comparison ───────────────────────────────────────────────────
    models = list(all_metrics.keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    palette = ["#6366f1", "#22d3ee", "#f59e0b", "#10b981", "#ef4444", "#a855f7"]

    fig_cmp = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m.replace("_", " ").upper() for m in metrics_to_plot]
    )
    for idx, (metric, color) in enumerate(zip(metrics_to_plot, palette)):
        r, c = divmod(idx, 3)
        vals = [all_metrics[m].get(metric, 0) for m in models]
        fig_cmp.add_trace(
            go.Bar(x=models, y=vals, name=metric,
                   marker_color=color, text=[f"{v:.3f}" for v in vals],
                   textposition="outside"),
            row=r + 1, col=c + 1
        )
        fig_cmp.update_yaxes(range=[0, 1.05], row=r + 1, col=c + 1)

    fig_cmp.update_layout(
        title="Model Benchmark — All Metrics Comparison",
        height=700, template="plotly_dark", showlegend=False
    )
    _write_html(fig_cmp, OUTPUT_DIR / "model_comparison_chart.html")

    # ── Overlaid ROC curves ───────────────────────────────────────────────────
    fig_roc = go.Figure()
    for (name, y_proba), color in zip(all_probas.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        rauc = roc_auc_score(y_test, y_proba)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={rauc:.4f})",
            line=dict(color=color, width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
        line=dict(color="gray", dash="dash")
    ))
    fig_roc.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="FPR", yaxis_title="TPR",
        template="plotly_dark", height=550
    )
    _write_html(fig_roc, OUTPUT_DIR / "all_roc_curves.html")
    print("   Saved: model_benchmarks/all_roc_curves.html")

    # ── Overlaid PR curves ───────────────────────────────────────────────────
    fig_pr = go.Figure()
    for (name, y_proba), color in zip(all_probas.items(), palette):
        prec_a, rec_a, _ = precision_recall_curve(y_test, y_proba)
        prauc = auc(rec_a, prec_a)
        fig_pr.add_trace(go.Scatter(
            x=rec_a, y=prec_a, mode="lines",
            name=f"{name} (PR-AUC={prauc:.4f})",
            line=dict(color=color, width=2.5)
        ))
    fig_pr.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall", yaxis_title="Precision",
        template="plotly_dark", height=550
    )
    _write_html(fig_pr, OUTPUT_DIR / "all_pr_curves.html")

    print("   Saved: model_benchmarks/model_comparison_chart.html")
    print("   Saved: model_benchmarks/all_pr_curves.html")

    return summary_df


def identify_best_model(all_metrics: dict, rank_metric: str = "roc_auc") -> str:
    """Return the model name with the highest rank_metric score."""
    best_name  = max(all_metrics, key=lambda n: all_metrics[n].get(rank_metric, 0))
    best_score = all_metrics[best_name][rank_metric]
    return best_name, best_score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  HEALTHCARE FRAUD DETECTION — MODEL BENCHMARK")
    print("=" * 55)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Load & evaluate all models
    all_metrics  = {}
    all_probas   = {}

    for display_name, stem in MODEL_REGISTRY.items():
        # Pass training data so stale PKLs can be retrained automatically
        model = load_model(stem, X_train=X_train, y_train=y_train)
        if model is None:
            print(f"\n⚠️  Could not load or retrain: models/{stem}.pkl — skipping.")
            continue

        try:
            metrics, y_proba, y_pred, feat_names = evaluate_model(
                display_name, model, X_train, X_test, y_train, y_test
            )
            save_model_results(display_name, metrics, y_test, y_proba, y_pred, feat_names)
            all_metrics[display_name] = metrics
            all_probas[display_name]  = y_proba
        except Exception as e:
            print(f"   ❌  {display_name} evaluation failed: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        print("\n❌  No models could be evaluated. Check models/ directory.")
        sys.exit(1)

    # 3. Master comparison report
    summary_df = generate_comparison_report(all_metrics, y_test, all_probas)

    # 4. Determine best model
    best_name, best_score = identify_best_model(all_metrics, RANK_METRIC)

    # 5. Save best_model.txt (used by Streamlit app)
    best_model_path = ROOT / "best_model.txt"
    with open(best_model_path, "w") as f:
        f.write(best_name)
    print(f"\n   best_model.txt → '{best_name}'")

    # 6. Final leaderboard
    print("\n" + "=" * 55)
    print("  LEADERBOARD  (ranked by ROC-AUC)")
    print("=" * 55)
    lb = summary_df[["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]].sort_values(
        "roc_auc", ascending=False
    )
    print(lb.round(4).to_string())

    print("\n" + "=" * 55)
    print(f"  🏆  BEST MODEL : {best_name}")
    print(f"  🎯  ROC-AUC   : {best_score:.4f}")
    print("=" * 55)
    print("\n  All results saved to:  model_benchmarks/")
    print("  ✅  Benchmark complete.\n")


if __name__ == "__main__":
    main()
