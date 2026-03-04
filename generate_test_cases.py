"""
generate_test_cases.py
══════════════════════════════════════════════════════════════
Generates a realistic simulated test dataset based on the actual
stats of the trained model's 30 features (provider-level aggregates).

Produces:
  data/simulated/simulated_test_cases.csv   — 20 labelled test cases
  data/simulated/simulated_results.csv      — model predictions on those cases

Run:
    python generate_test_cases.py
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data" / "processed"
SIM_DIR   = ROOT / "data" / "simulated"
SIM_DIR.mkdir(parents=True, exist_ok=True)

# ── Load real feature statistics from training data ────────────────────────────
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze().astype(int)
FEATURES = list(X_train.columns)

# Compute per-class mean & std so we can sample realistic values
fraud_rows = X_train[y_train == 1]
legit_rows = X_train[y_train == 0]

fraud_mean = fraud_rows.mean()
fraud_std  = fraud_rows.std().clip(lower=0.01)
legit_mean = legit_rows.mean()
legit_std  = legit_rows.std().clip(lower=0.01)

rng = np.random.default_rng(42)

def sample_provider(is_fraud: bool, scenario_name: str) -> dict:
    """Draw one realistic provider feature vector from real training distributions."""
    if is_fraud:
        row = {col: rng.normal(fraud_mean[col], fraud_std[col] * 0.5)
               for col in FEATURES}
        # Make Provider_Fraud_Rate the dominant signal (very high for fraud)
        if "Provider_Fraud_Rate_first" in row:
            row["Provider_Fraud_Rate_first"] = rng.uniform(2.0, 5.0)
        if "Claim_Amount_Anomaly" in row:
            row["Claim_Amount_Anomaly"] = rng.uniform(1.5, 4.0)
        if "Provider_Rejection_Likelihood_first" in row:
            row["Provider_Rejection_Likelihood_first"] = rng.uniform(1.0, 3.0)
    else:
        row = {col: rng.normal(legit_mean[col], legit_std[col] * 0.5)
               for col in FEATURES}
        if "Provider_Fraud_Rate_first" in row:
            row["Provider_Fraud_Rate_first"] = rng.uniform(-2.0, -0.5)
        if "Claim_Amount_Anomaly" in row:
            row["Claim_Amount_Anomaly"] = rng.uniform(-1.5, 0.0)
    row["Provider"] = int(rng.integers(1000, 9999))
    row["_scenario"] = scenario_name
    row["_true_label"] = int(is_fraud)
    return row

# ── Define 20 interpretable test scenarios ────────────────────────────────────
SCENARIOS = [
    # (is_fraud, scenario name)
    (True,  "TC-S01: Phantom billing ($-high-claim, many-patients)"),
    (True,  "TC-S02: Upcoding (inflated procedure codes)"),
    (True,  "TC-S03: Unnecessary hospitalisation"),
    (True,  "TC-S04: Duplicate claim filing"),
    (True,  "TC-S05: Unbundling of services"),
    (True,  "TC-S06: High-value claim with short LOS"),
    (True,  "TC-S07: Multiple inpatient claims same patient-day"),
    (True,  "TC-S08: Billing for non-covered services"),
    (True,  "TC-S09: Ghost patient — deceased beneficiary claim"),
    (True,  "TC-S10: Provider with 87% historical fraud rate"),
    (False, "TC-S11: Normal GP consultation (low amount)"),
    (False, "TC-S12: Routine diabetes management"),
    (False, "TC-S13: Maternity claim — standard delivery"),
    (False, "TC-S14: Single-day surgery — cataract"),
    (False, "TC-S15: Physiotherapy — 10 sessions"),
    (False, "TC-S16: Cancer chemotherapy (legitimate high cost)"),
    (False, "TC-S17: Emergency appendectomy"),
    (False, "TC-S18: Ayushman Bharat standard TB treatment"),
    (False, "TC-S19: Dialysis — renal failure patient"),
    (False, "TC-S20: Post-COVID rehabilitation (low fraud rate provider)"),
]

rows = [sample_provider(is_fraud, name) for is_fraud, name in SCENARIOS]
sim_df = pd.DataFrame(rows)

# Save the feature file (without meta columns)
feature_cols = FEATURES
X_sim = sim_df[feature_cols].astype(float)
X_sim.to_csv(SIM_DIR / "simulated_test_cases.csv", index=False)
print(f"✅  Simulated feature matrix saved → data/simulated/simulated_test_cases.csv  ({X_sim.shape})")

# ── Run model inference ────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
model_path = MODELS_DIR / "lightgbm.pkl"
if not model_path.exists():
    model_path = MODELS_DIR / "random_forest.pkl"
if not model_path.exists():
    print("❌  No trained model found. Run benchmark_models.py first.")
    sys.exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)

model_name = model_path.stem.replace("_", " ").title()
print(f"🤖  Using model: {model_name}")

# Align features
if hasattr(model, "feature_names_in_"):
    X_aligned = X_sim.reindex(columns=model.feature_names_in_, fill_value=0.0)
else:
    X_aligned = X_sim

probas = model.predict_proba(X_aligned)[:, 1]
preds  = (probas >= 0.5).astype(int)

# Build results table
results = []
for i, (row, prob, pred) in enumerate(zip(rows, probas, preds)):
    true_label = row["_true_label"]
    is_correct = (pred == true_label)
    risk_level = "🔴 High" if prob >= 0.6 else ("🟡 Medium" if prob >= 0.3 else "🟢 Low")
    results.append({
        "Test Case":       row["_scenario"],
        "True Label":      "🚨 FRAUD" if true_label else "✅ LEGIT",
        "Predicted":       "🚨 FRAUD" if pred else "✅ LEGIT",
        "Probability (%)": f"{prob*100:.1f}%",
        "Risk Level":      risk_level,
        "Correct?":        "✓ PASS" if is_correct else "✗ FAIL",
    })

results_df = pd.DataFrame(results)
results_df.to_csv(SIM_DIR / "simulated_results.csv", index=False)

# ── Print summary ──────────────────────────────────────────────────────────────
correct = sum(1 for r in results if r["Correct?"].startswith("✓"))
print(f"\n{'='*70}")
print(f"  SIMULATED TEST RESULTS  |  Model: {model_name}")
print(f"{'='*70}")
print(f"  {'Test Case':<52} {'True':^10} {'Pred':^10} {'Prob':^8} {'Result':^8}")
print(f"  {'-'*52} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
for r in results:
    name  = r["Test Case"][:50]
    true  = r["True Label"]
    pred  = r["Predicted"]
    prob  = r["Probability (%)"]
    chk   = r["Correct?"]
    print(f"  {name:<52} {true:^10} {pred:^10} {prob:^8} {chk:^8}")

print(f"\n{'='*70}")
print(f"  ✅  Passed: {correct}/20  |  ❌  Failed: {20-correct}/20")
print(f"  Accuracy on simulated cases: {correct/20*100:.1f}%")
print(f"{'='*70}")
print(f"\n  Results saved → data/simulated/simulated_results.csv")
