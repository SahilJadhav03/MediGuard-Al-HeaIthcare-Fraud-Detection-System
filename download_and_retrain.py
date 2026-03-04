"""
download_and_retrain.py
───────────────────────────────────────────────────────────────
Full pipeline:
  1. Install kagglehub if missing
  2. Download the Rohit Anand CMS Healthcare Fraud dataset
     (rohitrox/healthcare-provider-fraud-detection-analysis)
     which contains ~558 K+ provider claims with fraud labels
  3. Copy the 4 required CSVs into data/raw/
     (backs up any existing files first)
  4. Re-run preprocessing  →  data/processed/
  5. Re-run benchmark_models.py  →  models/  +  model_benchmarks/

Run:
    python download_and_retrain.py
"""

import subprocess, sys, os, shutil, time
from pathlib import Path

ROOT = Path(__file__).parent
RAW  = ROOT / "data" / "raw"

# ─── Step 0: ensure kagglehub is installed ───────────────────
def ensure_pkg(pkg):
    try:
        __import__(pkg.split("[")[0])
    except ImportError:
        print(f"📦 Installing {pkg}…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_pkg("kagglehub")

import kagglehub

# ─── Kaggle dataset info ──────────────────────────────────────
DATASET_HANDLE = "rohitrox/healthcare-provider-fraud-detection-analysis"

# The Rohit Anand dataset contains these files at root level:
#   Train-1542865627584.csv   → provider labels (fraud / not)
#   Train_Beneficiary*.csv    → beneficiary demographics
#   Train_Inpatient*.csv      → inpatient claims
#   Train_Outpatient*.csv     → outpatient claims
# We rename them to match config.yaml expectations.

FILE_MAP = {
    # (partial filename to search for)  : target name in data/raw/
    "Train-1542865627584.csv":          "Train.csv",
    "Train_Beneficiary":                "Beneficiary.csv",
    "Train_Inpatient":                  "Inpatient.csv",
    "Train_Outpatient":                 "Outpatient.csv",
}

# ─── Step 1: Download via kagglehub ──────────────────────────
print("\n" + "="*60)
print("STEP 1 — Downloading Kaggle dataset")
print("="*60)
print(f"Dataset : {DATASET_HANDLE}")
print("This may take a few minutes (~100 MB)…\n")

try:
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"\n✅ Downloaded to: {dataset_path}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nTROUBLESHOOTING:")
    print("  1. Make sure you have a Kaggle account and API token.")
    print("  2. Place your kaggle.json at C:\\Users\\<you>\\.kaggle\\kaggle.json")
    print("  3. Or run:  pip install kaggle  then  kaggle datasets download rohitrox/healthcare-provider-fraud-detection-analysis")
    sys.exit(1)

# ─── Step 2: Locate and copy files ───────────────────────────
print("\n" + "="*60)
print("STEP 2 — Copying files into data/raw/")
print("="*60)

RAW.mkdir(parents=True, exist_ok=True)

# Find all CSVs recursively inside the downloaded path
all_csvs = list(dataset_path.rglob("*.csv"))
print(f"Found {len(all_csvs)} CSV file(s) in download cache:")
for f in all_csvs:
    print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

copied = {}
for partial, target_name in FILE_MAP.items():
    matched = [f for f in all_csvs if partial.lower() in f.name.lower()]
    if not matched:
        print(f"⚠️  Could not find file matching '{partial}' — skipping")
        continue
    src = matched[0]
    dst = RAW / target_name
    # Backup old file if exists
    if dst.exists():
        bak = dst.with_suffix(f".bak_{int(time.time())}.csv")
        dst.rename(bak)
        print(f"  📦 Backed up old {target_name} → {bak.name}")
    shutil.copy2(src, dst)
    copied[target_name] = src
    print(f"  ✅ {src.name}  →  data/raw/{target_name}  ({dst.stat().st_size / 1e6:.1f} MB)")

if len(copied) < 4:
    print(f"\n⚠️  Only {len(copied)}/4 files copied. The pipeline may fail.")
    print("     Check the downloaded files above and copy manually if needed.")
else:
    print(f"\n✅ All 4 required files ready in data/raw/")

# ─── Step 3: Run preprocessing ───────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Running data preprocessing pipeline")
print("="*60)
os.chdir(ROOT)

preprocess_result = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0,'src'); "
     "from data_preprocessing import DataPreprocessor; "
     "p = DataPreprocessor(); p.run_pipeline()"],
    cwd=ROOT
)

if preprocess_result.returncode != 0:
    print("❌ Preprocessing failed. Check the error above.")
    print("   You can also run manually:  python run_pipeline.py")
    sys.exit(1)

print("\n✅ Preprocessing complete — data/processed/ is ready")

# ─── Step 4: Run benchmark_models.py ─────────────────────────
print("\n" + "="*60)
print("STEP 4 — Training & benchmarking models")
print("="*60)
print("Training 4 models on the new larger dataset…")
print("(This may take 5–20 minutes depending on dataset size)\n")

bench_result = subprocess.run(
    [sys.executable, "benchmark_models.py"],
    cwd=ROOT
)

if bench_result.returncode != 0:
    print("❌ Benchmark training failed. Check the error above.")
    sys.exit(1)

# ─── Done ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("🎉 COMPLETE! Your models are now trained on the full dataset.")
print("="*60)
print("\nWhat changed:")
print("  • data/raw/         — new larger raw CSVs from Kaggle")
print("  • data/processed/   — new X_train, X_test, y_train, y_test")
print("  • models/           — newly trained .pkl model files")
print("  • model_benchmarks/ — updated benchmark charts & metrics")
print("  • model_results.json / best_model.txt — updated best model")
print("\nRestart api_server.py to load the new models:")
print("  python api_server.py")
print("\nThe dashboard will automatically reflect the new data.")
