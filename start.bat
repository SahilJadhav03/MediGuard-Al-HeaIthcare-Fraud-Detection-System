@echo off
title MediGuard AI — Healthcare Fraud Detection System
color 0A
cls

echo.
echo  =============================================================
echo   MediGuard AI  ^|  Healthcare Fraud Detection System
echo   Powered by LightGBM ^| IRDAI / Ayushman Bharat PM-JAY
echo  =============================================================
echo.

:: ── Check Python ───────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  [OK] Found %%v

:: ── Check Node.js ──────────────────────────────────────────────
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Node.js not found. Install Node 18+ from https://nodejs.org
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo  [OK] Found Node.js %%v

:: ── Check npm ──────────────────────────────────────────────────
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] npm not found. Reinstall Node.js from https://nodejs.org
    pause & exit /b 1
)
echo.

:: ── Step 1: Install Python dependencies ────────────────────────
echo  [1/4] Installing Python requirements...
echo  ---------------------------------------------------------------
pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo  [ERROR] pip install failed. Check requirements.txt and try again.
    pause & exit /b 1
)
echo  [OK] Python packages installed.
echo.

:: ── Step 2: Install Node.js frontend dependencies ──────────────
echo  [2/4] Installing frontend (Node.js) packages...
echo  ---------------------------------------------------------------
cd frontend
npm install --silent
if %errorlevel% neq 0 (
    echo  [ERROR] npm install failed in frontend/.
    cd ..
    pause & exit /b 1
)
cd ..
echo  [OK] Frontend packages installed.
echo.

:: ── Step 3: Check if models exist, run pipeline if not ─────────
echo  [3/4] Checking ML models...
echo  ---------------------------------------------------------------
if not exist "models\lightgbm.pkl" (
    echo  [INFO] No trained models found. Running pipeline + training...
    echo  [INFO] This may take 2-5 minutes on first run.
    python run_pipeline.py
    python benchmark_models.py
    if %errorlevel% neq 0 (
        echo  [WARN] Model training had issues. API will still start.
    )
) else (
    echo  [OK] Pre-trained models found ^(lightgbm.pkl, xgboost.pkl, etc.^)
)
echo.

:: ── Step 4: Start both servers in separate windows ─────────────
echo  [4/4] Starting servers...
echo  ---------------------------------------------------------------
echo  [INFO] Opening API server on   http://localhost:8000
echo  [INFO] Opening Frontend UI on  http://localhost:5177
echo.

:: Start FastAPI backend in a new terminal window
start "MediGuard API (Backend :8000)" cmd /k "color 0B && echo. && echo  MediGuard AI — FastAPI Backend && echo  API running at: http://localhost:8000 && echo  API docs at:   http://localhost:8000/docs && echo  Press Ctrl+C to stop. && echo. && python api_server.py"

:: Wait 3 seconds for the API to start before launching frontend
timeout /t 3 /nobreak >nul

:: Start Vite frontend in a new terminal window
start "MediGuard UI (Frontend :5177)" cmd /k "color 0D && echo. && echo  MediGuard AI — React Frontend && echo  UI running at: http://localhost:5177 && echo  Press Ctrl+C to stop. && echo. && cd frontend && npm run dev"

:: Wait 4 seconds for Vite to start, then open browser
timeout /t 4 /nobreak >nul
echo  [OK] Opening browser at http://localhost:5177
start "" http://localhost:5177

echo.
echo  =============================================================
echo   All systems are running!
echo.
echo   Frontend  →  http://localhost:5177
echo   Backend   →  http://localhost:8000
echo   API Docs  →  http://localhost:8000/docs
echo.
echo   Two terminal windows have opened (API + Frontend).
echo   Close them to stop the servers.
echo  =============================================================
echo.
pause
