@echo off
REM -------------------------------------------------------------------
REM 1) Define your 10-city list:
set CITIES=Almaty Astana Karaganda Shymkent Aktobe Pavlodar Taraz Atyrau Kostanay Aktau

REM 2) Generate data (30 days, 5 routes per city)
python xfedformer.py generate-data --cities %CITIES% --days 90 --routes-per-city 30

REM 3) Start the federated server (50 rounds) in a new window
start "XFedFormer Server" cmd /k ^
  "python xfedformer.py server --rounds 50 & pause"

REM Give the server a moment to spin up
timeout /t 5 >nul

REM 4) Launch one client per city in its own window
for %%C in (%CITIES%) do (
  start "Client %%C" cmd /k ^
    "python xfedformer.py client --city %%C --days-data 90 --server_address 127.0.0.1:8080 & pause"
)

REM 5) Wait enough time for training to complete (adjust if needed)
timeout /t 600 >nul

REM 6) Evaluate the final global model
python xfedformer.py evaluate --cities %CITIES% --days-data 90 ^
  --model-path checkpoints/global_model.pt ^
  --report-file results/final_report.json

echo.
echo ==== All done! See results/final_report.json ====
pause
