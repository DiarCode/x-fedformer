@echo off
REM ——————————————————————————————————————————————
REM  X-FedFormer Enhanced Checkpoint Evaluation Script
REM ——————————————————————————————————————————————

REM Adjust the path to your Python executable if needed
set PYTHON=python

REM The analysis script filename
set SCRIPT=analyze_checkpoints.py

REM Checkpoint, data and output directories
set CKPT_DIR=.\checkpoints
set DATA_DIR=.\data_cache
set RESULTS_DIR=.\analysis_results

REM List of cities to evaluate
set CITIES=Almaty Astana Karaganda Shymkent Aktobe Pavlodar Taraz Atyrau Kostanay Aktau

REM Number of days of synthetic data per city
set DAYS=90

%PYTHON% %SCRIPT% ^
  --ckpt-dir %CKPT_DIR% ^
  --data-dir %DATA_DIR% ^
  --results-dir %RESULTS_DIR% ^
  --cities %CITIES% ^
  --days-data %DAYS%

if %ERRORLEVEL% EQU 0 (
  echo.
  echo === Evaluation Completed Successfully! ===
  echo Results written to %RESULTS_DIR%
) else (
  echo.
  echo *** Evaluation Failed (exit code %ERRORLEVEL%) ***
)

pause
