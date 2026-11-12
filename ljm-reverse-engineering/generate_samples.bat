@echo off
chcp 65001 > nul
cd /d "%~dp0"
".venv\Scripts\python.exe" samples\generate_samples.py
".venv\Scripts\python.exe" samples\evaluate_accuracy.py
pause
