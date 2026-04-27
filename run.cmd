@echo off
cls
set base=%~dp0
call %base%\venv\Scripts\activate.bat
 
python ./src/exercise_fifa.py

pause
