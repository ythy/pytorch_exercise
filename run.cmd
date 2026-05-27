@echo off
cls
set base=%~dp0
call %base%\venv\Scripts\activate.bat
 
rem python ./src/fifa/main.py --mode  predict train 
rem python ./src/chat/main.py  --mode pretrain  valid_pretrain tokenizer

rem python ./src/nlp/main.py --mode pretrain 
 
 
pause
