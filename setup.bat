@echo off
rem 以下指定路径，找不到的话用环境变量
SET PY=E:\Programs\Python\Python311\python.exe
if not exist %PY% ( SET PY=python ) 
SET PYTHONPATH=
SET curdir=%cd%\venv
echo "start check requirements.txt file"
if exist %cd%\requirements.txt (
   echo "check requirements.txt finish"
) else (
  echo "not exist requirements.txt"
)
echo "start init env"
echo %curdir%
if exist  %curdir% (
 RD /S /q %cd%\venv
 echo "delete old  venv"
 TIMEOUT /T 8
 echo "start create new venv"
 %PY% -m venv ./venv
 TIMEOUT /T 5
 echo "create new venv finish"
) else (
 %PY%  -m venv ./venv
 echo "finish create venv"
)
call %cd%\venv\Scripts\activate.bat
echo "source env finish"
echo "start install packages"
pip install -r requirements.txt
echo "install packages flowing:"
pip list
pause