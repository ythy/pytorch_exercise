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
call %cd%\venv\Scripts\activate.bat
echo "start install packages"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
rem pip config set global.index-url https://pypi.org/simple
pip install -r requirements.txt
echo "install packages flowing:"
pip list
pause