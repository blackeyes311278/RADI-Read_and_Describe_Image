Required APPS : 
==========================================

Python-3.11.9, VSCode, MiniConda, Git

BASH Commands (run in vsCode) :
==========================================

conda create --name radi_env python=3.11.9

conda activate radi_env
 
pip install -r requirements.txt

run model_training.ipynb (manually in vscode)

python manage.py makemigrations

python manage.py migrate

python manage.py runserver

(ctrl+click on http://127.0.0.1:8000/)