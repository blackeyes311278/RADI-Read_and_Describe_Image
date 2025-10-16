Required APPS : 
==========================================

Python-3.11.9, VSCode, MiniConda, Git

BASH Commands (run in vsCode) :
==========================================

for Conda Environment:

    conda create --name radi_env python=3.11.9

    conda activate radi_env

for Virtual Environment (Recommended):

    conda deactivate

    python -m venv ..radi_image_venv

    source ..radi_image_venv/Scripts/activate
 
pip install -r requirements.txt

run model.ipynb (manually in vscode)

run training.ipynb (manually in vscode)

python manage.py makemigrations

python manage.py migrate

python manage.py runserver

(ctrl+click on http://127.0.0.1:8000/)



Clear Separation:

Cell 1: Installation commands

Cell 2: All imports together

Cells 3-6: Base pipeline (always run)

Cells 7-13: Optional training pipeline

Logical Flow:

Base models → Pipeline functions → Testing

Optional: Download → Train → Save → Load trained model