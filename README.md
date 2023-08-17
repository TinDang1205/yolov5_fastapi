set -o errexit

pip install --upgrade pip
pip install -r requirements.txt


start server : gunicorn -k uvicorn.workers.UvicornWorker main:app