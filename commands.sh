celery -A app.celery worker --pool=gevent --loglevel=info
python app.py