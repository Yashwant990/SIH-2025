from celery import Celery
import os

CELERY_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery = Celery("career_app", broker=CELERY_BROKER, backend=CELERY_BACKEND)
from celery import Celery

celery = Celery('career_app', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

# Import the tasks so Celery knows about them
import tasks  # <-- make sure this imports the module containing parse_resume_task
