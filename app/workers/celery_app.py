from celery import Celery

from app.core.config import settings
from celery.signals import worker_ready
from prometheus_client import start_http_server, multiprocess, CollectorRegistry

celery_app = Celery(
    "ml_serving",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
@worker_ready.connect
def start_worker_metrics_server(**kwargs):
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(8001, registry=registry)