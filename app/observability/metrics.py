from prometheus_client import Counter, Histogram


predictions_total = Counter(
    "predictions_total",
    "Total number of predictions processed",
    ["prediction_type"],
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["prediction_type"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction inference latency in seconds",
    ["prediction_type"],
)

batch_jobs_submitted_total = Counter(
    "batch_jobs_submitted_total",
    "Total number of batch prediction jobs submitted",
)

batch_jobs_completed_total = Counter(
    "batch_jobs_completed_total",
    "Total number of batch prediction jobs completed",
)

batch_jobs_failed_total = Counter(
    "batch_jobs_failed_total",
    "Total number of batch prediction jobs failed",
)