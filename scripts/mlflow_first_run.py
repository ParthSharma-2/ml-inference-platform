import mlflow

mlflow.set_experiment("mlflow-learning")

with mlflow.start_run():
    mlflow.log_param("algorithm", "GradientBoostingClassifier")
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("roc_auc", 0.8415)

    print("Run ID:", mlflow.active_run().info.run_id)