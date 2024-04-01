from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonVirtualenvOperator
from tasks.anomaly_detection_train import train

with DAG(
    "anomaly_detection_train",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="anomaly detection example train",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False,
) as dag:
    train_task = PythonVirtualenvOperator(
        task_id="train_task",
        requirements=["plotly", "scikit-learn", "tensorflow", "numpy", "pandas", "kaleido"],
        python_callable=train
    )

    train_task
