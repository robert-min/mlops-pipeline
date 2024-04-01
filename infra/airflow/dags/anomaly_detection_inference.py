from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonVirtualenvOperator
from tasks.anomaly_detection_inference import inference

with DAG(
    "anomaly_detection_inference",
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="anomaly detection example inference",
    schedule=timedelta(days=30),
    start_date=datetime(2021, 1, 1),
    catchup=False
) as dag:
    inference_task = PythonVirtualenvOperator(
        task_id="inference_task",
        requirements=["plotly", "scikit-learn", "tensorflow", "numpy", "pandas", "kaleido"],
        python_callable=inference
    )

    inference_task