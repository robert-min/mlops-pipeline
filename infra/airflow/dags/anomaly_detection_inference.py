from __future__ import annotations

from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonVirtualenvOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from tasks.anomaly_detection_inference import inference
from tasks.anomaly_dtection_model_drift import is_model_drift

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

    model_drift_task = ShortCircuitOperator(
        task_id="is_model_drift_task",
        python_callable=is_model_drift,
    )

    train_trigger_task = TriggerDagRunOperator(
        task_id="trian_trigger_task",
        trigger_dag_id="train"
    )

    inference_task >> model_drift_task >> train_trigger_task