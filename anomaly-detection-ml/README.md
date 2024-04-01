## Solar Power Anomaly detection Example

- 태양렬 데이터를 활용하여 이상치를 탐지를 통해 모델 성능이 유지되도록 재학습하는 ML 파이프라인 구축
- ML Engineer가 ML 코드를 작업했음을 가정하고 프로젝트 진행(Kaggle에 업로드 된 코드 사용)

<br>

### 참고 링크
- datasets : https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
- kaggle code : https://www.kaggle.com/code/carlos1916/anomaly-detection-lstm-vs-isolation-forest

<br>


### 시나리오
1. 모델이 시간이 지남에 따라 성능이 저하되는 문제 발생
2. 모델 성능이 떨어짐을 inference를 통해 확인후 model drift로 탐지
3. 일정 threshold 이하 떨어지면 모델이 재학습하도록 Airflow ML Pipeline 구축


<br>


### Dags
```python
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
```

<br>


<img width="937" alt="스크린샷 2024-04-01 210502" src="https://github.com/robert-min/dataops-utils/assets/91866763/2bbe85af-c15e-4e8d-b5da-7156352ef0ec">



