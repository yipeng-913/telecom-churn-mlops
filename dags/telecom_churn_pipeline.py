"""Airflow 3 TaskFlow DAG; pandas/ML imports occur only inside task execution."""
from datetime import timedelta
import os
import pendulum
from airflow.sdk import dag, task, get_current_context

@dag(dag_id='telecom_churn_pipeline', start_date=pendulum.datetime(2026,1,1,tz='UTC'),
     schedule='0 3 * * 1', catchup=False, max_active_runs=1,
     default_args={'owner':'portfolio','retries':1,'retry_delay':timedelta(minutes=2)},
     tags=['portfolio','churn','ml'], doc_md='Prepare → train and pickle → reload and evaluate hold-out. Results are run-scoped.')
def churn_pipeline():
    @task(execution_timeout=timedelta(minutes=15))
    def prepare_data():
        from telecom_churn.pipeline import prepare
        return prepare(os.getenv('CHURN_PROJECT_ROOT','/opt/airflow'), get_current_context()['run_id'])

    @task(execution_timeout=timedelta(minutes=60))
    def train_model(prepared):
        from telecom_churn.pipeline import train
        return train(os.getenv('CHURN_PROJECT_ROOT','/opt/airflow'), prepared)

    @task(execution_timeout=timedelta(minutes=30))
    def evaluate_model(prepared, model):
        from telecom_churn.pipeline import evaluate
        return evaluate(os.getenv('CHURN_PROJECT_ROOT','/opt/airflow'),prepared,model)

    prepared=prepare_data()
    model=train_model(prepared)
    evaluate_model(prepared,model)

churn_pipeline()
