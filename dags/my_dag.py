from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.providers.openai.operators.openai import (
    OpenAIEmbeddingOperator
)
from pendulum import datetime
import requests


@dag(
    start_date=datetime(2023, 12, 1),
    schedule="@daily",
    catchup=False,
)
def my_dag():
    @task
    def get_cat_fact():
        r = requests.get("https://catfact.ninja/fact")
        return r.json()["fact"]

    get_cat_fact_obj = get_cat_fact()

    embed_cat_fact = OpenAIEmbeddingOperator(
        task_id="embed_cat_fact",
        conn_id="openai_ai_integrations_demo",
        model="text-embedding-ada-002",
        input_text=get_cat_fact_obj,
    )

    chain(get_cat_fact_obj, embed_cat_fact)


my_dag()
