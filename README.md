# AI integrations demo repository

This repository contains the DAG code used in the AI integrations demo on Cosmic Energy.

The DAGs in this repository showcases the following packages:

- apache-airflow-providers-cohere==1.0.0
- apache-airflow-providers-pinecone==1.0.0
- apache-airflow-providers-weaviate==1.0.0
- apache-airflow-providers-openai==1.0.0

:::info

Upcoming: 

- apache-airflow-providers-pgvector==1.0.0
- apache-airflow-providers-opensearch==1.0.0

:::

## Demo information

This demo contains 2 types of DAGs:

- DAGs showing how to use LLM endpoints: `cohere_example` and `openai_example`. 
- DAGs showing how to use a vector database: `pinecone_example`, `weaviate_example`, (upcoming: `pgvector_example`).

Upcoming:
- A DAG showing how to use the OpenSearch provider: `opensearch_example`.
- A use case DAG showing how to use the Cohere and OpenSearch provider together.