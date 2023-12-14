# AI integrations demo repository

This repository contains the DAG code used in the AI integrations webinar.

The DAGs in this repository showcases the following packages:

- [apache-airflow-providers-cohere](https://registry.astronomer.io/providers/apache-airflow-providers-cohere/versions/latest)
- [apache-airflow-providers-pinecone](https://registry.astronomer.io/providers/apache-airflow-providers-pinecone/versions/latest)
- [apache-airflow-providers-weaviate](https://registry.astronomer.io/providers/apache-airflow-providers-weaviate/versions/latest)
- [apache-airflow-providers-openai](https://registry.astronomer.io/providers/apache-airflow-providers-openai/versions/latest)
- [apache-airflow-providers-pgvector](https://registry.astronomer.io/providers/apache-airflow-providers-pgvector/versions/latest)
- [apache-airflow-providers-opensearch](https://registry.astronomer.io/providers/apache-airflow-providers-opensearch/versions/latest)

## Demo information

This demo contains the following DAGs:

- DAGs showing how to use LLM endpoints: `cohere_example` and `openai_example`. 
- DAGs showing how to use a vector database: `pinecone_example`, `weaviate_example`, (upcoming: `pgvector_example`).
- A DAG showing how to use the OpenSearch provider: `opensearch_example`.

You can find a full description of each of these DAGs in the relevant [Learn integration tutorial](https://docs.astronomer.io/learn/category/integrations).

Additionally, this repository contains a use case DAG `analyze_customer_feedback` showing how to use the Cohere and OpenSearch provider together, you can find a detailed code description in the [Use Cohere and OpenSearch to analyze customer feedback in an MLOps pipeline](https://docs.astronomer.io/learn/use-case-llm-customer-feedback) use case.