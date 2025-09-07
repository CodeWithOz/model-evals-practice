from opentelemetry.trace import StatusCode
import pandas as pd
import duckdb
import logging

from utils import get_model, get_tracer, invoke_model

logger = logging.getLogger(__name__)

tracer = get_tracer(__name__)

MODEL = "gpt-4o-mini"
TRANSACTION_DATA_FILE_PATH = "data/Store_Sales_Price_Elasticity_Promotions_Data.parquet"

SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""


def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, columns=columns, table_name=table_name
    )

    logger.info("Requesting sql query from model")
    response = invoke_model(get_model(), formatted_prompt)
    logger.info("Received sql query from model")

    return response.choices[0].message.content


@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    try:
        table_name = "sales"

        # initialize SQL table
        logger.info("Creating db from parquet file")
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        logger.info("Created db from parquet file")

        # get the SQL query to execute
        sql_query = generate_sql_query(prompt, df.columns.tolist(), table_name)

        # remove ``` from the query
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        with tracer.start_as_current_span(
            "execute_sql_query", openinference_span_kind="chain"
        ) as span:
            span.set_input(sql_query)
            # execute the chatbot's query
            logger.info("Executing query from model")
            result = duckdb.sql(sql_query)
            result_df = result.df()
            span.set_output(value=str(result_df))
            span.set_status(StatusCode.OK)
            result.show()
            logger.info("Executed query from model")

        return result.df().to_string()
    except Exception as e:
        logger.error(f"Error accessing data: {str(e)}")
        return f"Error accessing data: {str(e)}"
