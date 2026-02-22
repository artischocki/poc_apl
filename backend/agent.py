"""LangChain v1 agent factory.

Builds the data-exploration agent with all registered tools and an appropriate
system prompt. Add new tools to the `tools` list as they are implemented.
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools import ingest_mdf_to_db, query_sensor_data, plot_timeseries, plot_barchart


def build_agent():
    """Instantiate and return the configured LangChain agent."""
    model = init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
    )

    agent = create_agent(
        model=model,
        tools=[ingest_mdf_to_db, query_sensor_data, plot_timeseries, plot_barchart],
        system_prompt=(
            "You are a sensor data exploration assistant backed by a TimescaleDB database. "
            "The measurements table has columns: "
            "time (TIMESTAMPTZ), file_name (TEXT), channel (TEXT), value (DOUBLE PRECISION), unit (TEXT). "
            "Use query_sensor_data to answer questions with SQL. "
            "TimescaleDB functions like time_bucket(), first(), last() are available. "
            "Always include units in your answers and highlight notable patterns or anomalies. "
            "When the user asks for a chart or plot, use plot_timeseries for time-based data "
            "and plot_barchart for comparisons or aggregates. "
            "Downsample with time_bucket() before plotting to keep charts readable."
        ),
    )

    return agent
