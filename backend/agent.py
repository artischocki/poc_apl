"""LangChain v1 agent factory.

Builds the data-exploration agent with all registered tools and an appropriate
system prompt. Add new tools to the `tools` list as they are implemented.
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools import explore_mdf, read_mdf_channels, ingest_mdf_to_db, query_sensor_data


def build_agent():
    """Instantiate and return the configured LangChain agent."""
    model = init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
    )

    agent = create_agent(
        model=model,
        tools=[explore_mdf, read_mdf_channels, ingest_mdf_to_db, query_sensor_data],
        system_prompt=(
            "You are a sensor data exploration assistant. "
            "You can read MDF measurement files directly or ingest them into TimescaleDB "
            "for persistent SQL-based exploration. "
            "Workflow: use explore_mdf to inspect a file, ingest_mdf_to_db to store it, "
            "then query_sensor_data to analyse it with SQL. "
            "TimescaleDB time-series functions (time_bucket, first, last) are available. "
            "Always report units and highlight notable statistics."
        ),
    )

    return agent
