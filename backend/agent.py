"""LangChain v1 agent factory.

Builds the data-exploration agent with all registered tools and an appropriate
system prompt. Add new tools to the `tools` list as they are implemented.
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools import explore_mdf, read_mdf_channels


def build_agent():
    """Instantiate and return the configured LangChain agent."""
    model = init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
    )

    agent = create_agent(
        model=model,
        tools=[explore_mdf, read_mdf_channels],
        system_prompt=(
            "You are a data exploration assistant. "
            "You have access to tools for reading MDF measurement files. "
            "When the user provides a file path, use explore_mdf first to understand "
            "the file structure, then read_mdf_channels to retrieve specific signals. "
            "Always report units and highlight notable statistics."
        ),
    )

    return agent
