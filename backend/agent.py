import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


def build_agent():
    model = init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
    )

    agent = create_agent(
        model=model,
        tools=[],  # no tools yet
        system_prompt=(
            "You are a helpful assistant that interacts with data. "
            "Answer questions clearly and concisely."
        ),
    )

    return agent
