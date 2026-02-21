"""Chainlit frontend — streams responses from the FastAPI backend and persists
chat threads to a SQLite database via SQLAlchemyDataLayer."""

import json
import os

import httpx
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv

from db import ensure_tables

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_DB_URL = os.getenv("CHAINLIT_DB_URL", "sqlite+aiosqlite:///./chainlit.db")

# Create SQLite tables before the async event loop starts
ensure_tables(_DB_URL)


@cl.data_layer
def get_data_layer():
    """Return the data layer used for chat persistence."""
    return SQLAlchemyDataLayer(conninfo=_DB_URL)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session_id", cl.context.session.id)
    await cl.Message(content="Hello! I'm your data assistant. How can I help?").send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    msg = cl.Message(content="")
    await msg.send()

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BACKEND_URL}/chat/stream",
            json={"message": message.content, "session_id": session_id},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                data = json.loads(raw)
                token = data.get("token", "")
                if token:
                    await msg.stream_token(token)

    await msg.update()
