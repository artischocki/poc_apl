"""Chainlit frontend — streams responses from the FastAPI backend and persists
chat threads to a SQLite database via SQLAlchemyDataLayer."""

import asyncio
import json
import os
from typing import Optional

import aiohttp
import chainlit as cl
import plotly.io as pio
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


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Validate credentials from CHAINLIT_USERNAME / CHAINLIT_PASSWORD env vars."""
    expected_user = os.getenv("CHAINLIT_USERNAME", "admin")
    expected_pass = os.getenv("CHAINLIT_PASSWORD", "admin")
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username)
    return None


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("session_id", cl.context.session.id)


def _cancel_message_task() -> None:
    """Cancel the current on_message task if one is running."""
    task = cl.user_session.get("message_task")
    if task and not task.done():
        task.cancel()


@cl.on_stop
async def on_stop():
    """Cancel the running task when the user clicks the stop button."""
    _cancel_message_task()


@cl.on_chat_end
async def on_chat_end():
    """Cancel the running task when the user closes the tab or the session ends."""
    _cancel_message_task()


@cl.on_message
async def on_message(message: cl.Message):
    # Store the current asyncio task so on_stop can cancel it
    cl.user_session.set("message_task", asyncio.current_task())
    session_id = cl.user_session.get("session_id")

    msg = cl.Message(content="")
    await msg.send()

    # Tracks open cl.Step objects by run_id so tool_end can close them.
    active_steps: dict[str, cl.Step] = {}
    # Plotly figures to attach to the message once streaming completes.
    plot_elements: list[cl.Plotly] = []

    # aiohttp is used instead of httpx because httpx uses anyio internally,
    # which causes cancel-scope task-affinity errors when Chainlit cleans up.
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{BACKEND_URL}/chat/stream",
            json={"message": message.content, "session_id": session_id},
        ) as response:
            response.raise_for_status()
            # readline() gives clean SSE lines without chunk-boundary issues
            while True:
                line_bytes = await response.content.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode().strip()
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                data = json.loads(raw)
                event_type = data.get("type")

                if event_type == "token":
                    await msg.stream_token(data.get("token", ""))
                elif event_type == "tool_start":
                    run_id = data["run_id"]
                    step = cl.Step(name=data["name"], type="tool", parent_id=msg.id)
                    step.input = data.get("input", "")
                    await step.send()
                    active_steps[run_id] = step
                elif event_type == "tool_end":
                    run_id = data.get("run_id", "")
                    step = active_steps.pop(run_id, None)
                    if step:
                        output = data.get("output", "")
                        # Truncate very long outputs to keep the UI readable
                        step.output = output[:3000] if len(output) > 3000 else output
                        await step.update()
                elif event_type == "plotly":
                    plot_url = f"{BACKEND_URL}{data['path']}"
                    async with session.get(plot_url) as plot_resp:
                        figure_json = await plot_resp.text()
                    fig = pio.from_json(figure_json)
                    plot_elements.append(cl.Plotly(figure=fig, display="inline"))
                elif "token" in data:
                    # Backward-compatible: handle events without the type field
                    await msg.stream_token(data["token"])

    if plot_elements:
        msg.elements = plot_elements
    await msg.update()
