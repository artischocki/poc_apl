"""FastAPI backend — exposes the LangChain agent over HTTP with SSE streaming."""

import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import build_agent
from timescaledb import ensure_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run schema initialisation before accepting requests."""
    ensure_schema()
    yield


app = FastAPI(title="Data Agent API", lifespan=lifespan)

_plots_dir = Path(__file__).parent / "plots"
_plots_dir.mkdir(exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(_plots_dir)), name="plots")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

# Per-session in-memory message history: list of {"role": ..., "content": ...} dicts.
# Keyed by session_id, which comes from the Chainlit session.
_histories: dict[str, list[dict]] = defaultdict(list)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    history = _histories[request.session_id]
    history.append({"role": "user", "content": request.message})
    result = await agent.ainvoke({"messages": list(history)})
    last_message = result["messages"][-1]
    history.append({"role": "assistant", "content": last_message.content})
    return {"response": last_message.content}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    history = _histories[request.session_id]
    history.append({"role": "user", "content": request.message})

    async def generate():
        assistant_tokens: list[str] = []

        async for event in agent.astream_events(
            {"messages": list(history)},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if not content:
                    continue
                # content can be a string or a list of content blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            token = block["text"]
                            assistant_tokens.append(token)
                            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                else:
                    assistant_tokens.append(content)
                    yield f"data: {json.dumps({'type': 'token', 'token': content})}\n\n"
            elif event["event"] == "on_tool_start":
                tool_name = event.get("name", "unknown")
                run_id = event.get("run_id", "")
                tool_input = event["data"].get("input", "")
                try:
                    input_payload = (
                        json.dumps(tool_input) if not isinstance(tool_input, str) else tool_input
                    )
                except (TypeError, ValueError):
                    input_payload = str(tool_input)
                yield f"data: {json.dumps({'type': 'tool_start', 'run_id': run_id, 'name': tool_name, 'input': input_payload})}\n\n"
            elif event["event"] == "on_tool_end":
                run_id = event.get("run_id", "")
                tool_output = event["data"].get("output", "")
                output_str = (
                    str(tool_output.content)
                    if hasattr(tool_output, "content")
                    else str(tool_output)
                )
                if output_str.startswith("PLOT:"):
                    uid = output_str[5:]
                    yield f"data: {json.dumps({'type': 'plotly', 'run_id': run_id, 'path': f'/plots/{uid}.json'})}\n\n"
                    output_str = "Chart generated."
                yield f"data: {json.dumps({'type': 'tool_end', 'run_id': run_id, 'output': output_str})}\n\n"
        history.append({"role": "assistant", "content": "".join(assistant_tokens)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
