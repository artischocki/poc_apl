"""FastAPI backend — exposes the LangChain agent over HTTP with SSE streaming."""

import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import build_agent
from timescaledb import ensure_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run schema initialisation before accepting requests."""
    ensure_schema()
    yield


app = FastAPI(title="Data Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]}
    )
    last_message = result["messages"][-1]
    return {"response": last_message.content}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": request.message}]},
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
                            yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    yield f"data: {json.dumps({'token': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
