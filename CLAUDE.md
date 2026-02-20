# Project conventions for Claude

## Workflow

- After every implemented change, commit and push to origin. If unsure whether a change works, ask the user or test first before committing.

## Documentation

- Write a docstring or module-level comment for every file, function, and class you create.
- Before touching any part of the codebase you are unfamiliar with, read its existing docs (docstrings, comments, or external documentation) first.
- When you modify code, update the corresponding docs in the same change. Docs and code must stay in sync.

## Stack

- **Frontend**: Chainlit — docs at https://docs.chainlit.io
  - Theme: `frontend/public/theme.json` (HSL colors); dark mode is Chainlit's default so no config.toml needed
  - Persistence: `SQLAlchemyDataLayer` via `@cl.data_layer`; DB URL from `CHAINLIT_DB_URL` env var
- **Backend**: FastAPI + Uvicorn
- **Agent**: LangChain v1 (`create_agent`, `init_chat_model`) — docs at https://docs.langchain.com/oss/python/releases/langchain-v1
- **Packaging**: `uv` with `pyproject.toml` (`package = false`)
- **Deployment**: Docker + docker-compose; SQLite persisted via `chainlit_data` named volume
