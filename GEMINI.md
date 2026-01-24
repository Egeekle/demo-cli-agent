# Project Context: demo-cli-agent

## Overview
This project is a high-quality demo of a CLI-based personal AI assistant specializing in appointment management. It showcases a modern Python stack with a focus on observability, safety, and idiomatic use of the PydanticAI framework.

## Architecture & Technology
*   **Language:** Python 3.12+
*   **Package Manager:** [uv](https://github.com/astral-sh/uv) (configured with `pyproject.toml` and Hatchling).
*   **AI Agent:** [PydanticAI](https://pydantic.dev/pydantic-ai) with `OpenAIResponsesModel` (gpt-4.1-mini).
*   **Database & ORM:** SQLite with [SQLModel](https://sqlmodel.tiangolo.com/) for unified Pydantic/SQLAlchemy schemas.
*   **Observability:** [MLflow](https://mlflow.org/) with `mlflow.pydantic_ai.autolog()` for automatic agent tracing.
*   **CLI Framework:** [Typer](https://typer.tiangolo.com/) and [Rich](https://github.com/Textualize/rich) for a polished terminal UI.
*   **Configuration:** [PydanticSettings](https://docs.pydantic.dev/latest/usage/pydantic_settings/) for type-safe `.env` loading.

## Key Design Patterns
*   **Dependency Injection:** Agent tools utilize `RunContext[Deps]` to access the database session, enabling easy testing and clean resource management.
*   **Human-in-the-Loop Safety:** Destructive actions (`CREATE`, `DELETE`) require explicit user confirmation via the agent's system prompt.
*   **Unified Tracing:** Uses `mlflow.db` (SQLite) as the backend to capture detailed spans of agent reasoning and tool execution.

## Directory Structure
*   `src/main.py`: CLI entry point and interactive REPL.
*   `src/agent.py`: Agent configuration, system prompts, and tool registration.
*   `src/tools.py`: Business logic and database tools with dependency injection.
*   `src/models.py`: SQLModel definitions for Appointments.
*   `src/database.py`: Session management and table creation.
*   `src/mlflow_server.py`: Utility to launch the MLflow tracking UI.
*   `docs/`: Detailed `ARCHITECTURE.md` and `DEVELOPMENT.md`.
*   `tests/`: Comprehensive test suite for all modules.

## Development Commands
*   `uv run appointment-agent`: Start the assistant.
*   `uv run ml-ui`: Start the MLflow UI.
*   `uv run pytest`: Run all tests.
*   `uv sync`: Synchronize dependencies.