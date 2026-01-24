# Architecture & Design Patterns

This document outlines the key architectural decisions and design patterns used in the CLI Appointment Agent.

## 1. Dependency Injection with PydanticAI `Deps`

One of the most powerful features of PydanticAI is its dependency injection system via `RunContext`. Instead of tools reaching out to global states or opening their own database sessions, they receive a context object.

### Pattern:
- **`Deps` Class**: A simple container for shared resources (e.g., the database session).
- **`RunContext[Deps]`**: Injected into every tool call.
- **Benefits**:
    - **Testability**: Easily inject mock sessions or in-memory databases during tests.
    - **Session Management**: The lifecycle of the session is managed at the entry point (`main.py`), ensuring it's closed correctly.

## 2. Observability via MLflow Autologging

Rather than manually decorating every function, we use the `mlflow.pydantic_ai.autolog()` feature.

### Pattern:
- **Automatic Spans**: Every agent run and tool call is captured as a span in an MLflow trace.
- **SQLite Backend**: Configured to use a local `mlflow.db` to avoid network dependencies and resolve deprecation warnings for file-based storage.
- **Trace Visualization**: A custom `ml-ui` script provides a one-click way to view agent reasoning.

## 3. SQLite Persistence with SQLModel

We use `SQLModel` to bridge the gap between Pydantic models and SQLAlchemy tables.

### Pattern:
- **Single Source of Truth**: The `Appointment` class defines both the database schema and the validation logic.
- **Context Manager**: `get_session()` is wrapped in `@contextmanager` to ensure safe database interactions using the `with` statement.

## 4. Modern Package Management with `uv`

The project leverages `uv` for a unified and extremely fast development experience.

### Pattern:
- **`pyproject.toml`**: Centralized configuration for dependencies, build-system (Hatchling), and scripts.
- **Script Entry Points**: Commands like `appointment-agent` and `ml-ui` are registered as project scripts for easy execution via `uv run`.

## 5. Human-in-the-Loop Safety

Destructive actions are protected by the agent's system prompt rather than hardcoded logic in the tools.

### Pattern:
- **System Prompt Mandate**: The agent is instructed to summarize the proposed `CREATE` or `DELETE` action and wait for explicit user confirmation.
- **Reasoning**: This allows the agent to handle nuances (e.g., "Wait, I meant the park, not the office") before any data is persisted.
