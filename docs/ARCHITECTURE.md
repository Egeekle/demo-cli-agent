# Architecture & Design Patterns

This document outlines the key architectural decisions and design patterns used in the CLI Appointment Agent.

## 1. Multi-Agent System with Subagent Delegation

We use a "Supervisor-Worker" pattern to handle complex tasks like conflict resolution.

### Pattern:
- **Supervisor (Main Agent)**: Handles general user interaction, tool routing, and decision making.
- **Subagent (Negotiator)**: A specialized analytical agent that performs specific reasoning tasks without direct access to the database.
- **Delegation**: The Supervisor delegates to the Subagent by calling a specific tool (`resolve_scheduling_conflicts`) which wraps the Subagent's `run()` method.
- **Structured Output**: The Subagent returns a Pydantic model (`ConflictResolution`), allowing the Supervisor to programmatically understand the result and format it for the user.
- **Usage Propagation**: We pass the `usage` context from the Supervisor to the Subagent to ensure accurate token tracking across the entire interaction.

## 2. Dependency Injection with PydanticAI `Deps`

One of the most powerful features of PydanticAI is its dependency injection system via `RunContext`. Instead of tools reaching out to global states or opening their own database sessions, they receive a context object.

### Pattern:
- **`Deps` Class**: A container for shared resources, including the database session and the Negotiator subagent.
- **`RunContext[Deps]`**: Injected into every tool call.
- **Benefits**:
    - **Testability**: Easily inject mock sessions or subagents during tests.
    - **Session Management**: The lifecycle of the session is managed at the entry point (`main.py`), ensuring it's closed correctly.

## 3. Observability via MLflow Autologging

Rather than manually decorating every function, we use the `mlflow.pydantic_ai.autolog()` feature.

### Pattern:
- **Automatic Spans**: Every agent run and tool call is captured as a span in an MLflow trace.
- **SQLite Backend**: Configured to use a local `mlflow.db` to avoid network dependencies and resolve deprecation warnings for file-based storage.
- **Trace Visualization**: A custom `ml-ui` script provides a one-click way to view agent reasoning.

## 4. SQLite Persistence with SQLModel

We use `SQLModel` to bridge the gap between Pydantic models and SQLAlchemy tables.

### Pattern:
- **Single Source of Truth**: The `Appointment` class defines both the database schema and the validation logic.
- **Context Manager**: `get_session()` is wrapped in `@contextmanager` to ensure safe database interactions using the `with` statement.

## 5. Modern Package Management with `uv`

The project leverages `uv` for a unified and extremely fast development experience.

### Pattern:
- **`pyproject.toml`**: Centralized configuration for dependencies, build-system (Hatchling), and scripts.
- **Script Entry Points**: Commands like `appointment-agent` and `ml-ui` are registered as project scripts for easy execution via `uv run`.

## 6. Human-in-the-Loop Safety

Destructive actions are protected by the agent's system prompt rather than hardcoded logic in the tools.

### Pattern:
- **System Prompt Mandate**: The agent is instructed to summarize the proposed `CREATE` or `DELETE` action and wait for explicit user confirmation.
- **Reasoning**: This allows the agent to handle nuances (e.g., "Wait, I meant the park, not the office") before any data is persisted.