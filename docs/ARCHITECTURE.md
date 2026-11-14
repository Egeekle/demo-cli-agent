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

## 4. Async SQLite Persistence with SQLModel

We use `SQLModel` to bridge the gap between Pydantic models and SQLAlchemy tables, backed by a fully async engine.

### Pattern:
- **Single Source of Truth**: The `Appointment` class defines both the database schema and the validation logic.
- **Async Engine**: `create_async_engine` targets `sqlite+aiosqlite:///database.db`, so all reads/writes go through `await`.
- **`AsyncSession`**: Sourced from `sqlmodel.ext.asyncio.session`, not the sync `Session` — every tool in `src/tools.py` awaits `session.exec(...)`, `session.get(...)`, `session.commit()`, etc.
- **`async_sessionmaker(expire_on_commit=False)`**: Objects returned from a tool (e.g. a freshly created `Appointment`) stay usable after `commit()` without triggering a lazy (sync) reload, which would otherwise break under async.
- **Session-per-Turn**: `get_session()` is an `@asynccontextmanager` opened once per user turn in `src/main.py::_run_turn`; the same session (and `Deps`) is reused across the initial run and any approval round-trip within that turn, then closed.

## 5. Modern Package Management with `uv`

The project leverages `uv` for a unified and extremely fast development experience.

### Pattern:
- **`pyproject.toml`**: Centralized configuration for dependencies, build-system (Hatchling), and scripts.
- **Script Entry Points**: Commands like `appointment-agent` and `ml-ui` are registered as project scripts for easy execution via `uv run`.

## 6. Human-in-the-Loop Safety

Destructive actions are protected structurally, at the tool-registration level, not by asking the model nicely.

### Pattern:
- **`requires_approval=True`**: Every data-modifying tool (`create_appointment`, `update_appointment`, `delete_appointment`) is registered on the agent with `requires_approval=True` in `src/agent.py`. Read-only tools (`list_appointments`, `search_appointments`, `get_appointment`, `check_availability`, `resolve_scheduling_conflicts`) are not.
- **`DeferredToolRequests`**: When the model calls an approval-gated tool, PydanticAI halts execution and the agent's `output_type=[str, DeferredToolRequests]` surfaces the pending request instead of a tool result — the tool body never runs until approved.
- **`DeferredToolResults`**: `src/main.py::_run_turn` prompts the user, then calls `src/approvals.py::build_results` to construct a `ToolApproved()`/`ToolDenied(...)` decision per pending call and resumes the run via `deferred_tool_results=...`.
- **System Prompt Role**: The instructions in `src/agent.py` still tell the model to describe the proposed action before calling a mutating tool — but this is UX framing (a clear, human-readable summary), not the enforcement mechanism. Approval is enforced structurally even if the model forgets, misbehaves, or is adversarially prompted.

## 7. Conflict Engine: No Double-Booking, Enforced in Code

Scheduling conflicts are prevented by a deterministic check in the tool layer, not by trusting the model to remember to look before it books.

### Pattern:
- **Overlap Predicate**: `check_availability` (`src/tools.py`) selects existing appointments where `start_time < end AND end_time > start` — the standard half-open-interval overlap test — optionally excluding the appointment being rescheduled (`exclude_id`).
- **`ModelRetry` Backstop**: `create_appointment` and `update_appointment` both call `check_availability` themselves before writing, and raise `ModelRetry` with the names of the conflicting appointments if any overlap is found. This means no double-booking can reach the database even if the model skips the "check first" instruction — the backstop lives in the tool, not the prompt.
- **Structured Handoff to the Negotiator**: `resolve_scheduling_conflicts` fetches the real overlapping appointments from the database and passes them to the negotiator subagent as structured facts (title + start/end + location), not a model-generated paraphrase of what might be booked. The negotiator reasons over ground truth, not hearsay.

## 8. Provider Module: One Shared Model, One `autolog()` Call

`src/providers.py` is the single place that constructs model/provider objects and enables observability, so both agents in the system stay consistent and MLflow isn't instrumented twice.

### Pattern:
- **Single `autolog()` Call**: `mlflow.pydantic_ai.autolog()` is invoked exactly once, at import time in `src/providers.py`, instead of being scattered across `agent.py`/`negotiator.py`/`main.py`.
- **Shared Model Construction**: `build_model()` binds an `OpenAIResponsesModel` to one shared `OpenAIProvider`; both `appointment_agent` (`src/agent.py`) and `negotiator_agent` (`src/negotiator.py`) import `agent_model` and `default_model_settings` from here rather than each constructing their own.
- **Injected Clock for "Today"**: Instead of agents calling `datetime.now()` directly, `Deps.now` (default `datetime.now`) is injected via dependency injection and read in `src/agent.py::current_date` (an `@appointment_agent.instructions` function) and in `resolve_scheduling_conflicts`'s prompt. This makes "today" swappable in tests without patching the system clock.