# CLI Appointment Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a CLI-based personal assistant using Python, PydanticAI, and OpenAI to manage appointments with SQLite persistence and MLflow tracing.

**Architecture:**
- **CLI:** Typer for the command-line interface.
- **Agent:** PydanticAI `Agent` with `gpt-4.1-mini`.
- **Database:** SQLite accessed via SQLModel.
- **Monitoring:** MLflow for tracing agent execution using `@mlflow.trace`.
- **Package Manager:** `uv`.

**Tech Stack:** Python, PydanticAI, OpenAI, SQLModel, Typer, MLflow, PydanticSettings, uv.

---

### Task 1: Project Skeleton & Dependencies

**Files:**
- Create: `pyproject.toml` (via uv)
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/settings.py`

**Step 1: Initialize uv**
Run: `uv init`

**Step 2: Add Dependencies**
Run: `uv add pydantic-ai openai sqlmodel typer mlflow pydantic-settings python-dotenv`
Run: `uv add --dev pytest`

**Step 3: Create settings**
Define `Settings` class in `src/settings.py` loading `OPENAI_API_KEY`, `MLFLOW_TRACKING_URI`, `DATABASE_URL`.

**Step 4: Commit**

### Task 2: Database Models & Setup

**Files:**
- Create: `src/database.py`
- Create: `src/models.py`
- Test: `tests/test_database.py`

**Step 1: Define Model**
In `src/models.py`, define `Appointment` with `id`, `title`, `start_time`, `end_time`, `location`, `description`.

**Step 2: Database Engine**
In `src/database.py`, setup `create_db_and_tables()` and `get_session()`.

**Step 3: Test Connection**
Write a simple test to ensure tables are created in a temp SQLite db.

**Step 4: Commit**

### Task 3: Agent Tools (Read/Search)

**Files:**
- Create: `src/tools.py`
- Test: `tests/test_tools_read.py`

**Step 1: Implement Read Tools with Tracing**
- Decorate functions with `@mlflow.trace`.
- `list_appointments(limit: int, offset: int)`
- `search_appointments(query: str)`
- `get_appointment(id: int)`

**Step 2: Test Read Tools**
Verify they return data correctly.

**Step 3: Commit**

### Task 4: Agent Tools (Write with Confirmation Logic)

**Files:**
- Modify: `src/tools.py`
- Test: `tests/test_tools_write.py`

**Step 1: Implement Write Tools with Tracing**
- Decorate functions with `@mlflow.trace`.
- `create_appointment(appointment_data)`
- `delete_appointment(appointment_id)`

**Step 2: Test Write Tools**
Verify they persist/remove data.

**Step 3: Commit**

### Task 5: Agent Definition

**Files:**
- Create: `src/agent.py`
- Test: `tests/test_agent_structure.py`

**Step 1: Define Agent**
Initialize `Agent` with `gpt-4.1-mini`. Register tools.
Add system prompt ensuring it asks for permission before writing.

**Step 2: Trace Agent**
Ensure the primary entry point (e.g., a wrapper around `agent.run`) is decorated with `@mlflow.trace`.

**Step 3: Commit**

### Task 6: CLI Implementation

**Files:**
- Create: `src/main.py`

**Step 1: Typer App**
Create `app = Typer()`.
Command `chat`: Starts a REPL loop.
- Use `uv run` context.
- User input -> Agent.run() -> Print result.

**Step 2: Commit**

### Task 7: Integration & Refinement

**Files:**
- Modify: `src/agent.py`
- Manual Verification

**Step 1: Verify Confirmation Flow**
Test if the agent asks before deleting.

**Step 2: Verify MLflow**
Check if traces appear in MLflow UI.

**Step 3: Commit**
