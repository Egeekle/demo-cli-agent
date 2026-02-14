# Development Guide

## Environment Setup

This project uses `uv`. To set up your local development environment:

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    uv sync
    ```
3.  **Configure Environment**:
    - `OPENAI_API_KEY`: Required for agent reasoning.
    - `MLFLOW_TRACKING_URI`: Defaults to `sqlite:///mlflow.db`.
    - `DATABASE_URL`: Defaults to `sqlite:///database.db`.

## Running Tests

We use `pytest` for testing. To run all tests:

```bash
uv run pytest
```

### Test Structure
- `tests/test_settings.py`: Verifies configuration loading.
- `tests/test_database.py`: Ensures SQLModel schemas are valid.
- `tests/test_tools_read.py` & `tests/test_tools_write.py`: Tests the logic of individual agent tools using dependency injection mocks.
- `tests/test_agent_structure.py`: Verifies agent tool registration and system prompt.

## Project Structure

- `src/main.py`: CLI entry point and REPL loop.
- `src/agent.py`: PydanticAI Agent definition and system prompts.
- `src/tools.py`: Business logic and database operations (Agent Tools).
- `src/models.py`: SQLModel/Pydantic data schemas.
- `src/database.py`: Database engine and session management.
- `src/settings.py`: Configuration management.
- `src/mlflow_server.py`: Utility to launch the MLflow UI.

## Evaluations

This project includes MLflow-based evaluation pipelines to measure agent quality.

### Available Evals

| Eval                      | Description                                              | Metrics                                   |
|---------------------------|----------------------------------------------------------|-------------------------------------------|
| **Safety Confirmation**   | Verifies agent asks for confirmation before CREATE/DELETE | Pass/Fail                                 |
| **Happy Path Operations** | Tests CRUD operations quality                             | Retrieval Recall, Groundedness, Correctness |

### Eval Structure

Each eval follows a consistent pattern:

```text
tests/evals/<eval_name>/
├── generator.py      # Generates synthetic JSONL dataset
├── evaluate.py       # Runs evaluation with MLflow
├── test_generator.py # Tests for generator
├── test_evaluate.py  # Tests for evaluation pipeline
└── data/
    └── <eval>_dataset.jsonl
```

### Running Evaluations

**1. Generate synthetic test data:**

```bash
# Safety confirmation dataset
uv run python -m tests.evals.safety_confirmation.generator

# Happy path operations dataset
uv run python -m tests.evals.happy_path_operations.generator
```

**2. Run evaluations:**

```bash
# Safety confirmation eval
uv run python -m tests.evals.safety_confirmation.evaluate

# Happy path operations eval
uv run python -m tests.evals.happy_path_operations.evaluate
```

**3. View results in MLflow:**

```bash
uv run ml-ui
```

Visit `http://localhost:5050` and check the experiment runs.

## Key Commands

- `uv run appointment-agent`: Start the CLI.
- `uv run ml-ui`: Start MLflow.
- `uv run pytest`: Run tests.
- `uv sync`: Sync dependencies.
- `uv run python -m tests.evals.<eval_name>.generator`: Generate eval dataset.
- `uv run python -m tests.evals.<eval_name>.evaluate`: Run evaluation.
