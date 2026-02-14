# Happy Path Operations Eval - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement comprehensive CRUD evaluation suite with retrieval_recall, answer_groundedness, and answer_correctness metrics.

**Architecture:** Follow existing safety_confirmation eval pattern. Generator creates synthetic JSONL with inputs/expectations. Evaluate module defines three scorers (one code-based, two LLM judges) and runs them via `mlflow.genai.evaluate()`.

**Tech Stack:** PydanticAI, MLflow GenAI, SQLModel, pytest

---

## Task 1: Create Directory Structure and Empty Files

**Files:**
- Create: `tests/evals/happy_path_operations/__init__.py`
- Create: `tests/evals/happy_path_operations/generator.py`
- Create: `tests/evals/happy_path_operations/evaluate.py`
- Create: `tests/evals/happy_path_operations/test_generator.py`
- Create: `tests/evals/happy_path_operations/test_evaluate.py`
- Create: `tests/evals/happy_path_operations/data/.gitkeep`

**Step 1: Create directory and empty files**

```bash
mkdir -p tests/evals/happy_path_operations/data
touch tests/evals/happy_path_operations/__init__.py
touch tests/evals/happy_path_operations/generator.py
touch tests/evals/happy_path_operations/evaluate.py
touch tests/evals/happy_path_operations/test_generator.py
touch tests/evals/happy_path_operations/test_evaluate.py
touch tests/evals/happy_path_operations/data/.gitkeep
```

**Step 2: Commit**

```bash
git add tests/evals/happy_path_operations/
git commit -m "chore: scaffold happy_path_operations eval directory"
```

---

## Task 2: Implement Generator Pydantic Models

**Files:**
- Modify: `tests/evals/happy_path_operations/generator.py`
- Test: `tests/evals/happy_path_operations/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/evals/happy_path_operations/test_generator.py
from tests.evals.happy_path_operations.generator import (
    SyntheticDataset,
    SyntheticExample,
    SyntheticExpectations,
    SyntheticInputs,
)


def test_synthetic_example_structure():
    """Verify the nested structure is preserved for MLflow."""
    fields = SyntheticExample.model_fields
    assert "inputs" in fields
    assert "expectations" in fields
    assert fields["inputs"].annotation == SyntheticInputs
    assert fields["expectations"].annotation == SyntheticExpectations


def test_expectations_has_required_fields():
    """Verify expectations has all required fields."""
    fields = SyntheticExpectations.model_fields
    assert "operation_type" in fields
    assert "expected_tool" in fields
    assert "expected_appointments" in fields
    assert "expected_answer_contains" in fields
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evals/happy_path_operations/test_generator.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# tests/evals/happy_path_operations/generator.py
from typing import Literal

from pydantic import BaseModel


class SyntheticInputs(BaseModel):
    user_prompt: str


class SyntheticExpectations(BaseModel):
    operation_type: Literal["CREATE", "READ", "UPDATE", "DELETE"]
    expected_tool: str
    expected_appointments: list[str]
    expected_answer_contains: list[str]


class SyntheticExample(BaseModel):
    inputs: SyntheticInputs
    expectations: SyntheticExpectations


class SyntheticDataset(BaseModel):
    examples: list[SyntheticExample]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evals/happy_path_operations/test_generator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/evals/happy_path_operations/generator.py tests/evals/happy_path_operations/test_generator.py
git commit -m "feat(eval): add pydantic models for happy path operations"
```

---

## Task 3: Implement Generator Agent and Constants

**Files:**
- Modify: `tests/evals/happy_path_operations/generator.py`
- Test: `tests/evals/happy_path_operations/test_generator.py`

**Step 1: Add test for generator agent**

Add to `test_generator.py`:

```python
from tests.evals.happy_path_operations.generator import generator_agent


def test_generator_agent_output_schema():
    assert generator_agent.output_type == SyntheticDataset
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evals/happy_path_operations/test_generator.py::test_generator_agent_output_schema -v`
Expected: FAIL with ImportError

**Step 3: Add constants and generator agent to generator.py**

Replace entire `generator.py` with:

```python
# tests/evals/happy_path_operations/generator.py
import json
import os
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.settings import settings

_NUM_SYNTHETIC_SAMPLES: int = 20
_SYNTHETIC_DATASET_PATH: str = (
    "tests/evals/happy_path_operations/data/happy_path_dataset.jsonl"
)
_SEEDED_DATE: str = "Wednesday, February 11, 2026"
_SEEDED_APPOINTMENTS: str = """1. "Doctor's Appointment" - Thursday Feb 12, 2:00 PM (1 hour)
2. "Team Sync" - Thursday Feb 12, 9:00 AM (30 minutes)
3. "Lunch with Sarah" - Friday Feb 13, 12:00 PM (1 hour)
4. "Marketing Team Meeting" - Tuesday Feb 17, 3:00 PM (1 hour, Conference Room B)
5. "Dentist Appointment" - Tuesday May 5, 10:00 AM (1 hour, Downtown Clinic)"""

_GENERATOR_AGENT_INSTRUCTIONS: str = f"""You are an expert at generating synthetic datasets for testing AI appointment assistants.
Generate a diverse set of user prompts covering all CRUD operations.

IMPORTANT CONTEXT:
- Today's date is {_SEEDED_DATE}.
- The user's calendar contains ONLY these appointments:
{_SEEDED_APPOINTMENTS}

Generate examples for each operation type (aim for ~5 examples each):

READ operations (expected_tool: "search_appointments", "list_appointments", or "get_appointment"):
- Queries about existing appointments by date, name, or keyword
- Questions like "What do I have tomorrow?", "When is my dentist?", "Show me Friday's schedule"
- expected_appointments: titles that SHOULD be returned
- expected_answer_contains: key info that should appear (titles, dates, times)

CREATE operations (expected_tool: "create_appointment"):
- Requests to schedule NEW appointments at times that DO NOT conflict with existing ones
- Use gaps in the schedule: Feb 11 (today), Feb 14-16, Feb 18+, mornings/evenings
- expected_appointments: [] (no retrieval needed)
- expected_answer_contains: ["confirm"] (agent must ask for confirmation)

UPDATE operations (expected_tool: "update_appointment"):
- Requests to modify existing appointments (time, title, location)
- Reference existing appointments by name
- expected_appointments: [title being updated]
- expected_answer_contains: ["confirm"] (agent must ask for confirmation)

DELETE operations (expected_tool: "delete_appointment"):
- Requests to cancel/remove existing appointments
- Reference existing appointments by name or description
- expected_appointments: [title being deleted]
- expected_answer_contains: ["confirm"] (agent must ask for confirmation)

RULES:
- Use natural, varied phrasing ("tomorrow", "next Tuesday", "my dentist", etc.)
- READ operations MUST only reference appointments that exist in the seeded list
- CREATE operations MUST use times that don't conflict with existing appointments
- UPDATE/DELETE operations MUST reference existing appointments"""

# Initialize the model
openai_provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
model = OpenAIResponsesModel(settings.MODEL_NAME, provider=openai_provider)


class SyntheticInputs(BaseModel):
    user_prompt: str


class SyntheticExpectations(BaseModel):
    operation_type: Literal["CREATE", "READ", "UPDATE", "DELETE"]
    expected_tool: str
    expected_appointments: list[str]
    expected_answer_contains: list[str]


class SyntheticExample(BaseModel):
    inputs: SyntheticInputs
    expectations: SyntheticExpectations


class SyntheticDataset(BaseModel):
    examples: list[SyntheticExample]


generator_agent = Agent(
    model=model,
    output_type=SyntheticDataset,
    instructions=_GENERATOR_AGENT_INSTRUCTIONS,
)


def generate_data(count: int, output_path: str):
    print(f"Generating {count} synthetic examples...")
    result = generator_agent.run_sync(f"Generate {count} unique examples.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for ex in result.output.examples:
            f.write(json.dumps(ex.model_dump()) + "\n")

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_data(count=_NUM_SYNTHETIC_SAMPLES, output_path=_SYNTHETIC_DATASET_PATH)
```

**Step 4: Run all generator tests**

Run: `pytest tests/evals/happy_path_operations/test_generator.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add tests/evals/happy_path_operations/generator.py tests/evals/happy_path_operations/test_generator.py
git commit -m "feat(eval): add generator agent with instructions for happy path operations"
```

---

## Task 4: Implement evaluation_context

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`
- Test: `tests/evals/happy_path_operations/test_evaluate.py`

**Step 1: Write the failing test**

```python
# tests/evals/happy_path_operations/test_evaluate.py
from datetime import datetime

import pytest
from sqlmodel import select

import src.agent
import src.database
import src.negotiator
from src.models import Appointment
from tests.evals.happy_path_operations import evaluate


@pytest.mark.asyncio
async def test_evaluation_context_isolation():
    """Verify evaluation_context provides isolated DB and frozen time."""
    with evaluate.evaluation_context():
        # Check 1: Time is frozen for agent and negotiator
        assert src.agent.datetime.now() == datetime(2026, 2, 11, 10, 0, 0)
        assert src.negotiator.datetime.now() == datetime(2026, 2, 11, 10, 0, 0)

        # Check 2: DB is isolated and seeded with 5 appointments
        with src.database.get_session() as session:
            appts = session.exec(select(Appointment)).all()
            assert len(appts) == 5

            # Verify specific appointments exist
            titles = {a.title for a in appts}
            assert "Doctor's Appointment" in titles
            assert "Team Sync" in titles
            assert "Lunch with Sarah" in titles
            assert "Marketing Team Meeting" in titles
            assert "Dentist Appointment" in titles
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evals/happy_path_operations/test_evaluate.py::test_evaluation_context_isolation -v`
Expected: FAIL with ImportError

**Step 3: Implement evaluation_context in evaluate.py**

```python
# tests/evals/happy_path_operations/evaluate.py
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import src.database
from src.models import Appointment


@contextmanager
def evaluation_context():
    """
    Sets up a safe, isolated environment for evaluation.
    - Mocks the database to use an in-memory SQLite instance.
    - Seeds the database with test data.
    - Mocks datetime to a fixed point in time.
    """
    # 1. Setup In-Memory DB
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(test_engine)

    # 2. Seed Data
    with Session(test_engine) as session:
        appts = [
            Appointment(
                title="Doctor's Appointment",
                start_time=datetime.fromisoformat("2026-02-12T14:00:00"),
                end_time=datetime.fromisoformat("2026-02-12T14:00:00")
                + timedelta(minutes=60),
            ),
            Appointment(
                title="Team Sync",
                start_time=datetime.fromisoformat("2026-02-12T09:00:00"),
                end_time=datetime.fromisoformat("2026-02-12T09:00:00")
                + timedelta(minutes=30),
            ),
            Appointment(
                title="Lunch with Sarah",
                start_time=datetime.fromisoformat("2026-02-13T12:00:00"),
                end_time=datetime.fromisoformat("2026-02-13T12:00:00")
                + timedelta(minutes=60),
            ),
            Appointment(
                title="Marketing Team Meeting",
                start_time=datetime.fromisoformat("2026-02-17T15:00:00"),
                end_time=datetime.fromisoformat("2026-02-17T15:00:00")
                + timedelta(minutes=60),
                location="Conference Room B",
            ),
            Appointment(
                title="Dentist Appointment",
                start_time=datetime.fromisoformat("2026-05-05T10:00:00"),
                end_time=datetime.fromisoformat("2026-05-05T10:00:00")
                + timedelta(minutes=60),
                location="Downtown Clinic",
            ),
        ]
        session.add_all(appts)
        session.commit()

    # 3. Define the mocked get_session
    @contextmanager
    def mock_get_session():
        with Session(test_engine) as session:
            yield session

    # 4. Mock Datetime (Fixed to Wed, Feb 11, 2026, 10:00 AM)
    class MockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 11, 10, 0, 0)

    # Apply patches
    with (
        patch("src.database.get_session", side_effect=mock_get_session),
        patch("src.agent.datetime", MockDatetime),
        patch("src.negotiator.datetime", MockDatetime),
    ):
        yield
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evals/happy_path_operations/test_evaluate.py::test_evaluation_context_isolation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/evals/happy_path_operations/evaluate.py tests/evals/happy_path_operations/test_evaluate.py
git commit -m "feat(eval): add evaluation_context for happy path operations"
```

---

## Task 5: Implement predict_fn

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`
- Modify: `tests/evals/happy_path_operations/test_evaluate.py`

**Step 1: Write the failing test**

Add to `test_evaluate.py`:

```python
@pytest.mark.asyncio
async def test_predict_fn_returns_response():
    """Verify predict_fn returns a string response."""
    with evaluate.evaluation_context():
        response = await evaluate.predict_fn("What appointments do I have tomorrow?")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention tomorrow's appointments
        lowered = response.lower()
        assert "doctor" in lowered or "team sync" in lowered
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/evals/happy_path_operations/test_evaluate.py::test_predict_fn_returns_response -v`
Expected: FAIL with AttributeError (predict_fn not defined)

**Step 3: Add predict_fn to evaluate.py**

Add these imports at the top of `evaluate.py`:

```python
import src.database
from src.agent import appointment_agent
from src.negotiator import negotiator_agent
from src.tools import Deps
```

Add after `evaluation_context()`:

```python
async def predict_fn(user_prompt: str) -> str:
    """Run the appointment agent and return its response."""
    with src.database.get_session() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent)
        result = await appointment_agent.run(user_prompt, deps=deps)
        return result.output
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/evals/happy_path_operations/test_evaluate.py::test_predict_fn_returns_response -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/evals/happy_path_operations/evaluate.py tests/evals/happy_path_operations/test_evaluate.py
git commit -m "feat(eval): add predict_fn for happy path operations"
```

---

## Task 6: Implement LLM Judge Scorers

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`

**Step 1: Add imports and constants for judges**

Add to imports in `evaluate.py`:

```python
import mlflow
from mlflow.genai.judges import make_judge

from src.settings import settings
```

Add constants after imports:

```python
_MLFLOW_EXPERIMENT_NAME: str = "Happy Path Operations Eval"

_GROUNDEDNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating whether an agent's response is grounded in retrieved data.

Retrieved appointments from database:
{{retrieved_context}}

Agent's response:
{{outputs}}

Score 1 if the response ONLY mentions appointments, dates, times, or details that appear in the retrieved data.
Score 0 if the response mentions ANY appointment, date, time, or detail NOT present in the retrieved data (hallucination).

Special cases:
- If no appointments were retrieved and the agent correctly says "no appointments found" or similar, score 1.
- If the agent asks for confirmation (for create/update/delete), score 1 as long as no fake data is mentioned.
- Be strict: inventing appointment titles, times, or locations not in retrieved data = score 0.
"""

_CORRECTNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating if an agent's response correctly addresses the user's request.

User request: {{inputs}}
Expected key elements in response: {{expected_answer_contains}}
Agent's response: {{outputs}}

Score 1 if:
- The response addresses the user's request appropriately
- The response contains all expected key elements (or reasonable semantic equivalents)
- For CREATE/DELETE/UPDATE requests: asking for confirmation counts as correct behavior

Score 0 if:
- The response is missing expected information
- The response contains factually incorrect information
- The response doesn't address the user's request
"""
```

**Step 2: Add MLflow setup and judge definitions**

Add after constants:

```python
# Set MLflow experiment
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)

# Define LLM Judges
groundedness_judge = make_judge(
    name="answer_groundedness",
    instructions=_GROUNDEDNESS_JUDGE_INSTRUCTIONS,
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)

correctness_judge = make_judge(
    name="answer_correctness",
    instructions=_CORRECTNESS_JUDGE_INSTRUCTIONS,
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)
```

**Step 3: Commit**

```bash
git add tests/evals/happy_path_operations/evaluate.py
git commit -m "feat(eval): add groundedness and correctness LLM judges"
```

---

## Task 7: Implement retrieval_recall Scorer

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`

**Step 1: Add imports for custom scorer**

Add to imports:

```python
import json

from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai.scorers import scorer
```

**Step 2: Implement retrieval_recall scorer**

Add after the judge definitions:

```python
@scorer
def retrieval_recall(
    trace: Trace, expectations: dict
) -> Feedback:
    """
    Code-based scorer that measures retrieval recall.
    Checks if expected appointments were retrieved by tool calls.
    """
    expected_appointments = set(expectations.get("expected_appointments", []))

    # If no appointments expected (e.g., CREATE operation), recall is 1.0
    if not expected_appointments:
        return Feedback(
            value=1.0,
            rationale="No appointments expected to be retrieved (CREATE operation)",
            metadata={"expected": [], "retrieved": []}
        )

    # Extract retrieved appointment titles from TOOL spans
    retrieved_titles = set()

    if trace and trace.data and trace.data.spans:
        for span in trace.data.spans:
            # Look for tool spans that return appointment data
            if span.span_type == SpanType.TOOL and span.outputs:
                outputs = span.outputs
                # Handle list of appointments
                if isinstance(outputs, list):
                    for item in outputs:
                        if isinstance(item, dict) and "title" in item:
                            retrieved_titles.add(item["title"])
                # Handle single appointment
                elif isinstance(outputs, dict) and "title" in outputs:
                    retrieved_titles.add(outputs["title"])

    # Calculate recall
    matched = expected_appointments & retrieved_titles
    recall = len(matched) / len(expected_appointments)

    return Feedback(
        value=recall,
        rationale=f"Retrieved {len(matched)}/{len(expected_appointments)} expected appointments",
        metadata={
            "expected": list(expected_appointments),
            "retrieved": list(retrieved_titles),
            "matched": list(matched)
        }
    )
```

**Step 3: Commit**

```bash
git add tests/evals/happy_path_operations/evaluate.py
git commit -m "feat(eval): add retrieval_recall code-based scorer"
```

---

## Task 8: Implement run_evaluation Function

**Files:**
- Modify: `tests/evals/happy_path_operations/evaluate.py`

**Step 1: Add imports**

Add to imports:

```python
import os

from mlflow.genai import evaluate
```

**Step 2: Add dataset path constant and run_evaluation function**

Add constant at top with others:

```python
_SYNTHETIC_DATASET_PATH: str = (
    "tests/evals/happy_path_operations/data/happy_path_dataset.jsonl"
)
```

Add `run_evaluation` function at end of file:

```python
def run_evaluation(data_path: str = _SYNTHETIC_DATASET_PATH) -> None:
    """Run the full evaluation pipeline."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    with evaluation_context():
        print("Starting evaluation in isolated environment...")
        results = evaluate(
            data=data,
            scorers=[retrieval_recall, groundedness_judge, correctness_judge],
            predict_fn=predict_fn,
        )

    print(f"Evaluation complete. Metrics: {results.metrics}")
    return results


if __name__ == "__main__":
    run_evaluation()
```

**Step 3: Commit**

```bash
git add tests/evals/happy_path_operations/evaluate.py
git commit -m "feat(eval): add run_evaluation function for happy path operations"
```

---

## Task 9: Generate Synthetic Dataset

**Files:**
- Create: `tests/evals/happy_path_operations/data/happy_path_dataset.jsonl`

**Step 1: Run the generator**

```bash
python -m tests.evals.happy_path_operations.generator
```

Expected: "Dataset saved to tests/evals/happy_path_operations/data/happy_path_dataset.jsonl"

**Step 2: Verify the dataset**

```bash
wc -l tests/evals/happy_path_operations/data/happy_path_dataset.jsonl
head -3 tests/evals/happy_path_operations/data/happy_path_dataset.jsonl
```

Expected: ~20 lines, valid JSON per line

**Step 3: Commit**

```bash
git add tests/evals/happy_path_operations/data/happy_path_dataset.jsonl
git rm tests/evals/happy_path_operations/data/.gitkeep 2>/dev/null || true
git commit -m "feat(eval): generate synthetic dataset for happy path operations"
```

---

## Task 10: Run Full Evaluation and Verify

**Step 1: Run all tests**

```bash
pytest tests/evals/happy_path_operations/ -v
```

Expected: All tests PASS

**Step 2: Run the evaluation (optional, costs API credits)**

```bash
python -m tests.evals.happy_path_operations.evaluate
```

Expected: Evaluation completes with metrics output

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any issues from full eval run"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Scaffold directory | `tests/evals/happy_path_operations/*` |
| 2 | Pydantic models | `generator.py`, `test_generator.py` |
| 3 | Generator agent | `generator.py`, `test_generator.py` |
| 4 | evaluation_context | `evaluate.py`, `test_evaluate.py` |
| 5 | predict_fn | `evaluate.py`, `test_evaluate.py` |
| 6 | LLM judges | `evaluate.py` |
| 7 | retrieval_recall | `evaluate.py` |
| 8 | run_evaluation | `evaluate.py` |
| 9 | Generate dataset | `data/happy_path_dataset.jsonl` |
| 10 | Verify & fix | All files |
