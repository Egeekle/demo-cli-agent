import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import mlflow
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.genai import evaluate
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer
from pydantic_ai import DeferredToolRequests
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.agent import appointment_agent
from src.approvals import summarize_approvals
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.providers import judge_inference_params, judge_model_uri
from src.settings import settings
from src.tools import Deps

_MLFLOW_EXPERIMENT_NAME: str = "Happy Path Operations Eval"
_SYNTHETIC_DATASET_PATH: str = (
    "tests/evals/happy_path_operations/data/happy_path_dataset.jsonl"
)

# Fixed clock injected into Deps for deterministic evaluation (no datetime patching).
FIXED_NOW: datetime = datetime(2026, 2, 11, 10, 0, 0)

_GROUNDEDNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating whether an agent's response is grounded in the expected appointments.

Expected appointments that should be referenced:
{{expectations}}

Agent's response:
{{outputs}}

Score 1 if the response ONLY mentions appointments that appear in the expected list (or correctly indicates none exist).
Score 0 if the response mentions ANY appointment NOT in the expected list (hallucination).

Special cases:
- If expected_appointments is empty and the agent asks for confirmation, score 1.
- If expected_appointments is empty and the agent says "no appointments found", score 1.
- Be strict: inventing appointment titles not in expected_appointments = score 0.
"""

_RETRIEVAL_RECALL_JUDGE_INSTRUCTIONS: str = """You are evaluating whether the agent retrieved the expected appointments from the database.

Expected appointments to retrieve (from expectations.expected_appointments): {{expectations}}
Execution trace showing tool calls and their outputs: {{trace}}

Examine the trace for tool calls (search_appointments, list_appointments, get_appointment).
Look at the outputs of these tool calls to see which appointments were actually retrieved.

Score 1 if:
- All appointments listed in expected_appointments were retrieved by tool calls in the trace
- OR expected_appointments is empty (CREATE/UPDATE/DELETE operation - no retrieval needed)

Score 0 if:
- Any expected appointment is missing from the tool call outputs in the trace
- The agent failed to call the appropriate tool to retrieve expected appointments
"""

_CORRECTNESS_JUDGE_INSTRUCTIONS: str = """You are evaluating if an agent's response correctly addresses the user's request.

User request: {{inputs}}
Expected behavior (from expectations.expected_answer_contains): {{expectations}}
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

# Set MLflow experiment
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)

# Define LLM Judges (run on the dedicated judge model, not the agent's model).
groundedness_judge = make_judge(
    name="answer_groundedness",
    instructions=_GROUNDEDNESS_JUDGE_INSTRUCTIONS,
    model=judge_model_uri,
    inference_params=judge_inference_params,
    feedback_value_type=bool,
)

correctness_judge = make_judge(
    name="answer_correctness",
    instructions=_CORRECTNESS_JUDGE_INSTRUCTIONS,
    model=judge_model_uri,
    inference_params=judge_inference_params,
    feedback_value_type=bool,
)

retrieval_recall_judge = make_judge(
    name="retrieval_recall_judge",
    instructions=_RETRIEVAL_RECALL_JUDGE_INSTRUCTIONS,
    model=judge_model_uri,
    inference_params=judge_inference_params,
    feedback_value_type=bool,
)


@scorer
def retrieval_recall(trace: Trace, expectations: dict):
    """
    Deterministic, code-based scorer that measures retrieval recall.
    Checks if expected appointments were retrieved by tool calls.
    """
    expected_appointments = set(expectations.get("expected_appointments", []))

    # If no appointments expected (e.g., CREATE operation), recall is 1.0
    if not expected_appointments:
        return Feedback(
            value=1.0,
            rationale="No appointments expected to be retrieved (CREATE operation)",
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
    )


# Session factory for the isolated in-memory DB, populated by `evaluation_context`
# and consumed by `predict_fn`.
test_factory: async_sessionmaker | None = None


@asynccontextmanager
async def evaluation_context():
    """
    Sets up a safe, isolated environment for evaluation.
    - Async in-memory SQLite database, seeded with test data.
    - A fixed clock (FIXED_NOW) is injected into Deps by `predict_fn` (no datetime
      patching).
    """
    global test_factory

    # 1. Setup async in-memory DB
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    # 2. Seed Data
    async with factory() as s:
        s.add_all(
            [
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
        )
        await s.commit()

    test_factory = factory
    try:
        yield factory
    finally:
        test_factory = None
        await test_engine.dispose()


async def predict_fn(user_prompt: str) -> str:
    """Run the appointment agent and return its response."""
    async with test_factory() as session:
        deps = Deps(
            session=session, negotiator_agent=negotiator_agent, now=lambda: FIXED_NOW
        )
        result = await appointment_agent.run(user_prompt, deps=deps)
        # Destructive actions (CREATE/UPDATE/DELETE) surface as a DeferredToolRequests
        # approval gate rather than being auto-approved. Returning a confirmation
        # string (instead of executing the action) keeps the safety behavior visible
        # to the text-only judges and satisfies expected_answer_contains=["confirm"].
        if isinstance(result.output, DeferredToolRequests):
            return (
                "To proceed I need your confirmation for the following action(s):\n"
                f"{summarize_approvals(result.output)}"
            )
        return result.output


def run_evaluation(data_path: str = _SYNTHETIC_DATASET_PATH) -> None:
    """Run the full evaluation pipeline."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # `mlflow.genai.evaluate` is synchronous and internally drives the async
    # `predict_fn` via its own `asyncio.run()` per row, so the isolated DB must be
    # set up/torn down *outside* of any running event loop.
    ctx = evaluation_context()
    entered = False
    try:
        asyncio.run(ctx.__aenter__())
        entered = True
        print("Starting evaluation in isolated environment...")
        results = evaluate(
            data=data,
            scorers=[
                retrieval_recall,
                retrieval_recall_judge,
                groundedness_judge,
                correctness_judge,
            ],
            predict_fn=predict_fn,
        )
    finally:
        if entered:
            asyncio.run(ctx.__aexit__(None, None, None))

    print(f"Evaluation complete. Metrics: {results.metrics}")
    return results


if __name__ == "__main__":
    run_evaluation()
