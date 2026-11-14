import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import mlflow
from mlflow.genai import evaluate
from mlflow.genai.judges import make_judge
from pydantic_ai import DeferredToolRequests
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.agent import appointment_agent
from src.approvals import build_results
from src.models import Appointment
from src.negotiator import negotiator_agent
from src.providers import judge_inference_params, judge_model_uri
from src.settings import settings
from src.tools import Deps

_MLFLOW_EXPERIMENT_NAME: str = "Conflict Resolution Eval"

# Fixed clock injected into Deps for deterministic evaluation (no datetime patching).
FIXED_NOW: datetime = datetime(2026, 2, 11, 10, 0, 0)

# Every prompt overlaps the seeded "Doctor's Appointment" (Feb 12, 2026, 2:00-3:00pm).
_CONFLICT_DATASET: list[dict] = [
    {"inputs": {"user_prompt": "Book a 30-minute checkup on Feb 12 at 2:30pm"}},
    {"inputs": {"user_prompt": "Schedule a call on Feb 12 from 2:00pm to 2:45pm"}},
    {"inputs": {"user_prompt": "Set up a meeting on Feb 12 at 1:30pm for one hour"}},
    {"inputs": {"user_prompt": "Add a haircut appointment on Feb 12 from 2:15pm to 2:45pm"}},
]

_CONFLICT_JUDGE_INSTRUCTIONS: str = """You are evaluating whether an agent correctly handled a scheduling conflict.

The user's calendar already has "Doctor's Appointment" booked on Feb 12, 2026 from 2:00pm-3:00pm.
The user's request (which overlaps that existing appointment): {{inputs}}
Agent's response: {{outputs}}

Score 1 if the response:
- Clearly surfaces/acknowledges the scheduling conflict (mentions the existing appointment or the overlap), AND
- Offers a non-conflicting alternative time OR asks the user how they'd like to resolve it,
- AND does NOT silently confirm/create a new appointment that overlaps the existing one.

Score 0 if the response ignores the conflict, silently books the overlapping appointment, or
fails to mention any alternative/resolution path.
"""

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(_MLFLOW_EXPERIMENT_NAME)

# Judge runs on the dedicated judge model, not the agent's model.
conflict_judge = make_judge(
    name="conflict_resolution",
    instructions=_CONFLICT_JUDGE_INSTRUCTIONS,
    model=judge_model_uri,
    inference_params=judge_inference_params,
    feedback_value_type=bool,
)

# Session factory for the isolated in-memory DB, populated by `evaluation_context`
# and consumed by `predict_fn`.
test_factory: async_sessionmaker | None = None


@asynccontextmanager
async def evaluation_context():
    """
    Sets up a safe, isolated environment for evaluation.
    - Async in-memory SQLite database, seeded with a single Doctor's Appointment
      (Feb 12, 2026, 2:00pm-3:00pm) that every dataset prompt is designed to overlap.
    - A fixed clock (FIXED_NOW) is injected into Deps by `predict_fn` (no datetime
      patching).
    """
    global test_factory

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

    async with factory() as s:
        s.add(
            Appointment(
                title="Doctor's Appointment",
                start_time=datetime.fromisoformat("2026-02-12T14:00:00"),
                end_time=datetime.fromisoformat("2026-02-12T14:00:00")
                + timedelta(minutes=60),
            )
        )
        await s.commit()

    test_factory = factory
    try:
        yield factory
    finally:
        test_factory = None
        await test_engine.dispose()


async def predict_fn(user_prompt: str) -> str:
    """Run the appointment agent on a conflicting prompt and return its response.

    Deferred writes are approved automatically so the judge sees the agent's final
    text even if it walked through the approval gate; the overlap backstop
    (ModelRetry, enforced in src.tools) prevents an actual double-booked write from
    ever landing, so the response should still surface the conflict/alternatives.
    """
    async with test_factory() as session:
        deps = Deps(
            session=session, negotiator_agent=negotiator_agent, now=lambda: FIXED_NOW
        )
        result = await appointment_agent.run(user_prompt, deps=deps)
        while isinstance(result.output, DeferredToolRequests):
            result = await appointment_agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=build_results(result.output, approved=True),
                deps=deps,
            )
        return result.output


def run_evaluation(data: list[dict] | None = None) -> None:
    """Run the full conflict-resolution evaluation pipeline."""
    dataset = data if data is not None else _CONFLICT_DATASET

    # `mlflow.genai.evaluate` is synchronous and internally drives the async
    # `predict_fn` via its own `asyncio.run()` per row, so the isolated DB must be
    # set up/torn down *outside* of any running event loop.
    ctx = evaluation_context()
    entered = False
    try:
        asyncio.run(ctx.__aenter__())
        entered = True
        print("Starting conflict-resolution evaluation in isolated environment...")
        results = evaluate(
            data=dataset,
            scorers=[conflict_judge],
            predict_fn=predict_fn,
        )
    finally:
        if entered:
            asyncio.run(ctx.__aexit__(None, None, None))

    print(f"Evaluation complete. Metrics: {results.metrics}")
    return results


if __name__ == "__main__":
    run_evaluation()
