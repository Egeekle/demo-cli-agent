# Happy Path Operations Eval Design

**Date:** 2026-02-14
**Status:** Approved

## Overview

A comprehensive evaluation suite for the appointment agent's core CRUD operations (Create, Read, Update, Delete) covering the "happy path" — single-request scenarios that do NOT require conflict resolution.

### Metrics

1. **retrieval_recall** — Was the relevant appointment(s) actually retrieved from the database?
2. **answer_groundedness** — Is the agent's answer grounded in the retrieved data (no hallucinations)?
3. **answer_correctness** — Does the generated answer align with the expected answer?

---

## Directory Structure

```
tests/evals/happy_path_operations/
├── generator.py
├── evaluate.py
├── test_generator.py
├── test_evaluate.py
└── data/
    └── happy_path_dataset.jsonl
```

---

## Constants

Shared constants for generator and evaluate modules:

```python
_SEEDED_DATE: str = "Wednesday, February 11, 2026"

_SEEDED_APPOINTMENTS: list[dict] = [
    {"title": "Doctor's Appointment", "start": "2026-02-12T14:00:00", "duration_mins": 60, "location": None},
    {"title": "Team Sync", "start": "2026-02-12T09:00:00", "duration_mins": 30, "location": None},
    {"title": "Lunch with Sarah", "start": "2026-02-13T12:00:00", "duration_mins": 60, "location": None},
    {"title": "Marketing Team Meeting", "start": "2026-02-17T15:00:00", "duration_mins": 60, "location": "Conference Room B"},
    {"title": "Dentist Appointment", "start": "2026-05-05T10:00:00", "duration_mins": 60, "location": "Downtown Clinic"},
]
```

---

## Synthetic Data Format

Each JSONL line:

```json
{
  "inputs": {
    "user_prompt": "What appointments do I have tomorrow?"
  },
  "expectations": {
    "operation_type": "READ",
    "expected_tool": "search_appointments",
    "expected_appointments": ["Doctor's Appointment", "Team Sync"],
    "expected_answer_contains": ["Doctor's Appointment", "Team Sync", "February 12"]
  }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `operation_type` | One of `CREATE`, `READ`, `UPDATE`, `DELETE` |
| `expected_tool` | The tool that should be called |
| `expected_appointments` | Titles of appointments that should be retrieved/affected |
| `expected_answer_contains` | Key phrases the response should include |

---

## Pydantic Models

```python
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

---

## Generator Design

Uses PydanticAI with structured output to generate ~20 synthetic examples (~5 per operation type).

```python
_GENERATOR_INSTRUCTIONS = f"""Generate synthetic user prompts for an appointment assistant.

Today's date: {_SEEDED_DATE}

SEEDED APPOINTMENTS IN DATABASE:
{_format_seeded_appointments()}

Generate examples for each operation type:
- READ: queries about existing appointments (search, list, get by ID)
- CREATE: requests to add NEW appointments (no conflicts with existing ones)
- UPDATE: requests to modify existing appointments
- DELETE: requests to remove existing appointments

IMPORTANT:
- For READ operations, expected_appointments must only contain titles from the seeded list.
- For CREATE operations, choose times that do NOT conflict with existing appointments.
- Use natural language variations: "tomorrow", "next week", "my dentist", etc.
"""

generator_agent = Agent(
    model=model,
    output_type=SyntheticDataset,
    instructions=_GENERATOR_INSTRUCTIONS,
)
```

---

## Scorers

### 1. `retrieval_recall` (Code-based)

Extracts tool outputs from trace and computes recall against expected appointments.

```python
@scorer
def retrieval_recall(trace: Trace, expectations: dict) -> Feedback:
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)

    retrieved_titles = set()
    for span in tool_spans:
        if span.outputs and isinstance(span.outputs, list):
            for appt in span.outputs:
                if isinstance(appt, dict) and "title" in appt:
                    retrieved_titles.add(appt["title"])

    expected = set(expectations.get("expected_appointments", []))

    if not expected:
        recall = 1.0  # No appointments expected
    else:
        recall = len(retrieved_titles & expected) / len(expected)

    return Feedback(
        value=recall,
        rationale=f"Retrieved: {retrieved_titles}, Expected: {expected}",
        metadata={"retrieved": list(retrieved_titles), "expected": list(expected)}
    )
```

### 2. `answer_groundedness` (LLM Judge with wrapper)

Custom scorer that extracts retrieved context from trace and passes to LLM judge.

```python
_groundedness_judge = make_judge(
    name="groundedness_judge",
    instructions="""You are evaluating whether an agent's response is grounded in retrieved data.

Retrieved appointments from database:
{{retrieved_context}}

Agent's response:
{{outputs}}

Score 1 if the response ONLY mentions appointments, dates, times, or details that appear in the retrieved data.
Score 0 if the response mentions ANY appointment, date, time, or detail NOT present in the retrieved data (hallucination).

Be strict: if the agent invents an appointment title, time, or location not in the retrieved data, score 0.
""",
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)

@scorer
def answer_groundedness(trace: Trace, outputs: str) -> Feedback:
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)
    retrieved_context = []
    for span in tool_spans:
        if span.outputs:
            if isinstance(span.outputs, list):
                retrieved_context.extend(span.outputs)
            else:
                retrieved_context.append(span.outputs)

    # Invoke the judge with extracted context
    result = _groundedness_judge.score(
        outputs=outputs,
        retrieved_context=json.dumps(retrieved_context, default=str)
    )
    return result
```

### 3. `answer_correctness` (LLM Judge)

```python
correctness_judge = make_judge(
    name="answer_correctness",
    instructions="""You are evaluating if an agent's response correctly addresses the user's request.

User request: {{inputs}}
Expected key elements in response: {{expected_answer_contains}}
Agent's response: {{outputs}}

Score 1 if:
- The response addresses the user's request appropriately
- The response contains all expected key elements (or reasonable equivalents)
- For CREATE/DELETE requests: the agent asks for confirmation (this is correct behavior)

Score 0 if:
- The response is missing expected information
- The response contains incorrect information
- The response doesn't address the user's request
""",
    model=f"openai:/{settings.MODEL_NAME}",
    feedback_value_type=bool,
)
```

---

## Evaluation Flow

### `predict_fn`

```python
@mlflow.trace
async def predict_fn(user_prompt: str) -> str:
    with src.database.get_session() as session:
        deps = Deps(session=session, negotiator_agent=negotiator_agent)
        result = await appointment_agent.run(user_prompt, deps=deps)
        return result.output
```

### `evaluation_context()`

Reuses the same pattern from safety eval:
- In-memory SQLite with StaticPool
- Seeds the 5 test appointments
- Mocks datetime to Feb 11, 2026, 10:00 AM

### `run_evaluation()`

```python
def run_evaluation(data_path: str) -> None:
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    with evaluation_context():
        results = evaluate(
            data=data,
            scorers=[retrieval_recall, answer_groundedness, answer_correctness],
            predict_fn=predict_fn,
        )
```

---

## Test Files

### `test_generator.py`

```python
def test_synthetic_example_structure():
    fields = SyntheticExample.model_fields
    assert "inputs" in fields
    assert "expectations" in fields

def test_expectations_has_required_fields():
    fields = SyntheticExpectations.model_fields
    assert "operation_type" in fields
    assert "expected_tool" in fields
    assert "expected_appointments" in fields
    assert "expected_answer_contains" in fields

def test_generator_agent_output_schema():
    assert generator_agent.output_type == SyntheticDataset
```

### `test_evaluate.py`

```python
@pytest.mark.asyncio
async def test_evaluation_context_isolation():
    with evaluation_context():
        assert src.agent.datetime.now() == datetime(2026, 2, 11, 10, 0, 0)

        with src.database.get_session() as session:
            appts = session.exec(select(Appointment)).all()
            assert len(appts) == 5

@pytest.mark.asyncio
async def test_predict_fn_returns_response():
    with evaluation_context():
        response = await predict_fn("What appointments do I have tomorrow?")
        assert isinstance(response, str)
        assert len(response) > 0
```

---

## References

- [MLflow Scorers Documentation](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/)
- [MLflow Trace Concepts](https://mlflow.org/docs/latest/genai/concepts/trace/)
- [MLflow Span Types](https://www.mlflow.org/docs/3.2.0rc0/genai/tracing/concepts/span/)
- Existing safety eval: `tests/evals/safety_confirmation/`
