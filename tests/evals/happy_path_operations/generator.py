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
