# Conflict Resolution Subagent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Implement a "Scheduling Negotiator" subagent that provides structured alternative slots when an appointment conflict is detected.

**Architecture:** 
- **Subagent:** A specialized PydanticAI agent with a structured `result_type` (`ConflictResolution` model).
- **Delegation:** The main agent (Supervisor) will use the subagent as a "tool" or internal function when an overlap is detected.
- **Dependency Injection:** The subagent will receive context about the conflict and existing appointments.

**Tech Stack:** Python, PydanticAI, Pydantic (Structured Outputs), SQLModel.

---

### Task 1: Define Structured Output Schemas

**Files:**
- Create: `src/schemas.py`

**Step 1: Define ProposedSlot and ConflictResolution**
Implement the Pydantic models for structured output.

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ProposedSlot(BaseModel):
    start_time: datetime
    end_time: datetime
    reasoning: str

class ConflictResolution(BaseModel):
    has_conflict: bool
    conflict_summary: str
    alternative_suggestions: List[ProposedSlot]
    user_action_required: str
```

**Step 2: Commit**
`git add src/schemas.py && git commit -m "feat: add structured schemas for conflict resolution"`

### Task 2: Implement the Negotiator Subagent

**Files:**
- Create: `src/negotiator.py`
- Test: `tests/test_negotiator.py`

**Step 1: Write failing test**
Verify that the negotiator returns a `ConflictResolution` object when provided with a conflict scenario.

**Step 2: Define Negotiator Agent**
Use `result_type=ConflictResolution` and a specialized system prompt.

**Step 3: Run test and verify pass**
Ensure the subagent generates valid structured data.

**Step 4: Commit**
`git add src/negotiator.py tests/test_negotiator.py && git commit -m "feat: implement scheduling negotiator subagent with structured output"`

### Task 3: Integrate Subagent into Supervisor (Main Agent)

**Files:**
- Modify: `src/agent.py`
- Modify: `src/tools.py`
- Test: `tests/test_integration_negotiator.py`

**Step 1: Update Deps**
Add `negotiator_agent` to the `Deps` class.

**Step 2: Add Conflict Check Tool**
Create a new tool `check_for_conflicts(start_time, end_time)` that the main agent can use, or modify `create_appointment` to internally delegate to the Negotiator if a conflict is found.
*Preferred:* Add a tool `resolve_scheduling_conflicts` that the Supervisor calls when it sees a conflict.

**Step 3: Update Supervisor System Prompt**
Instruct the Supervisor to call the negotiator if `search_appointments` reveals an overlap.

**Step 4: Commit**
`git add src/agent.py src/tools.py tests/test_integration_negotiator.py && git commit -m "feat: integrate negotiator subagent into supervisor flow"`

### Task 4: UI Refinement in CLI

**Files:**
- Modify: `src/main.py`

**Step 1: Improve Display**
Since the Negotiator returns structured data, update the main loop to display suggestions in a more readable format (e.g., a Rich table) if the result indicates a conflict resolution.
*Note:* If the Supervisor just summarizes the subagent's text, this is easier. If we want raw data display, we need to handle the structured response.

**Step 2: Commit**
`git add src/main.py && git commit -m "feat: improve CLI display for conflict suggestions"`
