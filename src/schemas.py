from pydantic import BaseModel
from datetime import datetime
from typing import List

class ProposedSlot(BaseModel):
    start_time: datetime
    end_time: datetime
    reasoning: str

class ConflictResolution(BaseModel):
    has_conflict: bool
    conflict_summary: str
    alternative_suggestions: List[ProposedSlot]
    user_action_required: str
