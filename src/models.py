from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    description: Optional[str] = None
