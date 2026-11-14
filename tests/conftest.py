from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from src.models import Appointment

FIXED_NOW = datetime(2026, 2, 11, 10, 0, 0)


def fixed_clock() -> datetime:
    return FIXED_NOW


@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with factory() as s:
        s.add_all([
            Appointment(title="Team Sync",
                        start_time=datetime(2026, 2, 12, 9, 0),
                        end_time=datetime(2026, 2, 12, 9, 30)),
            Appointment(title="Doctor's Appointment",
                        start_time=datetime(2026, 2, 12, 14, 0),
                        end_time=datetime(2026, 2, 12, 15, 0)),
            Appointment(title="Lunch with Sarah",
                        start_time=datetime(2026, 2, 13, 12, 0),
                        end_time=datetime(2026, 2, 13, 13, 0)),
        ])
        await s.commit()
        yield s
