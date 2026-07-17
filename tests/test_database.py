import pytest
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.models import Appointment
from datetime import datetime


@pytest.mark.asyncio
async def test_get_session_yields_async_session_and_persists(monkeypatch):
    # Use a fresh in-memory async engine for isolation
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlmodel import SQLModel

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with factory() as session:
        appt = Appointment(
            title="X",
            start_time=datetime(2026, 2, 12, 9, 0),
            end_time=datetime(2026, 2, 12, 10, 0),
        )
        session.add(appt)
        await session.commit()
        await session.refresh(appt)
        assert appt.id is not None

        rows = (await session.exec(select(Appointment))).all()
        assert len(rows) == 1
