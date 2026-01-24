import pytest
from sqlmodel import create_engine, SQLModel, Session, select
from src.models import Appointment
from datetime import datetime, timedelta


@pytest.fixture(name="session")
def session_fixture():
    # Use in-memory database for testing
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    # implicit rollback/close when session context ends if not committed


def test_create_appointment(session: Session):
    now = datetime.now()
    appointment = Appointment(
        title="Meeting",
        start_time=now,
        end_time=now + timedelta(hours=1),
        location="Office",
        description="Discuss project",
    )
    session.add(appointment)
    session.flush()  # Send to DB so ID is generated
    session.refresh(appointment)

    assert appointment.id is not None

    statement = select(Appointment).where(Appointment.title == "Meeting")
    results = session.exec(statement)
    fetched_appointment = results.one()
    assert fetched_appointment.title == "Meeting"
    assert fetched_appointment.location == "Office"
