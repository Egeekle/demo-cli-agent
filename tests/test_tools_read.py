import pytest
import mlflow
import tempfile
import shutil
from sqlmodel import create_engine, SQLModel, Session
from datetime import datetime, timedelta
from src.models import Appointment
from src import tools
from dataclasses import dataclass
from typing import Any


@dataclass
class MockContext:
    deps: Any


@pytest.fixture(scope="module", autouse=True)
def setup_mlflow():
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    shutil.rmtree(temp_dir)


@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def mock_ctx(session):
    return MockContext(deps=tools.Deps(session=session))


@pytest.fixture
def sample_appointments(session):
    app1 = Appointment(
        title="Team Meeting",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1),
        location="Room A",
        description="Weekly sync",
    )
    app2 = Appointment(
        title="Doctor Appointment",
        start_time=datetime.now() + timedelta(days=1),
        end_time=datetime.now() + timedelta(days=1, hours=1),
        location="Clinic",
        description="Checkup",
    )
    app3 = Appointment(
        title="Lunch with Client",
        start_time=datetime.now() + timedelta(days=2),
        end_time=datetime.now() + timedelta(days=2, hours=2),
        location="Restaurant",
        description="Project discussion",
    )
    session.add(app1)
    session.add(app2)
    session.add(app3)
    session.commit()
    return [app1, app2, app3]


def test_list_appointments(mock_ctx, sample_appointments):
    results = tools.list_appointments(mock_ctx)
    assert len(results) == 3

    results = tools.list_appointments(mock_ctx, limit=1, offset=0)
    assert len(results) == 1
    assert results[0].title == "Team Meeting"


def test_search_appointments(mock_ctx, sample_appointments):
    results = tools.search_appointments(mock_ctx, "Doctor")
    assert len(results) == 1
    assert results[0].title == "Doctor Appointment"


def test_get_appointment(mock_ctx, sample_appointments):
    app = tools.get_appointment(mock_ctx, 1)
    assert app is not None
    assert app.title == "Team Meeting"
