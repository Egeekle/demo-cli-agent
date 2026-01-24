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


def test_create_appointment(session, mock_ctx):
    now = datetime.now()
    app = tools.create_appointment(
        mock_ctx,
        title="New Appointment",
        start_time=now,
        end_time=now + timedelta(hours=1),
        location="Office",
        description="Test desc",
    )

    assert app.id is not None
    assert app.title == "New Appointment"

    db_app = session.get(Appointment, app.id)
    assert db_app is not None
    assert db_app.title == "New Appointment"


def test_delete_appointment(session, mock_ctx):
    now = datetime.now()
    app = Appointment(
        title="To Delete", start_time=now, end_time=now + timedelta(hours=1)
    )
    session.add(app)
    session.commit()
    app_id = app.id

    success = tools.delete_appointment(mock_ctx, app_id)
    assert success is True

    db_app = session.get(Appointment, app_id)
    assert db_app is None
