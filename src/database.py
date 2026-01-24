from contextlib import contextmanager
from sqlmodel import create_engine, SQLModel, Session
from src.settings import settings

engine = create_engine(settings.DATABASE_URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
