from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# This connection string points SQLAlchemy at a local SQLite file named
# `portfolio_risk.db` in the project root. When the app later moves to
# Supabase PostgreSQL, this value will change to a PostgreSQL URL such as
# `postgresql+psycopg://user:password@host:5432/database`.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./portfolio_risk.db")

is_sqlite = DATABASE_URL.startswith("sqlite")

# SQLite needs `check_same_thread=False` because the same application can access
# the database from different threads during local development. PostgreSQL does
# not need this because it is a client/server database designed for concurrent use.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
)

# `autocommit=False` means SQLAlchemy will not save changes automatically; we
# explicitly call `commit()` when we are ready. `autoflush=False` means pending
# changes are not pushed to the database automatically before every query, which
# makes database behavior easier to reason about while learning.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# `Base` is the parent class for all ORM table models. Every SQLAlchemy model
# inherits from it so SQLAlchemy can collect table metadata in one shared place.
Base = declarative_base()


def get_db():
    """Yield a database session and always close it afterwards.

    Yields:
        A live SQLAlchemy session connected to the configured database.

    Notes:
        This is a generator function, which means it produces a value with
        `yield` and then resumes later to finish cleanup code. FastAPI uses this
        pattern for dependencies so it can give a route a database session and
        still guarantee the session is closed after the request ends.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
