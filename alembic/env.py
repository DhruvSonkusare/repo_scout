"""Alembic migration environment, using the app's settings and models."""

from logging.config import fileConfig

from alembic import context

from app.config import get_settings
from app.db import Base, engine

config = context.config

# disable_existing_loggers=False keeps the app's loggers on when migrations run in-process.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Autogenerate compares this metadata (our models) with the live database schema.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Write the migration SQL to stdout without connecting to a database."""
    context.configure(
        url=get_settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations on the app's own engine, so there is only one database URL."""
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
