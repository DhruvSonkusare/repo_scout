"""add pgvector: the extension and the vector tables

Vectors move from Chroma into Postgres. These two tables match what langchain-postgres's PGVector
expects, and tests/test_repos.py checks that they still do. The embedding column has no fixed size,
so each collection can use a different model: 768 dimensions locally, 3072 for gemini-embedding-001.

Revision ID: b1c2d3e4f5a6
Revises: 0a0eeb2112a5

"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = '0a0eeb2112a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        'langchain_pg_collection',
        sa.Column('uuid', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('cmetadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('uuid'),
        sa.UniqueConstraint('name'),
    )
    op.create_table(
        'langchain_pg_embedding',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('collection_id', sa.UUID(), nullable=True),
        sa.Column('embedding', Vector(), nullable=True),
        sa.Column('document', sa.String(), nullable=True),
        sa.Column('cmetadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['collection_id'], ['langchain_pg_collection.uuid'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_cmetadata_gin',
        'langchain_pg_embedding',
        ['cmetadata'],
        postgresql_using='gin',
        postgresql_ops={'cmetadata': 'jsonb_path_ops'},
    )
    # Vectors of repos indexed before this live in Chroma, not here. Forgetting their model makes
    # asking return "Re-index needed", instead of searching an empty collection and finding nothing.
    op.execute("UPDATE repos SET embedding_provider = NULL, embedding_model = NULL")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_cmetadata_gin', table_name='langchain_pg_embedding')
    op.drop_table('langchain_pg_embedding')
    op.drop_table('langchain_pg_collection')
    op.execute("DROP EXTENSION IF EXISTS vector")
