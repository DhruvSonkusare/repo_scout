"""make how_to_start a jsonb list

Revision ID: 230c2a3289d3
Revises: a232314711d0
Create Date: 2026-11-05 18:06:14.429709

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '230c2a3289d3'
down_revision: Union[str, Sequence[str], None] = 'a232314711d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Convert how_to_start from text to a JSONB list, one item per non-blank line."""
    # Postgres can't cast text to jsonb on its own, so postgresql_using says how to convert each row.
    op.alter_column(
        'search_results',
        'how_to_start',
        existing_type=sa.Text(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        postgresql_using=r"to_jsonb(array_remove(string_to_array(how_to_start, E'\n'), ''))",
    )


def downgrade() -> None:
    """Convert how_to_start back to text, one list item per line."""
    # USING can't contain a subquery, and unpacking a JSONB list needs one,
    # so fill a new column instead and swap it in.
    op.add_column('search_results', sa.Column('how_to_start_text', sa.Text(), nullable=True))
    op.execute(
        r"""
        UPDATE search_results
        SET how_to_start_text = COALESCE(
            (
                SELECT string_agg(step, E'\n' ORDER BY position)
                FROM jsonb_array_elements_text(how_to_start) WITH ORDINALITY AS steps(step, position)
            ),
            ''
        )
        """
    )
    op.drop_column('search_results', 'how_to_start')
    op.alter_column(
        'search_results',
        'how_to_start_text',
        new_column_name='how_to_start',
        existing_type=sa.Text(),
        nullable=False,
    )
