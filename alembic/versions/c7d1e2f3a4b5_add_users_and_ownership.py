"""add users and ownership

NULL owners mean public and repos default to public, so everything already here stays public.

Revision ID: c7d1e2f3a4b5
Revises: f9b2a625d412
Create Date: 2026-11-06 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d1e2f3a4b5'
down_revision: Union[str, Sequence[str], None] = 'f9b2a625d412'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OWNED_TABLES = ("conversations", "searches", "search_results")


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('github_id', sa.BigInteger(), nullable=False),
        sa.Column('login', sa.Text(), nullable=False),
        sa.Column('avatar_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('github_id', name=op.f('uq_users_github_id')),
    )
    op.create_table(
        'user_repos',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('repo_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['repo_id'], ['repos.id'], name=op.f('fk_user_repos_repo_id_repos'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_repos_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'repo_id', name=op.f('pk_user_repos')),
    )
    op.create_index(op.f('ix_user_repos_repo_id'), 'user_repos', ['repo_id'], unique=False)
    op.create_table(
        'user_daily_usage',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('requests', sa.Integer(), server_default='0', nullable=False),
        sa.Column('indexes', sa.Integer(), server_default='0', nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_daily_usage_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'day', name=op.f('pk_user_daily_usage')),
    )
    op.add_column('repos', sa.Column('is_public', sa.Boolean(), server_default=sa.text('true'), nullable=False))
    op.alter_column('daily_usage', 'requests', server_default='0')
    op.add_column('daily_usage', sa.Column('user_requests', sa.Integer(), server_default='0', nullable=False))
    op.add_column('daily_usage', sa.Column('user_chunks', sa.Integer(), server_default='0', nullable=False))
    for table in OWNED_TABLES:
        op.add_column(table, sa.Column('owner_user_id', sa.Integer(), nullable=True))
        op.create_index(op.f(f'ix_{table}_owner_user_id'), table, ['owner_user_id'], unique=False)
        op.create_foreign_key(
            op.f(f'fk_{table}_owner_user_id_users'), table, 'users', ['owner_user_id'], ['id'], ondelete='CASCADE'
        )


def downgrade() -> None:
    """Downgrade schema. Private rows are removed first, because without an owner they'd become public."""
    for table in OWNED_TABLES:
        op.execute(f'DELETE FROM {table} WHERE owner_user_id IS NOT NULL')
    op.execute('DELETE FROM repos WHERE NOT is_public')
    for table in OWNED_TABLES:
        op.drop_constraint(op.f(f'fk_{table}_owner_user_id_users'), table, type_='foreignkey')
        op.drop_index(op.f(f'ix_{table}_owner_user_id'), table_name=table)
        op.drop_column(table, 'owner_user_id')
    op.drop_column('daily_usage', 'user_chunks')
    op.drop_column('daily_usage', 'user_requests')
    op.alter_column('daily_usage', 'requests', server_default=None)
    op.drop_column('repos', 'is_public')
    op.drop_table('user_daily_usage')
    op.drop_index(op.f('ix_user_repos_repo_id'), table_name='user_repos')
    op.drop_table('user_repos')
    op.drop_table('users')
