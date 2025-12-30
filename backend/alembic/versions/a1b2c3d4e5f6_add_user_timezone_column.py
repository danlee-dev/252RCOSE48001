"""add user timezone column

Revision ID: a1b2c3d4e5f6
Revises: d5e2f1a3b4c6
Create Date: 2024-12-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'd5e2f1a3b4c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add timezone column to users table with default 'Asia/Seoul'
    op.add_column('users', sa.Column('timezone', sa.String(), nullable=True))

    # Set default value for existing rows
    op.execute("UPDATE users SET timezone = 'Asia/Seoul' WHERE timezone IS NULL")


def downgrade() -> None:
    op.drop_column('users', 'timezone')
