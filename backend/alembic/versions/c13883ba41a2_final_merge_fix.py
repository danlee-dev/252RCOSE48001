"""final merge fix

Revision ID: c13883ba41a2
Revises: 1d77e0c167b8, c8ad02008ca9
Create Date: 2025-12-09 17:26:35.426965

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c13883ba41a2'
down_revision: Union[str, Sequence[str], None] = ('1d77e0c167b8', 'c8ad02008ca9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
