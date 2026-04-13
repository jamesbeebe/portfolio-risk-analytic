"""create initial tables

Revision ID: e39cb9334d28
Revises: 
Create Date: 2026-04-13 23:18:41.274781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e39cb9334d28'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "saved_portfolios",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("tickers", sa.JSON(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("start_date", sa.String(length=10), nullable=False),
        sa.Column("end_date", sa.String(length=10), nullable=False),
        sa.Column("confidence_level", sa.Float(), nullable=False),
        sa.Column("simulations", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("portfolio_id", sa.Integer(), nullable=True),
        sa.Column("tickers", sa.JSON(), nullable=False),
        sa.Column("weights", sa.JSON(), nullable=False),
        sa.Column("mean_daily_return", sa.Float(), nullable=False),
        sa.Column("annualized_volatility", sa.Float(), nullable=False),
        sa.Column("var_95", sa.Float(), nullable=False),
        sa.Column("es_95", sa.Float(), nullable=False),
        sa.Column("var_99", sa.Float(), nullable=False),
        sa.Column("es_99", sa.Float(), nullable=False),
        sa.Column("simulation_count", sa.Integer(), nullable=False),
        sa.Column("ran_at", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["portfolio_id"], ["saved_portfolios.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("analysis_runs")
    op.drop_table("saved_portfolios")
