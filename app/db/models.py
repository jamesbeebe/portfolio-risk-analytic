from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


# This table stores user-defined portfolio presets so a person can save a named
# portfolio configuration and come back to it later without retyping every input.
class SavedPortfolio(Base):
    """ORM model for saved portfolio definitions.

    Attributes:
        id: Auto-incrementing primary key for the saved portfolio row.
        name: User-facing portfolio name.
        tickers: JSON string snapshot of ticker symbols.
        weights: JSON string snapshot of portfolio weights.
        start_date: Historical analysis start date.
        end_date: Historical analysis end date.
        confidence_level: Risk confidence level used for analysis.
        simulations: Monte Carlo simulation count.
        created_at: UTC timestamp recording when the portfolio was saved.
        notes: Optional free-text user note.
        analysis_runs: Related analysis history records for this saved portfolio.
    """

    __tablename__ = "saved_portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    tickers: Mapped[str] = mapped_column(Text, nullable=False)
    weights: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)
    end_date: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence_level: Mapped[float] = mapped_column(Float, nullable=False)
    simulations: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # A SQLAlchemy relationship links Python objects across tables without
    # manually writing a join every time. Here it lets us access all analysis
    # runs that belong to one saved portfolio as `saved_portfolio.analysis_runs`.
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship(
        back_populates="portfolio",
    )

    def __repr__(self) -> str:
        """Return a readable debug representation of the saved portfolio."""

        return f"<SavedPortfolio id={self.id} name={self.name!r}>"


# This table stores analysis history so the app can remember what was run, what
# portfolio snapshot was analyzed, and what risk metrics were produced at that time.
class AnalysisRun(Base):
    """ORM model for completed analysis runs.

    Attributes:
        id: Auto-incrementing primary key for the analysis run row.
        portfolio_id: Optional foreign key linking to a saved portfolio.
        tickers: JSON string snapshot of tickers used in the run.
        weights: JSON string snapshot of weights used in the run.
        mean_daily_return: Historical mean daily portfolio return.
        annualized_volatility: Annualized portfolio volatility.
        var_95: 95% Value at Risk.
        es_95: 95% Expected Shortfall.
        var_99: 99% Value at Risk.
        es_99: 99% Expected Shortfall.
        simulation_count: Number of Monte Carlo scenarios used.
        ran_at: UTC timestamp recording when the analysis finished.
        duration_ms: Optional runtime duration in milliseconds.
        portfolio: Optional linked saved portfolio object.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("saved_portfolios.id"),
        nullable=True,
    )
    tickers: Mapped[str] = mapped_column(Text, nullable=False)
    weights: Mapped[str] = mapped_column(Text, nullable=False)
    mean_daily_return: Mapped[float] = mapped_column(Float, nullable=False)
    annualized_volatility: Mapped[float] = mapped_column(Float, nullable=False)
    var_95: Mapped[float] = mapped_column(Float, nullable=False)
    es_95: Mapped[float] = mapped_column(Float, nullable=False)
    var_99: Mapped[float] = mapped_column(Float, nullable=False)
    es_99: Mapped[float] = mapped_column(Float, nullable=False)
    simulation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ran_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    portfolio: Mapped[SavedPortfolio | None] = relationship(
        back_populates="analysis_runs"
    )

    def __repr__(self) -> str:
        """Return a readable debug representation of the analysis run."""

        ran_at_value = self.ran_at.date().isoformat() if self.ran_at else "unknown"
        return f"<AnalysisRun id={self.id} var_95={self.var_95:.3f} ran_at={ran_at_value}>"
