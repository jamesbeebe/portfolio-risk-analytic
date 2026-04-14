"""CRUD helpers for the application's database layer.

CRUD stands for Create, Read, Update, Delete. These functions are the single
place where database rows are created, queried, or removed for this project.

We isolate database logic here instead of inside FastAPI routes so the API layer
can stay focused on HTTP concerns like request parsing and response formatting.
This also makes testing easier because database behavior can be exercised
directly, without having to go through the entire web stack for every case.
"""

from __future__ import annotations

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import AnalysisRun, SavedPortfolio
from app.models.api_models import AnalyzeRequest, AnalyzeResponse


def save_portfolio(
    db: Session,
    name: str,
    portfolio: AnalyzeRequest,
    notes: str | None = None,
) -> SavedPortfolio:
    """Create and persist a saved portfolio record.

    Args:
        db: Active SQLAlchemy database session.
        name: User-supplied display name for the portfolio.
        portfolio: API-layer portfolio definition to persist.
        notes: Optional free-text notes about the portfolio.

    Returns:
        The newly created SavedPortfolio ORM object after commit and refresh.
    """

    saved_portfolio = SavedPortfolio(
        name=name,
        tickers=portfolio.tickers,
        weights=portfolio.weights,
        start_date=portfolio.start_date,
        end_date=portfolio.end_date,
        confidence_level=portfolio.confidence_level,
        simulations=portfolio.simulations,
        notes=notes,
    )
    db.add(saved_portfolio)
    db.commit()
    # `db.refresh()` re-reads the row from the database so auto-generated fields
    # like the primary key and timestamp are populated on the Python object.
    db.refresh(saved_portfolio)
    return saved_portfolio


def save_analysis_run(
    db: Session,
    result: AnalyzeResponse,
    portfolio_id: int | None = None,
    duration_ms: int | None = None,
) -> AnalysisRun:
    """Create and persist an analysis history record.

    Args:
        db: Active SQLAlchemy database session.
        result: Completed analysis response to snapshot into history.
        portfolio_id: Optional linked saved portfolio ID.
        duration_ms: Optional runtime duration in milliseconds.

    Returns:
        The newly created AnalysisRun ORM object after commit and refresh.
    """

    analysis_run = AnalysisRun(
        portfolio_id=portfolio_id,
        tickers=result.tickers,
        weights=result.weights,
        mean_daily_return=result.mean_daily_return,
        annualized_volatility=result.annualized_volatility,
        var_95=result.var_95,
        es_95=result.es_95,
        var_99=result.var_99,
        es_99=result.es_99,
        simulation_count=result.simulation_count,
        duration_ms=duration_ms,
    )
    db.add(analysis_run)
    db.commit()
    db.refresh(analysis_run)
    return analysis_run


def get_all_portfolios(db: Session) -> list[SavedPortfolio]:
    """Return all saved portfolios ordered by newest first.

    Args:
        db: Active SQLAlchemy database session.

    Returns:
        A list of SavedPortfolio records ordered by `created_at` descending.
    """

    # "Descending" means newest timestamps appear first, which is useful because
    # people usually want to see their most recently saved portfolios at the top.
    return db.query(SavedPortfolio).order_by(desc(SavedPortfolio.created_at)).all()


def get_portfolio_by_id(db: Session, portfolio_id: int) -> SavedPortfolio | None:
    """Look up one saved portfolio by primary key.

    Args:
        db: Active SQLAlchemy database session.
        portfolio_id: Database ID of the portfolio to retrieve.

    Returns:
        The matching SavedPortfolio if found, otherwise None.
    """

    # Returning None is better than raising here because "not found" is a normal
    # application outcome that the API layer can translate into an HTTP response.
    return db.query(SavedPortfolio).filter(SavedPortfolio.id == portfolio_id).first()


def get_analysis_history(db: Session, limit: int = 20) -> list[AnalysisRun]:
    """Return the most recent analysis history rows.

    Args:
        db: Active SQLAlchemy database session.
        limit: Maximum number of rows to return.

    Returns:
        A list of AnalysisRun rows ordered by `ran_at` descending.
    """

    # History queries should always use a LIMIT so the API does not accidentally
    # return thousands of rows and become slow or memory-heavy as the table grows.
    return db.query(AnalysisRun).order_by(desc(AnalysisRun.ran_at)).limit(limit).all()


def get_runs_for_portfolio(db: Session, portfolio_id: int) -> list[AnalysisRun]:
    """Return all analysis runs linked to a specific saved portfolio.

    Args:
        db: Active SQLAlchemy database session.
        portfolio_id: Database ID of the saved portfolio.

    Returns:
        A list of AnalysisRun rows ordered by `ran_at` descending.
    """

    return (
        db.query(AnalysisRun)
        .filter(AnalysisRun.portfolio_id == portfolio_id)
        .order_by(desc(AnalysisRun.ran_at))
        .all()
    )


def delete_portfolio(db: Session, portfolio_id: int) -> bool:
    """Delete a saved portfolio by ID while preserving related analysis history.

    Args:
        db: Active SQLAlchemy database session.
        portfolio_id: Database ID of the portfolio to delete.

    Returns:
        True if a portfolio was deleted, otherwise False.
    """

    portfolio = get_portfolio_by_id(db, portfolio_id)
    if portfolio is None:
        return False

    # Linked AnalysisRun rows stay in the database so history is preserved. Their
    # `portfolio_id` is set to NULL before deletion because the foreign key column
    # is nullable and an analysis run can exist without a saved portfolio record.
    (
        db.query(AnalysisRun)
        .filter(AnalysisRun.portfolio_id == portfolio_id)
        .update({AnalysisRun.portfolio_id: None})
    )
    db.delete(portfolio)
    db.commit()
    return True


def deserialize_portfolio_to_request(portfolio: SavedPortfolio) -> AnalyzeRequest:
    """Convert a saved ORM portfolio row back into an API request model.

    Args:
        portfolio: SavedPortfolio ORM object loaded from the database.

    Returns:
        An AnalyzeRequest reconstructed from the saved portfolio fields.
    """

    return AnalyzeRequest(
        tickers=portfolio.tickers,
        weights=portfolio.weights,
        start_date=portfolio.start_date,
        end_date=portfolio.end_date,
        confidence_level=portfolio.confidence_level,
        simulations=portfolio.simulations,
        horizon_days=1,
    )
