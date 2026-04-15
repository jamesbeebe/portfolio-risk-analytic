🚀 Live Demo: [portfolio-risk-analytic-jeukzztecya9yqtujh5ish.streamlit.app](https://portfolio-risk-analytic-jeukzztecya9yqtujh5ish.streamlit.app)

⚙️ API Docs: [portfolio-risk-analytic.onrender.com/docs](https://portfolio-risk-analytic.onrender.com/docs)

Monte Carlo portfolio risk analysis with a Streamlit frontend, FastAPI backend, Supabase persistence, and Yahoo Finance market data.

# Portfolio Risk Analytics Platform

## Project Overview

This project is a Python-based portfolio risk analytics platform that combines historical market data, portfolio math, and Monte Carlo simulation to estimate volatility, Value at Risk (VaR), and Expected Shortfall (ES).

## Setup & Installation

Python requirement: `Python 3.12` is recommended.

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI development server from the project root:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI docs:

```text
http://127.0.0.1:8000/docs
```

Redoc docs:

```text
http://127.0.0.1:8000/redoc
```

## Endpoints Reference

| Method | Route | Description | Auth Required |
| --- | --- | --- | --- |
| `GET` | `/` | Returns a welcome message and points users to the docs UI. | No |
| `GET` | `/health` | Returns API health status and version information. | No |
| `POST` | `/analyze` | Runs the full portfolio risk pipeline and returns summary risk metrics. | No |
| `POST` | `/simulate` | Runs Monte Carlo simulation and returns richer percentile and distribution statistics. | No |
| `GET` | `/sample-portfolios` | Returns the bundled sample portfolios from `data/sample_portfolios.json`. | No |

## Example Requests

### POST `/analyze`

`curl` request using the balanced ETF sample portfolio:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
    "weights": [0.25, 0.25, 0.30, 0.20],
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
    "confidence_level": 0.95,
    "simulations": 10000,
    "horizon_days": 1,
    "random_seed": 42
  }'
```

Equivalent JSON body:

```json
{
  "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
  "weights": [0.25, 0.25, 0.30, 0.20],
  "start_date": "2021-01-01",
  "end_date": "2026-01-01",
  "confidence_level": 0.95,
  "simulations": 10000,
  "horizon_days": 1,
  "random_seed": 42
}
```

### POST `/simulate`

`curl` request using the tech heavy sample portfolio:

```bash
curl -X POST "http://127.0.0.1:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "weights": [0.30, 0.30, 0.20, 0.20],
    "start_date": "2021-01-01",
    "end_date": "2026-01-01",
    "confidence_level": 0.99,
    "simulations": 10000,
    "horizon_days": 1,
    "random_seed": 42
  }'
```

Equivalent JSON body:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
  "weights": [0.30, 0.30, 0.20, 0.20],
  "start_date": "2021-01-01",
  "end_date": "2026-01-01",
  "confidence_level": 0.99,
  "simulations": 10000,
  "horizon_days": 1,
  "random_seed": 42
}
```

## Example Response

Example JSON response for `POST /analyze`:

```json
{
  "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
  "weights": [0.25, 0.25, 0.30, 0.20],
  "mean_daily_return": 0.0007,
  "annualized_volatility": 0.1243,
  "var_95": 0.0182,
  "es_95": 0.0241,
  "var_99": 0.0274,
  "es_99": 0.0347,
  "correlation": {
    "tickers": ["AAPL", "MSFT", "SPY", "GLD"],
    "matrix": [
      [1.0, 0.86, 0.78, 0.11],
      [0.86, 1.0, 0.76, 0.09],
      [0.78, 0.76, 1.0, 0.04],
      [0.11, 0.09, 0.04, 1.0]
    ]
  },
  "simulation_count": 10000,
  "horizon_days": 1,
  "random_seed": 42
}
```

## Error Reference

| HTTP Status | `error` Field Value | When It Happens |
| --- | --- | --- |
| `422` | `validation_error` | Request body fails API validation or route logic raises a `ValueError`. |
| `422` | `data_error` | A required data key is missing while building a response or processing data. |
| `500` | `internal_error` | A route-level catch-all exception occurs inside `/analyze` or `/simulate`. |
| `500` | `unexpected_error` | An uncaught exception reaches the global exception handler. |
| `503` | `data_unavailable` | The sample portfolio file is missing when `/sample-portfolios` is requested. |

## Running Tests

Run the test suite from the project root:

```bash
python -m pytest -q
```

Passing output should look like:

```bash
...........                                                              [100%]
11 passed in 0.51s
```

## Running the Full Application

Terminal 1 — Start the API backend:

```bash
 cd portfolio-risk-analytics
uvicorn app.api.main:app --reload
```

Terminal 2 — Start the Streamlit UI:

```bash
 cd portfolio-risk-analytics
streamlit run app/ui/streamlit_app.py
```

Both must be running at the same time. The UI calls the API on `http://localhost:8000`.

## UI Walkthrough

1. Open `http://localhost:8501` in your browser.
2. Optionally load a sample portfolio from the dropdown.
3. Enter tickers (one per line) and weights (one per line).
4. Choose date range, confidence level, and simulation count.
5. Click `Run Analysis`.
6. Review the summary metrics, charts, and percentile table.

## Screenshots

(Add screenshots here after running the app locally)

## Architecture Overview

```text
[User Browser]
     ↓
[Streamlit Cloud]
     ↓ HTTPS
[Render — FastAPI]
     ↓            ↓
[Supabase DB]  [Yahoo Finance]
```

## Troubleshooting

| Problem | Likely Cause | Fix |
| --- | --- | --- |
| UI shows API offline | FastAPI backend is not running or is running on a different port. | Start the backend with `uvicorn app.api.main:app --reload` and confirm it is reachable at `http://localhost:8000/health`. |
| Analysis takes very long | Large simulation count, slow market-data fetch, or temporary Yahoo Finance delay. | Retry with a smaller simulation count first, then confirm your network and yfinance availability. |
| Ticker not found error | One or more ticker symbols are invalid, delisted, or unsupported by Yahoo Finance. | Double-check the symbol spelling and try widely known NYSE/NASDAQ tickers first. |
| Weights validation error in UI | Weights do not sum to `1.0`, include invalid numbers, or do not match ticker count. | Fix the entries manually or enable auto-normalize in the sidebar. |
| Charts don't render | Plotly may not be installed in the active virtual environment. | Install Plotly with `pip install plotly`, then restart Streamlit. |

## Public Demo Safeguards

This project includes lightweight hardening measures to make the public demo more stable and professional without adding unnecessary infrastructure. The current safeguards include backend-side validation caps, endpoint rate limiting, lightweight in-memory caching, deployment-aware CORS restrictions, and request logging for diagnostics.

These protections are intentionally simple and appropriate for a student or personal portfolio project. They reduce the chance of accidental or abusive overuse, keep the app responsive during demos, and help surface operational issues clearly without introducing enterprise-only complexity.

## Why These Protections Exist

Monte Carlo simulation and historical market-data retrieval are the most expensive parts of the application. The demo intentionally caps simulation settings and request shapes to keep the user experience responsive, protect free-tier or student-run infrastructure, and prevent abuse of expensive endpoints.

## Current Public-Demo Limits

| Setting | Limit |
| --- | --- |
| Max tickers | 10 |
| Allowed simulations | 1000, 5000, 10000, 50000 |
| Horizon days | 1 |
| Minimum history window | 180 days |
| Maximum history window | 10 years |

## Rate Limit Examples

| Endpoint | Rate Limit |
| --- | --- |
| `GET /health` | 60 requests per minute per IP |
| `GET /` | 30 requests per minute per IP |
| `GET /sample-portfolios` | 20 requests per minute per IP |
| `POST /analyze` | 10 requests per minute per IP |
| `POST /simulate` | 5 requests per minute per IP |

## Future Improvements

Possible future upgrades for a more production-oriented deployment include:

- Redis-backed caching
- authentication for private or admin use
- external monitoring and alerting
- container deployment hardening

## Database Schema

### SavedPortfolio

| Column | Purpose |
| --- | --- |
| `id` | Primary key for the saved portfolio record |
| `name` | User-facing portfolio name |
| `tickers` | JSON array of ticker symbols |
| `weights` | JSON array of portfolio weights |
| `start_date` | Historical analysis window start date |
| `end_date` | Historical analysis window end date |
| `confidence_level` | Confidence level used for risk analysis |
| `simulations` | Monte Carlo simulation count |
| `created_at` | UTC timestamp when the portfolio was saved |
| `notes` | Optional user notes |

### AnalysisRun

| Column | Purpose |
| --- | --- |
| `id` | Primary key for the analysis run record |
| `portfolio_id` | Optional foreign key to `SavedPortfolio` |
| `tickers` | JSON array snapshot of analyzed ticker symbols |
| `weights` | JSON array snapshot of analyzed weights |
| `mean_daily_return` | Historical mean daily return |
| `annualized_volatility` | Annualized portfolio volatility |
| `var_95` | 95% Value at Risk |
| `es_95` | 95% Expected Shortfall |
| `var_99` | 99% Value at Risk |
| `es_99` | 99% Expected Shortfall |
| `simulation_count` | Number of simulations used |
| `ran_at` | UTC timestamp when the run was recorded |
| `duration_ms` | Optional runtime duration in milliseconds |

## Deployment Guide

1. Fork the repository to your own GitHub account.
2. Create a Supabase project and copy the `DATABASE_URL` connection string.
3. Deploy the FastAPI backend to Render and set `DATABASE_URL` in the Render environment settings.
4. Deploy the Streamlit frontend to Streamlit Community Cloud and set `api.base_url` to your Render URL in the Secrets UI.
5. Open the public Streamlit URL and verify the app works end to end.

## Environment Variables Reference

| Variable | Required | Description | Example |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes for persistence | Database connection string for local SQLite or Supabase PostgreSQL | `postgresql://postgres.project-ref:password@aws-1-us-east-1.pooler.supabase.com:5432/postgres` |
| `EXTRA_CORS_ORIGINS` | No | Comma-separated deployed frontend origins allowed to call the API from a browser | `https://your-app.streamlit.app,https://portfolio-risk-analytic.onrender.com` |
| `API_HOST` | No | Local API bind host used for development scripts and examples | `0.0.0.0` |
| `API_PORT` | No | Local API port used for development scripts and examples | `8000` |

## Running Locally (Full Stack)

Terminal 1 — API backend:

```bash
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

Terminal 2 — Streamlit UI:

```bash
source .venv/bin/activate
streamlit run app/ui/streamlit_app.py
```

Terminal 3 — tests (optional):

```bash
source .venv/bin/activate
pytest
```

## Running All Tests

Run the full test suite:

```bash
pytest
```

Run unit and API tests only, skipping the live end-to-end smoke test file:

```bash
pytest tests/test_market_data.py tests/test_portfolio_math.py tests/test_monte_carlo.py tests/test_api.py tests/test_hardening.py tests/test_db.py
```

Expected passing summary:

```bash
============================= test session starts ==============================
collected ... items

... passed ...

============================== all tests passed ===============================
```

## Project Roadmap

Potential next steps for the platform include:

- User authentication with FastAPI Users or Supabase Auth
- Redis caching for market data and repeated analyses
- PDF report export for portfolio risk summaries
- Portfolio comparison view across saved allocations
- Docker containerization for reproducible local and cloud environments
- GitHub Actions CI/CD pipeline for tests and deployments
