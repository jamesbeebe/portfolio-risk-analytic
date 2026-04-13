# Portfolio Risk Analytics Platform

## Project Overview

This project is a Python-based portfolio risk analytics platform that combines historical market data, portfolio math, and Monte Carlo simulation to estimate volatility, Value at Risk (VaR), and Expected Shortfall (ES). Phase 2 adds a FastAPI backend so these analytics can be accessed through a documented HTTP API and consumed by future frontend or client applications.

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
