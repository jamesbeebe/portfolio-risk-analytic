from __future__ import annotations

from datetime import date
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# `st.secrets` is Streamlit's built-in secret/config store. On Streamlit Cloud,
# values are supplied through the Secrets UI instead of a committed file. We
# fall back to localhost when secrets are missing so the same app works for both
# local development and the deployed frontend without changing source code.
try:
    API_BASE_URL = st.secrets["api"]["base_url"]
except (KeyError, FileNotFoundError, StreamlitSecretNotFoundError):
    API_BASE_URL = "http://localhost:8000"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SIMULATIONS = 10000
HISTOGRAM_RECONSTRUCTION_POINTS = 10000
SECTION_DIVIDER = "---"
REQUEST_TIMEOUT_SECONDS = 10


def apply_portfolio_to_sidebar(portfolio: dict) -> None:
    """Load a portfolio-like dictionary into the sidebar widget state.

    Args:
        portfolio: Dictionary containing ticker, weight, and settings fields.

    Returns:
        None. The function mutates Streamlit session state directly.
    """

    st.session_state["sidebar_tickers"] = "\n".join(portfolio.get("tickers", []))
    st.session_state["sidebar_weights"] = "\n".join(
        str(weight) for weight in portfolio.get("weights", [])
    )
    st.session_state["sidebar_start_date"] = date.fromisoformat(
        portfolio.get("start_date", DEFAULT_START_DATE)
    )
    st.session_state["sidebar_end_date"] = date.fromisoformat(
        portfolio.get("end_date", DEFAULT_END_DATE)
    )
    st.session_state["sidebar_confidence_level"] = float(
        portfolio.get("confidence_level", DEFAULT_CONFIDENCE)
    )
    st.session_state["sidebar_simulations"] = int(
        portfolio.get("simulations", DEFAULT_SIMULATIONS)
    )
    st.session_state["sidebar_random_seed"] = int(portfolio.get("random_seed", 42))


def _extract_error_detail(response: requests.Response) -> str:
    """Convert an API error response into a user-friendly message.

    Args:
        response: HTTP response returned by the backend.

    Returns:
        A human-readable error message extracted from the JSON body when possible.
    """

    try:
        body = response.json()
    except ValueError:
        return "The API returned an unreadable error response."

    detail = body.get("detail")
    if isinstance(detail, str):
        return detail

    if isinstance(detail, list) and detail:
        first_error = detail[0]
        if isinstance(first_error, dict):
            return str(first_error.get("msg", "The request data was invalid."))

    if isinstance(detail, dict):
        return str(detail)

    if "error" in body and "detail" in body:
        if body["error"] == "rate_limit_exceeded":
            return "Too many requests — please wait a moment before trying again."
        return str(body["detail"])

    return "The API returned an unknown error."


def check_api_health() -> tuple[bool, str]:
    """Check whether the FastAPI backend is reachable and healthy.

    Returns:
        A tuple of `(is_healthy, message)` where `is_healthy` is `True` only when
        the backend responds successfully, and `message` explains the result.
    """

    # Free-tier Render services can sleep when idle, so the first health check
    # may fail during cold start. Retrying here hides that transient wake-up
    # delay from users and avoids treating a recoverable startup lag as an error.
    last_error_message = (
        f"Unable to connect to the Risk API at {API_BASE_URL}. "
        "Start the FastAPI backend and refresh this page."
    )

    for attempt in range(1, 4):
        try:
            response = requests.get(
                f"{API_BASE_URL}/health",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            response = None
        else:
            if response.status_code == 200:
                return True, "API is online"
            last_error_message = (
                f"The Risk API returned status code {response.status_code}."
            )

        if attempt < 3:
            with st.spinner(
                "⏳ Waking up the API server — this may take up to 30 seconds on first load..."
            ):
                st.warning(
                    "⏳ Waking up the API server — this may take up to 30 seconds on first load..."
                )
                time.sleep(10)

    return False, last_error_message


def fetch_sample_portfolios() -> tuple[list | None, str | None]:
    """Fetch the sample portfolio definitions from the backend API.

    Returns:
        A tuple of `(portfolios, error_message)` where `portfolios` is a list on
        success and `error_message` is populated on failure.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/sample-portfolios",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return (
            None,
            "Unable to load sample portfolios because the Risk API is unreachable.",
        )

    if response.status_code == 200:
        body = response.json()
        return body.get("portfolios", []), None

    return None, _extract_error_detail(response)


def fetch_saved_portfolios() -> tuple[list | None, str | None]:
    """Fetch saved portfolios from the persistence API.

    Returns:
        A tuple of `(portfolios, error_message)` where `portfolios` is a list on
        success and `error_message` is populated on failure.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/portfolios",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None, "Unable to load saved portfolios because the Risk API is unreachable."

    if response.status_code == 200:
        body = response.json()
        return body.get("portfolios", []), None

    return None, _extract_error_detail(response)


def save_portfolio_to_api(
    name: str,
    portfolio: dict,
    notes: str | None,
) -> tuple[dict | None, str | None]:
    """Persist a named portfolio preset through the backend API.

    Args:
        name: User-facing name for the saved portfolio.
        portfolio: JSON-serializable portfolio payload to save.
        notes: Optional free-text note.

    Returns:
        A tuple of `(saved_portfolio, error_message)` after the API request.
    """

    try:
        response = requests.post(
            f"{API_BASE_URL}/portfolios/save",
            json={"name": name, "notes": notes, "portfolio": portfolio},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None, "Unable to save the portfolio because the Risk API is unreachable."

    if response.status_code == 200:
        return response.json(), None

    return None, _extract_error_detail(response)


def delete_portfolio_from_api(portfolio_id: int) -> tuple[bool, str | None]:
    """Delete a saved portfolio through the backend API.

    Args:
        portfolio_id: Database ID of the saved portfolio to delete.

    Returns:
        A tuple of `(deleted, error_message)` indicating the outcome.
    """

    try:
        response = requests.delete(
            f"{API_BASE_URL}/portfolios/{portfolio_id}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False, "Unable to delete the portfolio because the Risk API is unreachable."

    if response.status_code == 200:
        return True, None

    return False, _extract_error_detail(response)


def fetch_analysis_history() -> tuple[list | None, str | None]:
    """Fetch recent persisted analysis history rows from the backend API.

    Returns:
        A tuple of `(runs, error_message)` where `runs` is a list on success.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/history",
            params={"limit": 20},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return None, "Unable to load analysis history because the Risk API is unreachable."

    if response.status_code == 200:
        body = response.json()
        return body.get("runs", []), None

    return None, _extract_error_detail(response)


def call_analyze(payload: dict) -> tuple[dict | None, str | None]:
    """Send a portfolio analysis request to the backend API.

    Args:
        payload: JSON-serializable request body for the `/analyze` endpoint.

    Returns:
        A tuple of `(response_dict, error_message)` where `response_dict` contains
        the API result on success and `error_message` is populated on failure.
    """

    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return (
            None,
            "Unable to reach the analysis engine. Confirm the FastAPI backend is running.",
        )

    if response.status_code == 200:
        return response.json(), None

    if response.status_code == 429:
        return None, "Too many requests — please wait a moment before trying again."

    if response.status_code == 422:
        return None, _extract_error_detail(response)

    if response.status_code >= 500:
        return None, "The analysis engine encountered an error."

    return None, _extract_error_detail(response)


def call_simulate(payload: dict) -> tuple[dict | None, str | None]:
    """Send a Monte Carlo simulation request to the backend API.

    Args:
        payload: JSON-serializable request body for the `/simulate` endpoint.

    Returns:
        A tuple of `(response_dict, error_message)` where `response_dict` contains
        the API result on success and `error_message` is populated on failure.
    """

    try:
        response = requests.post(
            f"{API_BASE_URL}/simulate",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return (
            None,
            "Unable to reach the simulation engine. Confirm the FastAPI backend is running.",
        )

    if response.status_code == 200:
        return response.json(), None

    if response.status_code == 429:
        return None, "Too many requests — please wait a moment before trying again."

    if response.status_code == 422:
        return None, _extract_error_detail(response)

    if response.status_code >= 500:
        return None, "The analysis engine encountered an error."

    return None, _extract_error_detail(response)


def parse_tickers(raw: str) -> list[str]:
    """Parse newline-delimited ticker input into a cleaned ticker list.

    Args:
        raw: Raw multiline text entered by the user.

    Returns:
        A list of uppercase ticker symbols with blank lines removed.
    """

    return [line.strip().upper() for line in raw.splitlines() if line.strip()]


def parse_weights(raw: str) -> tuple[list[float] | None, str | None]:
    """Parse newline-delimited weight input into a list of floats.

    Args:
        raw: Raw multiline text entered by the user.

    Returns:
        A tuple of `(weights, error_message)` where `weights` is a parsed float
        list on success and `error_message` is populated on failure.
    """

    cleaned_lines = [line.strip() for line in raw.splitlines() if line.strip()]

    try:
        parsed_weights = [float(line) for line in cleaned_lines]
    except ValueError:
        return None, "Could not parse weights — make sure each line is a number"

    return parsed_weights, None


def validate_and_build_payload(
    tickers: list[str],
    weights: list[float] | None,
    weight_parse_error: str | None,
    start_date: str,
    end_date: str,
    confidence_level: float,
    simulations: int,
    horizon_days: int,
    random_seed: int,
    auto_normalize: bool,
) -> tuple[dict | None, list[str]]:
    """Validate parsed form inputs and build an API payload when valid.

    Args:
        tickers: Cleaned ticker symbols entered by the user.
        weights: Parsed portfolio weights, or None if parsing failed.
        weight_parse_error: Parsing error message from the weights text area.
        start_date: Analysis start date in ISO format.
        end_date: Analysis end date in ISO format.
        confidence_level: Requested confidence level for risk metrics.
        simulations: Number of Monte Carlo simulation paths.
        horizon_days: Requested simulation horizon in trading days.
        random_seed: Seed used to make simulation output reproducible.
        auto_normalize: Whether the UI should auto-scale weights to sum to 1.0.

    Returns:
        A tuple of `(payload, errors)` where `payload` is a JSON-ready dict on
        success and `errors` contains every validation problem found.
    """

    errors: list[str] = []

    if weight_parse_error is not None:
        errors.append(weight_parse_error)

    if len(tickers) == 0:
        errors.append("Please enter at least one ticker symbol")

    if weights is not None and len(tickers) != len(weights):
        errors.append(
            f"Number of tickers ({len(tickers)}) does not match number of weights ({len(weights)})"
        )

    duplicate_tickers = sorted({ticker for ticker in tickers if tickers.count(ticker) > 1})
    if duplicate_tickers:
        errors.append(f"Duplicate tickers found: {', '.join(duplicate_tickers)}")

    start_date_obj = date.fromisoformat(start_date)
    end_date_obj = date.fromisoformat(end_date)

    if start_date_obj >= end_date_obj:
        errors.append("Start date must be before end date")

    if (end_date_obj - start_date_obj).days < 180:
        errors.append("Date range must be at least 6 months for reliable analysis")

    if weights is not None and any(weight <= 0 for weight in weights):
        errors.append("All weights must be positive numbers")

    if weights is not None and auto_normalize:
        total_weight = float(sum(weights))
        if total_weight <= 0:
            errors.append("Weights must sum to a positive value before normalization")
        else:
            normalized_weights = [weight / total_weight for weight in weights]
            st.info(f"Weights normalized from {total_weight:.3f} to 1.000")
            weights = normalized_weights

    if weights is not None and not auto_normalize:
        total_weight = float(sum(weights))
        if abs(total_weight - 1.0) > 0.001:
            errors.append(
                f"Weights sum to {total_weight:.4f} — must equal 1.0. Enable auto-normalize or adjust your weights."
            )

    if errors:
        return None, errors

    payload = {
        "tickers": tickers,
        "weights": weights,
        "start_date": start_date,
        "end_date": end_date,
        "confidence_level": confidence_level,
        "simulations": simulations,
        "horizon_days": horizon_days,
        "random_seed": random_seed,
    }
    return payload, []


def render_summary_metrics(result: dict, payload: dict) -> None:
    """Render the top-level portfolio summary metrics and composition table.

    Args:
        result: Analyze endpoint response payload stored in session state.
        payload: Last successfully submitted request payload.

    Returns:
        None. The function renders Streamlit components directly.
    """

    st.subheader("📈 Portfolio Summary")
    st.write(f"Analysis for: {' · '.join(payload['tickers'])}")

    mean_daily_return = float(result["mean_daily_return"])
    annualized_volatility = float(result["annualized_volatility"])
    var_95 = float(result["var_95"])
    es_95 = float(result["es_95"])
    var_99 = float(result["var_99"])
    es_99 = float(result["es_99"])
    confidence_level = float(payload["confidence_level"])
    confidence_pct = confidence_level * 100

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Mean Daily Return",
        f"{mean_daily_return:+.3%}",
        delta=f"{mean_daily_return:+.3%}",
        delta_color="normal",
    )
    metric_columns[1].metric(
        "Annualized Volatility",
        f"{annualized_volatility:.1%}",
    )
    metric_columns[2].metric(
        "Value at Risk (95%)",
        f"{var_95:.1%}",
        help="The estimated maximum daily loss at 95% confidence",
    )
    metric_columns[3].metric(
        "Expected Shortfall (95%)",
        f"{es_95:.1%}",
        help="Average loss in the worst 5% of scenarios",
    )

    st.info(
        "📘 How to read these results:\n"
        f"At {confidence_pct:.0f}% confidence, the estimated one-day VaR is {var_95:.1%}, "
        f"meaning losses larger than this are expected on roughly {100 - confidence_pct:.0f}% "
        "of trading days under the model's assumptions. "
        f"The Expected Shortfall of {es_95:.1%} shows the average loss on those worst-case days. "
        f"Annualized volatility of {annualized_volatility:.1%} reflects the overall return "
        "variability of this portfolio over a year."
    )

    tail_columns = st.columns(2)
    tail_columns[0].metric("VaR (99%)", f"{var_99:.1%}")
    tail_columns[1].metric("ES (99%)", f"{es_99:.1%}")
    st.caption(
        "99% figures represent more extreme but less frequent tail scenarios."
    )

    st.subheader("Portfolio Composition")
    weights_df = pd.DataFrame(
        {
            "Ticker": payload["tickers"],
            "Weight": payload["weights"],
            "Weight %": [f"{float(weight):.1%}" for weight in payload["weights"]],
        }
    )
    st.dataframe(weights_df, use_container_width=True, hide_index=True)


def render_api_error(error_message: str, endpoint: str) -> None:
    """Render a consistent API error state with troubleshooting guidance.

    Args:
        error_message: Human-readable error text to show the user.
        endpoint: Backend endpoint name associated with the failure.

    Returns:
        None. The function renders Streamlit components directly.
    """

    st.error(error_message)
    with st.expander("🔧 Troubleshooting"):
        st.markdown(
            "1. Make sure the FastAPI backend is running: "
            "`uvicorn app.api.main:app --reload`\n"
            "2. Check that your tickers are valid NYSE/NASDAQ symbols\n"
            "3. If the date range is very recent, yfinance data may have a short delay\n"
            f"4. Endpoint that failed: {endpoint}"
        )

    if st.button("↩ Go back and edit inputs"):
        for key in ["analyze_result", "simulate_result", "last_payload"]:
            st.session_state.pop(key, None)
        st.rerun()


def render_correlation_matrix(result: dict) -> None:
    """Render the portfolio correlation matrix as a Plotly heatmap.

    Args:
        result: Analyze endpoint response payload containing correlation data.

    Returns:
        None. The function renders Streamlit components directly.
    """

    correlation = result["correlation"]
    tickers = correlation["tickers"]
    matrix = correlation["matrix"]
    correlation_df = pd.DataFrame(matrix, index=tickers, columns=tickers)

    heatmap = go.Heatmap(
        z=correlation_df.values,
        x=tickers,
        y=tickers,
        zmin=-1,
        zmax=1,
        colorscale="RdYlGn",
        text=np.round(correlation_df.values, 2),
        texttemplate="%{text:.2f}",
        textfont={"color": "black"},
        hovertemplate="Asset X: %{x}<br>Asset Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>",
    )
    figure = go.Figure(data=[heatmap])
    figure.update_layout(
        title="Asset Correlation Matrix",
        width=max(500, len(tickers) * 100),
        height=max(400, len(tickers) * 80),
        xaxis_title="",
        yaxis_title="",
    )
    figure.update_yaxes(autorange="reversed")

    st.plotly_chart(figure, use_container_width=True)
    st.info(
        "📘 Correlation ranges from -1 (assets move in opposite directions) to +1 "
        "(assets move together). A well-diversified portfolio ideally has low or "
        "negative correlations between assets."
    )


def render_simulation_histogram(sim_result: dict, confidence_level: float) -> None:
    """Render an approximate simulation histogram reconstructed from percentiles.

    Args:
        sim_result: Simulate endpoint response payload with percentile anchors.
        confidence_level: Selected confidence level used to place the VaR marker.

    Returns:
        None. The function renders Streamlit components directly.
    """

    percentiles = sim_result["percentiles"]
    percentile_positions = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    percentile_values = np.array(
        [
            percentiles["p1"],
            percentiles["p5"],
            percentiles["p10"],
            percentiles["p25"],
            percentiles["p50"],
            percentiles["p75"],
            percentiles["p90"],
            percentiles["p95"],
            percentiles["p99"],
        ],
        dtype=float,
    )

    reconstruction_points = HISTOGRAM_RECONSTRUCTION_POINTS
    synthetic_percentile_grid = np.linspace(1, 99, reconstruction_points)
    synthetic_returns = np.interp(
        synthetic_percentile_grid,
        percentile_positions,
        percentile_values,
    )
    synthetic_df = pd.DataFrame({"Portfolio Return": synthetic_returns})

    figure = px.histogram(
        synthetic_df,
        x="Portfolio Return",
        nbins=30,
        color_discrete_sequence=["#2c7fb8"],
        title=(
            "Simulated Portfolio Return Distribution"
            "<br><sup>(Approximate distribution — reconstructed from percentiles "
            f"using {reconstruction_points:,} synthetic points)</sup>"
        ),
        labels={"Portfolio Return": "Portfolio Return", "count": "Frequency"},
    )

    var_value = percentiles["p5"] if confidence_level == 0.95 else percentiles["p1"]
    figure.add_vline(
        x=var_value,
        line_dash="dash",
        line_color="red",
        annotation_text="VaR threshold",
        annotation_position="top left",
    )
    figure.update_layout(
        xaxis_title="Portfolio Return",
        yaxis_title="Frequency",
        bargap=0.05,
    )

    st.plotly_chart(figure, use_container_width=True)
    st.info(
        "📘 This histogram shows the distribution of simulated one-day portfolio "
        f"returns across {sim_result['simulation_count']:,} Monte Carlo scenarios. "
        f"The chart shape is reconstructed from percentile summaries using a fixed "
        f"{reconstruction_points:,}-point synthetic sample for consistent UI performance. "
        "The red dashed line marks the VaR threshold — returns to the left of this "
        "line represent the tail losses used to compute Expected Shortfall."
    )


def render_simulation_details(sim_result: dict, confidence_level: float) -> None:
    """Render detailed percentile and summary statistics for the simulation run.

    Args:
        sim_result: Simulate endpoint response payload with percentile statistics.
        confidence_level: Selected confidence level used to determine the VaR row.

    Returns:
        None. The function renders Streamlit components directly.
    """

    st.subheader("🎲 Simulation Details")
    st.write(
        f"Based on {sim_result['simulation_count']:,} Monte Carlo scenarios · "
        f"{sim_result['horizon_days']}-day horizon"
    )

    percentile_rows = [
        ("p1", sim_result["percentiles"]["p1"], "Worst 1% of scenarios"),
        ("p5", sim_result["percentiles"]["p5"], "Worst 5% of scenarios (VaR 95%)"),
        ("p10", sim_result["percentiles"]["p10"], "Worst 10% of scenarios"),
        ("p25", sim_result["percentiles"]["p25"], "Bottom quartile"),
        ("p50", sim_result["percentiles"]["p50"], "Median scenario"),
        ("p75", sim_result["percentiles"]["p75"], "Top quartile"),
        ("p90", sim_result["percentiles"]["p90"], "Best 10% of scenarios"),
        ("p95", sim_result["percentiles"]["p95"], "Best 5% of scenarios"),
        ("p99", sim_result["percentiles"]["p99"], "Best 1% of scenarios"),
    ]
    percentile_df = pd.DataFrame(
        percentile_rows,
        columns=["Percentile", "Return", "Interpretation"],
    )
    percentile_df["Return"] = percentile_df["Return"].map(lambda value: f"{float(value):+.2%}")

    highlighted_percentile = "p5" if confidence_level == 0.95 else "p1"

    def highlight_var_row(row: pd.Series) -> list[str]:
        """Apply a highlight style to the row matching the active VaR percentile.

        Args:
            row: A row from the percentile DataFrame.

        Returns:
            A list of CSS styles for each cell in the row.
        """

        if row["Percentile"] == highlighted_percentile:
            return ["background-color: #ffe5e5"] * len(row)
        return [""] * len(row)

    styled_percentile_df = percentile_df.style.apply(highlight_var_row, axis=1)
    st.dataframe(styled_percentile_df, use_container_width=True, hide_index=True)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Best Case", f"{float(sim_result['best_case']):+.2%}")
    summary_columns[1].metric("Worst Case", f"{float(sim_result['worst_case']):+.2%}")
    summary_columns[2].metric("Mean Return", f"{float(sim_result['mean_return']):+.2%}")
    summary_columns[3].metric("Std Dev", f"{float(sim_result['std_dev']):.2%}")

    confidence_pct = confidence_level * 100
    st.info(
        "📘 Reading the percentile table: each row shows the simulated return at "
        "that point in the distribution. For example, the p5 row means that in 5% "
        "of simulated scenarios, the portfolio lost at least that much in a single "
        f"day. The highlighted row corresponds to your selected confidence level of {confidence_pct:.0f}%."
    )


def render_analysis_history() -> None:
    """Render recent persisted analysis history and support row-based reloads.

    Returns:
        None. The function renders Streamlit components directly.
    """

    history_runs, history_error = fetch_analysis_history()

    if history_error:
        st.error(history_error)
        return

    if not history_runs:
        st.info("No analysis history yet. Run an analysis to get started.")
        return

    history_rows: list[dict[str, str | float | int | None]] = []
    for run in history_runs:
        tickers_display = " · ".join(run["tickers"])
        if len(tickers_display) > 40:
            tickers_display = f"{tickers_display[:37]}..."

        history_rows.append(
            {
                "Date/Time": pd.to_datetime(run["ran_at"]).strftime("%b %d, %Y %H:%M"),
                "Tickers": tickers_display,
                "Volatility": f"{float(run['annualized_volatility']):.1%}",
                "VaR 95%": f"{float(run['var_95']):.1%}",
                "ES 95%": f"{float(run['es_95']):.1%}",
                "Simulations": int(run["simulation_count"]),
            }
        )

    history_df = pd.DataFrame(history_rows)
    selection = st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        selected_run = history_runs[selected_rows[0]]
        st.session_state["history_selected_run"] = selected_run
        st.session_state["analyze_result"] = {
            "tickers": selected_run["tickers"],
            "weights": selected_run["weights"],
            "mean_daily_return": selected_run["mean_daily_return"],
            "annualized_volatility": selected_run["annualized_volatility"],
            "var_95": selected_run["var_95"],
            "es_95": selected_run["es_95"],
            "var_99": selected_run["var_99"],
            "es_99": selected_run["es_99"],
        }
        st.session_state["simulate_result"] = None
        st.session_state["last_payload"] = {
            "tickers": selected_run["tickers"],
            "weights": selected_run["weights"],
            "start_date": st.session_state["sidebar_start_date"].isoformat(),
            "end_date": st.session_state["sidebar_end_date"].isoformat(),
            "confidence_level": float(st.session_state["sidebar_confidence_level"]),
            "simulations": int(selected_run["simulation_count"]),
            "horizon_days": 1,
            "random_seed": int(st.session_state["sidebar_random_seed"]),
        }
        st.session_state["active_main_tab"] = 0
        st.rerun()

    st.caption("Showing last 20 analyses. Analyses are saved automatically.")


api_is_healthy, health_message = check_api_health()
if api_is_healthy:
    st.success("✓ Connected to Risk API")
else:
    st.error(health_message)
    # st.stop() prevents the rest of the UI from rendering when the backend is
    # unavailable, which avoids showing controls that cannot actually work.
    st.stop()

st.title("📊 Portfolio Risk Analyzer")
st.caption(
    "Monte Carlo Value at Risk · Expected Shortfall · Correlation Analysis"
)
st.info(
    "Demo note: The backend is hosted on Render's free tier and may sleep when "
    "idle. The first request can take around 30 seconds to wake the service; "
    "after that, responses should be normal."
)
st.divider()
st.caption(
    "To keep the public demo stable, analysis settings are intentionally capped."
)

st.session_state.setdefault("sidebar_tickers", "")
st.session_state.setdefault("sidebar_weights", "")
st.session_state.setdefault("sidebar_start_date", date.fromisoformat(DEFAULT_START_DATE))
st.session_state.setdefault("sidebar_end_date", date.fromisoformat(DEFAULT_END_DATE))
st.session_state.setdefault("sidebar_confidence_level", DEFAULT_CONFIDENCE)
st.session_state.setdefault("sidebar_simulations", DEFAULT_SIMULATIONS)
st.session_state.setdefault("sidebar_random_seed", 42)
st.session_state.setdefault("sidebar_auto_normalize", False)
st.session_state.setdefault("selected_sample_portfolio", "— build manually —")
st.session_state.setdefault("selected_saved_portfolio", "— select —")
st.session_state.setdefault("show_success_toast", False)
st.session_state.setdefault("analysis_in_progress", False)
st.session_state.setdefault("active_main_tab", 0)

saved_portfolios, saved_portfolios_error = fetch_saved_portfolios()
saved_portfolio_options = ["— select —"]
saved_portfolio_lookup: dict[str, dict] = {}
if saved_portfolios:
    for portfolio in saved_portfolios:
        saved_portfolio_options.append(portfolio["name"])
        saved_portfolio_lookup[portfolio["name"]] = portfolio

sample_portfolios, sample_portfolios_error = fetch_sample_portfolios()
sample_portfolio_options = ["— build manually —"]
sample_portfolio_lookup: dict[str, dict] = {}

if sample_portfolios:
    for index, portfolio in enumerate(sample_portfolios, start=1):
        portfolio_name = portfolio.get("name", f"Sample Portfolio {index}")
        sample_portfolio_options.append(portfolio_name)
        sample_portfolio_lookup[portfolio_name] = portfolio


with st.sidebar:
    st.header("Portfolio Inputs")

    if saved_portfolios_error:
        st.warning(saved_portfolios_error)
    elif saved_portfolios:
        st.subheader("📂 My Saved Portfolios")
        saved_selector_column, delete_button_column = st.columns([4, 1])
        with saved_selector_column:
            selected_saved = st.selectbox(
                "Load a saved portfolio",
                options=saved_portfolio_options,
                key="selected_saved_portfolio",
            )
        selected_saved_portfolio = saved_portfolio_lookup.get(selected_saved)
        with delete_button_column:
            delete_saved_clicked = st.button("🗑 Delete", key="delete_saved")

        if selected_saved_portfolio:
            apply_portfolio_to_sidebar(selected_saved_portfolio)
            if selected_saved_portfolio.get("notes"):
                st.caption(selected_saved_portfolio["notes"])

        if delete_saved_clicked and selected_saved_portfolio:
            deleted, delete_error = delete_portfolio_from_api(selected_saved_portfolio["id"])
            if deleted:
                st.session_state["selected_saved_portfolio"] = "— select —"
                st.rerun()
            elif delete_error:
                st.error(delete_error)

        st.markdown(SECTION_DIVIDER)

    st.subheader("Load a Sample Portfolio")
    st.caption("Select a preset to auto-fill the form below.")

    if sample_portfolios_error:
        st.warning(sample_portfolios_error)
    else:
        selected_sample = st.selectbox(
            "Load a sample portfolio (optional)",
            options=sample_portfolio_options,
            key="selected_sample_portfolio",
        )
        selected_portfolio = sample_portfolio_lookup.get(selected_sample)

        if selected_portfolio:
            apply_portfolio_to_sidebar(selected_portfolio)

    st.markdown(SECTION_DIVIDER)

    st.subheader("Ticker and Weight Entry")
    st.text_area(
        "Ticker symbols (one per line)",
        key="sidebar_tickers",
        placeholder="AAPL\nMSFT\nSPY\nGLD",
        help="Enter NYSE/NASDAQ symbols. One ticker per line, uppercase.",
        height=140,
    )
    st.text_area(
        "Weights (one per line, must sum to 1.0)",
        key="sidebar_weights",
        placeholder="0.25\n0.25\n0.30\n0.20",
        help="Decimal weights. Example: 0.25 means 25%. Must sum to 1.0.",
        height=140,
    )

    st.markdown(SECTION_DIVIDER)

    st.subheader("Analysis Settings")
    st.date_input(
        "Start Date",
        key="sidebar_start_date",
    )
    st.date_input(
        "End Date",
        key="sidebar_end_date",
    )
    st.slider(
        "Confidence Level",
        min_value=0.80,
        max_value=0.99,
        step=0.01,
        format="%.2f",
        key="sidebar_confidence_level",
    )
    st.select_slider(
        "Monte Carlo Simulations",
        options=[1000, 5000, 10000, 50000],
        key="sidebar_simulations",
        help="Allowed demo tiers: 1000, 5000, 10000, or 50000 simulations.",
    )
    st.caption("Demo safeguard: only approved simulation tiers are allowed.")
    st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        step=1,
        key="sidebar_random_seed",
        help="Fix the seed for reproducible results.",
    )

    st.markdown(SECTION_DIVIDER)

    st.subheader("Weight Normalizer")
    auto_normalize = st.checkbox(
        "Auto-normalize weights to sum to 1.0",
        key="sidebar_auto_normalize",
        help="If checked, your weights will be scaled automatically.",
    )

    st.markdown(SECTION_DIVIDER)

    run_clicked = st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=st.session_state["analysis_in_progress"],
    )
    if "last_payload" in st.session_state:
        st.divider()
        st.subheader("💾 Save This Portfolio")
        save_portfolio_name = st.text_input(
            "Portfolio name",
            key="save_portfolio_name",
            max_chars=50,
        )
        save_portfolio_notes = st.text_area(
            "Notes (optional)",
            key="save_portfolio_notes",
            height=80,
        )
        if st.button("Save Portfolio", use_container_width=True):
            if not save_portfolio_name.strip():
                st.error("Portfolio name is required.")
            else:
                saved_portfolio, save_error = save_portfolio_to_api(
                    name=save_portfolio_name.strip(),
                    portfolio=st.session_state["last_payload"],
                    notes=save_portfolio_notes.strip() or None,
                )
                if save_error:
                    st.error(save_error)
                elif saved_portfolio is not None:
                    st.success(f"✓ Portfolio saved as '{saved_portfolio['name']}'")
    st.divider()
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        for key in list(st.session_state.keys()):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption("Portfolio Risk Analyzer v0.1.0")
    st.caption("Built with FastAPI · Streamlit · Monte Carlo simulation")


if run_clicked:
    st.session_state["analysis_in_progress"] = True
    parsed_tickers = parse_tickers(st.session_state["sidebar_tickers"])
    parsed_weights, weight_error = parse_weights(st.session_state["sidebar_weights"])

    payload, validation_errors = validate_and_build_payload(
        tickers=parsed_tickers,
        weights=parsed_weights,
        weight_parse_error=weight_error,
        start_date=st.session_state["sidebar_start_date"].isoformat(),
        end_date=st.session_state["sidebar_end_date"].isoformat(),
        confidence_level=float(st.session_state["sidebar_confidence_level"]),
        simulations=int(st.session_state["sidebar_simulations"]),
        horizon_days=1,
        random_seed=int(st.session_state["sidebar_random_seed"]),
        auto_normalize=auto_normalize,
    )

    if validation_errors:
        st.session_state["analysis_in_progress"] = False
        for error_message in validation_errors:
            st.error(error_message)
        st.stop()

    with st.spinner("⏳ Fetching market data and running simulations..."):
        analyze_result, analyze_error = call_analyze(payload)
        simulate_result, simulate_error = call_simulate(payload)

    if analyze_error:
        st.session_state["analysis_in_progress"] = False
        render_api_error(analyze_error, "/analyze")
        st.stop()

    if simulate_error:
        st.session_state["analysis_in_progress"] = False
        render_api_error(simulate_error, "/simulate")
        st.stop()

    st.session_state["analyze_result"] = analyze_result
    st.session_state["simulate_result"] = simulate_result
    st.session_state["last_payload"] = payload
    st.session_state["show_success_toast"] = True
    st.session_state["analysis_in_progress"] = False


if (
    "analyze_result" in st.session_state
    and st.session_state["analyze_result"] is not None
    and "last_payload" in st.session_state
):
    current_analysis_tab, history_tab = st.tabs(["📊 Current Analysis", "🕐 Analysis History"])
    with current_analysis_tab:
        if st.session_state.get("history_selected_run") is not None:
            st.info(
                "Loaded a saved analysis from history. Summary metrics are available below. "
                "Run the portfolio again to regenerate charts and full simulation details."
            )

        render_summary_metrics(
            result=st.session_state["analyze_result"],
            payload=st.session_state["last_payload"],
        )
        if (
            st.session_state.get("simulate_result") is not None
            and "correlation" in st.session_state["analyze_result"]
        ):
            chart_columns = st.columns(2)
            with chart_columns[0]:
                render_correlation_matrix(st.session_state["analyze_result"])
            with chart_columns[1]:
                render_simulation_histogram(
                    st.session_state["simulate_result"],
                    float(st.session_state["last_payload"]["confidence_level"]),
                )
            render_simulation_details(
                st.session_state["simulate_result"],
                float(st.session_state["last_payload"]["confidence_level"]),
            )
        else:
            st.caption(
                "Detailed charts are only available for the current live analysis run."
            )

    with history_tab:
        render_analysis_history()

    if st.session_state.get("show_success_toast"):
        # A toast is preferable here because it confirms success without taking up
        # permanent layout space the way a full success banner would.
        st.toast("✅ Analysis complete!", icon="📊")
        st.session_state["show_success_toast"] = False
else:
    current_analysis_tab, history_tab = st.tabs(["📊 Current Analysis", "🕐 Analysis History"])
    with current_analysis_tab:
        st.info("👈 Enter your portfolio in the sidebar and click Run Analysis to get started.")
        with st.expander("📖 How this works"):
            st.markdown(
                "1. Enter ticker symbols and portfolio weights in the sidebar\n"
                "2. Choose your date range and analysis settings\n"
                "3. Click Run Analysis — the app will call the risk engine\n"
                "4. Results include volatility, Monte Carlo VaR, Expected Shortfall, and a full simulation distribution"
            )
    with history_tab:
        render_analysis_history()
