from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SIMULATIONS = 10000
HISTOGRAM_RECONSTRUCTION_POINTS = 10000
SECTION_DIVIDER = "---"
REQUEST_TIMEOUT_SECONDS = 10


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
        return f"{body['error']}: {body['detail']}"

    return "The API returned an unknown error."


def check_api_health() -> tuple[bool, str]:
    """Check whether the FastAPI backend is reachable and healthy.

    Returns:
        A tuple of `(is_healthy, message)` where `is_healthy` is `True` only when
        the backend responds successfully, and `message` explains the result.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return False, (
            "Unable to connect to the Risk API at "
            f"{API_BASE_URL}. Start the FastAPI backend and refresh this page."
        )

    if response.status_code == 200:
        return True, "API is online"

    return False, f"The Risk API returned status code {response.status_code}."


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
st.divider()

st.session_state.setdefault("sidebar_tickers", "")
st.session_state.setdefault("sidebar_weights", "")
st.session_state.setdefault("sidebar_start_date", date.fromisoformat(DEFAULT_START_DATE))
st.session_state.setdefault("sidebar_end_date", date.fromisoformat(DEFAULT_END_DATE))
st.session_state.setdefault("sidebar_confidence_level", DEFAULT_CONFIDENCE)
st.session_state.setdefault("sidebar_simulations", DEFAULT_SIMULATIONS)
st.session_state.setdefault("sidebar_random_seed", 42)
st.session_state.setdefault("sidebar_auto_normalize", False)
st.session_state.setdefault("selected_sample_portfolio", "— build manually —")

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
            st.session_state["sidebar_tickers"] = "\n".join(
                selected_portfolio.get("tickers", [])
            )
            st.session_state["sidebar_weights"] = "\n".join(
                str(weight) for weight in selected_portfolio.get("weights", [])
            )
            st.session_state["sidebar_start_date"] = date.fromisoformat(
                selected_portfolio.get("start_date", DEFAULT_START_DATE)
            )
            st.session_state["sidebar_end_date"] = date.fromisoformat(
                selected_portfolio.get("end_date", DEFAULT_END_DATE)
            )
            st.session_state["sidebar_confidence_level"] = float(
                selected_portfolio.get("confidence_level", DEFAULT_CONFIDENCE)
            )
            st.session_state["sidebar_simulations"] = int(
                selected_portfolio.get("simulations", DEFAULT_SIMULATIONS)
            )
            st.session_state["sidebar_random_seed"] = int(
                selected_portfolio.get("random_seed", 42)
            )

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
    )
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
    )


if run_clicked:
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
        for error_message in validation_errors:
            st.error(error_message)
        st.stop()

    with st.spinner("Running analysis... this may take a few seconds"):
        analyze_result, analyze_error = call_analyze(payload)
        simulate_result, simulate_error = call_simulate(payload)

    if analyze_error:
        st.error(analyze_error)
        st.stop()

    if simulate_error:
        st.error(simulate_error)
        st.stop()

    st.session_state["analyze_result"] = analyze_result
    st.session_state["simulate_result"] = simulate_result
    st.session_state["last_payload"] = payload


if (
    "analyze_result" in st.session_state
    and st.session_state["analyze_result"] is not None
    and "last_payload" in st.session_state
):
    render_summary_metrics(
        result=st.session_state["analyze_result"],
        payload=st.session_state["last_payload"],
    )
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
