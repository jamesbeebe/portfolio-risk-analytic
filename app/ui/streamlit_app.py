from __future__ import annotations

from datetime import date

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
