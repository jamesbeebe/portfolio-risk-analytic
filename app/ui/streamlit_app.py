from __future__ import annotations

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
