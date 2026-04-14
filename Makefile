# `.PHONY` tells `make` these names are commands, not real files. Without this,
# a file named `test` or `lint` could accidentally stop the target from running.
.PHONY: lint format format-check test test-unit test-all dev-api dev-ui dev

# Run Ruff lint checks across the repository.
lint:
	ruff check .

# Auto-format Python files with Ruff.
format:
	ruff format .

# Check whether formatting is already correct without changing files.
format-check:
	ruff format --check .

# Run the default pytest suite using pyproject.toml settings.
test:
	pytest

# Run unit-style tests while skipping end-to-end smoke coverage.
test-unit:
	pytest tests/ -k "not e2e and not smoke"

# Run the full test suite with explicit verbose pytest output.
test-all:
	pytest tests/ -v

# Start the FastAPI development server locally.
dev-api:
	uvicorn app.api.main:app --reload

# Start the Streamlit frontend locally.
dev-ui:
	streamlit run app/ui/streamlit_app.py

# Running API and UI together in one Make target is awkward because both are
# long-lived processes that each want to control the terminal. Two terminals is
# the clearest workflow for development, so this target prints guidance only.
dev:
	@echo "Run 'make dev-api' in one terminal and 'make dev-ui' in another."
