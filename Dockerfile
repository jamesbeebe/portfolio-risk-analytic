# Use the official Python 3.12 slim image as our base.
# "slim" means it excludes unnecessary system packages, keeping the image smaller.
FROM python:3.12-slim

# Set the working directory inside the container.
# All subsequent commands run from this path.
WORKDIR /app

# Copy only requirements first — this is a Docker caching trick:
# Docker caches each layer. If requirements.txt hasn't changed,
# Docker reuses the cached pip install layer and skips re-installing packages.
# This makes rebuilds much faster during development.
COPY requirements.txt .

# Install Python dependencies into the container image.
# --no-cache-dir saves space by not storing pip's download cache in the image.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code.
# This happens AFTER pip install so code changes don't invalidate the package cache.
COPY . .

# Tell Docker this container listens on port 8000.
# This is documentation only — it doesn't actually open the port by itself.
EXPOSE 8000

# Set environment variables with safe defaults.
# These can be overridden at runtime with docker run -e or docker-compose.
ENV DATABASE_URL=sqlite:///./portfolio_risk.db
# Bind the API to all interfaces inside the container.
ENV API_HOST=0.0.0.0
# Keep the internal container port consistent with the local development default.
ENV API_PORT=8000

# Run database migrations then start the server.
# Using shell form lets us chain both commands with && in one container entrypoint.
CMD alembic upgrade head && uvicorn app.api.main:app --host $API_HOST --port $API_PORT
