# Use Python 3.12 slim as base image
FROM python:3.12-slim AS uv

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Prefer the system python
ENV UV_PYTHON_PREFERENCE=only-system

# Run without updating the uv.lock file like running with `--frozen`
ENV UV_FROZEN=true

# Copy the required files first
COPY pyproject.toml uv.lock uv-requirements.txt ./

# Python optimization and uv configuration
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and Python package manager
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    cargo \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Install the project's dependencies using the lockfile and settings
# Skip onnxruntime installation for Alpine compatibility
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --requirement uv-requirements.txt --no-cache-dir && \
    uv sync --python 3.12 --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.12 --frozen --no-dev --no-editable

# Make the directory just in case it doesn't exist
RUN mkdir -p /root/.local

# Final stage - runtime image
FROM python:3.12-slim

# Place executables in the environment at the front of the path and include other binaries
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Install runtime dependencies and create application user
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates && \
    groupadd -r app && \
    useradd -r -g app -d /app -s /bin/bash app

# Copy application artifacts from build stage
COPY --from=uv --chown=app:app /app/.venv /app/.venv
COPY --from=uv --chown=app:app /app/aws_orchestrator_agent /app/aws_orchestrator_agent

# Copy agent card for A2A protocol
COPY --from=uv --chown=app:app /app/aws_orchestrator_agent/card /app/aws_orchestrator_agent/card

# Get healthcheck script
COPY ./docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh
RUN chmod +x /usr/local/bin/docker-healthcheck.sh

# Run as non-root
USER app

# Expose port
EXPOSE 10102

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 CMD ["docker-healthcheck.sh"]

# Set working directory
WORKDIR /app

# Default arguments for A2A server
ENV A2A_HOST=0.0.0.0
ENV A2A_PORT=10102
ENV A2A_AGENT_CARD=aws_orchestrator_agent/card/aws_orchestrator_agent.json

# When running the container, the entrypoint will be the aws-orchestrator-agent command
ENTRYPOINT ["aws-orchestrator-agent", "--host", "0.0.0.0", "--port", "10102", "--agent-card", "aws_orchestrator_agent/card/aws_orchestrator_agent.json"]
