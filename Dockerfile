# Dockerfile for LLM-Driven OpenFOAM Orchestration System
# Based on design document Section 9.1
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: OpenFOAM Base
# ============================================================================
FROM ubuntu:22.04 AS openfoam-base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    software-properties-common \
    build-essential \
    cmake \
    git \
    flex \
    bison \
    zlib1g-dev \
    libboost-all-dev \
    libopenmpi-dev \
    openmpi-bin \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install OpenFOAM v2312
RUN wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash && \
    apt-get update && \
    apt-get install -y openfoam2312 && \
    rm -rf /var/lib/apt/lists/*

# Source OpenFOAM in bashrc
RUN echo "source /usr/lib/openfoam/openfoam2312/etc/bashrc" >> /etc/bash.bashrc

# ============================================================================
# Stage 2: Python Orchestration Layer
# ============================================================================
FROM openfoam-base AS orchestration

# Install Python 3.10+
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g 1000 cfduser && \
    useradd -u 1000 -g cfduser -m -s /bin/bash cfduser

# Set up working directory
WORKDIR /app

# Install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=cfduser:cfduser . /app/

# Create necessary directories
RUN mkdir -p /data/cases /data/results /app/templates /app/logs && \
    chown -R cfduser:cfduser /data /app

# Set up templates directory
RUN mkdir -p /app/templates/shared/turbulence_models

# Switch to non-root user
USER cfduser

# Environment variables
ENV PYTHONPATH=/app
ENV FOAM_TEMPLATES=/app/templates
ENV FOAM_WORK_DIR=/data/cases
ENV FOAM_RESULTS_DIR=/data/results
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 3: Execution Sandbox (for running simulations)
# ============================================================================
FROM openfoam-base AS sandbox

# Minimal Python for execution monitoring
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 cfduser && \
    useradd -u 1000 -g cfduser -m -s /bin/bash cfduser

# Work directory for cases
WORKDIR /case

# Switch to non-root user
USER cfduser

# Source OpenFOAM on startup
SHELL ["/bin/bash", "-c"]
RUN echo "source /usr/lib/openfoam/openfoam2312/etc/bashrc" >> ~/.bashrc

# Default to running a solver (overridden by orchestration layer)
ENTRYPOINT ["/bin/bash", "-c", "source /usr/lib/openfoam/openfoam2312/etc/bashrc && exec \"$@\"", "--"]
CMD ["simpleFoam"]
