"""
Quick start script for LLM-OpenFOAM Orchestrator
"""
#!/bin/bash

set -e

echo "🚀 LLM-Driven OpenFOAM Orchestrator - Quick Start"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install --quiet -r requirements.txt

# Setup environment file
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
fi

# Create necessary directories
mkdir -p logs tmp/foam_simulations

# Run tests
echo "🧪 Running tests..."
pytest tests/ --tb=short -q

# Start server
echo "🎯 Starting API server..."
python main.py
