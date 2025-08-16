#!/bin/bash

# Linting script for RAG system
# Runs all code quality checks

set -e

echo "🔍 Running flake8..."
uv run flake8 backend/ main.py

echo "🔧 Running mypy type checking..."
uv run mypy backend/ main.py

echo "📊 Running import order check..."
uv run isort --check-only backend/ main.py

echo "🎨 Running black format check..."
uv run black --check backend/ main.py

echo "✅ All linting checks passed!"