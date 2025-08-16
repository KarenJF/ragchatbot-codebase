#!/bin/bash

# Basic linting script for RAG system
# Runs essential code quality checks without strict typing

set -e

echo "🔍 Running flake8..."
uv run flake8 backend/ main.py

echo "📊 Checking import order..."
uv run isort --check-only backend/ main.py

echo "🎨 Checking code format..."
uv run black --check backend/ main.py

echo "✅ Basic linting checks passed!"