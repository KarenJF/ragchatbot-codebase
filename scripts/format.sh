#!/bin/bash

# Code formatting script for RAG system
# Formats code using black and sorts imports using isort

set -e

echo "🎨 Formatting code with black..."
uv run black backend/ main.py

echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"