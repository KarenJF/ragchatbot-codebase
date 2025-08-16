#!/bin/bash

# Complete code quality script for RAG system
# Formats code and runs all quality checks

set -e

echo "🚀 Starting code quality checks..."

echo ""
echo "📝 Step 1: Format code..."
./scripts/format.sh

echo ""
echo "🔍 Step 2: Run basic linting checks..."
./scripts/lint-basic.sh

echo ""
echo "🎉 All code quality checks completed successfully!"