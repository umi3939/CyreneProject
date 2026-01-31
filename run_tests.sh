#!/usr/bin/env bash
set -euo pipefail

echo "Running Cyrene test suite..."
python -m pytest tests/ -v --tb=short
echo ""
echo "Running legacy psyche flow test..."
python test_psyche_flow.py
echo ""
echo "All tests completed."
