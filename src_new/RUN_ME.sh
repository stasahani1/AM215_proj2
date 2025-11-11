#!/bin/bash

# Activation script for SIR Contagion Model Evaluation
# This script activates the virtual environment and runs the evaluation

echo "=========================================="
echo "SIR Contagion Model Evaluation"
echo "=========================================="
echo ""

# Navigate to project root
cd /Users/praghav/Desktop/am215/AM215_proj2

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Navigate to src_new
cd src_new

echo ""
echo "=========================================="
echo "Choose an option:"
echo "=========================================="
echo "1. Run quick test (3 stocks, ~5 minutes)"
echo "2. Run full evaluation (10 stocks, ~30 minutes)"
echo ""
read -p "Enter choice (1 or 2): " choice

echo ""

if [ "$choice" = "1" ]; then
    echo "Running quick test..."
    python quick_test.py
elif [ "$choice" = "2" ]; then
    echo "Running full evaluation..."
    python run_evaluation.py
else
    echo "Invalid choice. Please run again and choose 1 or 2."
fi

echo ""
echo "Done!"

