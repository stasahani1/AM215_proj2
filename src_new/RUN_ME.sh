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
echo "3. Run OPTIMAL test - 6-week + mixed (~7 minutes) ⭐ RECOMMENDED"
echo "4. Run OPTIMAL full evaluation (~35 minutes) ⭐⭐ BEST"
echo ""
echo "SECTOR-SPECIFIC TESTS (NEW!):"
echo "5. Tech sector only (10 tech stocks, ~8 minutes) 🔬"
echo "6. Finance sector only (10 finance stocks, ~8 minutes) 💰"
echo "7. Compare sectors (runs both #5 and #6, ~15 minutes) 📊"
echo ""
read -p "Enter choice (1-7): " choice

echo ""

if [ "$choice" = "1" ]; then
    echo "Running quick test..."
    python quick_test.py
elif [ "$choice" = "2" ]; then
    echo "Running full evaluation..."
    python run_evaluation.py
elif [ "$choice" = "3" ]; then
    echo "Running OPTIMAL quick test (6-week chunks + mixed training)..."
    python quick_test_optimal.py
elif [ "$choice" = "4" ]; then
    echo "Running OPTIMAL full evaluation..."
    python run_evaluation_optimal.py
elif [ "$choice" = "5" ]; then
    echo "Running TECH SECTOR test..."
    echo "Hypothesis: Contagion should be strong within tech sector"
    python quick_test_tech_sector.py
elif [ "$choice" = "6" ]; then
    echo "Running FINANCE SECTOR test..."
    echo "Hypothesis: Financial contagion (2008 crisis) should show highest β"
    python quick_test_finance_sector.py
elif [ "$choice" = "7" ]; then
    echo "Running SECTOR COMPARISON..."
    echo "This will compare contagion dynamics across sectors"
    python compare_sectors.py
else
    echo "Invalid choice. Please run again and choose 1-7."
fi

echo ""
echo "Done!"

