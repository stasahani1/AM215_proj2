#!/bin/bash

# SIR Contagion Model - Evaluation Script
# Provides an interactive menu to run various analyses

echo "=========================================="
echo "SIR Contagion Model Evaluation"
echo "Stock Price Prediction with Network Effects"
echo "=========================================="
echo ""

# Navigate to project root
PROJECT_ROOT="/Users/praghav/Desktop/am215/AM215_proj2"
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Please create one first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Navigate to src directory
cd src

echo ""
echo "=========================================="
echo "Choose an analysis to run:"
echo "=========================================="
echo ""
echo "⭐ RECOMMENDED (Quick Tests - 5-10 minutes):"
echo "  1. Quick Test (Optimal) - 6-week chunks + mixed training"
echo "  2. Finance Sector - Best results! (2008 + COVID crises)"
echo "  3. Tech Sector - Compare contagion across sectors"
echo ""
echo "📊 COMPREHENSIVE (Full Evaluations - 20-40 minutes):"
echo "  4. Full Evaluation (Optimal) - Complete analysis"
echo "  5. Sector Comparison - Finance vs Tech with visualizations"
echo ""
echo "🔬 ANALYSIS TOOLS:"
echo "  6. Window Size Comparison - Test different time windows"
echo "  7. Generate LaTeX Tables - For paper/report"
echo "  8. Visualize Networks - Correlation & contagion networks"
echo ""
echo "  0. Exit"
echo ""
read -p "Enter choice (0-8): " choice

echo ""
echo "=========================================="

case $choice in
    1)
        echo "Running: Quick Test (Optimal Settings)"
        echo "Configuration: 6-week chunks, mixed training, 6 stocks"
        echo "Expected runtime: ~5-8 minutes"
        echo "=========================================="
        python scripts/quick_test_optimal.py
        ;;
    2)
        echo "Running: Finance Sector Test"
        echo "Stocks: JPM, BAC, WFC, GS, MS, C, BLK, SCHW, AXP, USB"
        echo "Expected runtime: ~8-10 minutes"
        echo "Hypothesis: Strong contagion during financial crises"
        echo "=========================================="
        python scripts/quick_test_finance_sector.py
        ;;
    3)
        echo "Running: Tech Sector Test"
        echo "Stocks: AAPL, MSFT, GOOGL, NVDA, META, TSLA, AVGO, ORCL, ADBE, CRM"
        echo "Expected runtime: ~8-10 minutes"
        echo "Hypothesis: Moderate contagion within tech"
        echo "=========================================="
        python scripts/quick_test_tech_sector.py
        ;;
    4)
        echo "Running: Full Evaluation (Optimal)"
        echo "Configuration: 15 years data, 50 chunks, 500 simulations"
        echo "Expected runtime: ~30-40 minutes"
        echo "Will generate comprehensive visualizations"
        echo "=========================================="
        python scripts/run_evaluation_optimal.py
        ;;
    5)
        echo "Running: Sector Comparison with Visualizations"
        echo "Compares Finance vs Tech sectors"
        echo "Expected runtime: ~15-20 minutes"
        echo "Generates comparative plots and metrics"
        echo "=========================================="
        python scripts/compare_sectors_with_viz.py
        ;;
    6)
        echo "Running: Window Size Comparison"
        echo "Tests different time window configurations"
        echo "Expected runtime: ~20-25 minutes"
        echo "=========================================="
        python scripts/compare_window_sizes.py
        ;;
    7)
        echo "Generating LaTeX Tables"
        echo "Creates publication-ready tables"
        echo "Output: project_results/latex_tables/"
        echo "=========================================="
        python scripts/generate_latex_tables.py
        ;;
    8)
        echo "Visualizing Networks"
        echo "Creates correlation and adjacency matrices"
        echo "=========================================="
        python scripts/visualize_networks.py
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run again and choose 0-8."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  • results_optimal/ - Main results"
echo "  • results_sector_comparison/ - Sector analysis"
echo "  • project_results/ - Publication-ready outputs"
echo ""
echo "Next steps:"
echo "  • View generated PNG files for visualizations"
echo "  • Check CSV files for detailed metrics"
echo "  • See console output above for summary statistics"
echo ""
echo "For more info, see: ../README.md"
echo "=========================================="
