"""
Generate LaTeX formatted tables from the CSV results.
Creates publication-ready LaTeX table code.
"""

import pandas as pd
from pathlib import Path

results_dir = Path("project_results")
latex_dir = results_dir / "latex_tables"
latex_dir.mkdir(exist_ok=True)

print("="*80)
print("GENERATING LATEX TABLES")
print("="*80)

# ============================================================================
# TABLE 1: Overall Model Performance
# ============================================================================

print("\n1. Creating LaTeX for Overall Performance Table...")

df = pd.read_csv(results_dir / 'table1_overall_performance.csv')

latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Overall Model Performance Comparison (Finance Sector, 6-week windows)}
\label{tab:overall_performance}
\begin{tabular}{clrrrrrr}
\toprule
\textbf{Rank} & \textbf{Model} & \textbf{MSE} & \textbf{MAPE (\%)} & \textbf{R²} & \textbf{Vol. Error} & \textbf{Corr. Error} & \textbf{vs Best (\%)} \\
\midrule
"""

for _, row in df.iterrows():
    model_name = row['Model'].replace('_', r'\_')
    latex_table1 += f"{int(row['Rank'])} & {model_name} & {row['MSE']:.2f} & {row['MAPE (%)']:.1f} & {row['R²']:.2f} & {row['Volatility Error']:.3f} & {row['Correlation Error']:.2f} & {row['vs Best Baseline (%)']:+.1f} \\\\\n"

latex_table1 += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Save
with open(latex_dir / 'table1_overall_performance.tex', 'w') as f:
    f.write(latex_table1)

print(f"✅ Saved: {latex_dir / 'table1_overall_performance.tex'}")
print("\nLaTeX Code:")
print(latex_table1)

# ============================================================================
# TABLE 2: Sector Comparison
# ============================================================================

print("\n2. Creating LaTeX for Sector Comparison Table...")

df = pd.read_csv(results_dir / 'table2_sector_comparison.csv')

latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Sector-Specific Results: Finance vs Technology}
\label{tab:sector_comparison}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Sector} & \textbf{Model} & \textbf{MSE} & \textbf{MAPE (\%)} & \textbf{$\beta$} & \textbf{Density (\%)} & \textbf{Improvement (\%)} \\
\midrule
"""

for _, row in df.iterrows():
    sector = row['Sector']
    model = row['Model'].replace('_', r'\_')
    mse = row['MSE']
    mape = row['MAPE (%)']
    beta = f"{row['β (Contagion Rate)']:.4f}" if pd.notna(row['β (Contagion Rate)']) else "---"
    density = f"{row['Network Density (%)']:.0f}" if pd.notna(row['Network Density (%)']) else "---"
    improvement = f"{row['Improvement (%)']:+.1f}" if pd.notna(row['Improvement (%)']) else "---"

    latex_table2 += f"{sector} & {model} & {mse:.2f} & {mape:.1f} & {beta} & {density} & {improvement} \\\\\n"

latex_table2 += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $\beta$ = Contagion rate parameter; Density = Network connectivity percentage
\end{tablenotes}
\end{table}
"""

with open(latex_dir / 'table2_sector_comparison.tex', 'w') as f:
    f.write(latex_table2)

print(f"✅ Saved: {latex_dir / 'table2_sector_comparison.tex'}")
print("\nLaTeX Code:")
print(latex_table2)

# ============================================================================
# TABLE 3: Window Size Comparison
# ============================================================================

print("\n3. Creating LaTeX for Window Size Comparison Table...")

df = pd.read_csv(results_dir / 'table3_window_size_comparison.csv')

latex_table3 = r"""
\begin{table}[htbp]
\centering
\caption{Window Size Comparison: Impact on Model Performance}
\label{tab:window_comparison}
\begin{tabular}{lrrllrl}
\toprule
\textbf{Window Size} & \textbf{SIR MSE} & \textbf{Baseline MSE} & \textbf{Best Model} & \textbf{Improvement (\%)} & \textbf{p-value} & \textbf{Significant?} \\
\midrule
"""

for _, row in df.iterrows():
    window = row['Window Size'].replace('_', r'\_')
    sir_mse = row['SIR MSE']
    baseline_mse = row['Best Baseline MSE']
    baseline_name = row['Best Baseline Name'].replace('_', r'\_')
    improvement = f"{row['Improvement (%)']:+.1f}"
    pval = f"{row['p-value']:.4f}"
    sig = row['Significant?']

    latex_table3 += f"{window} & {sir_mse:.2f} & {baseline_mse:.2f} & {baseline_name} & {improvement} & {pval} & {sig} \\\\\n"

latex_table3 += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Significance determined by paired t-test at $\alpha = 0.05$ level
\end{tablenotes}
\end{table}
"""

with open(latex_dir / 'table3_window_size_comparison.tex', 'w') as f:
    f.write(latex_table3)

print(f"✅ Saved: {latex_dir / 'table3_window_size_comparison.tex'}")
print("\nLaTeX Code:")
print(latex_table3)

# ============================================================================
# TABLE 4: Contagion Parameters
# ============================================================================

print("\n4. Creating LaTeX for Contagion Parameters Table...")

df = pd.read_csv(results_dir / 'table4_contagion_parameters.csv')

latex_table4 = r"""
\begin{table}[htbp]
\centering
\caption{Learned Contagion Parameters by Sector}
\label{tab:contagion_parameters}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Sector} & \textbf{$\beta$} & \textbf{$\gamma$} & \textbf{$\alpha$} & \textbf{Density (\%)} & \textbf{Crisis (days)} & \textbf{Immunity (days)} \\
\midrule
"""

for _, row in df.iterrows():
    sector = row['Sector']
    beta = row['β (Contagion Rate)']
    gamma = row['γ (Recovery Rate)']
    alpha = row['α (Re-susceptibility)']
    density = row['Network Density (%)']
    crisis = row['Avg Crisis Duration (days)']
    immunity = row['Avg Immunity Period (days)']

    latex_table4 += f"{sector} & {beta:.4f} & {gamma:.2f} & {alpha:.2f} & {density:.0f} & {crisis} & {immunity} \\\\\n"

latex_table4 += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item $\beta$ = contagion rate; $\gamma$ = recovery rate; $\alpha$ = re-susceptibility rate
\item Crisis duration = $1/\gamma$; Immunity period = $1/\alpha$
\end{tablenotes}
\end{table}
"""

with open(latex_dir / 'table4_contagion_parameters.tex', 'w') as f:
    f.write(latex_table4)

print(f"✅ Saved: {latex_dir / 'table4_contagion_parameters.tex'}")
print("\nLaTeX Code:")
print(latex_table4)

# ============================================================================
# Create a combined LaTeX file with all tables
# ============================================================================

print("\n5. Creating combined LaTeX document...")

combined_latex = r"""\documentclass[12pt]{article}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{threeparttable}

\begin{document}

\section*{Results Tables}

""" + latex_table1 + "\n\n" + latex_table2 + "\n\n" + latex_table3 + "\n\n" + latex_table4 + r"""

\end{document}
"""

with open(latex_dir / 'all_tables_combined.tex', 'w') as f:
    f.write(combined_latex)

print(f"✅ Saved: {latex_dir / 'all_tables_combined.tex'}")

# ============================================================================
# Create a README with usage instructions
# ============================================================================

readme_content = """# LaTeX Tables Usage Instructions

## Files Generated

1. `table1_overall_performance.tex` - Overall model comparison
2. `table2_sector_comparison.tex` - Finance vs Tech sectors
3. `table3_window_size_comparison.tex` - Window size analysis
4. `table4_contagion_parameters.tex` - Learned parameters
5. `all_tables_combined.tex` - Complete document with all tables

## Required LaTeX Packages

Add these to your preamble:

```latex
\\usepackage{booktabs}      % For professional tables
\\usepackage{caption}        % For table captions
\\usepackage{threeparttable} % For table notes
```

## How to Include in Your Document

### Option 1: Copy individual tables
Copy the content from any `tableX_*.tex` file directly into your document.

### Option 2: Use \\input command
Place the .tex files in the same directory as your main document, then:

```latex
\\input{table1_overall_performance}
```

### Option 3: Compile standalone
Compile `all_tables_combined.tex` to see all tables together:

```bash
pdflatex all_tables_combined.tex
```

## Customization Tips

1. **Column widths**: Add `p{width}` specifier
   ```latex
   \\begin{tabular}{lp{3cm}rrr}
   ```

2. **Bold best values**: Use `\\textbf{}`
   ```latex
   219.40 & \\textbf{8.6} & 0.43
   ```

3. **Color coding**: Add `\\usepackage{xcolor}`
   ```latex
   \\textcolor{green}{+7.3}
   ```

4. **Adjust spacing**: Modify `\\arraystretch`
   ```latex
   \\renewcommand{\\arraystretch}{1.2}
   ```

## Notes

- All values are formatted with appropriate precision
- Greek symbols ($\\beta$, $\\gamma$, $\\alpha$) properly rendered
- Professional appearance using booktabs package
- Table notes included where relevant
"""

with open(latex_dir / 'README.md', 'w') as f:
    f.write(readme_content)

print(f"✅ Saved: {latex_dir / 'README.md'}")

print("\n" + "="*80)
print("LATEX GENERATION COMPLETE!")
print("="*80)
print(f"\nAll LaTeX files saved to: {latex_dir.absolute()}/")
print("\nFiles created:")
print("  1. table1_overall_performance.tex")
print("  2. table2_sector_comparison.tex")
print("  3. table3_window_size_comparison.tex")
print("  4. table4_contagion_parameters.tex")
print("  5. all_tables_combined.tex (complete document)")
print("  6. README.md (usage instructions)")
print("\nTo use: Copy the code or \\input{filename} in your LaTeX document")
print("="*80)
