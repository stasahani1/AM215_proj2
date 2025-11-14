# LaTeX Tables Usage Instructions

## Files Generated

1. `table1_overall_performance.tex` - Overall model comparison
2. `table2_sector_comparison.tex` - Finance vs Tech sectors
3. `table3_window_size_comparison.tex` - Window size analysis
4. `table4_contagion_parameters.tex` - Learned parameters
5. `all_tables_combined.tex` - Complete document with all tables

## Required LaTeX Packages

Add these to your preamble:

```latex
\usepackage{booktabs}      % For professional tables
\usepackage{caption}        % For table captions
\usepackage{threeparttable} % For table notes
```

## How to Include in Your Document

### Option 1: Copy individual tables
Copy the content from any `tableX_*.tex` file directly into your document.

### Option 2: Use \input command
Place the .tex files in the same directory as your main document, then:

```latex
\input{table1_overall_performance}
```

### Option 3: Compile standalone
Compile `all_tables_combined.tex` to see all tables together:

```bash
pdflatex all_tables_combined.tex
```

## Customization Tips

1. **Column widths**: Add `p{width}` specifier
   ```latex
   \begin{tabular}{lp{3cm}rrr}
   ```

2. **Bold best values**: Use `\textbf{}`
   ```latex
   219.40 & \textbf{8.6} & 0.43
   ```

3. **Color coding**: Add `\usepackage{xcolor}`
   ```latex
   \textcolor{green}{+7.3}
   ```

4. **Adjust spacing**: Modify `\arraystretch`
   ```latex
   \renewcommand{\arraystretch}{1.2}
   ```

## Notes

- All values are formatted with appropriate precision
- Greek symbols ($\beta$, $\gamma$, $\alpha$) properly rendered
- Professional appearance using booktabs package
- Table notes included where relevant
