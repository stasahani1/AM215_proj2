# Why the SIR Model Failed & How I Fixed It

## 🔴 The Problem

Your SIR contagion model wasn't working because **stocks never entered the "Infected" state**:

```
State occupancy:
ticker    pct_R     pct_S  pct_I
  AAPL 8.333333 91.666667    0.0    ← 0% Infected!
  MSFT 1.851852 98.148148    0.0
   JPM 3.703704 96.296296    0.0

Learning contagion parameters...
  Insufficient variation in S->I transitions    ← Can't learn contagion!
```

### Why This Is Catastrophic

1. **No contagion to learn**: If β (contagion rate) can't be learned, the model = Independent HMM
2. **Model becomes 2-state**: Only R ↔ S transitions, no crisis modeling
3. **Can't demonstrate contagion**: The whole point of the model is lost!

---

## 🔍 Root Causes

### Cause 1: Poor HMM Initialization
**Old method**: Linearly spaced means and stds
```python
means = [-2σ, 0, +2σ]
stds = [0.5σ, σ, 2σ]
```
**Problem**: Doesn't reflect actual data structure

### Cause 2: Too Quick Convergence
```
Iteration 0: log-likelihood = 0.00
Converged at iteration 1    ← Only 1 iteration!
```
**Problem**: Found local optimum, not true structure

### Cause 3: States Not Distinct
```
AAPL:
  R: σ=0.00389  ← Tiny outlier
  S: σ=0.02323  ← Normal
  I: σ=0.02440  ← Only 5% higher than S!
```
**Problem**: "Infected" and "Susceptible" have almost identical volatility

---

## ✅ Fixes Applied

### Fix 1: Smart Volatility-Based Initialization

**New method**: Initialize based on actual volatility regimes

```python
# Compute rolling volatility
rolling_vol = returns.rolling(20).std()

# Split into low/medium/high volatility periods
low_vol_periods = rolling_vol < 25th percentile
med_vol_periods = rolling_vol between 25th-75th percentile  
high_vol_periods = rolling_vol > 75th percentile

# Initialize each state from its own regime
State 0 (R): μ and σ from low_vol_periods
State 1 (S): μ and σ from med_vol_periods
State 2 (I): μ and σ from high_vol_periods
```

**Benefit**: States start with meaningful, distinct parameters

### Fix 2: Force Distinct Volatility Levels

```python
# Ensure at least 50% separation between states
if stds[1] / stds[0] < 1.5:
    stds = [σ_low, 1.5×σ_low, 3.0×σ_low]
```

**Benefit**: States can't collapse to similar values

### Fix 3: Require Minimum Iterations

```python
# Old: Could converge at iteration 1
# New: Must run at least 10 iterations
min_iter = 10
max_iter = 100
```

**Benefit**: Better chance to find true structure

---

## 📊 Expected Improvements

### Before (Old Initialization):
```
State occupancy:
  Infected: 0.0%    ← No infected states!

Contagion parameters:
  β: undefined      ← Can't learn
  γ: default 0.1
```

### After (New Initialization):
```
State occupancy:
  Recovered: 10-20%
  Susceptible: 50-70%
  Infected: 10-30%  ← NOW HAS INFECTED STATES!

Contagion parameters:
  β: 0.15-0.40      ← Learned from data
  γ: 0.10-0.25
  α: 0.05-0.15
```

---

## 🎯 What to Look For When You Rerun

### 1. Training Output - Check State Parameters
```
AAPL:
  R: μ=-0.001, σ=0.008   ← Low volatility
  S: μ=0.001,  σ=0.018   ← Medium volatility  
  I: μ=0.002,  σ=0.035   ← HIGH volatility (at least 1.5-2x higher!)
```

**Good sign**: σ_I should be at least 1.5-2x higher than σ_S

### 2. State Occupancy - Check Distribution
```
ticker    pct_R     pct_S  pct_I
  AAPL   15.2     68.5    16.3   ← Some infected states!
  MSFT   12.8     71.2    16.0
   JPM   18.3     65.1    16.6
```

**Good sign**: All three states should have >5% occupancy

### 3. Contagion Learning - Check Parameters
```
Learning contagion parameters...
  Baseline S->I rate: 0.0523        ← Base infection rate
  Contagion coefficient β: 0.2847   ← LEARNED (not undefined!)
  S->I transitions analyzed: 87     ← Actual transitions happened!
  Recovery rate γ: 0.1523
  Re-susceptibility rate α: 0.0892
```

**Good sign**: β is a real number (not undefined), transitions were analyzed

### 4. Performance - Check if SIR Wins
```
Backtest Summary
                     mse_mean  mape_mean
SIR Contagion Model     28.45       2.21   ← Lower is better
Independent HMM         31.83       2.45   ← Should be worse
Single Random Walk      34.67       2.68   ← Should be worst
```

**Good sign**: SIR < Independent HMM < Single RW

---

## 🔬 Why This Matters

### Without Infected States:
- ❌ Can't learn contagion (β)
- ❌ Can't model crises
- ❌ Model = Independent HMM (no advantage)
- ❌ Research contribution = zero

### With Infected States:
- ✅ Can learn contagion dynamics
- ✅ Can model crisis propagation
- ✅ Model truly different from baselines
- ✅ Can demonstrate contagion effects exist (or don't!)

---

## 🚀 Next Steps

### 1. Run Quick Test
```bash
./RUN_ME.sh
# Choose option 1
```

Look for:
- States with distinct σ values (check training output)
- Non-zero infected state percentages
- Actual contagion parameters learned

### 2. If Still No Infected States

Try these adjustments in `run_evaluation.py`:

**Option A: Lower correlation threshold (more connections)**
```python
CORRELATION_THRESHOLD = 0.3  # Was 0.5
```

**Option B: More extreme chunks (even higher VIX)**
```python
N_CHUNKS = 40  # Was 60 (selects only most extreme)
```

**Option C: Include 2008 crisis explicitly**
```python
YEARS_BACK = 20  # Was 15 (includes 2008 financial crisis)
```

### 3. If Infected States Exist But SIR Still Loses

This is actually **valuable research**! It means:
- States exist (volatility regimes are real)
- But contagion doesn't help predictions
- **Conclusion**: Co-movement is from correlation, not contagion

This is a legitimate finding! Not all stocks exhibit contagion effects.

---

## 📖 Understanding SIR Model Performance

### If SIR Wins:
✅ **Conclusion**: Contagion effects are real and predictive
- Stocks do "infect" each other during crises
- Network structure matters for modeling
- Your hypothesis is supported

### If Independent HMM Wins:
⚠️ **Conclusion**: States help but contagion doesn't
- Multiple regimes exist (volatility clustering)
- But transitions are mostly independent
- Stocks enter crises for idiosyncratic reasons

### If Single RW Wins:
❌ **Conclusion**: Even states don't help much
- Market is more efficient than expected
- Regime-switching is weak
- Simple models work best for these stocks/periods

**All three outcomes are scientifically interesting!**

---

## 🎓 Research Value

Even if SIR doesn't win, you've built:
1. ✅ Rigorous evaluation framework
2. ✅ Multiple baselines for comparison
3. ✅ Comprehensive metrics
4. ✅ Systematic methodology

You can report:
- "We tested SIR contagion model on X stocks over Y years"
- "Found Z% of time in infected states"
- "Contagion parameter β = W"
- "Model performance: [results]"
- "Conclusion: Contagion effects are [strong/weak/absent]"

This is **publishable research** regardless of outcome!

---

## 💡 Pro Tips

1. **Check state separation first**: If σ_I ≈ σ_S, the model can't work
2. **Need infected states**: At least 5-10% of time should be infected
3. **Contagion must be learned**: β must be a real number
4. **Network matters**: Try different correlation thresholds
5. **Crisis periods crucial**: Model works best during actual crises (2008, 2020)

---

**Ready to test?** Run `./RUN_ME.sh` and watch for these improvements! 🚀

