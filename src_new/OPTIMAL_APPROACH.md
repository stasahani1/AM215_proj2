# The Optimal Approach: 6-Week Chunks + Mixed Training

## 🎯 Why Combine Both?

Your experiments revealed something important:

### What We Learned:

**Experiment 1: 6-Week Chunks (High-VIX only)**
- ✅ SIR **WINS** by 20-24%
- ❌ But β (contagion) **not learned**
- Conclusion: Wins for time scale, not contagion

**Experiment 2: Mixed Periods (20-day chunks)**
- ✅ β (contagion) **learned** successfully (0.64)
- ❌ But SIR **LOSES** to baselines
- Conclusion: Good learning, wrong time scale

**The Insight**: Need BOTH good time scale AND good training!

---

## ⚡ The Optimal Configuration

### Combine the Best of Both:

1. **6-Week Chunks** (30 trading days)
   - Gives time for regime persistence
   - States can transition within chunks
   - Better prediction horizon

2. **Mixed VIX Training**
   - Train on calm + normal + crisis periods
   - Clear contrast between states
   - Better parameter learning

3. **High-VIX Testing**
   - Test where contagion matters most
   - Focus on crisis periods
   - Relevant for real applications

---

## 📊 Expected Results

### What Should Happen:

**State Learning** (from mixed training):
```
States will be DISTINCT:
  R: σ ≈ 0.009  (low)
  S: σ ≈ 0.015  (medium - 1.7x higher)
  I: σ ≈ 0.028  (high - 3x higher!) ✓
```

**Contagion Learning**:
```
β should be learned: 0.2-0.6 ✓
All transition types observed
S->I, I->R, R->S all present
```

**Prediction Performance**:
```
SIR should BEAT baselines by 20-30%
Better than 6-week alone (20%)
Better than mixed alone (loses)
```

---

## 🧪 Testing Options

### Option 1: Quick Test (~5 min)
```bash
python quick_test_optimal.py
```
- 3 stocks, 5 years
- Fast verification
- See if approach works

### Option 2: Full Evaluation (~30 min)
```bash
python run_evaluation_optimal.py
```
- 10 stocks, 15 years
- Complete analysis
- All visualizations
- Results in `results_optimal/`

---

## 🎓 What This Tests

### The Research Questions:

1. **Does time scale matter?**
   - 6-week chunks vs 2-week chunks
   - Answer: YES (6-week wins)

2. **Does training diversity matter?**
   - Mixed VIX vs high-VIX only
   - Answer: YES (for learning, need contrast)

3. **Does contagion help when properly learned?**
   - This is what optimal config tests!
   - If SIR wins: Contagion is real
   - If SIR loses: It's just regime-switching

---

## 💡 Possible Outcomes

### Outcome A: SIR Wins by 30%+ ✓✓✓
**Interpretation**:
- Contagion IS real and predictive
- Both time scale and network matter
- Multi-state + contagion > Multi-state alone

**Research Conclusion**:
> "SIR contagion model outperforms baselines by 30% when using optimal 6-week chunks with mixed training. This demonstrates that contagion dynamics are both learnable and predictive."

### Outcome B: SIR Wins by 20% (same as before) ✓
**Interpretation**:
- Time scale matters most
- Contagion adds little beyond regime-switching
- States help, networks don't

**Research Conclusion**:
> "Multi-state regime-switching models outperform baselines, but contagion effects don't add predictive value beyond state structure. Improvement comes from regime persistence, not network effects."

### Outcome C: SIR Loses ❌
**Interpretation**:
- Mixed training causes some issue
- Overfitting or distribution mismatch
- Need to investigate further

**Research Conclusion**:
> "Optimal configuration failed. Need to investigate training/test distribution mismatch."

---

## 📈 Comparison Table

| Configuration | Time Scale | Training | β Learned? | SIR Performance |
|---------------|-----------|----------|------------|-----------------|
| Original | 2 weeks | High-VIX | ❌ No | Loses (NaN) |
| 6-Week Only | 6 weeks | High-VIX | ❌ No | **Wins 20%** ✓ |
| Mixed Only | 20 days | Mixed | ✅ Yes | Loses |
| **OPTIMAL** | **6 weeks** | **Mixed** | **✅ Should be** | **??? TEST THIS** |

---

## 🔬 The Scientific Method

This is actually perfect scientific process:

1. **Hypothesis**: Contagion improves predictions
2. **Test 1**: Failed (no infected states)
3. **Insight**: Time scale matters (6-week wins)
4. **Insight**: Training matters (mixed learns β)
5. **Refined Test**: Combine both (optimal)
6. **Result**: Will tell us if contagion truly helps

---

## 🚀 Run It Now!

### Quick Test (Recommended First):
```bash
cd /Users/praghav/Desktop/am215/AM215_proj2/src_new
source ../venv/bin/activate
python quick_test_optimal.py
```

Look for:
1. ✅ Distinct σ values (check training)
2. ✅ β learned (not "Insufficient variation")
3. ✅ SIR beats baselines in test
4. ✅ Improvement > 20% (better than 6-week alone)

### Full Evaluation:
```bash
python run_evaluation_optimal.py
```

Generates visualizations in `results_optimal/`

---

## 📝 Key Takeaways

### Why This Matters:

**Previous tests were incomplete**:
- 6-week: Good predictions, poor learning
- Mixed: Good learning, poor predictions

**Optimal combines both**:
- Good predictions (from time scale)
- Good learning (from diverse training)
- Tests if contagion TRULY helps

**Whatever the result, it's valuable**:
- If wins: Contagion is real ✓
- If same: Time scale is key, networks don't matter ✓
- If loses: Need more investigation ✓

All outcomes advance our understanding!

---

## 🎯 Bottom Line

**YES, you definitely want to try ALL periods + 6-week chunks!**

This is the **methodologically correct** approach:
- Proper time scale (6 weeks)
- Proper training (diverse data)
- Proper testing (crisis periods)

This will give you the **definitive answer** about whether contagion helps.

---

**Ready to test?** Run `quick_test_optimal.py` now! 🚀

