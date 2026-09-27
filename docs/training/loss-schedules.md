# Loss Schedules: Standard vs Inverted (Boost)

## Two Schedule Engines

The system implements **dual loss scheduling** for experimental comparison:

| Schedule | Script | Engine | Consistency Range |
|----------|--------|--------|-------------------|
| **Standard** | `train.sh` | `low_rank_adaptation.py` | 0.1 → 0.5 (Regime 1) |
| **Inverted/Boost** | `boost.sh` | `adaptive_moment_estimation.py` | 1.0 → 0.5 (Regime 2) |

## Standard Schedule (Low-Rank Adaptation)

### Philosophy
Progressive consistency weighting from low to medium, allowing policy/value learning to establish first, then adding representation alignment.

### Regime 1: λ_cons ∈ [0.1, 0.5]

| Weight | Start | End | Direction | Progress Curve |
|--------|-------|-----|-----------|----------------|
| Consistency | 0.10 | 0.50 | **up** | p/0.60 |
| Policy | high | low | **down** | (p-0.15)/0.70 |
| Value | low | high | **up** | p |
| Reward | high | low | **down** | p/0.40 |
| CQL | low | high | **up** | p/0.80 |

### Base Anchors
```python
# From low_rank_adaptation.py:223-232
high_val = end_max           # 0.9637
high_pol = 0.8970
high_rew = 0.7357
high_cql = max(cql_weight, 0.07857)

low_rew = start_min          # 0.1357
low_pol = start_min + 0.0400 # 0.1757
low_val = start_min + 0.0600 # 0.1957
low_cql = start_min * 0.1    # 0.01357
```

### Progress Curves (Staggered)
```python
p_rew = min(1.0, p / 0.40)      # Reward finishes first
p_cons = min(1.0, p / 0.60)     # Consistency at 60%
p_pol = 0.0 if p < 0.15 else min(1.0, (p - 0.15) / 0.70)  # Delayed start
p_val = p                        # Linear
p_cql = min(1.0, p / 0.80)      # CQL slowest
```

### Temperature
Softmax temperature τ = 0.25 (configurable via `schedule_temperature`)

### Distinctness Enforcement
All 5 weights validated mutually distinct at every step:
```python
# From low_rank_adaptation.py:87-112
chain_pairs = [
    ("consistency-weight", "policy-weight"),
    ("policy-weight", "value-weight"),
    ("value-weight", "reward-weight"),
    ("reward-weight", "cql-weight"),
]
# + all-pairs check
# Auto-adjust: +0.0157*(index+1) on collision
```

---

## Inverted Schedule (Adaptive Moment Estimation / Boost)

### Philosophy
Start with **high consistency** (strong representation alignment), then decay while inverting other weight directions — tests "antagonistic" gradient dynamics.

### Regime 2: λ_cons ∈ [1.0, 0.5] (starts HIGH, goes down)

| Weight | Start | End | Direction | Rationale |
|--------|-------|-----|-----------|-----------|
| Consistency | 1.00 | 0.50 | **down** | Strong initial alignment |
| Policy | low | high | **up** | Late policy focus |
| Value | high | low | **down** | Early value, late policy |
| Reward | low | high | **up** | Delayed reward learning |
| CQL | high | low | **down** | Early conservative, late exploratory |

### Anchors (Regime 2)
```python
# From adaptive_moment_estimation.py (similar structure)
high_cons = max(0.5057, min(1.0, cons_base))  # ~0.85-1.0
low_cons = 0.5000

# Directions inverted from Regime 1
dir_cons = "down"      # vs "up"
dir_pol = "up"         # vs "down"
dir_val = "down"       # vs "up"
dir_rew = "up"         # vs "down"
dir_cql = "down"       # vs "up"
```

### Manual Override Support
```python
# From low_rank_adaptation.py:241-251
directions = base_weights.get("schedule_directions")
# Can be dict: {"consistency": "up", "policy": "down", ...}
# Or string: "consistency:up,policy:down,value:up,reward:down,cql:up"
```

---

## Comparison Matrix

| Aspect | Standard (train.sh) | Boost (boost.sh) |
|--------|---------------------|------------------|
| **Engine** | `low_rank_adaptation.py` | `adaptive_moment_estimation.py` |
| **Consistency** | 0.1 → 0.5 (build-up) | 1.0 → 0.5 (decay) |
| **Regime** | 1 (low-cons) | 2 (high-cons) |
| **Policy** | Early focus, then decay | Late focus, build-up |
| **Value** | Build-up | Early focus, then decay |
| **Reward** | Early focus, then decay | Late focus, build-up |
| **CQL** | Build-up | Early focus, then decay |
| **Gradient Flow** | Cooperative | Antagonistic |
| **Use Case** | Standard curriculum | Exploration, escaping local optima |
| **Champion Candidates** | `submission_train_cand_*.zip` | `submission_boost_cand_*.zip` |

---

## Mathematical Foundation

### Softmax Transition (Both Engines)
```python
def softmax_weights_high_low(high, low, progress, direction="down", τ=0.25):
    p = clip(progress, 0, 1)
    z_high = (1-p) / τ
    z_low = p / τ
    m = max(z_high, z_low)
    π_high = exp(z_high - m) / (exp(z_high - m) + exp(z_low - m))
    
    # Boundary calibration
    π_0 = 1 / (1 + exp(-1/τ))
    π_1 = 1 - π_0
    α = (π_high - π_1) / (π_0 - π_1)
    α = clip(α, 0, 1)
    
    if direction in ("down", "high_to_low"):
        return low + (high - low) * α
    else:
        return low + (high - low) * (1 - α)
```

### Properties
- **Exact endpoints**: At p=0 → high (for "down"), at p=1 → low
- **Smooth**: C∞ transition
- **Temperature τ**: Controls sharpness (lower = sharper)
- **Calibration**: Corrects softmax boundary bias

---

## Candidate Scores (Empirical)

### Boost Schedule Candidates
| Candidate | Score | Iteration | Notes |
|-----------|-------|-----------|-------|
| `submission_boost_cand016.zip` | $108,092.65 | 16 | Champion snapshot |
| `submission_boost_cand003.zip` | $106,236.50 | 3 | Population pool |
| `submission_boost_cand014.zip` | $105,679.50 | 14 | Population pool |
| `submission_boost_cand034.zip` | $104,067.65 | 34 | Final champion |

### Standard Schedule Candidates
| Candidate | Score | Iteration | Notes |
|-----------|-------|-----------|-------|
| `submission_train_cand042.zip` | $110,301.60 | 42 | Champion (highest) |
| `submission_train_cand048.zip` | $107,477.65 | 48 | Champion |
| `submission_train_cand056.zip` | $106,908.15 | 56 | Champion |
| `submission_train_cand012.zip` | $106,755.65 | 12 | Champion |

**Key Finding**: Standard schedule produced highest single score ($110,301), but boost schedule provides diversity for league training.

---

## Running Both Schedules

```bash
# Standard curriculum
./train.sh hybrid --eval-episodes 8

# Inverted/Boost curriculum  
./boost.sh hybrid --eval-episodes 4

# Both can run in parallel (separate workspaces)
# Results compared via tournament.py
```

---

## Code References

- `pipelines/low_rank_adaptation.py:1` - Standard engine
- `pipelines/adaptive_moment_estimation.py:1` - Boost engine
- `pipelines/loss_schedule.py:1` - Base schedule class
- `pipelines/boost_schedule.py:1` - Boost config
- `train.sh:1` - Standard launch
- `boost.sh:1` - Boost launch
- `docs/prioritized_muzero_system.py:850-1115` - Full schedule documentation