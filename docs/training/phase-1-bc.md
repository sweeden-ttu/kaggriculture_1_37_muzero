# Phase 1: Behavioral Cloning (BC)

## Overview

Phase 1 establishes a competent initial policy by supervised learning on gold-standard Kaggle replays. This provides non-random priors so MCTS exploration doesn't waste thousands of steps on random actions.

## Data Source

**Gold Replays**: 50 canonical replays in `replays/` (`^[0-9]+\.json$` format, pruning-immune)
- Expert human games with scores $100k–$147k
- Contains optimal opening sequences, crop strategies, expansion timing

## Distillation Process

```bash
# Extract policy artifacts from replays
python learn_from_replay.py --all
```

**Output**: `artifacts/ep<id>_policy.json` with strategic parameters:
- `sw_day_min`, `skip_se` (expansion decisions)
- `tomato_seed_total_target`, `carrot_post_ne_qty` (crop targets)
- `hire_target` (labor management)
- Shed liquidation thresholds

## Training Configuration

### Loss Function
Standard supervised loss on demonstrations:
```
L_BC = CE(π_expert, π_network) + MSE(v_expert, v_network)
```

### Hyperparameters (from `pipelines/low_rank_adaptation.py`)
| Parameter | Value | Schedule |
|-----------|-------|----------|
| Consistency weight (λ_cons) | 0.1 → 0.5 | Regime 1, direction="up" |
| Policy weight | 1.0 (fixed) | - |
| Value weight | 0.25 (fixed) | - |
| Reward weight | 1.0 (fixed) | - |
| CQL weight | 0.0 (disabled) | - |

### Distinctness Rule Enforcement
All 5 weights must be mutually distinct (validated in `low_rank_adaptation.py:43-114`):
```
consistency-weight ≠ policy-weight ≠ value-weight ≠ reward-weight ≠ cql-weight
```

Auto-adjusted by +0.0157×(index+1) if collision detected.

## Implementation

**Entry Point**: `pipelines/train_muzero.py`
**Launch Script**: `train.sh` (standard) or `boost.sh` (inverted)

```bash
# Standard BC phase
./train.sh hybrid --eval-episodes 8

# Inverted schedule (antagonistic gradients)
./boost.sh hybrid --eval-episodes 4
```

## Network State After Phase 1

- Representation h_θ maps observations to meaningful latent space
- Dynamics g_θ learns basic transition structure
- Prediction f_θ produces sensible policy/value priors
- **But**: No MCTS improvement yet; pure imitation

## Validation

- Policy accuracy vs expert actions > 60%
- Value MSE on demonstration states < threshold
- Loss curves: policy_loss ↓, value_loss ↓

## Next Phase

Phase 2 uses this initialized network for analytical bootstrapping with self-consistent unroll targets.

## Code References

- `learn_from_replay.py:1` - Replay distillation entry
- `pipelines/low_rank_adaptation.py:188-308` - Loss scheduling
- `pipelines/train_muzero.py:1` - Training orchestration
- `train.sh:1` / `boost.sh:1` - Launch scripts