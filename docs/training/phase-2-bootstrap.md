# Phase 2: Analytical Bootstrapping

## Overview

Phase 2 grounds the representation and dynamics models using self-consistent unroll targets. The network learns to predict its own unrolled latent states, aligning h_θ and g_θ without MCTS.

## Objective

Minimize the discrepancy between:
- **Predicted unroll**: g_θ^k(h_θ(o_t)) — latent state after K dynamics steps
- **Ground truth encoding**: h_θ(o_{t+k}) — direct encoding of future observation

## Loss Components

### 1. K-Step Unroll Loss
```
L_unroll = Σ_{k=1}^K [CE(π_k, π_target_k) + 0.25×MSE(v_k, z_k) + MSE(r_k, r_target_k)]
```
Where z_k = n-step bootstrap target, r_target_k = environment reward.

### 2. SimSiam Consistency Loss (Primary Signal)
```
L_cons = -cosine_sim(predict(proj(s_k)), stopgrad(proj(h_θ(o_{t+k}))))
```
- **Target branch**: h_θ(o_{t+k}) → proj → **stop-gradient** (no grad to encoder)
- **Predictor branch**: s_k (unrolled) → proj → predict (grad flows to g_θ)
- Forces g_θ to produce latents matching h_θ's observation encodings

### 3. Total Phase 2 Loss
```
L_total = L_policy + 0.25×L_value + L_reward + λ_cons×L_cons
```

## Hyperparameter Schedule (Regime 1: λ_cons ∈ [0.1, 0.5])

| Step Progress | λ_cons | Policy | Value | Reward | CQL |
|---------------|--------|--------|-------|--------|-----|
| 0%            | 0.10   | high→low | low→high | high→low | low→high |
| 50%           | 0.30   | mid    | mid   | mid    | mid |
| 100%          | 0.50   | low    | high  | low    | high |

**Directions** (from `low_rank_adaptation.py:254-266`):
- Consistency: "up" (0.1 → 0.5)
- Policy: "down"
- Value: "up"  
- Reward: "down"
- CQL: "up"

## Softmax Transition Formula

```python
# From low_rank_adaptation.py:117-158
def softmax_weights_high_low(high, low, progress, direction="down", temperature=0.25):
    p = clip(progress, 0, 1)
    z_high = (1-p) / τ
    z_low = p / τ
    π_high = softmax([z_high, z_low])[0]
    # Boundary calibration ensures exact endpoints
    return low + (high - low) * π_high  # for "down"
```

Temperature τ=0.25 provides smooth but decisive transitions.

## Training Loop

```python
# Pseudocode from muzero/trainer.py:240-311
for step in range(total_steps):
    batch = buffer.sample_k_step_batch(batch_size=16, k_steps=5)
    
    # K-step unroll with consistency
    s_0 = h_θ(batch.obs[:, 0])
    for k in 1..5:
        s_k, r_k = g_θ(s_{k-1}, batch.actions[:, k-1])
        preds_k = f_θ(s_k)
        
        # Standard losses
        L_policy += CE(preds_k.farmer, batch.target_f[:, k]) + ...
        L_value += MSE(preds_k.value, batch.values[:, k])
        L_reward += MSE(r_k, batch.rewards[:, k-1])
        
        # Consistency
        target_z = h_θ_target(batch.obs[:, k])  # Target network!
        L_cons += SimSiamLoss(s_k, target_z)
    
    # Composite loss
    loss = mean(L_policy) + 0.25*mean(L_value) + mean(L_reward) + λ_cons*mean(L_cons)
    
    # Update
    loss.backward()
    clip_grad_norm(5.0)
    optimizer.step()
    
    # EMA target network
    θ_target ← 0.995 × θ_target + 0.005 × θ
```

## Why This Works

1. **Dense supervision**: Every unroll step provides gradient signal (vs sparse rewards)
2. **Latent alignment**: Consistency loss prevents g_θ drift from h_θ manifold
3. **Representation learning**: h_θ learns features predictive of future states
4. **No MCTS needed**: Pure gradient-based; faster iteration

## Validation Metrics

| Metric | Target |
|--------|--------|
| Consistency loss (cosine sim) | → -1.0 (perfect alignment) |
| Latent drift: ‖h_θ(o_{t+k}) - g_θ^k(s_0)‖ | Decreasing |
| Policy CE on unroll steps | < Phase 1 final |
| Value MSE on bootstrapped targets | < Phase 1 final |

## Code References

- `muzero/chassis.py:368-451` - `compute_muzero_unroll_loss_with_consistency`
- `muzero/chassis.py:315-365` - `SimSiamProjectionPredictionHead`, `SimSiamConsistencyLoss`
- `muzero/trainer.py:240-311` - `MuZeroTrainer.train_step` (uniform)
- `pipelines/low_rank_adaptation.py:188-308` - Loss scheduling logic
- `docs/Consistency_Loss.md` - Full theoretical background