# Phase 3: MCTS Distillation

## Overview

Phase 3 uses Gumbel MCTS lookahead to generate improved policy/value targets, then distills them back into the network. This is where the agent surpasses behavioral cloning by learning from its own search.

## Pipeline

```
Self-Play (Phase 1-2 policy) 
    │
    ▼
Sampled MCTS Search (25-50 sims, K=8)
    │
    ▼
Visit-count Policy π_MCTS + Root Value v_MCTS
    │
    ▼
Store in Replay Buffer (trajectory + MCTS targets)
    │
    ▼
Training: Distill π_MCTS → f_θ, v_MCTS → f_θ.value
    │
    ▼
Improved Network → Better Self-Play → Repeat
```

## MCTS Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Simulations | 25 (train) / 50 (eval) | `muzero/mcts.py:106` |
| Samples per node (K) | 8 | `muzero/mcts.py:112` |
| PUCT c1 | 1.25 | `muzero/mcts.py:109` |
| PUCT c2 | 19652.0 | `muzero/mcts.py:110` |
| Discount γ | 0.997 | `muzero/mcts.py:111` |
| Dirichlet α | 0.25 | `muzero/mcts.py:114` |
| Exploration ε | 0.25 | `muzero/mcts.py:115` |

## Distillation Loss

### Policy Distillation
Cross-entropy from network logits to MCTS visit distributions:
```
L_policy = CE(π_farmer, π_MCTS_farmer) + CE(π_hands, π_MCTS_hands) + CE(π_market, π_MCTS_market)
```
Where π_MCTS_* are marginalized from joint visit counts.

### Value Distillation
MSE from network value to MCTS root value (with n-step bootstrap):
```
L_value = MSE(v_network, z_MCTS)
z_MCTS = Σ γ^j r_{t+j} + γ^n v_MCTS(s_{t+n})
```

### Consistency Loss (Continued)
SimSiam consistency maintained throughout:
```
L_cons = -cosine_sim(predict(proj(g_θ^k(s_0))), stopgrad(proj(h_θ(o_{t+k}))))
```

### Total Loss
```
L = L_policy + 0.25×L_value + L_reward + λ_cons×L_cons
```

## Replay Buffer Integration

### Trajectory Structure
```python
GameTrajectory:
    observations: List[np.ndarray]      # [28, 10, 10] each step
    actions: List[Tuple[f,h,m]]         # Executed joint actions
    rewards: List[float]                # Environment rewards
    policies: List[Dict[tuple, float]]  # MCTS visit distributions
    values: List[float]                 # MCTS root values
    priorities: List[float]             # TD-error for PER
```

### Reanalyze (Critical for Phase 3)
As network improves, stored MCTS targets become stale. Periodic refresh:

```python
# From muzero/buffer.py:205-212, muzero/trainer.py:186-194
def reanalyze_trajectory(traj_idx, mcts_engine, device):
    traj = buffer.trajectories[traj_idx]
    for t in range(len(traj)):
        obs_tensor = torch.from_numpy(traj.observations[t]).unsqueeze(0).to(device)
        best_act, fresh_policy, fresh_val = mcts_engine.search(obs_tensor, add_dirichlet_noise=False)
        traj.policies[t] = fresh_policy    # Overwrite stale π
        traj.values[t] = fresh_val         # Overwrite stale v
```

**Frequency**: Every 100-500 training steps (configurable)
**Budget**: 1-4 trajectories per reanalyze pass

## Self-Play Data Generation

### Continuous Loop (from `pipelines/self_improving_loop.py`)
```python
while not converged:
    # 1. Generate self-play games using CURRENT champion
    trajectories = self_play_worker.generate_games(champion_model, num_games=8)
    
    # 2. Store in replay buffer (FIFO, disk budget enforced)
    for traj in trajectories:
        buffer.save_trajectory(traj)
    
    # 3. Train on mixed buffer (BC data + self-play)
    for _ in range(train_steps_per_generation):
        metrics = trainer.train_step(batch_size=16, k_steps=5)
    
    # 4. Reanalyze stale targets
    trainer.run_reanalyze_pass(num_trajectories=2)
    
    # 5. Evaluate candidate vs champion
    if evaluate(candidate, champion) > threshold:
        promote_candidate_to_champion()
    
    # 6. Replay manager prunes old selfplay_*.json by mtime
    replay_manager.enforce_budget()
```

## Disk Budget Management

| Workspace | Budget | Pruning |
|-----------|--------|---------|
| This repo | 10.0 GB | `replay_manager.py` |
| External | 24.0 GB | Same logic |

**Pruning Rules** (`pipelines/replay_manager.py`):
1. Canonical replays (`^[0-9]+\.json$`) — **NEVER** deleted
2. Self-play (`selfplay_*.json`) — sorted by `mtime` ascending, oldest first
3. Prune until directory ≤ budget
4. No score-based filtering (prevents stagnation)

## Training Schedules

### Standard (train.sh) — `pipelines/low_rank_adaptation.py`
- λ_cons: 0.1 → 0.5 (Regime 1)
- Smooth softmax transitions
- Distinctness enforced

### Inverted / Boost (boost.sh) — `pipelines/adaptive_moment_estimation.py`
- λ_cons: 1.0 → 0.5 (Regime 2, antagonistic)
- Tests opposite gradient directions
- Different policy/value/reward/CQL directions

## Validation

| Metric | Phase 2 → Phase 3 Improvement |
|--------|-------------------------------|
| Policy CE vs MCTS | Decreasing (distillation working) |
| Self-play win rate vs Phase 2 | > 55% |
| MCTS root value accuracy | Improving |
| Reanalyze loss reduction | Fresh targets < stale targets |

## Code References

- `muzero/mcts.py:103-245` - Sampled MCTS implementation
- `muzero/buffer.py:38-336` - Trajectory buffer + PER + Reanalyze
- `muzero/trainer.py:51-194` - PER Trainer with Reanalyze
- `muzero/trainer.py:197-320` - Uniform Trainer with Reanalyze
- `pipelines/self_improving_loop.py:1` - Continuous loop orchestration
- `pipelines/replay_manager.py:1` - Disk budget enforcement
- `train.sh:1` / `boost.sh:1` - Launch configurations