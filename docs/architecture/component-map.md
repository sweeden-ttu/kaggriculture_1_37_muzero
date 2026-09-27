# Component Interaction Map

## Component Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTION MAP                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │  OBSERVATION        │
                    │  ENCODER            │
                    │  (28-ch encoder)    │
                    └──────────┬──────────┘
                               │ [B, 28, 10, 10]
                               ▼
                    ┌─────────────────────┐
                    │  SPATIAL            │
                    │  REPRESENTATION     │
                    │  NETWORK (h_θ)      │
                    │  Conv28→64 + 3 Res  │
                    └──────────┬──────────┘
                               │ s_0 [B, 64, 10, 10]
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ SPATIAL        │ │ SAMPLED MCTS   │ │ TARGET NETWORK │
     │ PREDICTION     │ │ (Search)       │ │ (EMA copy)     │
     │ NETWORK (f_θ)  │ │                │ │                │
     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
             │                  │                  │
             │ logits_farmer    │                  │
             │ logits_hands     │                  │
             │ logits_market    │                  │
             │ value_logits     │                  │
             │                  │                  │
             ▼                  ▼                  ▼
     ┌─────────────────────────────────────────────────────┐
     │              FACTORIZED POLICY HEADS                │
     │  • Farmer (15) ──┐                                  │
     │  • Hands (32) ───┤──► Joint Action Sampling (K=8)  │
     │  • Market (20) ──┘                                  │
     │  • Value (601)                                      │
     └─────────────────────────────────────────────────────┘
             │                          │
             │ Sample K joint           │ Root policy + value
             │ actions                  │ for MCTS root
             ▼                          ▼
     ┌─────────────────────────────────────────────────────┐
     │              SAMPLED MCTS SEARCH TREE               │
     │  • MinMaxStats Q-normalization                      │
     │  • PUCT Selection (c1=1.25, c2=19652)              │
     │  • Recurrent Expansion via g_θ                      │
     │  • Discounted Backup (γ=0.997)                      │
     │  • Dirichlet Noise at Root (α=0.25)                │
     └─────────────────────────────────────────────────────┘
             │                          │
             │ best_action              │ π(a|s₀) visit dist
             ▼                          ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ ACTION           │      │ REPLAY BUFFER    │
     │ TRANSLATION      │      │ (PER + Reanalyze)│
     │ (Chassis Pipeline)      │                  │
     └────────┬─────────┘      └────────┬─────────┘
              │                         │
              ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ KAGGLE ENV       │      │ TRAINER LOOP     │
     │ (Step execution) │      │ (PER + EMA +     │
     └────────┬─────────┘      │  SimSiam)        │
              │               └────────┬─────────┘
              │                        │
              ▼                        ▼
     ┌──────────────────┐      ┌──────────────────┐
     │ GAME TRAJECTORY  │◄─────│ LOSS COMPUTATION │
     │ (obs, act, rew,  │      │ • Policy CE (×3) │
     │  π, v, priority) │      │ • Value MSE      │
     └──────────────────┘      │ • Reward MSE     │
                               │ • Consistency    │
                               └────────┬─────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │ OPTIMIZER STEP   │
                               │ • AdamW + Clip   │
                               │ • EMA Update     │
                               │ • Priority Update│
                               └──────────────────┘
```

## Interface Contracts

### 1. Observation Encoder → Representation Network
```python
# Input
obs: Dict[str, Any]  # Raw Kaggle observation

# Output
tensor: torch.Tensor  # [1, 28, 10, 10] float32
action_masks: Dict[str, torch.Tensor]  # {"farmer": [15], "hands": [32], "market": [20]}
```

### 2. Representation Network (h_θ)
```python
# Input
obs: torch.Tensor  # [B, 28, 10, 10]

# Output
latent: torch.Tensor  # [B, 64, 10, 10] L2-normalized on dim=1
```

### 3. Dynamics Network (g_θ)
```python
# Input
s_prev: torch.Tensor  # [B, 64, 10, 10]
action_plane: torch.Tensor  # [B, 16, 10, 10]

# Output
s_next: torch.Tensor  # [B, 64, 10, 10] L2-normalized
reward_logits: torch.Tensor  # [B, 601]
```

### 4. Prediction Network (f_θ)
```python
# Input
s_k: torch.Tensor  # [B, 64, 10, 10] or [B, 64*10*10] (auto-reshape)

# Output
{
    "logits_farmer": torch.Tensor,  # [B, 15]
    "logits_hands": torch.Tensor,   # [B, 32]
    "logits_market": torch.Tensor,  # [B, 20]
    "value_logits": torch.Tensor    # [B, 601]
}
```

### 5. Chassis Unified API
```python
# Initial inference (root)
initial_inference(obs) → (latent, preds_dict, value_scalar)

# Recurrent inference (tree)
recurrent_inference(latent, action_plane) → (next_latent, reward_scalar, preds_dict, value_scalar)
```

### 6. MCTS Interface
```python
# Search
search(obs_tensor, add_dirichlet_noise=True, masks=None) 
    → (best_action_tuple, policy_dict, root_value)

# Where:
#   best_action_tuple: (farmer_int, hands_int, market_int)
#   policy_dict: {(f,h,m): visit_prob}
#   root_value: float
```

### 7. Replay Buffer
```python
# Store
append_step(obs: np.ndarray, action: tuple, reward: float, 
            policy_dict: Dict[tuple, float], value: float, priority: float=1.0)

# Sample (uniform)
sample_k_step_batch(batch_size, k_steps) → Dict[str, torch.Tensor]

# Sample (PER)
sample_prioritized_k_step_batch(batch_size, k_steps) 
    → (batch_dict, slice_keys, is_weights)

# Reanalyze
reanalyze_trajectory(traj_idx, mcts_engine, device)
```

### 8. Trainer
```python
# PER Trainer
train_step(batch_size, k_steps, global_step) → Dict[str, float]

# Uniform Trainer  
train_step(batch_size, k_steps) → Dict[str, float]

# Both have:
run_reanalyze_pass(num_trajectories)
```

### 9. SimSiam Consistency
```python
# Loss computation
simsiam_loss_fn(unrolled_latent, target_obs_embedding, simsiam_head) → scalar loss

# Where simsiam_head has:
#   project(latent) → projected [B, proj_dim]
#   predict(projected) → predicted [B, proj_dim]
```

### 10. League / PFSP
```python
# League
add_checkpoint(checkpoint_id)
record_match_outcome(agent_id, opponent_id, outcome)  # outcome: 1.0/0.5/0.0
get_win_rate(agent_id, opponent_id) → float  # Laplace smoothed

# PFSP
sample_opponent(active_agent_id, mode="hard"|"var"|"uniform", power=2.0) → opponent_id

# Matchmaking
select_matchmaking_opponent(agent_role, active_agent_id, pfsp) → (opponent_id, mode_str)
```

## Data Flow Matrix

| From \ To | Encoder | h_θ | g_θ | f_θ | MCTS | Buffer | Trainer | SimSiam | League |
|-----------|---------|-----|-----|-----|------|--------|---------|---------|--------|
| Encoder   | -       | ✓   | -   | -   | -    | ✓      | -       | -       | -      |
| h_θ       | -       | -   | ✓   | ✓   | ✓    | -      | ✓       | ✓       | -      |
| g_θ       | -       | -   | -   | -   | ✓    | -      | ✓       | -       | -      |
| f_θ       | -       | -   | -   | -   | ✓    | -      | ✓       | -       | -      |
| MCTS      | -       | -   | -   | -   | -    | ✓      | -       | -       | ✓      |
| Buffer    | -       | -   | -   | -   | -    | -      | ✓       | -       | -      |
| Trainer   | -       | -   | -   | -   | -    | ✓      | -       | ✓       | ✓      |
| SimSiam   | -       | -   | -   | -   | -    | -      | ✓       | -       | -      |
| League    | -       | -   | -   | -   | -    | -      | ✓       | -       | -      |

## File Location Map

```
muzero/
├── observation_encoder.py    # Encoder + action masks
├── chassis.py                # h_θ, g_θ, f_θ, SimSiam, loss
├── mcts.py                   # SampledMuZeroMCTS, MinMaxStats, MCTSNode
├── buffer.py                 # GameTrajectory, MuZeroTrajectoryBuffer, PrioritizedMuZeroBuffer
├── trainer.py                # PrioritizedMuZeroTrainer, MuZeroTrainer
├── _core.py                  # Legacy macro exports
├── spatial_constants.py      # All constants
├── encoding.py               # Macro encoding (legacy)
├── legacy_macro.py           # Legacy MLP MuZero
├── lora.py                   # LoRA injection/merge
├── ppo.py                    # PPO microcontroller
├── meta.py                   # Meta-parameters
├── types.py                  # Types, DiscreteSupport, MacroOption
└── __init__.py               # Public API exports

pipelines/
├── train_muzero.py           # Main training entry
├── low_rank_adaptation.py    # Standard loss scheduling (train.sh)
├── adaptive_moment_estimation.py  # Inverted scheduling (boost.sh)
├── replay_manager.py         # Disk budget, pruning
├── self_improving_loop.py    # Continuous loop orchestration
├── boost_schedule.py         # Boost schedule config
├── loss_schedule.py          # Loss schedule base
└── phases/
    └── phase2_distill.py     # Phase 2 distillation

evaluation/
├── league.py                 # LeagueTracker, PFSPSampler, tournament
├── harness.py                # Game execution
└── __init__.py

hybrid_chassis/
├── core/chassis.py           # Reactive chassis (route replay + safety)
├── pipeline.py               # Macro → Kaggle action translation
├── builder.py                # Agent factory
├── compiler.py               # Single-file compilation
├── layers/                   # 10 reactive layers
└── routes/                   # Route definitions

packaging/
├── builder.py                # Submission packaging
└── __init__.py

scripts/
├── build_head_submissions.py # 3 archetype compilation
├── package_submission.py     # Single submission packaging
└── ...

Root scripts:
├── train.sh                  # Standard curriculum
├── boost.sh                  # Inverted curriculum
├── tournament.py             # Benchmark vs baselines
├── tournament_candidates_vs_baselines.py  # Cross-tournament
└── learn_from_replay.py      # Replay distillation
```

## Cross-Reference Index

| Component | Primary File | Key Classes | Key Functions |
|-----------|-------------|-------------|---------------|
| Observation Encoder | `muzero/observation_encoder.py` | `KaggricultureObservationEncoder` | `encode()`, `get_action_mask()` |
| Factorized Head | `muzero/chassis.py` | `SpatialPredictionNetwork` (=`FactorizedKaggriculturePredictionHead`) | `forward()`, `sample_joint_actions()` |
| SimSiam | `muzero/chassis.py` | `SimSiamProjectionPredictionHead`, `SimSiamConsistencyLoss` | `project()`, `predict()`, `forward()` |
| Sampled MCTS | `muzero/mcts.py` | `SampledMuZeroMCTS`, `MinMaxStats`, `MCTSNode` | `search()`, `_select_child()`, `_puct_score()` |
| Core Chassis | `muzero/chassis.py` | `KaggricultureMuZeroChassis` | `initial_inference()`, `recurrent_inference()` |
| PER Buffer | `muzero/buffer.py` | `PrioritizedMuZeroBuffer`, `GameTrajectory` | `sample_prioritized_k_step_batch()`, `reanalyze_trajectory()` |
| Trainer | `muzero/trainer.py` | `PrioritizedMuZeroTrainer`, `MuZeroTrainer` | `train_step()`, `run_reanalyze_pass()` |
| League | `evaluation/league.py` | `LeagueTracker`, `PFSPSampler` | `sample_opponent()`, `select_matchmaking_opponent()` |
| Consolidated | `pipelines/train_muzero.py` | (script) | `main()` |