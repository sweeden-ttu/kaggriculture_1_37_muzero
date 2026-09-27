# Complete Hyperparameter Reference

## Neural Network Architecture

### Spatial Encoder (Observation)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `OBS_CHANNELS` | 28 | `spatial_constants.py` | Input observation channels |
| `BOARD_SIZE` | 10 | `observation_encoder.py:82` | Grid dimensions |
| `LATENT_CHANNELS` | 64 | `spatial_constants.py` | Latent state channels |

### Representation Network (h_θ)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `in_channels` | 28 | `chassis.py:78` | Observation channels |
| `latent_channels` | 64 | `chassis.py:78` | Output latent channels |
| `num_blocks` | 3 | `chassis.py:78` | ResNet blocks |
| `stem_kernel` | 3×3 | `chassis.py:83` | Initial convolution |
| `res_kernel` | 3×3 | `chassis.py:58` | ResNet block kernels |
| `normalization` | L2 | `chassis.py:95` | Channel-wise L2 norm |

### Dynamics Network (g_θ)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `latent_channels` | 64 | `chassis.py:112` | Latent state channels |
| `action_channels` | 16 | `spatial_constants.py` | Action plane channels |
| `num_blocks` | 3 | `chassis.py:114` | ResNet blocks |
| `support_size` | 601 | `chassis.py:115` | Reward distribution bins |
| `reward_head_channels` | 16 | `chassis.py:131` | Reward head conv channels |

### Prediction Network (f_θ)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `latent_channels` | 64 | `chassis.py:165` | Input latent channels |
| `trunk_conv_channels` | 32 | `chassis.py:179` | Trunk conv output |
| `trunk_fc_dim` | 256 | `chassis.py:183` | Trunk FC dimension |
| `num_farmer_actions` | 15 | `spatial_constants.py` | Farmer policy dim |
| `num_hand_assignments` | 32 | `spatial_constants.py` | Hands policy dim |
| `num_market_orders` | 20 | `spatial_constants.py` | Market policy dim |
| `support_size` | 601 | `chassis.py:169` | Value distribution bins |

### Chassis (Unified)
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `obs_channels` | 28 | `chassis.py:267` | Observation channels |
| `latent_channels` | 64 | `chassis.py:268` | Latent channels |
| `action_channels` | 16 | `chassis.py:269` | Action plane channels |
| `support_size` | 601 | `chassis.py:273` | Discrete support size |

---

## Discrete Support
| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `SUPPORT_SIZE` | 601 | `types.py:133` | Number of bins |
| `SUPPORT_B` | 300 | `types.py:133` | Support bound |
| `REWARD_SCALE` | 1/1000 | `types.py:131` | Reward scaling |
| Transform φ(x) | sign(x)(√(|x|+1)-1+ε|x|) | `types.py` | Invertible transform |

---

## MCTS Configuration

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `c1` | 1.25 | `mcts.py:109` | PUCT exploration constant |
| `c2` | 19652.0 | `mcts.py:110` | PUCT exploration constant |
| `discount` (γ) | 0.997 | `mcts.py:111` | Discount factor |
| `num_samples` (K) | 8 | `mcts.py:112` | Candidate actions per node |
| `num_simulations` | 25 (train) / 50 (eval) | `mcts.py:113` | MCTS iterations |
| `dirichlet_alpha` | 0.25 | `mcts.py:114` | Root noise concentration |
| `exploration_fraction` | 0.25 | `mcts.py:115` | Root noise weight |

---

## Replay Buffer

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `max_trajectories` | 1000 | `buffer.py:79` | Buffer capacity |
| `discount` (γ) | 0.997 | `buffer.py:82` | n-step bootstrap |
| `n_steps` | 5 | `buffer.py:83` | Bootstrap horizon |
| `k_steps` | 5 | `buffer.py:84` | Unroll length |
| `alpha` (PER) | 0.6 | `buffer.py:224` | Priority exponent |
| `beta_start` | 0.4 | `buffer.py:225` | IS weight start |
| `beta_frames` | 100000 | `buffer.py:226` | Beta annealing frames |
| `eps` | 1e-5 | `buffer.py:227` | Priority floor |

---

## Trainer

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `lr` (PER) | 1e-3 | `trainer.py:60` | Learning rate |
| `lr` (uniform) | 3e-4 | `trainer.py:206` | Learning rate |
| `ema_tau` | 0.995 | `trainer.py:61` | Polyak averaging |
| `consistency_weight` | 0.25 | `trainer.py:62` | SimSiam λ_cons |
| `grad_clip` | 5.0 | `trainer.py:63` | Gradient clipping |
| `weight_decay` | 1e-4 | `trainer.py:83` | AdamW weight decay |
| `T_max` (cosine) | 1000 | `trainer.py:230` | LR schedule period |
| `eta_min` | 1e-5 | `trainer.py:230` | Min learning rate |

---

## Loss Weights (Scheduled)

### Standard Schedule (Regime 1: λ_cons ∈ [0.1, 0.5])
| Weight | Start | End | Direction | Progress |
|--------|-------|-----|-----------|----------|
| Consistency | 0.10 | 0.50 | up | p/0.60 |
| Policy | high | low | down | (p-0.15)/0.70 |
| Value | low | high | up | p |
| Reward | high | low | down | p/0.40 |
| CQL | low | high | up | p/0.80 |

### Boost Schedule (Regime 2: λ_cons ∈ [1.0, 0.5])
| Weight | Start | End | Direction |
|--------|-------|-----|-----------|
| Consistency | 1.00 | 0.50 | down |
| Policy | low | high | up |
| Value | high | low | down |
| Reward | low | high | up |
| CQL | high | low | down |

### Base Anchors
```python
high_val = 0.9637
high_pol = 0.8970
high_rew = 0.7357
high_cql = 0.07857

low_rew = 0.1357
low_pol = 0.1757
low_val = 0.1957
low_cql = 0.01357
```

### Temperature
| Parameter | Value | Source |
|-----------|-------|--------|
| `schedule_temperature` | 0.25 | `low_rank_adaptation.py:252` |

---

## LoRA Configuration

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `LORA_RANK` | 16 | `meta.py:9`, `lora.py:99` | Adapter rank |
| `LORA_ALPHA` | 32 | `lora.py:103` | Scaling (2×rank) |
| `LORA_DROPOUT` | 0.1 | `lora.py:103` | Dropout rate |
| Target modules | All Linear | `lora.py:103` | In h_θ, g_θ, f_θ |

---

## PPO Microcontroller

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `clip_epsilon` | 0.2 | `ppo.py:113` | PPO clip |
| `gae_lambda` | 0.95 | `ppo.py:113` | GAE λ |
| `epochs` | 4 | `ppo.py:113` | Epochs per batch |
| `minibatch_size` | 64 | `ppo.py:113` | Minibatch size |
| `entropy_coef` | 0.01 | `ppo.py:113` | Entropy bonus |
| `value_coef` | 0.5 | `ppo.py:113` | Value loss weight |
| `max_grad_norm` | 0.5 | `ppo.py:113` | Grad clipping |
| `lr` | 3e-4 | `ppo.py:114` | Learning rate |

---

## League / PFSP

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| `smoothing` | 1.0 | `league.py:187` | Laplace prior |
| `main_self_play` | 0.35 | `league.py:291` | Self-play ratio |
| `main_hard_pfsp` | 0.50 | `league.py:294` | Hard PFSP ratio |
| `main_var_pfsp` | 0.15 | `league.py:297` | Var PFSP ratio |
| `exploiter_hard_power` | 3.0 | `league.py:309` | League exploiter p |
| `collapse_threshold` | 0.70 | `league.py` comment | Exploiter reset |

---

## Training Schedule (4 Phases)

| Phase | Name | Key Config | Duration |
|-------|------|------------|----------|
| 1 | Behavioral Cloning | Replay distill, λ_cons=0.1→0.5 | Until convergence |
| 2 | Analytical Bootstrap | Consistency loss, no MCTS | Until latent aligned |
| 3 | MCTS Distillation | Self-play + Reanalyze | Continuous |
| 4 | LoRA Fine-tune | Frozen backbone, rank=16 | Until plateau |

---

## Storage & Disk

| Workspace | Budget | Pruning | Immune |
|-----------|--------|---------|--------|
| This repo | 10.0 GB | mtime asc | `^[0-9]+\.json$` |
| External | 24.0 GB | mtime asc | `^[0-9]+\.json$` |

---

## Evaluation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `games_per_baseline` | 4 (gate) / 8-16 (cert) | Tournament games |
| `promotion_threshold` | 0.55 | Champion win rate |
| `min_replay_score` | 80000.0 | Save threshold |
| `episode_steps` | 720 | Full game length |

---

## Constants (from `spatial_constants.py`)

```python
OBS_CHANNELS = 28
LATENT_CHANNELS = 64
ACTION_CHANNELS = 16
NUM_FARMER_ACTIONS = 15
NUM_HAND_ASSIGNMENTS = 32
NUM_MARKET_ORDERS = 20
SUPPORT_SIZE = 601
SUPPORT_B = 300
DISCOUNT = 0.997
```

## Action Plane Encoding (16 channels)

| Channels | Meaning | Size |
|----------|---------|------|
| 0-4 | Farmer one-hot (5 actions) | 5 |
| 5-9 | Hands one-hot (5 actions) | 5 |
| 10-15 | Market one-hot (6 actions) | 6 |

## Code References

- `muzero/spatial_constants.py:1` - All spatial constants
- `muzero/types.py:1` - Types, DiscreteSupport, macros
- `muzero/chassis.py:1` - Network architectures
- `muzero/mcts.py:106-115` - MCTS config
- `muzero/buffer.py:79-84` - Buffer config
- `muzero/trainer.py:51-63` - Trainer config
- `pipelines/low_rank_adaptation.py:1` - Loss scheduling
- `pipelines/adaptive_moment_estimation.py:1` - Boost scheduling
- `muzero/lora.py:99` - LoRA config
- `muzero/ppo.py:113` - PPO config
- `evaluation/league.py:187` - League config