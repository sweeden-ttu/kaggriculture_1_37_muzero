# Phase 4: LoRA Fine-Tuning + PPO Microcontroller

## Overview

Phase 4 fine-tunes the champion foundation model using Low-Rank Adaptation (LoRA) and a PPO microcontroller for online adaptation. The champion weights are frozen; only rank-r adapters are trained.

## LoRA Fine-Tuning

### Architecture

```
Champion Foundation (frozen)
    │
    ├── h_θ (SpatialRepresentationNetwork)  ── LoRA rank=16
    ├── g_θ (SpatialDynamicsNetwork)        ── LoRA rank=16
    └── f_θ (SpatialPredictionNetwork)      ── LoRA rank=16
```

### Injection (from `muzero/lora.py:103-108`)

```python
# Inject LoRA into linear layers
inject_lora(model, rank=16, alpha=32, dropout=0.1)

# Only LoRA params are trainable
for param in model.parameters():
    param.requires_grad = False  # Freeze base
for param in lora_parameters(model):
    param.requires_grad = True   # Train adapters
```

### Merge for Deployment (from `muzero/lora.py:106-108`)

```python
# Fold adapters into base weights for standalone submission
merge_lora(model)
save_muzero_checkpoint(model, "champion_lora_merged.pt")
```

### Export Adapters Only (for transfer)
```python
save_lora_adapters(model, "champion_lora_adapters.pt")
load_lora_adapters(model, "champion_lora_adapters.pt")
```

### Hyperparameters
| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank (r) | 16 | `muzero/meta.py:9`, `muzero/lora.py:99` |
| LoRA alpha | 32 (2×r) | `muzero/lora.py:103` |
| LoRA dropout | 0.1 | `muzero/lora.py:103` |
| Target modules | All Linear in h_θ, g_θ, f_θ | `muzero/lora.py:103` |

### Training Command
```bash
# Fine-tune champion with LoRA
./train.sh --use-lora --lora-rank 16
```

## PPO Microcontroller

### Purpose
Online policy adjustment during self-play for:
- Exploration bonus calibration
- Action masking compliance
- Real-time value correction

### Architecture (from `muzero/ppo.py`)

```python
ActorCritic:
    shared_encoder: Linear(64*10*10 → 256) + ReLU
    actor_head: Linear(256 → 15+32+20)  # Factorized policy
    critic_head: Linear(256 → 1)        # Scalar value
```

### PPO Hyperparameters (from `muzero/ppo.py:113-114`)

| Parameter | Value |
|-----------|-------|
| Clip ε | 0.2 |
| GAE λ | 0.95 |
| Epochs per batch | 4 |
| Minibatch size | 64 |
| Entropy coef | 0.01 |
| Value coef | 0.5 |
| Max grad norm | 0.5 |
| LR | 3e-4 |

### Rollout Collection
```python
# From muzero/ppo.py:116-117
ppo_rollout_episode(env, actor_critic, max_steps=720)
ppo_rollout_batch(env, actor_critic, num_episodes=8)
```

### Checkpoint Management
```python
save_ppo_checkpoint(actor_critic, optimizer, "ppo_checkpoint.pt")
load_ppo_checkpoint("ppo_checkpoint.pt") → (actor_critic, optimizer)
resolve_ppo_checkpoint() → path
```

## Champion Checkpointing

### Promotion Gate
```python
# From pipelines/self_improving_loop.py
def evaluate_and_promote(candidate, champion, eval_episodes=8):
    results = tournament(
        agent_a=candidate,
        agent_b=champion,
        games_per_matchup=eval_episodes
    )
    win_rate = results["candidate_wins"] / eval_episodes
    
    if win_rate >= 0.55:  # 55% threshold
        promote_candidate(candidate)
        return True
    return False
```

### Checkpoint Artifacts
Each promotion saves:
```
artifacts/
├── muzero_checkpoints_champion.pt      # Full model (champion)
├── muzero_checkpoints.pt               # Latest trained
├── muzero_policy_head.pt               # Policy head only
├── muzero_value_head.pt                # Value head only
├── candidate_iter_XXX.pt               # Candidate snapshots
└── lora_adapters_champion.pt           # LoRA adapters (if used)
```

### Resolution Logic (from `muzero/legacy_macro.py:86-87`)
```python
resolve_muzero_checkpoint() → path  # Finds latest champion
load_muzero_checkpoint(path) → model_state
```

## Consolidated Training Flow

```
Phase 1: BC (Replay Distill)
    │
    ▼
Phase 2: Bootstrap (Consistency)
    │
    ▼
Phase 3: MCTS Distill (Self-Play + Reanalyze)
    │
    ▼  [Champion reaches threshold]
Phase 4a: LoRA Fine-Tune (Frozen backbone, rank=16)
    │
    ▼  [Merge adapters]
Phase 4b: PPO Microcontroller (Online adaptation)
    │
    ▼
Continuous Loop:
    Self-Play → Buffer → Train (LoRA) → Evaluate → Promote
```

## Loss Schedule in Phase 4

### Standard (train.sh)
```bash
# LoRA training uses same scheduler but only adapter params
--use-lora --lora-rank 16
```

### Boost (boost.sh) — Antagonistic
```bash
# Inverted consistency: 1.0 → 0.5
# Tests opposite gradient directions
./boost.sh hybrid --eval-episodes 4
```

### Weight Distinctness (Always Enforced)
```python
# From pipelines/low_rank_adaptation.py:43-114
validate_loss_weights_rule(
    consistency_weight, policy_weight, value_weight, reward_weight, cql_weight,
    auto_adjust=True  # Jitters colliding weights
)
# Rule: consistency ≠ policy ≠ value ≠ reward ≠ cql
```

## Submission Packaging

### Three Archetypes (from `scripts/build_head_submissions.py`)
```bash
python scripts/build_head_submissions.py
```

| Archetype | Entry Point | Weights | Description |
|-----------|-------------|---------|-------------|
| `head_pl_submission.zip` | `main_pl.py` | None (pure replay) | Distilled expert lessons only |
| `head_hybrid_submission.zip` | `main_hpm.py` | `muzero_checkpoints.pt` | Hybrid: 2033 chassis + MuZero NeML |
| `head_muzero_submission.zip` | `main_mz.py` | `muzero_checkpoints_champion.pt` | Pure MuZero pipeline |

### Compilation Process
1. `hybrid_chassis/compiler.py` compiles layers → single `main.py`
2. `packaging/builder.py` bundles weights + code → zip
3. Weight priority: `champion.pt` > `latest.pt` > `base.pt`

## Code References

- `muzero/lora.py:1` - LoRA injection, merge, save/load
- `muzero/ppo.py:1` - ActorCritic, PPOBuffer, rollout, checkpoint
- `pipelines/self_improving_loop.py:1` - Promotion gate
- `scripts/build_head_submissions.py:1` - Three archetype build
- `packaging/builder.py:1` - Submission packaging
- `hybrid_chassis/compiler.py:1` - Single-file compilation
- `muzero/meta.py:9` - `LORA_RANK = 16`
- `muzero/legacy_macro.py:86-87` - Checkpoint resolution