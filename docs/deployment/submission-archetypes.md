# Submission Packaging & Archetypes

## Overview

The system produces **three canonical submission archetypes** from trained checkpoints, each targeting different deployment strategies.

## Three Archetypes

| Archetype | Zip File | Entry Point | Weights | Description |
|-----------|----------|-------------|---------|-------------|
| **Pure Replay (PL)** | `head_pl_submission.zip` | `main_pl.py` | None | Distilled expert lessons only |
| **Hybrid (HPM)** | `head_hybrid_submission.zip` | `main_hpm.py` | `muzero_checkpoints.pt` | 2033 chassis + MuZero NeML |
| **Pure MuZero (MZ)** | `head_muzero_submission.zip` | `main_mz.py` | `muzero_checkpoints_champion.pt` | Full MuZero pipeline |

## Build Process

```bash
# Build all three archetypes
python scripts/build_head_submissions.py

# Or legacy single target
python package_submission.py
```

### Output
```
dist/
├── main_pl.py              # Pure replay control
├── main_hpm.py             # Hybrid 2033 + MuZero
├── main_mz.py              # Pure MuZero
├── compiled_submission.py  # Build metadata
├── head_pl_submission.zip
├── head_hybrid_submission.zip
└── head_muzero_submission.zip
```

## Archetype Details

### 1. Pure Replay (PL) - `head_pl_submission.zip`

**Entry**: `main_pl.py`  
**Weights**: None (no neural network)  
**Strategy**: Monolithic reactive chassis with distilled expert lessons

**Layers** (from `hybrid_chassis/layers/`):
1. `base_patches.py` - Engine compatibility fixes
2. `resource_guards.py` - Resource protection
3. `v9_expansion.py` - Expansion logic
4. `tactical_planners.py` - Tactical planning
5. `market_guards.py` - Market safety
6. `v7_stack.py` - Outer execution stack
7. `replay_lessons.py` - Distilled expert parameters (sw_day_min, skip_se, etc.)
8. `action_masking.py` - Survival masks
9. `muzero_options.py` - **Empty** (no MuZero)

**Use Case**: Control baseline, zero neural dependency, fast inference

---

### 2. Hybrid (HPM) - `head_hybrid_submission.zip`

**Entry**: `main_hpm.py`  
**Weights**: `muzero_checkpoints.pt` (standard, not champion)  
**Strategy**: 2033 monolithic chassis + MuZero NeML integration

**Integration Points** (from `hybrid_chassis/pipeline.py`):
- **NeML Step 0 Opening**: MuZero controls first turn arbitrage
- **SE Expansion Evaluation**: MuZero evaluates SE land unlock
- **Trickle Liquidation Gate**: MuZero controls endgame sell decisions
- **Rest of Game**: 2033 reactive chassis (proven 2033 score)

**Weight Loading**:
```python
# Strictly loads standard checkpoint
checkpoint = "muzero_checkpoints.pt"  # NOT champion
```

**Use Case**: Best of both worlds — proven chassis + neural lookahead at key decisions

---

### 3. Pure MuZero (MZ) - `head_muzero_submission.zip`

**Entry**: `main_mz.py`  
**Weights**: `muzero_checkpoints_champion.pt` (PRIORITY)  
**Strategy**: Full modular single-file compilation of all 8 chassis layers + MuZero

**Architecture**:
```
main_mz.py (compiled single file)
├── Layer 1: base_patches
├── Layer 2: resource_guards
├── Layer 3: v9_expansion
├── Layer 4: tactical_planners
├── Layer 5: market_guards
├── Layer 6: v7_stack
├── Layer 7: replay_lessons
├── Layer 8: muzero_options (FULL planner)
└── MuZero Networks: h_θ, g_θ, f_θ (embedded weights)
```

**Weight Priority**:
```python
# From packaging/builder.py
WEIGHT_PRIORITY = [
    "muzero_checkpoints_champion.pt",  # 1st choice
    "muzero_checkpoints.pt",           # 2nd choice
    "muzero_checkpoints_base.pt",      # 3rd choice
]
```

**Use Case**: Maximum neural control, highest ceiling, requires GPU for MCTS

---

## Compilation Pipeline

### 1. Layer Compilation (from `hybrid_chassis/compiler.py`)

```python
def compile_agent(layers, routes, chassis_config):
    # 1. Inline all layer functions
    # 2. Resolve route references
    # 3. Inject constants
    # 4. Generate single main.py
    # 5. Verify no external imports (except stdlib + kaggle-env)
```

### 2. Weight Bundling (from `packaging/builder.py`)

```python
def package_submission(entry_point, checkpoint_path, output_zip):
    # 1. Copy entry_point → main.py
    # 2. Embed checkpoint weights as base64 / torch.save
    # 3. Include DiscreteSupport, encoding constants
    # 4. Verify total size < 50MB (Kaggle limit)
    # 5. Create zip with main.py + metadata
```

### 3. Verification

```bash
# Test submission locally
kaggle competitions submit -c kaggriculture -f dist/head_muzero_submission.zip -m "Test"

# Or run local simulation
python -m kaggle_environments evaluate \
    --agent dist/main_mz.py \
    --agent dist/main_mz.py
```

## Weight Export (Per Checkpoint)

Every training checkpoint automatically exports:

```python
# From muzero/trainer.py (on checkpoint save)
def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    
    # ALSO export discrete heads
    torch.save(model.prediction.farmer_head.state_dict(), 
               path.replace(".pt", "_policy_head.pt"))
    torch.save(model.prediction.value_head.state_dict(),
               path.replace(".pt", "_value_head.pt"))
```

### Artifacts Created
```
artifacts/
├── muzero_checkpoints.pt              # Latest training
├── muzero_checkpoints_champion.pt     # Promoted champion
├── muzero_policy_head.pt              # Farmer+Hands+Market heads
├── muzero_value_head.pt               # Value head
├── candidate_iter_012.pt              # Snapshots
├── candidate_iter_042.pt
└── lora_adapters_champion.pt          # If LoRA used
```

## Kaggle Submission Requirements

| Requirement | Value |
|-------------|-------|
| Max zip size | 50 MB |
| Entry point | `main.py` at root |
| Python version | 3.9+ |
| Allowed imports | stdlib, numpy, torch (CPU), kaggle-env |
| Inference time | < 1 sec/step (720 steps = 12 min) |
| Memory limit | 16 GB |

## Submission Checklist

- [ ] `main.py` at zip root
- [ ] All weights embedded (base64 or torch tensor)
- [ ] No external file dependencies
- [ ] Inference < 1s/step on CPU
- [ ] Handles both player 0 and 1
- [ ] Graceful fallback on errors
- [ ] Returns valid Kaggle action format

## Code References

- `scripts/build_head_submissions.py:1` - Three archetype build
- `packaging/builder.py:1` - Submission packaging
- `hybrid_chassis/compiler.py:1` - Single-file compilation
- `hybrid_chassis/builder.py:1` - Agent factory
- `hybrid_chassis/pipeline.py:1` - Macro→Kaggle action translation
- `hybrid_chassis/layers/muzero_options.py:1` - MuZero integration layer
- `hybrid_chassis/layers/replay_lessons.py:1` - Distilled parameters
- `package_submission.py:1` - Legacy single build