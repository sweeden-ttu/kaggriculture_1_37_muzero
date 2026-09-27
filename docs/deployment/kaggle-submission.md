# Kaggle Submission Guide

## Overview

Complete workflow for submitting trained agents to the Kaggriculture Kaggle competition.

## Prerequisites

### Kaggle CLI Setup
```bash
# Install
pip install kaggle

# Configure (one-time)
kaggle config set -n competition -v kaggriculture
kaggle config set -n username -v YOUR_USERNAME
kaggle config set -n key -v YOUR_API_KEY

# Or use kaggle.json in ~/.kaggle/
```

### Environment
```bash
# Verify competition access
kaggle competitions list -s kaggriculture

# Check current leaderboard
kaggle competitions leaderboard kaggriculture --show
```

## Submission Pipeline

### 1. Build Submission Package

```bash
# Build all three archetypes
python scripts/build_head_submissions.py

# Output in dist/
ls dist/*.zip
# head_pl_submission.zip
# head_hybrid_submission.zip
# head_muzero_submission.zip
```

### 2. Validate Locally

```bash
# Quick smoke test (2 games)
python -m kaggle_environments evaluate \
    --agent dist/main_mz.py \
    --agent dist/main_mz.py \
    --config '{"episodeSteps": 720}'

# Or use tournament harness
python tournament.py benchmark \
    --agent dist/main_mz.py \
    --baselines baselines \
    --games-per-baseline 2
```

### 3. Submit to Kaggle

```bash
# Submit hybrid (recommended for first submissions)
kaggle competitions submit -c kaggriculture \
    -f dist/head_hybrid_submission.zip \
    -m "Hybrid 2033 + MuZero NeML, iter 42"

# Submit pure MuZero
kaggle competitions submit -c kaggriculture \
    -f dist/head_muzero_submission.zip \
    -m "Pure MuZero, champion iter 42, LoRA rank 16"

# Submit pure replay (control)
kaggle competitions submit -c kaggriculture \
    -f dist/head_pl_submission.zip \
    -m "Pure replay distilled lessons, control"
```

### 4. Monitor Submission

```bash
# Check status
kaggle competitions submissions kaggriculture

# View specific submission
kaggle competitions submissions kaggriculture -v

# Get leaderboard position
kaggle competitions leaderboard kaggriculture --show
```

## Submission Metadata

### Required Files in Zip
```
submission.zip
├── main.py                 # Entry point (REQUIRED at root)
├── *.pt                    # Model weights (embedded or alongside)
├── *.json                  # Config / metadata (optional)
└── ...                     # Any other dependencies
```

### Size Limits
| Limit | Value |
|-------|-------|
| Zip size | 50 MB max |
| Inference time | < 1 sec/step |
| Total episode | < 12 min (720 steps) |
| Memory | 16 GB |

### Weight Embedding Strategy

**Option A: Base64 in main.py** (for small weights)
```python
# In main.py
import base64
import torch
import io

WEIGHTS_B64 = "..."
weights = torch.load(io.BytesIO(base64.b64decode(WEIGHTS_B64)))
```

**Option B: Separate .pt files** (for large weights)
```
submission.zip
├── main.py
├── model.pt
└── ...
```

**Option C: TorchScript** (for production)
```python
# Export
scripted = torch.jit.script(model)
torch.jit.save(scripted, "model.ts")

# Load
model = torch.jit.load("model.ts")
```

## Kaggle Environment Details

### Observation Format
```python
observation = {
    "step": 0,
    "player": 0,
    "farms": [
        {
            "farmer": {"x": 5, "y": 5},
            "hands": [{"x": 4, "y": 5}, ...],
            "tiles": [[...], ...],  # 10x10
            "animals": [...],
            "money": 1000.0,
            "shed": {"WHEAT": 10, ...},
            "unlocked_quadrants": ["NW"],
        },
        {...}  # Opponent farm
    ],
    "market_prices": {"WHEAT": 10, "CARROT": 15, ...},
    "private": {"shed": {...}}  # Your private info
}
```

### Action Format (Return from `agent()`)
```python
action = {
    "farmer": ["PLANT", "WHEAT"],           # Or ["MOVE", "UP"], ["PASS"]
    "hands": [
        ["PLANT", "CARROT"],                # Hand 0
        ["WATER"],                          # Hand 1
        ["PASS"],                           # Hand 2
    ],
    "market": [
        ["SELL", "WHEAT", 5],               # Sell 5 wheat
        ["BUY_SEED", "STRAWBERRY", 3],      # Buy 3 strawberry seeds
    ]
}
```

### Configuration
```python
configuration = {
    "episodeSteps": 720,
    "actTimeout": 1000,  # ms per step
    "runTimeout": 720000,  # ms total
}
```

## Best Practices

### 1. Error Handling
```python
def agent(observation, configuration):
    try:
        return my_policy(observation)
    except Exception as e:
        # ALWAYS return valid action on error
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in range(num_hands)],
            "market": []
        }
```

### 2. Deterministic Inference
```python
# Set seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 3. CPU Optimization
```python
# Use torch.compile if available (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# Or use ONNX for faster CPU inference
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 4. Memory Management
```python
# Clear cache between steps
torch.cuda.empty_cache()  # If GPU
# For CPU: avoid accumulating tensors in lists
```

## Debugging Failed Submissions

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `main.py not found` | Zip structure wrong | Ensure `main.py` at root |
| `ImportError: torch` | Missing dependency | Kaggle has torch preinstalled |
| `Timeout` | Inference too slow | Optimize model, reduce MCTS sims |
| `OOM` | Memory leak | Clear caches, smaller batch |
| `Invalid action` | Wrong format | Validate action dict structure |

### Debug Locally
```bash
# Run with Kaggle environment simulation
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['dist/main_mz.py', 'dist/main_mz.py'])
print(env.steps[-1][0].reward)
"
```

## Post-Submission Analysis

### 1. Download Replay
```bash
# After submission completes
kaggle competitions download -c kaggriculture -f submission_XXXXX_replay.json
```

### 2. Analyze Replay
```bash
# Learn from high-scoring replays
python learn_from_replay.py --replay submission_XXXXX_replay.json
```

### 3. Compare Scores
```bash
# Track progression
python -c "
import json
with open('submission_history.json') as f:
    hist = json.load(f)
for s in hist:
    print(f\"{s['date']}: {s['public_score']} / {s['private_score']}\")
"
```

## Automated Submission Script

```bash
#!/bin/bash
# submit.sh - Automated submission with validation

set -e

ARCHETYPE=${1:-hybrid}  # pl, hybrid, muzero
MESSAGE=${2:-"Auto-submit $(date)"}

echo "Building $ARCHETYPE submission..."
python scripts/build_head_submissions.py

ZIP="dist/head_${ARCHETYPE}_submission.zip"
if [ ! -f "$ZIP" ]; then
    echo "ERROR: $ZIP not found"
    exit 1
fi

echo "Validating..."
python -m kaggle_environments evaluate \
    --agent dist/main_${ARCHETYPE}.py \
    --agent dist/main_${ARCHETYPE}.py \
    --config '{"episodeSteps": 100}' > /dev/null

echo "Submitting..."
kaggle competitions submit -c kaggriculture -f "$ZIP" -m "$MESSAGE"

echo "Done! Check status with:"
echo "  kaggle competitions submissions kaggriculture"
```

## Code References

- `scripts/build_head_submissions.py:1` - Build all archetypes
- `packaging/builder.py:1` - Zip packaging
- `hybrid_chassis/compiler.py:1` - Single-file compilation
- `package_submission.py:1` - Legacy single build
- `tournament.py:1` - Local validation
- `learn_from_replay.py:1` - Replay analysis