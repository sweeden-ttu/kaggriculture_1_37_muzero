# Continuous Self-Play & Reinforcement Learning Loop

## Overview

The autonomous training loop runs indefinitely, cycling through self-play, training, evaluation, and champion promotion until performance plateaus.

## Loop Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS SELF-IMPROVING LOOP                               │
└────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │  CHAMPION   │◄──────────────────────────────────────┐
    │  MODEL      │                                       │
    └──────┬──────┘                                       │
           │                                              │
           ▼                                              │
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  SELF-PLAY  │────►│ REPLAY BUF  │────►│   TRAINER   │
    │  WORKERS    │     │  (PER, 10GB)│     │  (EMA,      │
    │  (MCTS)     │     │  Reanalyze  │     │   LoRA)     │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │  EVALUATOR  │
                                            │ (Tournament)│
                                            └──────┬──────┘
                                                   │
                                    ┌──────────────┴──────────────┐
                                    │                             │
                              ┌─────┴─────┐                 ┌─────┴─────┐
                              │  PROMOTE  │                 │  REJECT   │
                              │  NEW      │                 │  KEEP     │
                              │ CHAMPION  │                 │  OLD      │
                              └─────┬─────┘                 └───────────┘
                                    │
                                    └──────────────────────► (back to CHAMPION)
```

## Component Details

### 1. Self-Play Workers

```python
# From pipelines/self_improving_loop.py
class SelfPlayWorker:
    def __init__(self, champion_checkpoint, num_workers=4):
        self.champion = load_champion(champion_checkpoint)
        self.mcts = SampledMuZeroMCTS(self.champion, num_simulations=50)
    
    def generate_games(self, num_games=8):
        trajectories = []
        for _ in range(num_games):
            traj = self.play_game()
            trajectories.append(traj)
        return trajectories
    
    def play_game(self):
        traj = GameTrajectory()
        obs = env.reset()
        done = False
        while not done:
            # MCTS search with current champion
            action, policy, value = self.mcts.search(encode(obs))
            
            # Execute in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store step
            traj.append_step(
                obs=encode(obs).numpy(),
                action=action,
                reward=reward,
                policy_dict=policy,      # MCTS visit distribution
                value=value,             # MCTS root value
                priority=1.0
            )
            obs = next_obs
        traj.finalize(encode(obs).numpy())
        return traj
```

### 2. Replay Buffer (Dynamic)

**Capacity**: 10.0 GB (this repo) / 24.0 GB (external)
**Pruning**: Oldest `selfplay_*.json` by `mtime`, canonical replays immune

```python
# From pipelines/replay_manager.py
class ReplayManager:
    def enforce_budget(self, budget_gb=10.0):
        replay_dir = Path("replays")
        selfplay_files = sorted(
            replay_dir.glob("selfplay_*.json"),
            key=lambda f: f.stat().st_mtime  # Oldest first
        )
        current_gb = sum(f.stat().st_size for f in selfplay_files) / 1e9
        
        for f in selfplay_files:
            if current_gb <= budget_gb:
                break
            f.unlink()
            current_gb -= f.stat().st_size / 1e9
```

### 3. Trainer (Continuous)

```python
# From pipelines/self_improving_loop.py
class ContinuousTrainer:
    def __init__(self, model, buffer, mcts):
        self.trainer = PrioritizedMuZeroTrainer(
            online_model=model,
            simsiam_head=SimSiamProjectionPredictionHead(),
            buffer=buffer,
            mcts_engine=mcts,
            lr=3e-4,
            ema_tau=0.995,
            consistency_weight=0.25,  # Scheduled
        )
    
    def train_iteration(self, steps=100):
        for step in range(steps):
            metrics = self.trainer.train_step(
                batch_size=16, 
                k_steps=5, 
                global_step=step
            )
            
            # Periodic reanalyze
            if step % 100 == 0:
                self.trainer.run_reanalyze_pass(num_trajectories=2)
        
        return metrics
```

### 4. Evaluator (Tournament)

```python
# From evaluation/league.py, tournament.py
def evaluate_candidate(candidate_path, baseline_paths, games_per=4):
    results = {}
    for baseline in baseline_paths:
        result = play_match(candidate_path, baseline, games=games_per)
        results[baseline] = result["win_rate"]
    return results

# Promotion threshold
PROMOTION_THRESHOLD = 0.55  # 55% win rate vs champion
```

### 5. Champion Promotion

```python
# Atomic promotion
def promote_champion(candidate_path):
    # 1. Copy candidate to champion location
    shutil.copy2(candidate_path, "artifacts/muzero_checkpoints_champion.pt")
    
    # 2. Export discrete heads for deployment
    export_heads(candidate_path)
    
    # 3. Notify self-play workers (hot reload)
    for worker in self_play_workers:
        worker.reload_champion()
    
    # 4. Log promotion
    log_promotion(candidate_path, timestamp, eval_results)
```

## Disk Budget Management

### Two-Tier Storage

| Tier | Location | Budget | Contents |
|------|----------|--------|----------|
| **Hot** | `replays/` | 10 GB | Recent self-play + canonical |
| **Cold** | External | 24 GB | Long-run self-play archive |

### Pruning Invariants (Strict)
1. **Canonical Immunity**: `^[0-9]+\.json$` never deleted (50 gold replays)
2. **Timestamp Order**: `selfplay_*.json` sorted by `mtime` ascending
3. **No Score Filtering**: Prevents stagnation traps
4. **Atomic**: Prune completes or rolls back

### Monitoring
```python
def get_disk_usage():
    replay_dir = Path("replays")
    total_gb = sum(f.stat().st_size for f in replay_dir.glob("*.json")) / 1e9
    selfplay_gb = sum(f.stat().st_size for f in replay_dir.glob("selfplay_*.json")) / 1e9
    canonical_gb = sum(f.stat().st_size for f in replay_dir.glob("[0-9]*.json")) / 1e9
    return {"total": total_gb, "selfplay": selfplay_gb, "canonical": canonical_gb}
```

## Launch Commands

### Standard Loop (train.sh)
```bash
#!/bin/bash
# train.sh - Standard curriculum
python pipelines/train_muzero.py \
    --mode hybrid \
    --use-lora \
    --lora-rank 4 \
    --consistency-weight 0.4357 \
    --eval-episodes 8 \
    --train-steps 1000 \
    --reanalyze-freq 100
```

### Boost Loop (boost.sh)
```bash
#!/bin/bash
# boost.sh - Inverted/antagonistic curriculum
python pipelines/train_muzero.py \
    --mode hybrid \
    --use-lora \
    --lora-rank 16 \
    --boost-schedule \
    --eval-episodes 4 \
    --train-steps 1000 \
    --reanalyze-freq 50
```

### Online Continuous Mode
```bash
# Runs indefinitely until Ctrl+C or threshold met
./train.sh online
# or
./boost.sh hybrid  # Runs continuous with boost schedule
```

## Stopping Criteria

The loop continues until:
- [ ] Champion win rate vs baselines plateaus (< 0.5% improvement over 10 promotions)
- [ ] Target score achieved (e.g., > $150k)
- [ ] Compute budget exhausted
- [ ] Manual stop

## Monitoring & Logging

### Key Metrics Tracked
| Metric | Source | Frequency |
|--------|--------|-----------|
| Self-play games generated | SelfPlayWorker | Per game |
| Buffer size / trajectories | ReplayManager | Per prune |
| Training loss (policy/value/reward/consistency) | Trainer | Per step |
| EMA weight distance | Trainer | Per step |
| Reanalyze loss delta | Trainer | Per reanalyze |
| Candidate vs champion win rate | Evaluator | Per eval |
| Promotion count | Promoter | Per promotion |
| Disk usage | ReplayManager | Per prune |

### TensorBoard / Logging
```python
# Add to trainer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs/train")

def log_metrics(metrics, step):
    for k, v in metrics.items():
        writer.add_scalar(f"train/{k}", v, step)
```

## Code References

- `pipelines/self_improving_loop.py:1` - Loop orchestration
- `pipelines/replay_manager.py:1` - Disk budget enforcement
- `pipelines/train_muzero.py:1` - Training entry point
- `train.sh:1` / `boost.sh:1` - Launch scripts
- `evaluation/league.py:56-174` - Tournament engine
- `evaluation/league.py:181-311` - League + PFSP
- `muzero/trainer.py:51-320` - Trainers with Reanalyze