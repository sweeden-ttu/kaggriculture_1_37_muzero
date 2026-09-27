# Frequently Asked Questions

## Architecture & Design

### Why Spatial ResNet instead of MLP?

The 10×10 board has spatial structure (adjacency, quadrants, local patterns). Convolutions preserve this topology and provide translation equivariance. The legacy MLP macro-option MuZero (`muzero/legacy_macro.py`) is kept for ablation/packaging only.

### Why Factorized Policy Heads?

Joint action space = 15 × 32 × 20 = 9,600 combinations. Flat softmax would be:
- Memory: ~38KB per logits vector (manageable)
- MCTS: Impossible to expand all 9,600 actions per node
- Sample efficiency: Most actions never visited

Factorization reduces to 15+32+20=67 logits + independent sampling.

### Why Sampled MCTS (K=8) instead of Full Expansion?

Full expansion would require evaluating f_θ 9,600 times per node. With K=8:
- 8 forward passes per expansion vs 9,600
- 1200× speedup per node
- Sufficient coverage if policy heads are calibrated

### Why Discrete Support (601 bins) instead of Regression?

- Handles multi-scale returns (small rewards ±10 to large harvests ±10,000)
- Categorical cross-entropy more stable than MSE for heavy-tailed distributions
- φ-transform compresses dynamic range: φ(x) ≈ sign(x)√|x| for large x
- Enables distributional RL semantics (risk-sensitive)

---

## Training

### What's the difference between train.sh and boost.sh?

| Aspect | train.sh | boost.sh |
|--------|----------|----------|
| Schedule Engine | `low_rank_adaptation.py` | `adaptive_moment_estimation.py` |
| Consistency Range | 0.1 → 0.5 (Regime 1) | 1.0 → 0.5 (Regime 2) |
| Gradient Flow | Cooperative | Antagonistic (inverted) |
| LoRA Rank | 4 | 16 |
| Candidates | `submission_train_cand_*.zip` | `submission_boost_cand_*.zip` |

Both can run in parallel; they explore different loss landscapes.

### Why Two Consistency Regimes?

- **Regime 1 (0.1→0.5)**: Gentle alignment, lets policy/value learn first
- **Regime 2 (1.0→0.5)**: Strong initial representation lock, tests if forced alignment helps

The distinctness rule (all 5 weights different) prevents collapse in both.

### What is Reanalyze and why is it needed?

MCTS targets (π, v) stored in buffer become stale as network improves. Reanalyze:
1. Samples trajectory from buffer
2. Re-runs MCTS with **current** network weights
3. Overwrites π_t, v_t in-place
4. Resets priorities (recomputed on next sample)

Without Reanalyze: training chases moving targets from old network versions.

### How does PER priority work?

Priority = max_k |v_{t+k} - z_{t+k}| + ε over K-step unroll.
- Focuses on transitions where value prediction is wrong
- α=0.6 moderates prioritization (α=1 = full, α=0 = uniform)
- IS weights correct bias: w_i = (N×P(i))^{-β}, β: 0.4→1.0

### Why Polyak EMA (τ=0.995) instead of Hard Target Updates?

- EMA: smooth tracking, no sudden target shifts
- Hard updates (every N steps): can cause instability
- τ=0.995 means target moves 0.5% toward online per step
- Effective horizon ≈ 1/(1-τ) = 200 steps

---

## League & PFSP

### What is PFSP and why use it?

**Prioritized Fictitious Self-Play** (AlphaStar): Instead of uniform opponent sampling, weight by win rate.
- **Hard**: (1-wr)^p — practice against opponents you lose to
- **Var**: wr(1-wr) — practice against competitive ~50% opponents
- **Uniform**: baseline exploration

Creates adaptive curriculum targeting weaknesses.

### What are the agent roles?

| Role | Purpose | Mix |
|------|---------|-----|
| **Main** | Primary learner | 35% self, 50% hard, 15% var |
| **Main Exploiter** | Target current main | 50% vs main, 50% var past |
| **League Exploiter** | Find any weakness | 100% hard (p=3.0) |

### What is Population Collapse?

Exploiters overfit to current main (>70% win rate), stop discovering new strategies. Detected by win-rate threshold; fixed by re-initializing exploiter with noise.

---

## Submissions

### Which archetype should I submit?

| Archetype | Best For | Risk |
|-----------|----------|------|
| **Hybrid (HPM)** | First submissions, reliability | Lower ceiling |
| **Pure MuZero (MZ)** | Maximum score attempts | Needs GPU, slower |
| **Pure Replay (PL)** | Control baseline, ablation | No neural adaptation |

**Recommendation**: Start with Hybrid. Switch to Pure MuZero when champion consistently beats Hybrid in tournaments.

### How are weights embedded in submission zip?

Three strategies (from `packaging/builder.py`):
1. **Base64 in main.py** — For small weights (<1MB)
2. **Separate .pt files** — For larger weights, loaded via `torch.load()`
3. **TorchScript (.ts)** — For production, `torch.jit.save/load`

The builder automatically chooses based on checkpoint size.

### What's the 50MB zip limit?

Kaggle hard limit. Our checkpoints:
- Full model: ~50MB (may exceed)
- Policy head only: ~5MB
- Value head only: ~1MB
- LoRA adapters: ~2MB

**Solution**: Submit `muzero_checkpoints_champion.pt` for Pure MuZero; the builder strips optimizer states and compresses.

---

## Debugging

### My training loss isn't decreasing. What to check?

1. **Replay buffer empty?** → Run self-play first or check `replays/` directory
2. **Learning rate too high?** → Try 1e-4, check grad norms
3. **Consistency weight too high?** → Should be 0.1-0.5 (Regime 1) or 0.5-1.0 (Regime 2)
4. **Target network stale?** → Verify EMA updates in trainer logs
5. **Reanalyze not running?** → Check `run_reanalyze_pass` frequency

### MCTS is slow. How to speed up?

```python
# Reduce simulations
mcts = SampledMuZeroMCTS(chassis, num_simulations=10, num_samples=4)

# Use GPU
device = torch.device("cuda")
chassis.to(device)
mcts.chassis = chassis

# Compile (PyTorch 2.0+)
chassis = torch.compile(chassis, mode="reduce-overhead")
```

### Out of Memory on GPU

```python
# Reduce batch size
trainer.train_step(batch_size=4, k_steps=3)

# Gradient accumulation
accum_steps = 4
for step in range(accum_steps):
    loss = trainer.train_step(batch_size=2) / accum_steps
    loss.backward()
optimizer.step()
```

### "Invalid action" errors in Kaggle

1. Check action format matches exactly:
```python
{
    "farmer": ["PLANT", "WHEAT"],  # List of strings
    "hands": [["PASS"], ["WATER"]],  # List of lists
    "market": [["SELL", "WHEAT", 5]]  # List of lists
}
```

2. Ensure survival masks applied (action_masking.py)
3. Handle empty hands list: `hands = [["PASS"]] * num_hands`

---

## Performance

### What scores are achievable?

| Configuration | Typical Score | Best Known |
|---------------|---------------|------------|
| Pure Replay (PL) | $80k-$95k | $95k |
| Hybrid (HPM) | $95k-$115k | $110,301 |
| Pure MuZero (MZ) | $100k-$130k | $115k+ |
| Kaggle Baselines | 771-2033 | 2033 |

**Note**: Dollar score ≈ Kaggle Score × 50 + offset (non-linear).

### How many self-play games per training iteration?

Default: 8 games per iteration, 100 training steps, then eval.
- 8 games × 720 steps = 5,760 transitions
- Buffer holds ~1000 trajectories
- Training samples 16×5-step = 80 transitions/step

### When does the loop stop?

Manual or automatic criteria:
- Champion win rate plateaus (<0.5% improvement over 10 promotions)
- Target score reached (e.g., $150k)
- Compute budget exhausted
- `Ctrl+C` → graceful shutdown saves checkpoint

---

## Code Navigation

### Where is the main training loop?

```
pipelines/train_muzero.py          # Entry point
pipelines/self_improving_loop.py   # Loop orchestration
pipelines/replay_manager.py        # Disk budget
muzero/trainer.py                  # Train step + Reanalyze
```

### Where are hyperparameters defined?

```
muzero/spatial_constants.py        # Architecture constants
muzero/types.py                    # DiscreteSupport, macros
muzero/meta.py                     # LoRA, PPO defaults
pipelines/low_rank_adaptation.py   # Loss schedules
pipelines/adaptive_moment_estimation.py  # Boost schedules
```

### Where are the 3 submission archetypes built?

```
scripts/build_head_submissions.py  # Main build script
hybrid_chassis/compiler.py         # Single-file compilation
packaging/builder.py               # Zip packaging
dist/                              # Output directory
```

---

## Environment

### Do I need GPU?

- **Training**: Strongly recommended (10-50× speedup)
- **MCTS Inference**: GPU recommended for 25-50 sims/step
- **Submission**: CPU only (Kaggle runs on CPU)

### Python Version?

3.10+ recommended. Tested on 3.10, 3.11, 3.12.

### Key Dependencies

```
torch >= 2.0
numpy >= 1.24
langgraph, langchain-core (for spec agent only)
kaggle-environments (for local testing)
```

---

## Getting Help

### Where to find logs?

```
logs/train/           # TensorBoard (if enabled)
artifacts/            # Checkpoints
replays/              # Self-play data
dist/                 # Submission packages
```

### How to report issues?

Check `AGENTS.md` for repository topology, then file issue with:
1. Component (trainer, MCTS, buffer, etc.)
2. Error message + stack trace
3. Config (train.sh vs boost.sh, hyperparameters)
4. Steps to reproduce