# A Peak Under the Hood - A Trace Language theory of Agents

**A**-R-**C** meaning A - or - C is the choice the an intelligent logical Automaton acting upon the boundaries of a language model will choose in the missing Chain Link of Artificial General Intelligence or A-R-C AGI. Through rigorous mathematical definition we provide critically missing description of "what" an agent is and "how" it operates under a dual belief significance ranked question-learning confidence and follow its natural conclusion until the completion of its understanding allowing a model to choose it's agent. By demonstrating the language model horizons which  turn autonomous agents to to sub-autonomous question and learning, we find Chomsky, Nash, and Turing theoretical application and model our own predictions about how autonomous farmers (acting as autonomous nation states) within the game of Kaggriculture will naturally find a Nash equilibrium given the rules and environment play of Kaggriculture.

A note to reader: this demonstration ties theory and practice to real world domains of economics. Find evidence of general intelligence and meaning from autonomous and sub autonomous agents we describe real world dilemmas and situations that match our experimental outcomes. Should your goal be to WIN your contest rather than PROVE the usefulness of a model I Nash equilibrium will still be your beginning, but focused on the arbitrarily absurd environment one finds yourself competing in. There you may find the limits which afford a different strategy. We begin with the NeML move first demonstrating that money from nothing is one strategy being played in the current game, and according to Nash rules it should be the first in discovery. We choose MuZero for an architectural basis as it fits nicely with our goal of defining a free-agent which acts and understands language through the context of its lens and outer boundary goals. The definition of a deep belief network is no more than our implementation of the MCTS tree in MuZero matched to the Q-learning subqueries of significant consequence to our understanding of artificial general intelligence

## 1. System Architecture

```
                                 Raw Farm State (Dict)
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   KaggricultureObservationEncoder     │  ← 28 channels, 10×10 grid
                      │   (quadrant mask, entities, crops,    │     (unlocked NW, NE, SW, SE)
                      │    livestock, units, global context)  │
                      └───────────────────┬───────────────────┘
                                          │  [Batch, 28, 10, 10]
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    SpatialRepresentationNetwork (h_θ) │  ← 3-block 2D ResNet
                      │    preserves 10×10 grid, L2 normalize │     (no downsampling)
                      └───────────────────┬───────────────────┘
                                          │  s⁰ root latent [Batch, 64, 10, 10]
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
      ┌──────────────────────────────┐          ┌──────────────────────────────┐
      │ FactorizedPredictionNetwork  │          │    SpatialDynamicsNetwork    │
      │           (f_θ)              │          │             (g_θ)            │
      ├──────────────────────────────┤          ├──────────────────────────────┤
      │ • Farmer Head:  15 logits    │          │ Inputs:                      │
      │ • Hands Head:   32 logits    │          │   • sᵏ⁻¹ latent tensor       │
      │ • Market Head:  20 logits    │          │   • aᵏ action plane (16-ch)  │
      │ • Value Head:   601 atoms    │          │ Outputs:                     │
      │   ([-300, 300] discrete)     │          │   • sᵏ next latent tensor    │
      └──────────────┬───────────────┘          │   • rᵏ immediate reward      │
                     │                          │     (601-atom categorical)   │
                     ▼                          └──────────────┬───────────────┘
      ┌──────────────────────────────┐                         │
      │      SampledMuZeroMCTS       │                         ▼
      │ • Candidate Sampling (K=16)  │          ┌──────────────────────────────┐
      │ • Global MinMaxStats         │          │   SimSiam Consistency Head   │
      │ • P-UCT Search + Dirichlet   │          │   Asymmetric g_proj / p_pred │
      │ • Multi-Agent PFSP Opponents │          │   Stop-Gradient on target h  │
      └──────────────────────────────┘          └──────────────────────────────┘
```

### Key Architectural Invariants
1. **Fixed Maximum-Grid Feature Stacking ($10 \times 10 \times 28$)**: Farm quadrants (`NW`, `NE`, `SW`, `SE`) are initialized on a fixed $10\times 10$ plane from turn 0. Progressive land unlocks flip the land mask channel without altering tensor shapes.
2. **Combinatorial Solvency via Factorized Prediction**: Disentangles the $15 \times 32 \times 20 \approx 9,600$ joint action combinations into independent marginal heads:
   - **Farmer Head** (15 actions): Coarse movement (N/E/S/W), water, harvest, plant crops, pass.
   - **Hired Hands Head** (32 chore assignments): Priority dispatch tokens for autonomous field hands.
   - **Market Head** (20 orders): Scaled seed purchases, crop liquidation orders, herd acquisitions, land expansions (`NE`, `SW`, `SE`).
3. **SimSiam Latent Regularization**: Prevents representation drift between encoder $h_\theta(o_{t+k})$ and unrolled dynamics $g_\theta(s^{k-1}, a^k)$ using asymmetric projection/prediction heads with explicit target stop-gradients (`detach()`).
4. **Sampled MCTS with Tree-Wide MinMaxStats**: Expands $K=16$ candidate joint actions per leaf; normalizes $Q$-values globally across the tree for stable P-UCT exploration.
5. **Prioritized Experience Replay (PER)**: Computes TD errors with importance sampling (IS) correction and $n$-step bootstrapping.
6. **Canonical Replay Immunity**: Canonical Kaggle ground-truth replays (`replays/^[0-9]+\.json$`) are permanently protected from pruning; self-play replays (`selfplay_*.json`) are dynamically pruned by `mtime` under disk volume budgets.

---

## 2. Getting Started & Prerequisites

### Environment Setup
The codebase runs on **macOS / Linux** with Python 3.12 (Miniforge / Conda environment named `kagg`):

```bash
# 1. Clone / Navigate to workspace
cd ~/kaggriculture_1_37_muzero

# 2. Activate conda environment
conda activate kagg
# or ensure /Users/sweeden/miniforge3/envs/kagg/bin/python is on PATH

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Verify test suite & type checker
make test
pyright muzero evaluation hybrid_chassis pipelines tests
```

---

## 3. The Rank #1 Champion Build Workflow ($185,000 Playbook)

To produce the highest-performing autonomous champion agent that commands the leaderboard, follow the structured 6-step workflow below using the canonical `make` targets:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RANK #1 CHAMPION BUILD PIPELINE                        │
│                                                                             │
│   Step 1: Ingest Gold Kaggle Replays          →  make gold                  │
│   Step 2: Distill Strategic Micro-Rules       →  make distill               │
│   Step 3: Run Full 4-Phase Curriculum         →  make phases BATCH=32       │
│           (or high-capacity competition)      →  make weights-competition   │
│   Step 4: Continuous Self-Play Refinement     →  make loop-spatial          │
│   Step 5: Certify Against Score 2033 Gate     →  make gate-buoy             │
│   Step 6: Compile Standalone Submission Zips  →  make build                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Ingest Top Gold Replays
Acquires canonical dual-$100k+ Kaggle replays:
```bash
make gold
```
*Checks canonical count against threshold (35). If needed, fetches top matches from leaderboard Gold teams.*

### Step 2: Distill Expert Strategic Policies
Extracts operational constants (`sw_day_min`, seed allocations, liquidation timings):
```bash
make distill
```

### Step 3: Train Spatial MuZero Weights
Train the complete 4-phase curriculum end-to-end:
```bash
# Standard training run:
make phases BATCH=32 SIMS=16

# High-capacity competition run (deeper rollouts, larger batch size):
make weights-competition
```

### Step 4: Continuous Online Self-Play Loop
Runs autonomous self-play iterations, expanding the PER buffer and promoting champion weights upon positive validation:
```bash
make loop-spatial ITERATIONS=10 PROMOTE_EVERY=1 SIMS=16 BATCH=32
```

### Step 5: Validate Against Kaggle Scored Baselines & Gate 2033
Certifies the new weights in Head-to-Head matches against the official 2033 benchmark:
```bash
make gate-buoy
make benchmark
```

### Step 6: Build Standalone Submission Packages
Packages self-contained runnable `.zip` bundles ready for Kaggle submission:
```bash
make submission
```
*Builds and validates `dist/head_muzero_submission.zip`, `dist/head_hybrid_submission.zip`, and `dist/head_pl_submission.zip`.*

---

## 4. Complete Makefile Command Reference

Below is every `make` target implemented in [`Makefile`](file:///Users/sweeden/kaggriculture_1_37_muzero/Makefile) with its dials, default settings, and usage examples.

### A. Data & Replay Management

| Command | Description | Configurable Variables |
| :--- | :--- | :--- |
| `make gold` | Downloads top-tier Gold Kaggle replays (both players $> \$100\text{k}$) | `GOLD_TARGET=50`, `GOLD_THRESHOLD=35` |
| `make replays` | Alias for `make gold` | `GOLD_TARGET=50` |
| `make distill` | Analyzes replays and distills policy parameters into `artifacts/ep*_policy.json` | None |
| `make prune` | Safely prunes stale `selfplay_*.json` by `mtime` while protecting canonical replays | `PRUNE_MAX_GB=10.0`, `PRUNE_ARGS` |
| `make clean` | Wipes non-canonical self-play replays from `replays/` | None |

**Examples:**
```bash
make gold GOLD_TARGET=60
make prune PRUNE_MAX_GB=8.0
make distill
```

---

### B. Spatial Curriculum Phases (Isolated Phase Execution)

Run each training curriculum phase in isolation to iterate, inspect loss curves, or debug specific subsystems:

| Command | Curriculum Phase | Description | Key Variables |
| :--- | :--- | :--- | :--- |
| `make phase-0` | **Data Bootstrap** | Validates presence of gold Kaggle replays in `replays/` | `GOLD_THRESHOLD=35` |
| `make phase-1` | **Expert BC** | Behavioral cloning on gold replays with TD-error PER into spatial ResNet | `PHASE1_STEPS=120`, `BATCH=32`, `LR=1e-3` |
| `make phase-2` | **SimSiam Bootstrap** | SimSiam dynamics bootstrapping with stop-gradients on trajectory buffer | `PHASE2_STEPS=30`, `BATCH=32`, `LR=1e-3` |
| `make phase-3` | **Sampled-MCTS Distill** | Self-play trajectory harvest + policy/value target distillation | `PHASE3_STEPS=40`, `SELFPLAY_EPISODES=4`, `SIMS=16` |
| `make phase-4` | **Reanalyze Refresh** | Reanalyzes historical trajectory slices with latest weights; updates PER priorities | `PHASE4_STEPS=20`, `BATCH=32` |
| `make phases` | **Full Sequence** | Executes `phase-0` $\to$ `phase-1` $\to$ `phase-2` $\to$ `phase-3` $\to$ `phase-4` $\to$ `promote-weights` | `BATCH=32`, `SIMS=16`, `PHASE1_STEPS=120`, etc. |

**Examples:**
```bash
# Run isolated SimSiam bootstrap with 30 steps:
make phase-2 PHASE2_STEPS=30

# Run full sequence with custom batch size and MCTS simulations:
make phases BATCH=32 SIMS=16

# Run isolated MCTS distillation with 8 selfplay games:
make phase-3 PHASE3_STEPS=50 SELFPLAY_EPISODES=8 SIMS=24
```

---

### C. Weight Building Presets

| Command | Architecture | Description |
| :--- | :--- | :--- |
| `make weights-spatial` | Spatial (Production) | Executes complete spatial Sampled MuZero pipeline, saves to `artifacts/muzero_spatial_checkpoints.pt`, and promotes champion |
| `make weights` | Spatial (Production) | Alias for `make weights-spatial` |
| `make weights-competition` | Spatial (High-Capacity) | Competitive training run: `COMP_PHASE1_STEPS=200`, `COMP_PHASE2_STEPS=80`, `COMP_PHASE3_STEPS=100`, `COMP_PHASE4_STEPS=40`, `COMP_SELFPLAY=16`, `COMP_BATCH=64`, `COMP_SIMS=50`, `COMP_K_STEPS=5` |
| `make weights-macro` | Legacy Macro MLP | Trains the legacy 8-macro-option MLP backbone with LoRA rank-16 adapters |
| `make train-weights` | Dynamic Dispatch | Automatically executes `weights-macro` if `ARCH=macro`, otherwise `weights-spatial` |
| `make smoke-spatial` | Spatial Smoke Test | Fast 1-step verification of spatial representation, candidate sampling, and checkpoint I/O |
| `make smoke-macro` | Legacy Smoke Test | Fast 1-step verification of legacy macro MLP network |

**Examples:**
```bash
# Train competitive champion model:
make weights-competition

# Fast smoke test before long runs:
make smoke-spatial
```

---

### D. Unified Hybrid Online/Offline Trainer & Continuous Reinforcement Learning

The production RL engine integrates offline hyperparameter sweeps, online Sampled MCTS self-play harvesting, dynamic timestamp-ordered replay buffer governance (10GB budget), and multi-stage reanalysis:

| Command | Mode | Description | Key Variables / Defaults |
| :--- | :--- | :--- | :--- |
| `make hybrid` | **Hybrid** | Interleaved offline tuning sweep + online self-play + reanalysis distillation + champion gate | `HYBRID_ITERATIONS=24`, `HYBRID_BASE_ITERS=719`, `HYBRID_SELFPLAY=50`, `HYBRID_EVAL_EPISODES=30`, `HYBRID_STRATEGY=adaptive` |
| `make sweep-offline` | **Offline** | High-velocity hyperparameter search across replay demonstrations without live game simulation | `HYBRID_ITERATIONS=24`, `HYBRID_BASE_ITERS=719`, `HYBRID_STRATEGY=adaptive` |
| `make online-spatial` | **Online** | Directed MCTS self-play game harvesting into `replays/` and continuous online learning | `HYBRID_ITERATIONS=24`, `HYBRID_SELFPLAY=50`, `HYBRID_BASE_ITERS=719` |
| `make loop-spatial` | **Sequential Loop** | Continuous 4-phase sequential loop over spatial curriculum | `ITERATIONS=10`, `PROMOTE_EVERY=1`, `SIMS=16`, `BATCH=32` |

**Competitive Rank #1 Production Training Command:**
```bash
# Canonical 24-iteration hybrid optimization run:
python -m pipelines.hybrid_trainer \
  --iterations 24 \
  --base-iterations 719 \
  --eval-episodes 30 \
  --selfplay-episodes 50 \
  --mode hybrid

# Or equivalently via make:
make hybrid HYBRID_ITERATIONS=24 HYBRID_BASE_ITERS=719 HYBRID_SELFPLAY=50 HYBRID_EVAL_EPISODES=30
```

**Alternative Sweep Strategies:**
```bash
# Cartesian grid sweep co-tuning PPO dials every 3rd trial:
python -m pipelines.hybrid_trainer --mode offline --sweep-strategy grid --iterations 12

# Antagonistic gradient pairs (+cons/-val vs -cons/+val):
python -m pipelines.hybrid_trainer --mode hybrid --sweep-strategy gradient --iterations 10
```

---

### E. Checkpoint Promotion & Weight Management

| Command | Description |
| :--- | :--- |
| `make promote-weights` | Validates loadability of `artifacts/muzero_spatial_checkpoints.pt` and atomically links it into packaging aliases (`muzero_checkpoints_champion.pt` and `muzero_checkpoints.pt`) |
| `make champion` | Archives the current champion submission and main script with a timestamp into `baselines/submission_champion_<timestamp>.zip` |
| `make embed-weights` | Encodes neural weights directly as standalone base64 strings into self-contained runner scripts |

**Example:**
```bash
make promote-weights
make champion
```

---

### F. Packaging & Submission Compilation

| Command | Description |
| :--- | :--- |
| `make build` | Promotes latest spatial weights and compiles the 3 canonical submission archives into `dist/` |
| `make submission` | Full production submission pipeline: `gold` $\to$ `distill` $\to$ `weights-spatial` $\to$ `build` $\to$ zip integrity assertion |

**Example:**
```bash
make submission
```

---

### G. Gating, Benchmarking & Tournaments

| Command | Target Benchmark | Description |
| :--- | :--- | :--- |
| `make test` | Full Test Suite | Runs `pytest tests/` (63 comprehensive unit and integration tests) |
| `make gate-buoy` | Kaggle 2033 Gate | Runs strict head-to-head veto certification match against Kaggle-scored `submission_2033` |
| `make benchmark` | Local Top-8 | Evaluates active agent against top local candidates (`baselines/`) |
| `make tournament` | Candidate League | Runs multi-worker round-robin tournament between candidates and champions |
| `make league` | PFSP League | Prioritized Fictitious Self-Play evaluation matching against historical snapshots |
| `make dashboard` | Telemetry Dashboard | Renders terminal telemetry dashboard showing win rates, net revenue, and ELOs |

**Examples:**
```bash
make test
make gate-buoy
make tournament TOURNAMENT_ARGS="--workers 8 --games-per-matchup 4"
```

---

### H. Economic Diagnostics & Microcontroller Tuning

| Command | Description | Key Variables |
| :--- | :--- | :--- |
| `make diagnostics` | Analyzes farm labor tradeoffs, land unlock thresholds, crop margins, and shed liquidation rules | `DIAGNOSTICS_ARGS="--step 194 --seat 0"` |
| `make concurrency` | Stress-tests multi-threaded MCTS simulation throughput and tensor allocators | `DIAGNOSTICS_ARGS` |
| `make ppo-grid` | Executes offline hyperparameter sweep across PPO microcontroller dials | `PPO_TRIALS=12`, `PPO_ARGS` |
| `make ppo-train` | Trains standalone PPO microcontroller policy for daily macro-option routing | `PPO_ROUNDS=20`, `PPO_ARGS` |

**Examples:**
```bash
make diagnostics
make concurrency
make ppo-grid PPO_TRIALS=18
```

---

## 5. Distilled Strategic Rules for the $185,000 Farm

Empirical analysis of 100k–147k Kaggle replays yields the following non-negotiable rules for reaching Rank #1:

1. **Turn 1 NeML Opening Arbitrage**:
   - Execute exact sequence: `["BUY_PRODUCT", "WHEAT", 9]`, `["SELL", "WHEAT", 9]`, `["BUY_PRODUCT", "WHEAT", 11]`, `["SELL", "WHEAT", 6]`, `["BUY_SEED", "WHEAT", 1]`.
   - Generates instant +$20–$25 cash delta with zero price fluctuation risk.
2. **Turn 2 Conversion**:
   - Reinvest initial cash immediately into 5 workers (`5x HIRE`) + 2 Cows + 2 Sheep.
3. **Crop Strategy & Growth Waves**:
   - **Melon (12–18 seeds)**: Harvested on Day 10 for a massive $12,000–$16,000 cash injection. Immediately reinvest into replacement seeds for Wave 2.
   - **Strawberry (33–75 seeds)**: Purchased incrementally from Day 7 to Day 25, scaling with herd size.
   - **Carrot (18–40 seeds auxiliary post-NE, scaling to 100–125 on Days 20–26)**: Leverages 3-day maturation for an end-game terminal cash boost of +$8,000–$10,000.
   - **Tomato (0 seeds)**: Excluded from top games unless late SE land is unlocked.
4. **Livestock & Auxiliary Poultry**:
   - Maintain 16–22 Cows and 15–20 Sheep to supply high-margin recurring Milk and Wool revenue.
   - 2 Geese provide ~$3,500 in supplementary Egg revenue using shared wheat feed.
   - Worker hands capped at 11–12 hands to avoid wage inflation.
5. **Land Expansions**:
   - **North-East (NE)**: Day 6, Hours 2–7 ($1,000).
   - **South-West (SW)**: Days 8–11 ($2,000 + $500 cash buffer).
   - **South-East (SE)**: Skip by default. Only expand if cash $\ge \$35,000$ after Day 18 with 14+ hands.
6. **Liquidation & Labor Efficiency**:
   - Hours 1 & 2 morning dump for Wool, Milk, Strawberry, and Melon.
   - Shed Capacity Guard: When shed inventory $\ge 85$ items, dump down to 75.
   - Avoid the Fertilizer Labor Trap: Never assign workers to collect low-margin ($7–$8) fertilizer when high-margin planting slots are open.

---

## 6. Submission Archetypes

Compiled into `dist/` via `make build` or `make submission`:

| Package Archive | Runner Script | Description | Weights Used |
| :--- | :--- | :--- | :--- |
| **`head_muzero_submission.zip`** | `main_mz.py` | **Full Production Spatial MuZero**: Complete 8-layer autonomous chassis with Sampled MCTS, spatial action plane projections, and factorized heads | `muzero_spatial_checkpoints.pt` / `muzero_checkpoints_champion.pt` |
| **`head_hybrid_submission.zip`** | `main_hpm.py` | **Hybrid Monolith + MuZero**: Combines the 2033 reference chassis with MuZero NeML opening, SE gate, and trickle liquidation | `muzero_checkpoints.pt` |
| **`head_pl_submission.zip`** | `main_pl.py` | **Control (Pure Replay)**: Non-neural control baseline with distilled expert replay rules (zero neural dependencies) | None |

---

## 7. Repository Layout

```
├── Makefile                       # Developer automation (all phase, train, build, & eval targets)
├── muzero/                        # RL core engine
│   ├── chassis.py                 # ResNet h_θ, dynamics g_θ, factorized f_θ, SimSiam loss
│   ├── mcts.py                    # Sampled MuZero MCTS, MinMaxStats, Dirichlet noise
│   ├── buffer.py                  # Prioritized Experience Replay (PER), IS weights, n-step returns
│   ├── trainer.py                 # PrioritizedMuZeroTrainer, Polyak EMA, reanalyze worker
│   ├── observation_encoder.py     # 28-channel 10x10 spatial feature encoder
│   └── prioritized_system.py      # Consolidated system exports
├── pipelines/                     # 4-phase curriculum & training loops
│   ├── train_muzero.py            # Unified trainer CLI (--arch spatial | --arch macro)
│   ├── low_rank_adaptation.py     # Standard loss schedule & distinctness validation
│   ├── adaptive_moment_estimation.py # Boost schedule testing opposing gradients
│   ├── ppo_tuning.py              # PPO microcontroller grid search and training
│   └── phases/                    # Modular phase scripts (phase_spatial, phase1-4)
├── hybrid_chassis/                # Autonomous agent framework (router, action masks, compiler)
├── evaluation/                    # Benchmarking sandbox, H2H gates, & PFSP league engine
├── packaging/                     # Submission packager & single-file compiler
├── scripts/                       # Replay downloader, pruning, head builder, diagnostics
├── replays/                       # 55 canonical gold Kaggle replays (pruning-immune)
├── baselines/                     # Official Kaggle-scored (2033, 2028, etc.) + candidate zips
├── artifacts/                     # Trained checkpoints, snapshots, summaries, & policy JSONs
└── tests/                         # Full pytest verification suite (100% passing)
```
