# Agent Guidelines & Consolidated Repository Architecture (Kaggriculture MuZero)

This repository (`~/kaggriculture_1_37_muzero`) is the consolidated, self-contained rebuild of the Kaggriculture MuZero stack, merged from three legacy workspaces:

- **SCRATCH** (`/Users/sweeden/muzero`) — rapid prototypes, exclusive scripts (gold replay download, candidate search, tournaments)
- **READONLY** (`/Volumes/BASELINES/muzero`) — Kaggle-scored gold-standard baselines + reference chassis
- **READ+WRITE** (`/Volumes/TRAINREPLAYBOOST/muzero`) — canonical refactored code base (latent-only MCTS, 4-phase curriculum, continuous loop)

The canonical code comes from READ+WRITE; exclusive scripts and the newest root launchers come from SCRATCH; scored baselines come from READONLY.

---

## 1. Repository Topology & Storage Governance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ~/kaggriculture_1_37_muzero  (READ+WRITE)               │
│                                                                             │
│  muzero/          RL core: h_θ encoder, g_θ dynamics, f_θ decoder,         │
│                   latent MCTS, PER, LoRA, PPO microcontroller               │
│  pipelines/       4-phase curriculum, Adam loss schedules, self-improving  │
│                   loop, PPO tuning, replay manager                          │
│  hybrid_chassis/  Autonomous agent framework (layers, routes, compiler)     │
│  evaluation/      Tournament engine, benchmarks, leagues                    │
│  packaging/       Submission packaging & weight bundling                    │
│  scripts/         Gold replay download, pruning, tournaments, head builder  │
│  replays/         50 canonical gold Kaggle replays (pruning-immune)         │
│  baselines/       21 Kaggle-scored + candidate submission zips              │
│  artifacts/       Champion checkpoints, snapshots, LoRA adapters, policies  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Storage Matrix

| Attribute | This repository |
| :--- | :--- |
| **Role & Designation** | **Consolidated READ+WRITE** |
| **Storage Location** | Local APFS Volume (Host NVMe) |
| **Replay Buffer Budget** | **10.0 GB** maximum capacity (SCRATCH-equivalent) |
| **Replay Pruning Policy** | Auto-prunes oldest `selfplay_*.json` by `mtime`; canonical `^[0-9]+\.json$` replays immune |
| **Baselines Directory** | Official Kaggle-Scored Submissions + Top Candidates |
| **Primary Workflow** | Continuous `train.sh` & `boost.sh` loops + PPO grid tuning |

---

## 2. Baselines Comparison: Local Candidates vs. Kaggle-Scored Standards

The baselines folders across the workspaces serve complementary roles in the iterative optimization and validation pipeline:

### A. Local Candidates (`/Users/sweeden/muzero/baselines` & `/Volumes/TRAINREPLAYBOOST/muzero/baselines`)
Contains the top 4 candidates discovered by `boost.sh` (inverted loss schedule) and top 4 candidates discovered by `train.sh` (standard loss schedule), compiled with standalone weights into runnable submission packages:
1. `submission_boost_cand016.zip` — Score: **$108,092.65** (Iter 16, champion snapshot)
2. `submission_boost_cand003.zip` — Score: **$106,236.50** (Iter 3, population pool)
3. `submission_boost_cand014.zip` — Score: **$105,679.50** (Iter 14, population pool)
4. `submission_boost_cand034.zip` — Score: **$104,067.65** (Iter 34, final champion snapshot)
5. `submission_train_cand042.zip` — Score: **$110,301.60** (Iter 42, champion snapshot)
6. `submission_train_cand048.zip` — Score: **$107,477.65** (Iter 48, champion snapshot)
7. `submission_train_cand056.zip` — Score: **$106,908.15** (Iter 56, champion snapshot)
8. `submission_train_cand012.zip` — Score: **$106,755.65** (Iter 12, champion snapshot)

### B. Official Kaggle-Scored Baselines (`/Volumes/BASELINES/muzero/baselines`)
Contains the authoritative baseline submissions already evaluated on the official Kaggle leaderboard, with public/private scores directly encoded into their prefixes:
1. `submission_2033.zip` — Kaggle Score: **2033**
2. `submission_2028.zip` — Kaggle Score: **2028**
3. `submission_2025.zip` — Kaggle Score: **2025**
4. `submission_2021.zip` — Kaggle Score: **2021**
5. `submission_0771.tar.gz` — Kaggle Score: **771**
6. `main.py` — Reference chassis runner

*Evaluation Principle*: Candidates discovered in `SCRATCH` or `READ+WRITE` are continuously tested against the local top-8 suite for rapid gating, and periodically validated against `/Volumes/BASELINES/muzero` to certify leaderboard progression.

---

## 3. Replay Storage & Dynamic Disk Space Governance

- **SCRATCH (`/Users/sweeden/muzero`)**:
  - Replay folder capped at **10.0 GB** to prevent disk pressure on the main host drive.
  - Automatic pruning triggers on candidate evaluation and promotion cycles via `get_default_replay_budget_gb()`.
- **READ+WRITE (`/Volumes/TRAINREPLAYBOOST/muzero`)**:
  - Replay folder allowed to scale up to **24.0 GB**, capturing long multi-day self-play runs across hundreds of stochastic market environments.
- **Strict Invariants for Both Writable Workspaces**:
  1. **Canonical Replay Immunity**: Canonical Kaggle ground-truth replays (`^[0-9]+\.json$`, 72 files) are permanently immune from deletion.
  2. **Timestamp Order Pruning**: Stale non-canonical self-play replays (`selfplay_*.json`) are sorted strictly by file system update timestamps (`mtime` ascending) and pruned oldest-first until directory size $\le$ target budget.
  3. **No Score-Filter Traps**: Pruning no longer relies on arbitrary score filters, ensuring steady replay rotation and preventing stale data stagnation.

---

## 4. Replay Learning & Distillation Workflow

1. **Distill Replays to Policy Artifacts**:
   ```bash
   python learn_from_replay.py --all
   ```
   - Automatically detects winner seats and extracts strategic parameters (`sw_day_min`, `skip_se`, `tomato_seed_total_target`, `carrot_post_ne_qty`, `hire_target`, shed liquidation thresholds) into `artifacts/ep<id>_policy.json`.

2. **Train MuZero on Replay Demonstrations**:
   ```bash
   # Standard training schedule:
   ./train.sh hybrid --eval-episodes 8
   # Inverted high-weight boost schedule:
   ./boost.sh hybrid --eval-episodes 4
   # Fine-tune champion foundation with LoRA:
   ./train.sh --use-lora --lora-rank 4
   ```
   - **Top Tier Replay (Training Data)**: Raw fuel fed into the network to calculate loss and gradients (never confused with Adam or LoRA).
   - **Trained Champion Weights (Model State)**: Acts as the foundation model parameters from prior best runs. When `--use-lora` is set, champion weights are frozen, rank-$r$ LoRA adapter matrices are injected, and Adam trains exclusively the adapters. Adapters are seamlessly merged upon checkpoint save for standalone deployment.
   - **Hyperparameters (Fixed Dials)**: Learning rates, LoRA rank $r$, and loss weights are fixed dials configured prior to training, tested via experimental sweeps (grid/adaptive) across phases—never as gradients.
   - **Canonical 3-Phase Execution Order**:
     - **Phase 1 (Behavioral Cloning - BC)**: Mimics expert replay data with PER using Adam.
     - **Phase 2 (Analytical Bootstrapping)**: Grounds representation and dynamics models with self-consistent unroll targets.
     - **Phase 3 (MCTS Distillation)**: Uses Gumbel MCTS lookahead to generate fresh targets and distills them back into the network with Adam.
   - **Discrete Head Weights**: Every checkpoint save automatically exports separate `muzero_policy_head.pt` and `muzero_value_head.pt` files alongside `muzero_checkpoints.pt` and candidate checkpoints.

3. **Continuous Self-Play & Reinforcement Learning Loop**:
   - **Supervised Jump-Start (Phases 1–3)**: Establishes a competent initial representation and non-random priors so exploration does not waste thousands of steps on random actions.
   - **Self-Play Data Generation**: Guided by Phase 3 policy and value weights, MCTS interacts with the environment (or plays games against itself and snapshot opponents) to explore state spaces and make decisions.
   - **Dynamic Replay Buffer**: Newly harvested self-play episodes (`selfplay_*.json`) are continuously streamed into the replay buffer. Over time, fresh self-generated data phases out older BC data via timestamp-ordered disk pruning (`mtime` ascending) under volume budgets (10 GB SCRATCH, 24 GB READ+WRITE) while permanently protecting canonical replays.
   - **Continuous RL Training**: Batches sampled from this fresh replay data are optimized via Adam (or LoRA adapter fine-tuning on frozen champion foundation) to align predictions with actual self-play outcomes and MCTS visit distributions.
   - **Champion Checkpointing & Promotion Gate**: Periodic evaluation runs the latest trained candidate against saved champion weights. Candidates that consistently outperform the champion are promoted atomically, and self-play workers immediately load the new champion to generate higher-quality data for future iterations.
   - This autonomous loop runs indefinitely (`./train.sh online` or `./boost.sh hybrid`) until performance plateaus or target metrics are achieved.

4. **Loss Schedule Engines**:
   - **Low-Rank Adaptation (`pipelines/low_rank_adaptation.py`, `train.sh`)**: Standard loss schedule engine with consistency regimes ($[0.1, 0.5]$ vs $[0.5, 1.0]$) and smooth softmax transitions.
   - **Adaptive Moment Estimation (`pipelines/adaptive_moment_estimation.py`, `boost.sh`)**: Adaptive Moment Estimation schedule engine testing antagonistic gradient directions with smooth softmax transitions.
   - Both schedules strictly enforce mutual distinctness (`consistency != policy != value != reward != cql`).

5. **Build Canonical Head Submissions (The Three Build Archetypes)**:
   ```bash
   python scripts/build_head_submissions.py
   # or legacy single target:
   python package_submission.py
   ```
   Compiles and packages the three canonical submission archetypes into `dist/`:
   - **`head_pl_submission.zip` (`main_pl.py`)** — **Control (Pure Replay)**: Non-neural control baseline with distilled expert policy lessons (Layers 1–7: base patches, resource guards, V9 expansion, tactical planners, market guards, V7 outer stack, replay lessons). Zero neural weights or dependencies.
   - **`head_hybrid_submission.zip` (`main_hpm.py`)** — **Hybrid Control + MuZero**: Combines the monolithic 2033 chassis (`main_2033.py`) with the active MuZero NeML Step 0 opening, SE expansion evaluation, and trickle liquidation gate. Strictly loads standard `muzero_checkpoints.pt`.
   - **`head_muzero_submission.zip` (`main_mz.py`)** — **Pure MuZero Pipeline**: Full modular single-file compilation of all 8 chassis layers, integrating the complete MuZero macro-options planner. Prioritizes `muzero_checkpoints_champion.pt`.

6. **Tournament Evaluation vs Baselines**:
   ```bash
   # Evaluate against local Top-8 baselines:
   python tournament.py benchmark --agent dist/main.py --baselines baselines --games-per-baseline 4
   # Evaluate against Kaggle READONLY gold standard:
   python tournament.py benchmark --agent dist/main.py --baselines /Volumes/BASELINES/muzero/baselines --games-per-baseline 4
   # Cross-tournament: Official Baselines vs Candidate League with Promotion to scratch:
   python tournament_candidates_vs_baselines.py --workers 8 --games-per-matchup 2
   ```


---

## 5. Domain, Phase & Tier Organization

- **Domains**:
  - `muzero/`: RL Core (Networks $h_\theta, g_\theta, f_\theta$, analytical economic model, MCTS planner, DiscreteSupport)
  - `hybrid_chassis/`: Autonomous Agent Framework (views, reactive layers, routes, single-file compiler)
  - `pipelines/`: Training & RL Pipelines (Phase 1 bootstrap, Phase 2 distill, Phase 3 BC, self-improving loop, replay manager)
  - `evaluation/`: Benchmark sandbox & tournament engine (Tiers 1–3 baselines, H2H, round-robin league)
  - `packaging/`: Submission packaging, compilation, and weight bundling
- **Phases**: Phase 1 (Analytical Bootstrap) $\to$ Phase 2 (MCTS Distillation & Reanalyse) $\to$ Phase 3 (Expert BC with PER)
- **Tiers**: Tier 1 (Heuristic Baselines) $\to$ Tier 2 (Hybrid Reactive Chassis) $\to$ Tier 3 (Neural MuZero & League Champions)

---

## 6. Strategic Rules Distilled from 100k–147k Replays

1. **Step 1 NeML Opening Arbitrage**:
   - `["BUY_PRODUCT", "WHEAT", 9]`, `["SELL", "WHEAT", 9]`, `["BUY_PRODUCT", "WHEAT", 11]`, `["SELL", "WHEAT", 6]`, `["BUY_SEED", "WHEAT", 1]`
   - Yields +$20–$25 cash delta on turn 1 at zero price risk.
2. **Turn 2 Conversion**:
   - Convert cash immediately into 5 hands (`5x HIRE`) + `2 Cows + 2 Sheep`.
3. **Crop Targets**:
   - **Melon**: 12–18 seeds. Harvests around Day 10 for a massive cash injection ($12k–$16k). Re-invest post-harvest liquidity on Day 10 into 3–4 replacement seeds for a second wave.
   - **Strawberry**: 33–75 seeds. Incremental purchases from Day 7 to Day 25, scaling with herd size.
   - **Carrot**: 18–40 seeds auxiliary post-NE, scaling up to 100–125 seeds on Days 20–26 to exploit the 3-day maturation cycle for a terminal cash boost of +$8,000–$10,000.
   - **Tomato**: 0 seeds in top games unless late SE land is unlocked.
4. **Livestock & Auxiliary Poultry**:
   - Cows (up to 16–22) and Sheep (up to 15–20) generate the recurring high-margin Milk and Wool that drive scores $\ge \$140,000$.
   - 2 Geese add ~$3,500 in auxiliary Egg revenue at near-zero marginal cost using pooled wheat feed.
   - Max worker hands capped at 11–12 hands to control wage inflation.
5. **Land Expansions**:
   - **NE**: Day 6, Hour 2–7 ($1,000).
   - **SW**: Day 8–11 ($2,000 + $500 buffer).
   - **SE**: Skip by default. Only expand if cash $\ge \$35,000$ post-Day 18 with 14+ hands.
6. **Liquidation & Labor Efficiency**:
   - Hours 1 & 2 morning dump for Wool, Milk, Strawberry, Melon.
   - Shed cap guard: When shed $\ge 85$ items, dump down to 75. Clamped to actual stock.
   - **Avoid the Fertilizer Labor Trap**: Hands should prioritize `PLANT` and `WATER` over `COLLECT_FERTILIZER`. Do not burn worker actions gathering low-value ($7–$8) fertilizer when high-margin crop planting slots are open.

See [`notes/expert_replay_lessons.md`](notes/expert_replay_lessons.md) for full empirical analysis.

---

## Production Architecture Note (ADR 0001)

**Spatial Sampled MuZero is production.** Prefer `muzero.chassis.KaggricultureMuZeroChassis`, `muzero.mcts.SampledMuZeroMCTS`, and `pipelines/train_muzero.py --arch spatial`. Legacy MLP / 8-`MacroOption` code is under `muzero.legacy_macro` (`--arch macro` for ablation only).
