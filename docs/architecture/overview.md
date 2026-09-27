# Kaggriculture MuZero - System Architecture Overview

## High-Level Architecture

Kaggriculture MuZero is a **spatial Sampled MuZero** implementation for the Kaggle "Kaggriculture" competition. It combines:

1. **Neural Networks** (spatial ResNet-based): Representation (h_θ), Dynamics (g_θ), Prediction (f_θ)
2. **Sampled MCTS** in latent space with factorized action sampling
3. **Prioritized Experience Replay** with Reanalyze target refreshing
4. **Polyak EMA Target Networks** for stable training
5. **SimSiam Self-Supervised Consistency** for latent space alignment
6. **League Training** with Prioritized Fictitious Self-Play (PFSP)
7. **4-Phase Curriculum**: BC → Bootstrap → MCTS Distill → LoRA/PPO

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KAGGRICULTURE MUZERO SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   REPLAYS    │    │  BASELINES   │    │  ARTIFACTS   │                  │
│  │  (10-24 GB)  │    │  (21 scored) │    │  (checkpoints)               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONTINUOUS SELF-PLAY LOOP                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ SELF-   │  │ REPLAY  │  │ TRAIN   │  │ EVAL    │  │PROMOTE  │  │   │
│  │  │ PLAY    │──►│ BUFFER  │──►│ STEP    │──►│ TOURNA- │──►│CHAMPION │  │   │
│  │  │ (MCTS)  │  │ (PER)   │  │ (EMA)   │  │ MENT    │  │         │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      NEURAL NETWORK CHASSIS                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ h_θ: Spatial│  │ g_θ: Spatial│  │ f_θ: SpatialPredictionNet   │  │   │
│  │  │ Represent-  │  │ DynamicsNet │  │ (Factorized Heads)          │  │   │
│  │  │ ationNet    │  │             │  │  • Farmer (15)              │  │   │
│  │  │  Conv28→64  │  │ Conv80→64   │  │  • Hands (32)               │  │   │
│  │  │  +3 ResNet  │  │ +3 ResNet   │  │  • Market (20)              │  │   │
│  │  │  L2 norm    │  │ Reward Head │  │  • Value (601)              │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      MCTS SEARCH (Sampled)                           │   │
│  │  • MinMaxStats Q-normalization                                       │   │
│  │  • PUCT with c1=1.25, c2=19652                                       │   │
│  │  • K=8 samples, 25-50 simulations                                   │   │
│  │  • Dirichlet noise (α=0.25, ε=0.25)                                 │   │
│  │  • Factorized action sampling                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                             │
│                              ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SIM-SIAM CONSISTENCY                              │   │
│  │  • Projection head (3-layer) + Prediction head (2-layer bottleneck) │   │
│  │  • Stop-gradient on target: z_target = proj(h_θ(o)).detach()        │   │
│  │  • Loss: -cosine_sim(predict(proj(s_unrolled)), z_target)           │   │
│  │  • Weight λ_cons ∈ [0.1, 0.5] or [0.5, 1.0]                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
Observation (dict)
    │
    ▼
KaggricultureObservationEncoder.encode() → [Batch, 28, 10, 10] tensor
    │
    ▼
SpatialRepresentationNetwork (h_θ) → s_0 [Batch, 64, 10, 10] (L2 normalized)
    │
    ├──────────────────────┬──────────────────────┐
    ▼                      ▼                      ▼
SpatialPredictionNet    Sampled MCTS          Target Network
(f_θ)                   (Search)              (EMA copy)
    │                      │                      │
    │              ┌───────┴───────┐             │
    │              ▼               ▼             │
    │         Root Policy       Recurrent        │
    │         + Value           Inference        │
    │                        (g_θ → f_θ)         │
    │                        (K steps)           │
    │                          │                 │
    └──────────────────────────┼─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   LOSS COMPUTATION  │
                    │  • Policy CE (×3)   │
                    │  • Value MSE        │
                    │  • Reward MSE       │
                    │  • Consistency      │
                    └─────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Spatial ResNet (not MLP)** | Preserves 10x10 board topology; convolutional inductive bias |
| **Factorized Policy Heads** | Joint action space = 15×32×20 = 9,600; flat softmax infeasible for MCTS |
| **Sampled MCTS (K=8)** | Expands only K candidates per node; tractable search in large action space |
| **Discrete Support (601 bins)** | Categorical value/reward regression; handles multi-scale returns |
| **SimSiam Consistency** | Prevents latent drift between h_θ and g_θ unrolls; dense supervision |
| **PER + Reanalyze** | Focuses training on high-TD-error transitions; refreshes stale MCTS targets |
| **Polyak EMA (τ=0.995)** | Stable target network for bootstrapping; slower than hard updates |
| **4-Phase Curriculum** | BC → Bootstrap → Distill → LoRA progressively increases difficulty |
| **PFSP League** | Adaptive opponent sampling targets agent weaknesses |

## Repository Topology

```
~/kaggriculture_1_37_muzero/ (READ+WRITE, Consolidated)
├── muzero/                    # RL Core
│   ├── observation_encoder.py
│   ├── chassis.py             # h_θ, g_θ, f_θ networks
│   ├── mcts.py                # Sampled MCTS
│   ├── buffer.py              # PER + Reanalyze
│   ├── trainer.py             # Training loop + EMA
│   └── ...
├── pipelines/                 # Training Pipelines
│   ├── train_muzero.py        # Main training entry
│   ├── low_rank_adaptation.py # LoRA loss scheduling
│   ├── adaptive_moment_estimation.py # Inverted schedule (boost.sh)
│   ├── replay_manager.py      # Disk budget management
│   └── self_improving_loop.py # Continuous loop orchestration
├── evaluation/                # Benchmarks & Tournaments
│   └── league.py              # League + PFSP
├── hybrid_chassis/            # Reactive Agent Framework
├── packaging/                 # Submission Packaging
├── scripts/                   # Gold replay download, tournaments
├── replays/                   # 50 canonical gold Kaggle replays
├── baselines/                 # 21 Kaggle-scored + candidate submissions
├── artifacts/                 # Champion checkpoints, LoRA adapters
├── train.sh                   # Standard loss schedule
├── boost.sh                   # Inverted (antagonistic) schedule
└── AGENTS.md                  # This architecture document
```

## Storage Governance

| Workspace | Role | Replay Budget | Pruning Policy |
|-----------|------|---------------|----------------|
| This repo | Consolidated READ+WRITE | 10.0 GB | mtime ascending, canonical replays immune |
| SCRATCH | Prototypes | 10.0 GB | Auto-prune selfplay_*.json |
| READ+WRITE (external) | Canonical | 24.0 GB | Long multi-day runs |

**Invariants:**
1. Canonical replays (`^[0-9]+\.json$`) never deleted
2. Self-play replays pruned oldest-first by `mtime`
3. No score-based filtering (prevents stagnation)

## Baselines Comparison

### Local Candidates (this repo + external)
- `submission_boost_cand016.zip` — $108,092.65 (Iter 16, champion)
- `submission_train_cand042.zip` — $110,301.60 (Iter 42, champion)
- Top 4 from boost.sh + top 4 from train.sh

### Official Kaggle-Scored (/Volumes/BASELINES/muzero/baselines)
- `submission_2033.zip` — Score: 2033
- `submission_2028.zip` — Score: 2028
- `submission_2025.zip` — Score: 2025
- `submission_2021.zip` — Score: 2021
- `submission_0771.tar.gz` — Score: 771

**Evaluation Principle**: Candidates tested against local top-8 for rapid gating; periodically validated against Kaggle baselines for leaderboard certification.

## Next Steps

- [Component Specifications](../components/) - Detailed per-component docs
- [Training Phases](../training/) - 4-phase curriculum details
- [Evaluation & League](../evaluation/) - Tournament engine, PFSP
- [Deployment](../deployment/) - Submission archetypes, GitHub Pages
- [Reference](../reference/) - Hyperparameters, API, Glossary