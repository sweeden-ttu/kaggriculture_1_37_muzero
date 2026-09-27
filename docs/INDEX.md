# Kaggriculture MuZero Documentation

**Complete architecture, training, evaluation, and deployment documentation for the Kaggriculture MuZero system.**

---

## Quick Navigation

### 🏗️ Architecture
- [System Overview](architecture/overview.md) — High-level component diagram, data flow, design decisions
- [End-to-End Data Flow](architecture/data-flow.md) — Step-by-step tensor shapes, code references
- [Component Interaction Map](architecture/component-map.md) — Dependency graph, interface contracts

### 🧩 Components
| Component | Description |
|-----------|-------------|
| [Observation Encoder](components/observation-encoder.md) | 28-channel 10×10 spatial feature encoder |
| [Factorized Prediction Head](components/factorized-prediction-head.md) | Farmer/Hands/Market sub-heads + joint sampling |
| [SimSiam Consistency](components/simsiam-consistency.md) | Asymmetric stop-gradient self-supervision |
| [Sampled MCTS](components/sampled-mcts.md) | PUCT with Dirichlet, MinMaxStats, K=8 sampling |
| [MuZero Core Chassis](components/muzero-core-chassis.md) | h_θ, g_θ, f_θ spatial ResNet networks |
| [PER Buffer & Reanalyze](components/per-buffer-reanalyze.md) | Trajectory buffer, K-step unroll, n-step bootstrap |
| [Trainer Loop & EMA](components/trainer-loop-ema.md) | Dual network, IS-weighted loss, Reanalyze passes |
| [League Tracker & PFSP](components/league-tracker-pfsp.md) | AlphaStar-style matchmaking, win-rate prioritization |
| [Consolidated System](components/consolidated-system.md) | End-to-end pipeline: 4 phases, LoRA, continuous loop |

### 🎯 Training
| Phase | Description |
|-------|-------------|
| [Phase 1: Behavioral Cloning](training/phase-1-bc.md) | Replay distillation, expert policy extraction |
| [Phase 2: Analytical Bootstrap](training/phase-2-bootstrap.md) | Consistency loss, latent alignment |
| [Phase 3: MCTS Distillation](training/phase-3-mcts-distill.md) | Self-play, Reanalyze, policy/value distillation |
| [Phase 4: LoRA + PPO](training/phase-4-lora-ppo.md) | Champion fine-tuning, online microcontroller |
| [Loss Schedules](training/loss-schedules.md) | Standard vs Boost (inverted) curriculum |
| [Continuous Loop](training/continuous-loop.md) | Autonomous self-play → train → eval → promote |

### 🏆 Evaluation
| Topic | Description |
|-------|-------------|
| [Tournament Engine](evaluation/tournament-engine.md) | Round-robin benchmarking, seat alternation |
| [Baseline Comparison](evaluation/baseline-comparison.md) | Local top-8 vs Kaggle gold standard (2021-2033) |
| [League PFSP](evaluation/league-pfsp.md) | Prioritized Fictitious Self-Play, agent roles |

### 🚀 Deployment
| Topic | Description |
|-------|-------------|
| [Submission Archetypes](deployment/submission-archetypes.md) | PL (pure replay), HPM (hybrid), MZ (pure MuZero) |
| [GitHub Pages Setup](deployment/github-pages-setup.md) | MkDocs Material auto-deploy via GitHub Actions |
| [Kaggle Submission](deployment/kaggle-submission.md) | Build, validate, submit, debug workflow |

### 📚 Reference
| Resource | Description |
|----------|-------------|
| [Hyperparameter Table](reference/hyperparameter-table.md) | Complete config: network, MCTS, buffer, trainer, LoRA, PPO, league |
| [API Reference](reference/api-reference.md) | All classes, methods, signatures, constants |
| [Glossary](reference/glossary.md) | Terminology: A-Z with code references |
| [FAQ](reference/faq.md) | Common questions: architecture, training, debugging, submissions |

---

## Quick Start

### Build Documentation Locally
```bash
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-minify-plugin
mkdocs serve  # http://127.0.0.1:8000
```

### Generate Component Specs (Auto)
```bash
# All components
python docs/vsa/kaggriculture_muzero_spec_agent.py all --output-dir docs/components

# Single component
python docs/vsa/kaggriculture_muzero_spec_agent.py observation_encoder --output-dir docs/components
```

### Train Models
```bash
# Standard curriculum (train.sh)
./train.sh hybrid --eval-episodes 8

# Inverted/Boost curriculum (boost.sh)
./boost.sh hybrid --eval-episodes 4

# With LoRA fine-tuning
./train.sh --use-lora --lora-rank 16
```

### Build Submissions
```bash
# All three archetypes
python scripts/build_head_submissions.py

# Output in dist/
ls dist/*.zip
```

### Submit to Kaggle
```bash
kaggle competitions submit -c kaggriculture \
    -f dist/head_hybrid_submission.zip \
    -m "Hybrid 2033 + MuZero NeML"
```

---

## Repository Structure

```
kaggriculture_1_37_muzero/
├── muzero/                    # RL Core (h_θ, g_θ, f_θ, MCTS, Buffer, Trainer)
├── pipelines/                 # Training (4 phases, LoRA, schedules, loop)
├── evaluation/                # Tournament, League, PFSP
├── hybrid_chassis/            # Reactive agent framework
├── packaging/                 # Submission compilation
├── scripts/                   # Gold replays, tournaments, head builds
├── replays/                   # 50 canonical gold Kaggle replays
├── baselines/                 # 21 scored + candidate submissions
├── artifacts/                 # Champion checkpoints, LoRA adapters
├── docs/                      # This documentation
├── train.sh / boost.sh        # Launch scripts
└── AGENTS.md                  # Repository topology & governance
```

---

## Key Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| `OBS_CHANNELS` | 28 | Observation tensor channels |
| `LATENT_CHANNELS` | 64 | Latent state channels |
| `NUM_FARMER_ACTIONS` | 15 | Farmer policy dimension |
| `NUM_HAND_ASSIGNMENTS` | 32 | Hands policy dimension |
| `NUM_MARKET_ORDERS` | 20 | Market policy dimension |
| `SUPPORT_SIZE` | 601 | Discrete support bins |
| `DISCOUNT (γ)` | 0.997 | MCTS/training discount |
| `LORA_RANK` | 16 | Adapter rank |

---

## Baselines

### Local Candidates (Rapid Gating)
| Candidate | Score | Source |
|-----------|-------|--------|
| `submission_train_cand042.zip` | $110,301.60 | Iter 42 champion |
| `submission_boost_cand016.zip` | $108,092.65 | Iter 16 champion |

### Official Kaggle Baselines (Certification)
| Baseline | Kaggle Score |
|----------|--------------|
| `submission_2033.zip` | 2033 |
| `submission_2028.zip` | 2028 |
| `submission_2025.zip` | 2025 |
| `submission_2021.zip` | 2021 |

---

## Links

- **Repository**: [github.com/sweeden/kaggriculture_1_37_muzero](https://github.com/sweeden/kaggriculture_1_37_muzero)
- **Kaggle Competition**: [kaggle.com/c/kaggriculture](https://www.kaggle.com/c/kaggriculture)
- **MuZero Paper**: [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- **Sampled MuZero**: [arXiv:2006.07803](https://arxiv.org/abs/2006.07803)
- **EfficientZero**: [arXiv:2111.02512](https://arxiv.org/abs/2111.02512)
- **AlphaStar League**: [DeepMind Blog](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii)

---

## License

Apache 2.0 — See `LICENSE` file.

**Author**: Scott Weeden  
**Generated**: 2026-09-27