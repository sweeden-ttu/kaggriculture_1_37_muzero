---
description: Build comprehensive documentation for the Kaggriculture MuZero codebase
agent: documentation-builder
model: anthropic/claude-sonnet-4-6
---

Build complete documentation for the Kaggriculture MuZero architecture.

## Instructions

Use the `documentation-builder` agent to analyze the codebase at `/Users/sweeden/kaggriculture_1_37_muzero` and generate comprehensive markdown documentation for all 9 core components.

### Target Components

1. **Observation Encoder** - Spatial 28-channel 10x10 encoder
2. **Factorized Prediction Head** - Farmer/Hands/Market sub-heads with joint sampling
3. **Latent SimSiam Consistency** - Asymmetric stop-gradient self-supervision
4. **Sampled MCTS Search Tree** - PUCT with Dirichlet, MinMaxStats, factorized actions
5. **MuZero Core Chassis** - h_θ, g_θ, f_θ spatial ResNet networks
6. **PER Buffer & Reanalyze** - Trajectory-based, K-step unroll, n-step bootstrap, IS weights
7. **Trainer Loop & Polyak EMA** - Dual network, multi-objective loss, Reanalyze passes
8. **League Tracker & PFSP** - AlphaStar-style matchmaking, win-rate prioritization
9. **Consolidated Prioritized System** - End-to-end pipeline: 4-phase curriculum, LoRA, continuous loop

### Output Requirements

Generate the full documentation structure under `docs/`:

```
docs/
├── INDEX.md
├── architecture/
│   ├── overview.md
│   ├── data-flow.md
│   └── component-map.md
├── components/
│   ├── observation-encoder.md
│   ├── factorized-prediction-head.md
│   ├── simsiam-consistency.md
│   ├── sampled-mcts.md
│   ├── muzero-core-chassis.md
│   ├── per-buffer-reanalyze.md
│   ├── trainer-loop-ema.md
│   ├── league-tracker-pfsp.md
│   └── consolidated-system.md
├── training/
│   ├── phase-1-bc.md
│   ├── phase-2-bootstrap.md
│   ├── phase-3-mcts-distill.md
│   ├── phase-4-lora-ppo.md
│   ├── loss-schedules.md
│   └── continuous-loop.md
├── evaluation/
│   ├── tournament-engine.md
│   ├── baseline-comparison.md
│   ├── league-pfsp.md
│   └── submission-packaging.md
├── deployment/
│   ├── submission-archetypes.md
│   ├── github-pages-setup.md
│   └── kaggle-submission.md
└── reference/
    ├── hyperparameter-table.md
    ├── api-reference.md
    ├── glossary.md
    └── faq.md
```

### Quality Standards

- Each component doc must include: Overview, Architecture, API Reference, Hyperparameters, Training Workflow, Testing Strategy, Evaluation Metrics, Code References
- Use `file_path:line_number` format for code references
- Cross-reference between components
- Include tensor shapes, loss formulas, algorithm pseudocode
- Match actual hyperparameters from source (e.g., `SUPPORT_SIZE=601`, `LATENT_CHANNELS=64`, `NUM_FARMER_ACTIONS=15`, etc.)

### GitHub Pages Setup

If `$ARGUMENTS` contains `--publish` or `--github-pages`:
1. Create `mkdocs.yml` with Material theme
2. Create `.github/workflows/docs.yml` for auto-deploy
3. Configure navigation from INDEX.md
4. Output instructions for enabling GitHub Pages

### Execution

Start by reading `AGENTS.md` for repository context, then use the spec agent script at `docs/vsa/kaggriculture_muzero_spec_agent.py` or analyze source files directly.

Report progress and final summary with file counts and any gaps.