---
name: docs-builder
description: Build comprehensive documentation for Kaggriculture MuZero from source code. Use when user wants to generate architecture docs, component specs, API references, or publish to GitHub Pages. Keywords: documentation, docs, markdown, architecture, spec, github pages, mkdocs, component documentation.
---

# Documentation Builder Skill

This skill provides utilities and context for building comprehensive documentation for the Kaggriculture MuZero codebase.

## When to Use

- User says "build docs", "generate documentation", "create documentation"
- User wants architecture documentation for MuZero components
- User wants to publish docs to GitHub Pages
- User needs API reference, hyperparameter tables, or component specs
- User asks about "documentation folder" or "online docs"

## Key Context

The repository is at `/Users/sweeden/kaggriculture_1_37_muzero` and contains:

### 9 Core Components

| Component | Source | Docs |
|-----------|--------|------|
| Observation Encoder | `muzero/observation_encoder.py` | `docs/Observation_Encoder.MD` |
| Factorized Prediction Head | `muzero/chassis.py` | `docs/FACTORIZED_PREDICTION_HEAD.md` |
| SimSiam Consistency | `muzero/chassis.py`, `simsiam_consistency.py` | `docs/Consistency_Loss.md` |
| Sampled MCTS | `muzero/mcts.py` | `docs/MCTS_Search_Tree.md` |
| MuZero Core Chassis | `muzero/chassis.py`, `muzero/_core.py` | `docs/MuZero_Core.md` |
| PER Buffer & Reanalyze | `muzero/buffer.py` | `docs/prioritized_muzero_system.py` |
| Trainer Loop & EMA | `muzero/trainer.py` | `docs/prioritized_muzero_system.py` |
| League Tracker & PFSP | `evaluation/league.py` | `docs/League_Tracker.md` |
| Consolidated System | `pipelines/train_muzero.py`, `train.sh`, `boost.sh` | `AGENTS.md` |

### Spec Generation Script

A ready-to-use spec agent exists at `docs/vsa/kaggriculture_muzero_spec_agent.py`:

```bash
# Generate single component
python docs/vsa/kaggriculture_muzero_spec_agent.py observation_encoder --output-dir docs/components

# Generate all components
python docs/vsa/kaggriculture_muzero_spec_agent.py all --output-dir docs/components
```

### Repository Structure

```
kaggriculture_1_37_muzero/
├── muzero/                 # Core RL implementation
│   ├── observation_encoder.py
│   ├── chassis.py          # h_θ, g_θ, f_θ networks
│   ├── mcts.py             # Sampled MCTS
│   ├── buffer.py           # PER + Reanalyze
│   ├── trainer.py          # Training loop + EMA
│   └── ...
├── pipelines/              # Training pipelines
│   ├── train_muzero.py
│   ├── low_rank_adaptation.py
│   └── ...
├── evaluation/             # Tournament & league
│   └── league.py
├── hybrid_chassis/         # Reactive agent framework
├── docs/                   # Existing documentation
├── train.sh / boost.sh     # Launch scripts
└── AGENTS.md               # Architecture overview
```

## Documentation Standards

Each component doc must include:

1. **Overview** - Purpose, design rationale, integration points
2. **Architecture** - Class diagram, data flow, tensor shapes, algorithms
3. **API Reference** - Classes, methods, signatures, config options
4. **Hyperparameters** - Table with defaults, ranges, impact
5. **Training Workflow** - Loss contributions, gradient flow
6. **Testing Strategy** - Unit tests, integration tests, invariants
7. **Evaluation Metrics** - Key metrics, benchmarks, expected ranges
8. **Code References** - `file_path:line_number` mappings

### Cross-Reference Format

Use `file_path:line_number` for code references:
- `muzero/chassis.py:71` for `SpatialRepresentationNetwork`
- `muzero/mcts.py:103` for `SampledMuZeroMCTS`
- `pipelines/train_muzero.py:1` for training entry point

## GitHub Pages Publishing

To publish online docs:

1. **Create `mkdocs.yml`**:
```yaml
site_name: Kaggriculture MuZero Documentation
theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
nav:
  - Home: INDEX.md
  - Architecture: architecture/overview.md
  - Components: components/
  - Training: training/
  - Evaluation: evaluation/
  - Deployment: deployment/
  - Reference: reference/
markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - toc:
      permalink: true
```

2. **Create `.github/workflows/docs.yml`**:
```yaml
name: Deploy Docs
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install mkdocs-material
      - run: mkdocs gh-deploy --force
```

3. **Enable GitHub Pages**: Settings → Pages → Source: `gh-pages` branch

## Common Tasks

### Generate All Component Docs
```bash
python docs/vsa/kaggriculture_muzero_spec_agent.py all --output-dir docs/components
```

### Generate Single Component
```bash
python docs/vsa/kaggriculture_muzero_spec_agent.py factorized_prediction_head --output-dir docs/components
```

### Validate Output
```bash
# Check all components documented
ls docs/components/*.md | wc -l
# Should be 9
```

### Quick Architecture Summary
```bash
# Get component overview
grep -r "class.*:" muzero/*.py | head -30
```

## Integration with Opencode

- Use `agent: documentation-builder` for the main agent
- Use `command: build-docs` for the slash command
- This skill provides context and utilities

## Environment

- Repository: `/Users/sweeden/kaggriculture_1_37_muzero`
- Python: 3.11+ (check `pyproject.toml`)
- Key deps: torch, numpy, langgraph, langchain-ollama (for spec agent)