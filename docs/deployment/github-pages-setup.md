# GitHub Pages Documentation Deployment

## Overview

This guide sets up automatic documentation deployment to GitHub Pages using **MkDocs Material** theme.

## Quick Setup

### 1. Create `mkdocs.yml` at Repo Root

```yaml
# mkdocs.yml
site_name: Kaggriculture MuZero Documentation
site_description: Complete architecture, training, and deployment docs for Kaggriculture MuZero
site_url: https://sweeden.github.io/kaggriculture_1_37_muzero/
repo_url: https://github.com/sweeden/kaggriculture_1_37_muzero
repo_name: kaggriculture_1_37_muzero

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.share
    - content.code.copy
    - content.tabs.link

nav:
  - Home: INDEX.md
  - Architecture:
      - Overview: architecture/overview.md
      - Data Flow: architecture/data-flow.md
      - Component Map: architecture/component-map.md
  - Components:
      - Observation Encoder: components/observation-encoder.md
      - Factorized Prediction Head: components/factorized-prediction-head.md
      - SimSiam Consistency: components/simsiam-consistency.md
      - Sampled MCTS: components/sampled-mcts.md
      - MuZero Core Chassis: components/muzero-core-chassis.md
      - PER Buffer & Reanalyze: components/per-buffer-reanalyze.md
      - Trainer Loop & EMA: components/trainer-loop-ema.md
      - League Tracker & PFSP: components/league-tracker-pfsp.md
      - Consolidated System: components/consolidated-system.md
  - Training:
      - Phase 1: Behavioral Cloning: training/phase-1-bc.md
      - Phase 2: Analytical Bootstrap: training/phase-2-bootstrap.md
      - Phase 3: MCTS Distillation: training/phase-3-mcts-distill.md
      - Phase 4: LoRA + PPO: training/phase-4-lora-ppo.md
      - Loss Schedules: training/loss-schedules.md
      - Continuous Loop: training/continuous-loop.md
  - Evaluation:
      - Tournament Engine: evaluation/tournament-engine.md
      - Baseline Comparison: evaluation/baseline-comparison.md
      - League PFSP: evaluation/league-pfsp.md
  - Deployment:
      - Submission Archetypes: deployment/submission-archetypes.md
      - GitHub Pages Setup: deployment/github-pages-setup.md
      - Kaggle Submission: deployment/kaggle-submission.md
  - Reference:
      - Hyperparameter Table: reference/hyperparameter-table.md
      - API Reference: reference/api-reference.md
      - Glossary: reference/glossary.md
      - FAQ: reference/faq.md

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  - toc:
      permalink: true
      title: On this page

plugins:
  - search
  - git-revision-date-localized:
      enable_creation_date: true
  - minify:
      minify_html: true

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/sweeden/kaggriculture_1_37_muzero
  analytics:
    provider: google
    property: G-XXXXXXXXXX
  version:
    provider: mike

copyright: Copyright &copy; 2026 Scott Weeden
```

### 2. Create GitHub Actions Workflow

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Needed for git-revision-date

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-minify-plugin

      - name: Build Documentation
        run: mkdocs build --strict

      - name: Upload Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 3. Enable GitHub Pages

1. Go to **Settings** → **Pages**
2. **Source**: "GitHub Actions"
3. The workflow will auto-deploy on push to main

### 4. Verify Deployment

- URL: `https://sweeden.github.io/kaggriculture_1_37_muzero/`
- Check **Actions** tab for deployment status
- Custom domain: Add `CNAME` file to `docs/` if needed

## Local Development

```bash
# Install
pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-minify-plugin

# Serve locally
mkdocs serve
# → http://127.0.0.1:8000

# Build static site
mkdocs build
# → ./site/ folder
```

## Documentation Structure for MkDocs

Ensure your `docs/` folder matches the `nav` structure:

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
│   └── league-pfsp.md
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

## Component File Naming

**Rename generated spec files** to match nav:
```bash
# From spec agent output (component_spec.md) → docs/components/component.md
mv docs/components/observation_encoder_spec.md docs/components/observation-encoder.md
mv docs/components/factorized_prediction_head_spec.md docs/components/factorized-prediction-head.md
mv docs/components/simsiam_consistency_spec.md docs/components/simsiam-consistency.md
mv docs/components/sampled_mcts_spec.md docs/components/sampled-mcts.md
mv docs/components/muzero_core_chassis_spec.md docs/components/muzero-core-chassis.md
mv docs/components/per_buffer_reanalyze_spec.md docs/components/per-buffer-reanalyze.md
mv docs/components/trainer_loop_ema_spec.md docs/components/trainer-loop-ema.md
mv docs/components/league_tracker_pfsp_spec.md docs/components/league-tracker-pfsp.md
mv docs/components/consolidated_system_spec.md docs/components/consolidated-system.md
```

## Advanced Configuration

### Custom Domain
```bash
# Add to docs/CNAME
docs.kaggriculture-muzero.com
```

### Analytics (Google Analytics 4)
```yaml
# In mkdocs.yml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
```

### Versioning (mike)
```bash
pip install mike
mike deploy --push --update-aliases 1.0 latest
mike set-default latest
```

### PDF Export
```bash
pip install mkdocs-pdf-export-plugin
# Add to mkdocs.yml plugins:
plugins:
  - pdf-export:
      enabled: true
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails on `git-revision-date` | Ensure `fetch-depth: 0` in checkout |
| Pages not updating | Check Actions tab; verify `gh-pages` branch exists |
| Navigation broken | Verify file paths in `nav` match actual files |
| Theme not loading | Clear browser cache; check `site_url` in mkdocs.yml |

## Code References

- `mkdocs.yml` - Configuration (create at root)
- `.github/workflows/docs.yml` - CI/CD (create)
- `docs/` - Source folder (already structured)