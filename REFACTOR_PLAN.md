

 
# Refactoring Guide: Towards Canonical Kaggriculture MuZero

This guide details the structural and architectural migration from a heuristic-dominant framework to a mathematically canonical MuZero implementation. The primary objective is isolating the analytical rules engine, moving MCTS entirely into the latent space, and enabling continuous online fine-tuning via LoRA.

## Phase 1: Repository Scaffolding

Execute the following bash script to generate the canonical directory topology and stub the necessary files. This will establish the strict boundaries required for the new architecture.

```bash  
#!/usr/bin/env bash

echo "Scaffolding Canonical MuZero Topology..."

# 1. Core RL Engine  
mkdir -p muzero  
touch muzero/__init__.py  
touch muzero/_core.py       # h(s), g(s,a), f(s) networks  
touch muzero/economy.py     # Analytical teacher (isolated for Phase 1)  
touch muzero/encoding.py    # Tensor serialization  
touch muzero/lora.py        # LoRA injections (r=16)  
touch muzero/meta.py        # Hyperparameters (d=256)  
touch muzero/types.py       # Discrete support (B=601)

# 2. 4-Phase Training Curriculum  
mkdir -p pipelines/phases  
touch pipelines/__init__.py  
touch pipelines/train_muzero.py  
touch pipelines/replay_manager.py  
touch pipelines/phases/phase1_bootstrap.py  
touch pipelines/phases/phase2_distill.py  
touch pipelines/phases/phase3_bc.py  
touch pipelines/phases/phase4_reanalysis.py

# 3. Action Masking & Macro Options  
mkdir -p hybrid_chassis/layers  
touch hybrid_chassis/__init__.py  
touch hybrid_chassis/pipeline.py

# 4. Tournament Engine & Gating  
mkdir -p evaluation  
touch evaluation/__init__.py  
touch evaluation/league.py  
touch evaluation/harness.py  
touch evaluation/benchmarks.py

# 5. Model Weights & Telemetry  
mkdir -p artifacts/snapshots  
mkdir -p artifacts/lora_adapters

# 6. Prioritized Replay Storage  
mkdir -p replays

echo "Topology generated successfully."
```

## **Phase 2: Core Engine Migration (muzero/)**

The core engine must be stripped of all environment-specific game logic outside of tensor encoding and the Phase 1 teacher.

* **_core.py**: Migrate your neural architectures here. Consolidate into three explicit classes: RepresentationNetwork, DynamicsNetwork, and PredictionNetwork. Ensure the dynamics network relies solely on the latent dimension ($d=256$) and never calls out to the environment state.  
* **economy.py**: Isolate your existing analytical simulator here. It must **only** be imported and utilized by phase1_bootstrap.py to generate supervised targets.  
* **lora.py**: Implement the adapter logic. Freeze the base weights of your dynamics and prediction layers, and inject rank $r=16$ trainable matrices for continuous learning.  
* **types.py**: Implement the value-equivalent discrete support transformations. Set $B=601$ for your categorical bins.

## **Phase 3: Curriculum Pipeline (pipelines/)**

Break your monolithic training loop into four strictly separated execution phases orchestrated by train_muzero.py.

* **Phase 1 (Bootstrap)**: Route raw observations through economy.py to pre-train latent representations and dynamics before attempting any tree search.  
* **Phase 2 (Distill)**: Initialize Gumbel MCTS. Run the search against the pre-trained networks and distill the resulting policy/value targets back into the prediction head.  
* **Phase 3 (Behavior Cloning)**: Wire replay_manager.py to ingest expert .json replays from the replays/ folder. Use Prioritized Experience Replay (PER) to map high-ELO observation sequences to macro-actions.  
* **Phase 4 (Re-analysis)**: Build the continuous self-play loop. Re-evaluate historical trajectories with current weights to refresh MCTS targets, updating *only* the LoRA adapters stored in artifacts/lora_adapters/.

## **Phase 4: Chassis Reduction (hybrid_chassis/)**

The legacy hybrid chassis must be downgraded from a decision-maker to a state-router and constraint-enforcer.

* **pipeline.py**: Strip out heuristic planning logic. This file should now only sequence the translation of MuZero's abstract macro-options into raw Kaggle environment actions.  
* **Action Masking**: Move all hard survival rules (e.g., preventing zero-water states) into the chassis layers. These rules will output a mask that sets invalid action logits to $-infty$ before the MuZero policy head applies its softmax activation.

## **Phase 5: Evaluation & Artifacts (evaluation/, artifacts/)**

* **benchmarks.py**: Set up automated gating. New candidate weights should automatically run a head-to-head match against the benchmark 2033 ELO champion.  
* **Artifact Routing**: Configure your pipelines so that snapshots/ only receives frozen base weights, while lora_adapters/ receives the lightweight, continuous updates from Phase 4.