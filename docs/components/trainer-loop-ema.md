# Trainer Loop Ema - Architecture Specification

**Source**: muzero/trainer.py, muzero_trainer_loop.py
**Documentation**: docs/MCTS_Search_Tree.md, docs/prioritized_muzero_system.py
**Generated**: 2026-09-27T03:38:05.476951

## Description
Trainer with Polyak EMA target network, IS-weighted PER loss, SimSiam consistency, Reanalyze passes

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Dual network: online (θ) + target (θ⁻) with Polyak EMA τ=0.995
- Joint optimizer over online model + SimSiam heads (AdamW, lr=3e-4)
- K-step unroll loss: policy + 0.25*value + reward + λ_cons*consistency
- PER: sample batch with IS weights, scale per-sample loss, update priorities
- Reanalyze pass: periodic (e.g. every 100 steps) refresh stale targets
- CosineAnnealingLR for uniform trainer, constant LR for PER trainer

## Testing Approach
- Unit test: train_step() returns loss dict with all components
- Verify EMA update: target params = τ*target + (1-τ)*online
- Test IS weighting: per-sample loss scaled by is_weights
- Verify gradient clipping at grad_clip=5.0
- Test Reanalyze pass updates buffer targets in-place
- Integration: 3 training steps reduce total loss

## Evaluation Metrics
- Training curves: policy/value/reward/consistency loss over steps
- EMA tracking: online vs target network weight distance
- Learning rate schedule adherence
- Gradient norm statistics
- Reanalyze impact: loss on refreshed vs stale targets