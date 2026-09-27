# Consolidated System - Architecture Specification

**Source**: pipelines/train_muzero.py, pipelines/low_rank_adaptation.py, pipelines/adaptive_moment_estimation.py, pipelines/replay_manager.py, pipelines/self_improving_loop.py, train.sh, boost.sh
**Documentation**: docs/prioritized_muzero_system.py, AGENTS.md
**Generated**: 2026-09-27T03:38:05.499240

## Description
End-to-end training pipeline: 4-phase curriculum, LoRA fine-tuning, continuous self-play loop

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Phase 1 (BC): Behavioral cloning on gold replays with PER
- Phase 2 (Bootstrap): Analytical unroll targets, consistency loss
- Phase 3 (MCTS Distill): Gumbel MCTS targets distilled back to network
- Phase 4 (LoRA/PPO): LoRA rank-16 adapters on frozen champion + PPO microcontroller
- train.sh: standard loss schedule (consistency 0.1→0.5)
- boost.sh: inverted schedule (consistency 1.0→0.5, antagonistic gradients)
- Continuous loop: self-play → buffer → train → eval → promote champion
- Discrete head exports: policy_head.pt, value_head.pt per checkpoint
- Submission archetypes: head_pl (pure replay), head_hybrid, head_muzero

## Testing Approach
- Integration test: train.sh runs 1 epoch without errors
- Verify checkpoint saves: muzero_checkpoints.pt + policy_head.pt + value_head.pt
- Test LoRA fine-tune: --use-lora --lora-rank 4 freezes backbone
- Test submission packaging: build_head_submissions.py creates 3 zips
- Tournament test: benchmark against baselines/ and /Volumes/BASELINES/muzero/baselines
- Score validation: candidate scores vs Kaggle-scored baselines (2021-2033)

## Evaluation Metrics
- Kaggle submission scores: public/private leaderboard
- Tournament vs baselines: head-to-head win rates
- Self-play Elo progression over iterations
- Replay buffer rotation: canonical replay immunity, disk budget adherence
- Champion promotion gate: eval-episodes threshold
- LoRA merge: adapter weights folded into base for deployment