# Per Buffer Reanalyze - Architecture Specification

**Source**: muzero/buffer.py, prioritized_muzero_buffer.py, muzero_replay_buffer.py
**Documentation**: docs/prioritized_muzero_system.py, docs/MCTS_Search_Tree.md
**Generated**: 2026-09-27T03:38:05.466444

## Description
Trajectory-based PER buffer with K-step unroll, n-step bootstrap, Reanalyze target refreshing

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- GameTrajectory stores: obs, actions, rewards, policies (visit dist), values, priorities
- K-step slice sampling: random trajectory, random start, extract K+1 obs, K actions
- n-step bootstrap targets: z_t = Σγ^j r_{t+j} + γ^n v_{t+n}
- Factorized policy targets: marginalize joint visit dist to 3 sub-heads
- PER: priority = max_k |v_{t+k} - z_{t+k}| + ε, P(i) ∝ p_i^α
- IS weights: w_i = (N·P(i))^{-β}, β annealed 0.4→1.0
- Reanalyze: re-run MCTS on stored obs with latest weights, overwrite π_t, v_t

## Testing Approach
- Unit test: sample_k_step_batch() returns dict with correct tensor shapes
- Verify n-step targets use discount γ=0.997 correctly
- Test factorized target conversion: joint dict → 3 marginal arrays
- PER test: priorities updated after train_step, sampling probability ∝ p^α
- Reanalyze test: policies/values overwritten, priorities reset
- Integration: buffer + trainer + mcts end-to-end training step

## Evaluation Metrics
- Buffer utilization: trajectory count vs max_trajectories
- Priority distribution: histogram of TD-errors
- Reanalyze frequency: target staleness vs compute budget
- Sample efficiency: loss reduction per environment step