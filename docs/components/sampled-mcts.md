# Sampled Mcts - Architecture Specification

**Source**: muzero/mcts.py, sampled_muzero_mcts.py
**Documentation**: docs/MCTS_Search_Tree.md
**Generated**: 2026-09-27T03:38:05.444520

## Description
Sampled MCTS in latent space with PUCT, MinMaxStats, Dirichlet noise, factorized action sampling

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Root: initial_inference(obs) → s_0, priors, root_value
- Sample K=8 joint actions from factorized heads with Dirichlet noise
- Selection: PUCT with MinMaxStats Q-normalization
- Expansion: recurrent_inference(s, action_plane) → s_next, r, preds, v
- Backup: discounted value propagation, update MinMaxStats
- Output: visit-count policy π, best action, root value

## Testing Approach
- Unit test: search() returns (best_action, probs_dict, root_value)
- Verify PUCT selection prefers high prior + high value
- Verify MinMaxStats normalizes Q-values across tree
- Test Dirichlet noise at root adds exploration
- Integration: chassis.recurrent_inference called for each expansion
- Stress test: num_simulations=50 completes in <1s on CPU

## Evaluation Metrics
- Search quality: root value vs final game outcome correlation
- Policy improvement: visit-count policy vs raw network policy (KL divergence)
- Simulation efficiency: nodes/sec, latency per search
- Ablation: num_samples (K=4,8,16), num_simulations (25,50,100)