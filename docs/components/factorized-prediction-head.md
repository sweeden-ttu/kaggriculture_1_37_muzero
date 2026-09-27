# Factorized Prediction Head - Architecture Specification

**Source**: muzero/chassis.py, kaggriculture_muzero_chassis.py
**Documentation**: docs/FACTORIZED_PREDICTION_HEAD.md
**Generated**: 2026-09-27T03:38:05.423375

## Description
Factorized policy heads (Farmer 15, Hands 32, Market 20) with joint action sampling

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Shared trunk processes latent state (64 channels, 10x10) to 256-dim features
- Three independent linear heads: farmer(15), hands(32), market(20)
- Value head outputs 601-bin distributional support
- sample_joint_actions() draws K candidates for MCTS expansion
- Loss: sum of 3 cross-entropy terms + value MSE

## Testing Approach
- Unit test: forward() returns dict with 4 logit tensors
- Verify shapes: farmer[B,15], hands[B,32], market[B,20], value[B,601]
- Test sample_joint_actions(): returns [B, K, 3] actions + [B, K] log_probs
- Verify masking: masked logits → -inf, softmax → 0 probability
- Gradient check: backward() flows through all three heads independently

## Evaluation Metrics
- Policy accuracy: cross-entropy vs MCTS visit distributions
- Value accuracy: MSE vs bootstrapped n-step returns
- Joint action diversity: entropy across K sampled candidates
- Sub-head correlation: measure independence assumption validity