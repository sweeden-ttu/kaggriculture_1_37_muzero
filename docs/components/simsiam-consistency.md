# Simsiam Consistency - Architecture Specification

**Source**: muzero/chassis.py, simsiam_consistency.py
**Documentation**: docs/Consistency_Loss.md
**Generated**: 2026-09-27T03:38:05.433329

## Description
Asymmetric SimSiam projection/prediction heads with explicit stop-gradient for latent consistency

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Projection head: 3-layer MLP (latent→proj→proj→proj) with BatchNorm
- Prediction head: 2-layer bottleneck MLP (proj→pred→proj)
- Stop-gradient on target branch: target_z = proj(h_θ(o)).detach()
- Loss: -cosine_similarity(predict(proj(s_unrolled)), target_z)
- Weight λ_cons ∈ [0.1, 0.5] (Regime 1) or [0.5, 1.0] (Regime 2)

## Testing Approach
- Unit test: SimSiamConsistencyLoss forward returns scalar loss ∈ [-1, 0]
- Verify stop-gradient: target branch has no grad_fn
- Verify predictor head breaks symmetry (non-zero grad wrt unrolled latent)
- Integration: compute_muzero_unroll_loss_with_consistency() runs K-step loop
- Test L2 normalization prevents collapse to constant vectors

## Evaluation Metrics
- Consistency loss magnitude: track λ_cons * L_cons vs other losses
- Latent drift: cosine similarity between h_θ(o_{t+k}) and g_θ^k(s_0)
- Representation collapse check: variance of projected embeddings > threshold
- Ablation: training with/without consistency loss