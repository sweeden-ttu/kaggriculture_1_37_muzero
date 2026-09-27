# Muzero Core Chassis - Architecture Specification

**Source**: muzero/chassis.py, kaggriculture_muzero_chassis.py, muzero/_core.py
**Documentation**: docs/MuZero_Core.md
**Generated**: 2026-09-27T03:38:05.455282

## Description
Unified spatial ResNet chassis: h_θ (representation), g_θ (dynamics), f_θ (prediction) networks

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- h_θ: SpatialRepresentationNetwork - Conv2d(28→64) + 3 ResNet blocks → L2 norm
- g_θ: SpatialDynamicsNetwork - Conv2d(80→64) + 3 ResNet blocks → reward_head + L2 norm
- f_θ: SpatialPredictionNetwork - Conv2d(64→32) + FC(3200→256) → 3 policy heads + value head
- initial_inference: obs → h_θ → f_θ
- recurrent_inference: (s, action_plane) → g_θ → f_θ
- DiscreteSupport: 601 bins, B=300, φ(x)=sign(x)(√(|x|+1)-1+ε|x|)

## Testing Approach
- Unit test: initial_inference(obs[B,28,10,10]) → s_0[B,64,10,10], preds, v_scalar
- Unit test: recurrent_inference(s[B,64,10,10], action[B,16,10,10]) → s_next, r, preds, v
- Verify L2 normalization on latent states (dim=1)
- Verify DiscreteSupport scalar conversion matches φ⁻¹(softmax(logits) @ support)
- Test LoRA injection: inject_lora(model, rank=4) adds trainable adapters

## Evaluation Metrics
- Parameter count: h_θ/g_θ/f_θ total ~1.2M params
- Inference latency: initial_inference + recurrent_inference on M1/M2
- Gradient flow: verify all params receive gradients in K-step unroll
- Checkpoint compatibility: load/save round-trip preserves weights