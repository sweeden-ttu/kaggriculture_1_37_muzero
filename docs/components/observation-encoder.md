# Observation Encoder - Architecture Specification

**Source**: muzero/observation_encoder.py, kaggriculture_observation_encoder.py
**Documentation**: docs/Observation_Encoder.MD
**Generated**: 2026-09-27T03:38:05.412053

## Description
Fixed maximum-grid 10x10 spatial feature encoder producing 28-channel tensors

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- Encode raw game observations to 28-channel 10x10 spatial tensors
- Used as input to RepresentationNetwork (h_θ) during initial_inference
- Action masks generated per factorized head (farmer/hands/market)

## Testing Approach
- Unit test: encode() produces [1, 28, 10, 10] tensor
- Verify channel semantics: quadrant mask, crop types, livestock, units, global context
- Test action masks: land expansion gating, price thresholds
- Integration: feed into chassis.initial_inference() without shape errors

## Evaluation Metrics
- Encoding fidelity: reconstruction accuracy of key game state features
- Action mask coverage: % of illegal actions correctly masked
- Latency: <5ms per encode on CPU