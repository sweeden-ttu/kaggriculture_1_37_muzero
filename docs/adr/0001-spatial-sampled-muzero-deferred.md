# ADR 0001: Spatial Sampled MuZero is Production

- **Status:** Accepted
- **Date:** 2026-09-27
- **Supersedes:** Deferred draft that kept spatial Sampled MuZero off the train path

## Context

The repository previously trained an MLP / 8-`MacroOption` MuZero
(`muzero.legacy_macro`). Specs in `docs/` and the reference implementation
described a spatial Sampled MuZero stack (10×10×28 obs, factorized
farmer/hands/market heads, Sampled MCTS, SimSiam, trajectory PER).

## Decision

**Spatial Sampled MuZero is the production architecture.**

Canonical modules:

| Component | Module |
|---|---|
| Chassis h/g/f | `muzero.chassis.KaggricultureMuZeroChassis` |
| Observation | `muzero.observation_encoder.encode_spatial_observation` |
| Action planes | `muzero.spatial_constants.joint_action_to_plane` (`ACTION_CHANNELS=67`) |
| MCTS | `muzero.mcts.SampledMuZeroMCTS` |
| PER + trainer | `muzero.buffer.PrioritizedMuZeroBuffer`, `muzero.trainer.PrioritizedMuZeroTrainer` |
| Train entry | `pipelines/train_muzero.py --arch spatial` (default) |
| Checkpoint | `artifacts/muzero_spatial_checkpoints.pt` |

Legacy MLP / macro-option MuZero remains under `muzero.legacy_macro` and is
reachable via `--arch macro` for packaging/ablation only.

## Consequences

- Spatial is production for **both training and inference** (`hybrid_chassis/layers/muzero_options.py`
  loads `KaggricultureMuZeroChassis` + `SampledMuZeroMCTS` + `translate_joint_action`).
- New spatial checkpoints are **not** loadable into old `MuZeroNetwork` weights; packaging
  refuses to embed macro MLP weights under spatial names.
- Promote copies `muzero_spatial_checkpoints.pt` → `muzero_checkpoints*.pt` only after a
  successful spatial load probe.
- Action-plane encoding is lossless (no `%5` aliasing).
- Soft MCTS visit targets use soft cross-entropy, not hard `F.cross_entropy`.
- Continuous loop entry: `make loop-spatial` / `pipelines.phases.phase_spatial_loop`.
- Local buoy gate: `make gate-buoy` / `scripts/gate_buoy.py`.

## Archive

Historical monolithic sketch:
[`archive/prioritized_muzero_system.py`](archive/prioritized_muzero_system.py)
(now implemented across `muzero/chassis.py`, `mcts.py`, `buffer.py`, `trainer.py`).
