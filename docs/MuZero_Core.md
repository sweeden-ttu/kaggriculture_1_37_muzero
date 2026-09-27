# MuZero Core (Canonical)

Source of truth: [`muzero/_core.py`](../muzero/_core.py), [`muzero/types.py`](../muzero/types.py),
[`muzero/encoding.py`](../muzero/encoding.py), [`muzero/meta.py`](../muzero/meta.py).

This repository’s production MuZero is a **macro-option MLP** stack — not a spatial
Conv Sampled MuZero. The deferred spatial design lives in
[ADR 0001](adr/0001-spatial-sampled-muzero-deferred.md).

## Decomposition

Three learned networks, plus latent MCTS over eight daily macros:

1. **Representation `h_θ` (`RepresentationNetwork`)**  
   Maps a 32-dim observation vector to a latent state `s⁰ ∈ ℝ^{256}`.

2. **Dynamics `g_θ` (`DynamicsNetwork`)**  
   Given `(s^{k-1}, a^k)` with `a` as a one-hot over 8 `MacroOption`s, predicts
   reward support logits and next latent `s^k`.

3. **Prediction `f_θ` (`PredictionNetwork`)**  
   Given `s^k`, predicts policy logits over 8 macros and value support logits.

Unified container: **`MuZeroNetwork`** with `initial_inference` / `recurrent_inference`
(and `*_logits` variants used by the K-step training loss).

## Hyperparameters (locked)

| Symbol | Value | Constant |
|---|---|---|
| Observation dim | 32 | `OBS_DIM` |
| Latent dim `d` | 256 | `HIDDEN_DIM` |
| Actions | 8 macros | `NUM_MACRO_OPTIONS` / `MacroOption` |
| Discrete support | 601 bins, `B=300` | `SUPPORT_SIZE`, `SUPPORT_B` |
| LoRA rank (Phase 4) | 16 | `muzero.meta.LORA_RANK` |

## Network shapes

### RepresentationNetwork

```
Linear(32 → 128) → ReLU → Linear(128 → 256) → Tanh
```

Input: `[B, 32]` (or `[32]` auto-unsqueezed).  
Output: latent `[B, 256]`.

### DynamicsNetwork

- Concatenate latent with one-hot action: `[B, 256 + 8]`
- Shared MLP → reward head `Linear → SUPPORT_SIZE` and next-state head `→ 256 + Tanh`
- `forward` returns `(reward_scalar, next_latent)` via `DiscreteSupport.support_to_scalar`
- `forward_logits` returns `(reward_logits, next_latent)` for categorical training

### PredictionNetwork

- Shared `Linear(256 → 128) → ReLU`
- Policy head → 8 logits
- Value head → `SUPPORT_SIZE` logits
- `forward` returns `(policy_logits, value_scalar)`; `forward_logits` keeps value categorical

### MuZeroNetwork API

| Method | Returns |
|---|---|
| `initial_inference(obs)` | `(latent, policy_probs, value_scalar)` |
| `recurrent_inference(latent, action)` | `(reward_scalar, next_latent, policy_probs, value_scalar)` |
| `initial_inference_logits(obs)` | `(latent, policy_logits, value_logits)` |
| `recurrent_inference_logits(latent, action)` | `(reward_logits, next_latent, policy_logits, value_logits)` |

## Discrete support

`DiscreteSupport` in `muzero/types.py` uses invertible transform φ:

- `φ(x) = sign(x)·(√(|x|+1) − 1 + ε·|x|)`
- Targets are projected onto two adjacent bins; heads train with categorical CE
- Scalars recovered with `support_to_scalar` (softmax expectation then `φ⁻¹`)

Support span after φ is `[-300, 300]`. Environment wealth deltas are scaled by
`REWARD_SCALE` (`1/1000`) before targets enter the support path.

## Inference vs search

- **Root:** `h_θ(o)` then `f_θ(s⁰)` — observation legality mask applied at the root only.
- **Interior:** `g_θ` only (no analytical economy / env step inside the tree).
- Planner facade: `MuZeroPlanner` wrapping `LatentMacroOptionMCTS`
  (see [MCTS_Search_Tree.md](MCTS_Search_Tree.md)).

## Chassis handoff

Macro indices are translated to raw Kaggle actions by
`hybrid_chassis/pipeline.py` (`translate_macro_option`) with survival masks from
`hybrid_chassis/layers/action_masking.py` (invalid logits → −∞ before softmax).
