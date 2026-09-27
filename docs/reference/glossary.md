# Glossary

## A

**Action Plane** — 16-channel 10×10 spatial tensor encoding a joint action (farmer+hands+market) for MCTS expansion. Channels 0-4: farmer one-hot, 5-9: hands one-hot, 10-15: market one-hot.

**AdamW** — Adam optimizer with decoupled weight decay. Used for all training with lr=3e-4 (uniform) or 1e-3 (PER), weight_decay=1e-4.

**AlphaStar** — DeepMind's StarCraft II agent that introduced League Training and PFSP. Our league system is based on this architecture.

---

## B

**Backpropagation (MCTS)** — Discounted value propagation from leaf to root: `value = reward + γ × value`. Updates visit counts and value sums along search path.

**Baseline** — A reference agent with known performance. Two tiers: Local (top-8 candidates, rapid gating) and Gold (Kaggle-scored, certification).

**Behavioral Cloning (BC)** — Phase 1 supervised training on expert replay demonstrations. Establishes initial policy priors.

**Boost Schedule** — Inverted loss schedule (consistency 1.0→0.5, antagonistic gradients). Implemented in `adaptive_moment_estimation.py`, launched via `boost.sh`.

**Buffer** — See Replay Buffer.

---

## C

**Canonical Replay** — Gold-standard Kaggle replays (`^[0-9]+\.json$`, 50 files). Pruning-immune, permanent fixtures in replay directory.

**Champion** — The currently promoted best model. Used for self-play generation and submission packaging. Stored at `artifacts/muzero_checkpoints_champion.pt`.

**Checkpoint** — Serialized model state dict. Types: champion (promoted), candidate (eval), snapshot (periodic), LoRA adapters.

**Chassis** — Two meanings:
1. **MuZero Chassis** (`KaggricultureMuZeroChassis`): Unified h_θ, g_θ, f_θ container.
2. **Reactive Chassis** (`hybrid_chassis.core.Chassis`): Route-replay + safety layers.

**Consistency Loss** — SimSiam self-supervised loss aligning unrolled latents g_θ^k(s_0) with encoded observations h_θ(o_{t+k}). Weight λ_cons.

**Continuous Loop** — Autonomous training: Self-Play → Buffer → Train → Evaluate → Promote → Repeat.

**Cosine Annealing** — Learning rate schedule for uniform trainer: CosineAnnealingLR(T_max=1000, eta_min=1e-5).

---

## D

**Dirichlet Noise** — Exploration noise at MCTS root: `priors = (1-ε)×softmax(log_probs) + ε×Dirichlet(α)`. α=0.25, ε=0.25.

**Discrete Support** — Categorical value/reward representation with 601 bins spanning [-300, 300] after φ-transform. φ(x) = sign(x)(√(|x|+1)-1+ε|x|).

**Dynamics Network (g_θ)** — Predicts next latent state s^k and reward distribution from (s^{k-1}, action). Spatial ResNet.

---

## E

**EMA (Exponential Moving Average)** — Target network update: θ⁻ ← τ×θ⁻ + (1-τ)×θ with τ=0.995 (Polyak averaging).

**Evaluation** — Tournament benchmarking against baselines. Two tiers: local (4 games) and gold (8-16 games).

**Exploiter** — League agent role targeting specific opponents. Types: main_exploiter (vs current main), league_exploiter (broad).

---

## F

**Factorized Policy Head** — Three independent policy sub-heads (Farmer 15, Hands 32, Market 20) instead of flat 9,600-dim joint space. Enables tractable MCTS.

**FIFO** — First-In-First-Out eviction for replay buffer when max_trajectories reached.

---

## G

**GameTrajectory** — Complete episode storage: observations, actions, rewards, MCTS policies, values, priorities.

**Gold Baseline** — Kaggle-scored submission zips in `/Volumes/BASELINES/muzero/baselines/` (scores 771-2033).

**Gradient Clipping** — max_norm=5.0 applied before optimizer step.

---

## H

**Hard PFSP** — PFSP mode weighting opponents by (1 - win_rate)^p. Focuses on difficult opponents (low win rate).

**Hybrid Submission** — `head_hybrid_submission.zip`: 2033 reactive chassis + MuZero NeML opening/SE eval/trickle gate.

---

## I

**Importance Sampling (IS)** — PER bias correction weights: w_i = (N×P(i))^{-β} / max(w). β annealed 0.4→1.0.

**Initial Inference** — Root MCTS step: h_θ(obs) → s_0, then f_θ(s_0) → policy + value.

---

## K

**K-Step Unroll** — Recurrent training unroll over K=5 steps: s_0 → g_θ → f_θ → ... → s_K.

**Kaggriculture** — Kaggle competition: agricultural simulation with farmers, crops, livestock, markets.

---

## L

**Laplace Smoothing** — Win rate estimation: (wins + 1) / (games + 2). Ensures 0.5 for unplayed matchups.

**Latent Drift** — Divergence between h_θ(o_{t+k}) and g_θ^k(h_θ(o_t)). Prevented by SimSiam consistency.

**League** — Collection of frozen checkpoints with pairwise win-rate matrix. Enables PFSP.

**LoRA (Low-Rank Adaptation)** — Rank-r adapters injected into frozen champion weights. r=16, α=32. Only adapters trained.

**LoRA Merge** — Folding adapter weights into base model for standalone deployment: W' = W + (α/r)×A×B.

---

## M

**Macro Option** — One of 8 high-level daily strategies (legacy MLP MuZero). Replaced by spatial factorized actions.

**Main Agent** — Primary league learner: 35% self-play, 50% hard PFSP, 15% var PFSP.

**MinMaxStats** — Global Q-value tracking across MCTS tree for PUCT normalization.

**MCTS (Monte Carlo Tree Search)** — Sampled MuZero MCTS in latent space with factorized action sampling.

**MCTS Distillation** — Phase 3: Use MCTS visit distributions as training targets for policy/value networks.

---

## N

**NeML** — Net Economic Marginal Loss: Step-1 wheat arbitrage opening (+$20-25). Learned by MuZero.

**n-Step Bootstrap** — Value target: z_t = Σ_{j=0}^{n-1} γ^j r_{t+j} + γ^n v_{t+n}. n=5.

---

## O

**Observation Encoder** — Fixed 28-channel 10×10 spatial feature extractor from raw Kaggle dict.

---

## P

**PFSP (Prioritized Fictitious Self-Play)** — AlphaStar opponent sampling: hard/var/uniform modes based on win rates.

**PER (Prioritized Experience Replay)** — Sampling proportional to TD-error^α (α=0.6). IS weights for bias correction.

**Policy Distillation** — Cross-entropy from network logits to MCTS marginal visit distributions.

**Polyak Averaging** — EMA target network update with τ=0.995.

**Prediction Network (f_θ)** — Factorized policy heads + value head from latent state.

**Promotion Gate** — Candidate must achieve ≥55% win rate vs champion over 8 eval episodes.

**PUCT** — Predictor Upper Confidence Bound: Q_norm + P × √(ΣN)/(1+N) × (c1 + log((ΣN+c2+1)/c2)).

**Pure Replay Submission** — `head_pl_submission.zip`: Zero neural, distilled expert lessons only.

**Pure MuZero Submission** — `head_muzero_submission.zip`: Full neural pipeline, champion weights.

---

## R

**Reanalyze** — Periodic MCTS re-search on stored observations to refresh stale policy/value targets.

**Recurrent Inference** — MCTS tree step: g_θ(s, action) → s_next, then f_θ(s_next) → policy, value.

**Replay Buffer** — Trajectory-based storage supporting K-step slicing, n-step bootstrap, PER, Reanalyze.

**Representation Network (h_θ)** — Encodes 28-channel observation to 64-channel latent state. Spatial ResNet.

**Reward Scale** — 1/1000 scaling before discrete support projection.

---

## S

**Sampled MCTS** — Expands only K=8 sampled joint actions per node (not full action space).

**Self-Play** — Agent plays against itself or league opponents to generate training data.

**SimSiam** — Asymmetric self-supervised consistency: stop-gradient on target branch, predictor head on predictor branch.

**Submission Archetype** — One of three canonical packages: PL (pure replay), HPM (hybrid), MZ (pure MuZero).

---

## T

**Target Network** — EMA copy of online model (θ⁻). Used for bootstrapped value targets and consistency targets.

**TD-Error** — Temporal Difference error: |v_predicted - z_target|. Used for PER priorities.

**Trainer** — Coordinates training step: sample batch → K-step unroll → loss → backward → EMA → priority update.

---

## U

**Unroll** — See K-Step Unroll.

---

## V

**Var PFSP** — PFSP mode weighting by win_rate × (1 - win_rate). Focuses on ~50% win rate opponents.

**Value Distillation** — MSE from network value to MCTS root value (with n-step bootstrap).

---

## W

**Win Rate (Smoothed)** — Laplace-smoothed: (wins + 1) / (games + 2). Used for PFSP weights.

---

## Code References

All terms link to implementations in:
- `muzero/observation_encoder.py`
- `muzero/chassis.py`
- `muzero/mcts.py`
- `muzero/buffer.py`
- `muzero/trainer.py`
- `muzero/lora.py`
- `muzero/ppo.py`
- `evaluation/league.py`
- `pipelines/low_rank_adaptation.py`
- `pipelines/adaptive_moment_estimation.py`
- `pipelines/replay_manager.py`
- `pipelines/self_improving_loop.py`
- `hybrid_chassis/core/chassis.py`
- `hybrid_chassis/pipeline.py`
- `scripts/build_head_submissions.py`